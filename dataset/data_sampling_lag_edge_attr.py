# -*- coding: utf-8 -*-
import os
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

# =========================================================
# 工具：鲁棒读取 CSV，返回 ndarray [T, D]
# =========================================================
def _robust_load_csv(path: str) -> np.ndarray:
    try:
        arr = np.genfromtxt(path, delimiter=',', skip_header=1)
        if np.isnan(arr).all():
            arr = np.genfromtxt(path, delimiter=',', skip_header=0)
    except Exception:
        try:
            arr = np.loadtxt(path, delimiter=',')
        except Exception:
            df = pd.read_csv(path)
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                raise RuntimeError(f"[CSV] 文件 {path} 无数值列")
            arr = df[num_cols].to_numpy()
    arr = np.array(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr.astype(np.float32, copy=False)

def _pad_or_trunc_cols(arr: np.ndarray, target_cols: int) -> np.ndarray:
    cur = arr.shape[1]
    if cur == target_cols:
        return arr
    if cur > target_cols:
        return arr[:, :target_cols]
    pad_w = target_cols - cur
    return np.pad(arr, ((0, 0), (0, pad_w)), mode='constant', constant_values=0.0)

def _zscore_per_row(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-6
    return x / std

# =========================================================
# 构建同模态相关性边
# =========================================================
def _pairwise_topk_corr(X: np.ndarray, topk: int, min_abs_corr: float) -> Tuple[np.ndarray, np.ndarray]:
    N, T = X.shape
    if N == 0 or T < 2 or topk <= 0:
        return np.zeros((2, 0), np.int64), np.zeros((0,), np.float32)
    Xz = X - X.mean(axis=1, keepdims=True)
    Xs = Xz.std(axis=1, keepdims=True) + 1e-8
    Xn = Xz / Xs
    C = (Xn @ Xn.T) / max(1, T - 1)
    np.fill_diagonal(C, 0.0)
    edges = []
    for i in range(N):
        idx = np.argsort(-np.abs(C[i]))[:topk]
        for j in idx:
            if i < j and abs(C[i, j]) >= min_abs_corr:
                edges.append((i, j, C[i, j]))
    if not edges:
        return np.zeros((2, 0), np.int64), np.zeros((0,), np.float32)
    ei = np.array([[i, j] for i, j, _ in edges], np.int64).T
    ew = np.array([w for _, _, w in edges], np.float32)
    return ei, ew

# =========================================================
# TFF 数据集工具
# =========================================================
_TFF_LABEL_ORDER = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
_TFF_LABEL2ID = {k: i for i, k in enumerate(_TFF_LABEL_ORDER)}

def _scan_tff_folders(root: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"[TFF] 根目录不存在: {root}")
    res = []
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        lab = sub.strip().lower()
        if lab not in _TFF_LABEL2ID:
            lab = 'others'
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith('.csv'):
                res.append((os.path.join(d, fn), lab))
    if not res:
        raise RuntimeError(f"[TFF] 未发现任何 CSV: {root}")
    return res

def _read_one_csv_align_numeric(csv_path: str, op_interval: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if 'time' not in df.columns:
        df.insert(0, 'time', np.arange(len(df)) * float(op_interval))
    keep = ['time'] + [c for c in df.columns if c != 'time' and pd.api.types.is_numeric_dtype(df[c])]
    df = df[keep].copy()
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
    return df

def _union_columns(dfs: List[pd.DataFrame]) -> List[str]:
    cols = set()
    for df in dfs:
        cols |= set([c for c in df.columns if c != 'time'])
    return sorted(cols)

def _df_to_np(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    T = len(df)
    X = np.zeros((len(cols), T), np.float32)
    for i, c in enumerate(cols):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors='coerce').to_numpy(np.float32)
            v[np.isnan(v)] = 0.0
            X[i] = v
    t = df['time'].to_numpy(np.float32)
    return X, t

# =========================================================
# 主数据集类
# =========================================================
class MultiModalHeteroDatasetV2(Dataset):
    def __init__(self,
                 dataset: str = 'NPS',
                 op_dir: str = 'Data/Operation_csv_data',
                 dose_dir: str = 'Data/Dose_csv_data',
                 op_interval: int = 10,
                 dose_interval: int = 60,
                 dose_window_steps: int = 10,
                 dose_stride_steps: int = 1,
                 task_type: str = 'multiclass',
                 clip_val: float = 6.0,
                 op_topk: int = 10,
                 dose_topk: int = 10,
                 min_abs_corr: float = 0.2,
                 max_lag_dose: int = 5,
                 topk_cross: int = 0,
                 downsample_mode: str = 'pick'):
        super().__init__()
        self.dataset_type = dataset.upper()
        self.op_dir = op_dir
        self.dose_dir = dose_dir
        self.op_interval = int(op_interval)
        self.dose_interval = int(dose_interval)
        self.ratio = int(round(self.dose_interval / max(1, self.op_interval)))
        self.dose_steps = int(dose_window_steps)
        self.op_steps = self.dose_steps * self.ratio + (self.ratio - 1)
        self.dose_stride = int(dose_stride_steps)
        self.task_type = task_type
        self.clip_val = float(clip_val)
        self.op_topk = int(op_topk)
        self.dose_topk = int(dose_topk)
        self.min_abs_corr = float(min_abs_corr)
        self.max_lag_dose = int(max_lag_dose)
        self.topk_cross = int(topk_cross)
        assert downsample_mode in ('pick', 'mean')
        self.downsample_mode = downsample_mode

        # TFF 单模态判断
        self.tff_single = (self.dataset_type == 'TFF'
                           and self.topk_cross <= 0
                           and self.dose_topk <= 0)

        # —— 加载原始 NPS 或 TFF 数据 ——
        if self.dataset_type == 'TFF':
            self._load_tff()
        else:
            self._load_nps()

        # —— binary NPS 标签映射 ——
        if self.task_type == 'binary' and self.dataset_type == 'NPS':
            if 'Normal' not in self._label2id:
                raise KeyError("[NPS-binary] 找不到标签 'Normal'")
            safe_id = self._label2id['Normal']
            binary_samples = [0 if lab == safe_id else 1 for lab in self._labels]
            self._window_labels = [binary_samples[fidx] for fidx in self._window2file]
            self._label2id = {'safe': 0, 'unsafe': 1}
            self.id2label    = {0: 'safe', 1: 'unsafe'}

        # 构建滑窗索引
        self._build_windows()

        # 同步维度属性
        self.num_op   = getattr(self, 'num_op', getattr(self, 'op_dim', 0))
        self.num_dose = 0 if self.tff_single else getattr(self, 'num_dose', getattr(self, 'dose_dim', 0))
        self.op_dim   = self.num_op
        self.dose_dim = self.num_dose

        self.label2id = getattr(self, '_label2id', {})
        self.id2label = {v: k for k, v in self.label2id.items()}

        mode = "TFF-Single" if self.tff_single else self.dataset_type
        print(f"[DatasetV2-{mode}] windows={len(self._win_meta)} | "
              f"dose_steps={self.dose_steps} | op_steps={self.op_steps} (ratio={self.ratio}) | "
              f"N_op={self.num_op}, N_dose={self.num_dose} | labels={len(self.label2id)}")

    # ---------------- NPS 相关 ----------------
    def _collect_pairs_nps(self) -> List[Dict[str, str]]:
        pairs = []
        if not os.path.isdir(self.op_dir):
            raise FileNotFoundError(f"[NPS] OP 根目录不存在: {self.op_dir}")
        if not os.path.isdir(self.dose_dir):
            print(f"[NPS] WARN: Dose 根目录不存在，将使用 pseudo-dose: {self.dose_dir}")
        for label in sorted(os.listdir(self.op_dir)):
            op_sub = os.path.join(self.op_dir, label)
            dose_sub = os.path.join(self.dose_dir, label)
            if not os.path.isdir(op_sub):
                continue
            for fname in sorted(os.listdir(op_sub)):
                if not fname.lower().endswith('.csv'):
                    continue
                op_path = os.path.join(op_sub, fname)
                dose_path = None
                if os.path.isdir(dose_sub):
                    cand = os.path.join(dose_sub, fname.replace('.csv', 'dose.csv'))
                    if os.path.exists(cand):
                        dose_path = cand
                pairs.append({'op_path': op_path, 'dose_path': dose_path, 'label': label})
        return pairs

    def _load_nps(self):
        samples = self._collect_pairs_nps()
        print(f"[NPS] OP 文件数={len(samples)}, 真实 Dose={sum(1 for s in samples if s['dose_path'])}")
        if not samples:
            raise RuntimeError("[NPS] 未找到任何样本")
        uniq = sorted({s['label'] for s in samples})
        self._label2id = {lb: i for i, lb in enumerate(uniq)}
        self._labels = [self._label2id[s['label']] for s in samples]

        max_op = max((_robust_load_csv(s['op_path']).shape[1] for s in samples), default=0)
        max_dose = max((_robust_load_csv(s['dose_path']).shape[1]
                        for s in samples if s['dose_path']), default=0)
        if max_op == 0:
            raise RuntimeError("[NPS] 所有 OP 列数=0")
        self.num_op = max_op
        self.num_dose = max_dose or min(53, max_op)
        self.samples = samples

        # 构建窗口索引
        self._window2file = []
        self._window2dose_start = []
        for idx, s in enumerate(samples):
            if s['dose_path']:
                Td = _robust_load_csv(s['dose_path']).shape[0]
            else:
                Td = _robust_load_csv(s['op_path']).shape[0] // self.ratio
            max_start = Td - self.dose_steps
            if max_start < 0:
                continue
            for st in range(0, max_start + 1, self.dose_stride):
                self._window2file.append(idx)
                self._window2dose_start.append(st)
        self._window_labels = [self._labels[f] for f in self._window2file]

        # 全量拼接
        op_list, dose_list = [], []
        for s in samples:
            op_arr = _pad_or_trunc_cols(_robust_load_csv(s['op_path']), self.num_op)
            op_list.append(op_arr)
            if s['dose_path']:
                d_arr = _pad_or_trunc_cols(_robust_load_csv(s['dose_path']), self.num_dose)
            else:
                opA = op_arr
                T = opA.shape[0] // self.ratio
                if T <= 0:
                    d_arr = np.zeros((0, self.num_dose), np.float32)
                else:
                    if self.downsample_mode == 'pick':
                        tmp = opA[:T*self.ratio:self.ratio]
                    else:
                        tmp = opA[:T*self.ratio].reshape(T, self.ratio, self.num_op).mean(axis=1)
                    if self.num_dose <= self.num_op:
                        tmp = tmp[:, :self.num_dose]
                    else:
                        tmp = np.pad(tmp, ((0,0),(0,self.num_dose-self.num_op)), mode='constant')
                    d_arr = tmp
            dose_list.append(d_arr)
        self._op_all = np.concatenate(op_list, axis=0).T.astype(np.float32)
        self._dose_all = np.concatenate(dose_list, axis=0).T.astype(np.float32)

        # 同模态相关性边
        ei_op, ew_op = _pairwise_topk_corr(self._op_all, self.op_topk, self.min_abs_corr)
        ei_dose, ew_dose = _pairwise_topk_corr(self._dose_all, self.dose_topk, self.min_abs_corr)
        self.edge_index_op = torch.tensor(ei_op, dtype=torch.long)
        self.edge_index_dose = torch.tensor(ei_dose, dtype=torch.long)

        # 跨模态滞后边
        if self.topk_cross > 0:
            (d2o_e, d2o_a, o2d_e, o2d_a) = self._build_lagged_cross_edges_with_attr(
                max_lag_dose=self.max_lag_dose,
                topk_cross=self.topk_cross,
                downsample_mode=self.downsample_mode
            )
            self.edge_index_dose2op = d2o_e
            self.edge_attr_dose2op = d2o_a
            self.edge_index_op2dose = o2d_e
            self.edge_attr_op2dose = o2d_a
        else:
            self.edge_index_dose2op = torch.empty((2,0),dtype=torch.long)
            self.edge_attr_dose2op = torch.empty((0,3),dtype=torch.float32)
            self.edge_index_op2dose = torch.empty((2,0),dtype=torch.long)
            self.edge_attr_op2dose = torch.empty((0,3),dtype=torch.float32)

    # ---------------- TFF 加载 ----------------
    def _load_tff(self):
        pairs = _scan_tff_folders(self.op_dir)
        dfs, labs = [], []
        for path, lab in pairs:
            df = _read_one_csv_align_numeric(path, self.op_interval)
            dfs.append(df)
            labs.append(lab)
        cols = _union_columns(dfs)
        self.num_op = len(cols)
        self._ops, self._ops_t, self._labels = [], [], []
        self._label2id = {k:i for i,k in enumerate(_TFF_LABEL_ORDER)}
        for df, lab in zip(dfs, labs):
            X, t = _df_to_np(df, cols)
            self._ops.append(X.astype(np.float32))
            self._ops_t.append(t.astype(np.float32))
            self._labels.append(self._label2id.get(lab,0))
        if not self.tff_single:
            doses = []
            for X in self._ops:
                N, T = X.shape
                K = T // self.ratio
                if K <= 0:
                    Y = np.zeros((min(53,N),0),np.float32)
                else:
                    if self.downsample_mode=='pick':
                        tmp = X[:,:K*self.ratio:self.ratio]
                    else:
                        tmp = X[:,:K*self.ratio].reshape(N,K,self.ratio).mean(axis=2)
                    if N>=53:
                        Y = tmp[:53]
                    else:
                        Y = np.vstack([tmp, np.zeros((53-N,tmp.shape[1]),np.float32)])
                doses.append(Y.astype(np.float32))
            self._doses = doses
            self.num_dose = 53
        else:
            self._doses = None
            self.num_dose = 0

        self._window2file, self._window2dose_start = [], []
        if self.tff_single:
            for sid,X in enumerate(self._ops):
                Td = X.shape[1]//self.ratio
                for st in range(0,Td-self.dose_steps+1,self.dose_stride):
                    self._window2file.append(sid)
                    self._window2dose_start.append(st)
        else:
            for sid,Y in enumerate(self._doses):
                Td = Y.shape[1]
                for st in range(0,Td-self.dose_steps+1,self.dose_stride):
                    self._window2file.append(sid)
                    self._window2dose_start.append(st)
        self._window_labels = [self._labels[f] for f in self._window2file]

        # 全局相关性
        if self.tff_single:
            Xcat = np.concatenate([x.T for x in self._ops],axis=0).T
            ei_op,_ = _pairwise_topk_corr(Xcat, self.op_topk, self.min_abs_corr)
            self.edge_index_op = torch.tensor(ei_op,dtype=torch.long)
            self.edge_index_dose = torch.empty((2,0),dtype=torch.long)
            self.edge_index_dose2op = torch.empty((2,0),dtype=torch.long)
            self.edge_attr_dose2op = torch.empty((0,3),dtype=torch.float32)
            self.edge_index_op2dose = torch.empty((2,0),dtype=torch.long)
            self.edge_attr_op2dose = torch.empty((0,3),dtype=torch.float32)
        else:
            Xcat = np.concatenate([x.T for x in self._ops],axis=0).T
            Ycat = np.concatenate([y.T for y in self._doses],axis=0).T
            ei_op,_ = _pairwise_topk_corr(Xcat,self.op_topk,self.min_abs_corr)
            ei_dose,_ = _pairwise_topk_corr(Ycat,self.dose_topk,self.min_abs_corr)
            self.edge_index_op = torch.tensor(ei_op,dtype=torch.long)
            self.edge_index_dose = torch.tensor(ei_dose,dtype=torch.long)
            if self.topk_cross>0:
                d2o_e,d2o_a,o2d_e,o2d_a = self._build_lagged_cross_edges_with_attr(
                    max_lag_dose=self.max_lag_dose,
                    topk_cross=self.topk_cross,
                    downsample_mode=self.downsample_mode
                )
                self.edge_index_dose2op = d2o_e
                self.edge_attr_dose2op = d2o_a
                self.edge_index_op2dose = o2d_e
                self.edge_attr_op2dose = o2d_a
            else:
                self.edge_index_dose2op = torch.empty((2,0),dtype=torch.long)
                self.edge_attr_dose2op = torch.empty((0,3),dtype=torch.float32)
                self.edge_index_op2dose = torch.empty((2,0),dtype=torch.long)
                self.edge_attr_op2dose = torch.empty((0,3),dtype=torch.float32)

    # 构建滑窗元信息
    def _build_windows(self):
        self._win_meta   = list(zip(self._window2file, self._window2dose_start))
        self._win_labels = list(self._window_labels)

    def __len__(self):
        return len(self._win_meta)

    # =====================================================
    # 标准化 & 裁剪（保证 float32，避免 Double vs Half 问题）
    # =====================================================
    def _std_clip(self, A: np.ndarray) -> torch.Tensor:
        if A.size == 0:
            h = A.shape[0] if A.ndim>0 else 0
            w = A.shape[1] if A.ndim>1 else 0
            return torch.zeros((h,w), dtype=torch.float32)
        mu = A.mean(axis=1, keepdims=True)
        sd = A.std(axis=1, keepdims=True) + 1e-8
        Z = (A - mu) / sd
        if self.clip_val>0:
            Z = np.clip(Z, -self.clip_val, self.clip_val)
        return torch.tensor(Z.astype(np.float32), dtype=torch.float32)

    def __getitem__(self, idx: int) -> HeteroData:
        sid, st = self._win_meta[idx]
        data = HeteroData()
        # —— NPS 或 TFF 双模态 ——
        if self.dataset_type=='NPS' or (self.dataset_type=='TFF' and not self.tff_single):
            s = self.samples[sid]
            op_arr = _pad_or_trunc_cols(_robust_load_csv(s['op_path']), self.num_op)
            if s['dose_path']:
                dose_arr = _pad_or_trunc_cols(_robust_load_csv(s['dose_path']), self.num_dose)
            else:
                T = op_arr.shape[0]//self.ratio
                if T<=0:
                    dose_arr = np.zeros((0,self.num_dose),np.float32)
                else:
                    if self.downsample_mode=='pick':
                        tmp = op_arr[:T*self.ratio:self.ratio]
                    else:
                        tmp = op_arr[:T*self.ratio].reshape(T,self.ratio,self.num_op).mean(axis=1)
                    if self.num_dose<=self.num_op:
                        tmp = tmp[:,:self.num_dose]
                    else:
                        tmp = np.pad(tmp,((0,0),(0,self.num_dose-self.num_op)),mode='constant')
                    dose_arr = tmp
            # dose window
            ds_idx = np.arange(st, st+self.dose_steps)
            ds_idx = np.clip(ds_idx, 0, dose_arr.shape[0]-1 if dose_arr.shape[0]>0 else 0)
            dose_win = dose_arr[ds_idx] if dose_arr.size>0 else np.zeros((self.dose_steps,self.num_dose),np.float32)
            # op window aligned
            t0 = st*self.ratio
            need = self.op_steps
            t1 = t0+need
            if t0>=op_arr.shape[0]:
                last = op_arr[-1:]
                op_win = np.repeat(last, need, axis=0)
            else:
                sl = op_arr[t0:min(t1,op_arr.shape[0])]
                if sl.shape[0]<need:
                    pad = np.repeat(sl[-1:], need-sl.shape[0], axis=0)
                    op_win = np.vstack([sl,pad])
                else:
                    op_win = sl
            op_x   = self._std_clip(op_win.T).float()
            dose_x = self._std_clip(dose_win.T).float()
            data['op'].x   = op_x
            data['dose'].x = dose_x
            # 同模态边
            for nodetype in ('op','dose'):
                ei = getattr(self, f'edge_index_{nodetype}', torch.zeros((2,0),dtype=torch.long))
                if ei.numel()>0:
                    w = torch.ones((ei.shape[1],1),dtype=torch.float32)
                    data[nodetype, 'intra', nodetype].edge_index = ei
                    data[nodetype, 'intra', nodetype].edge_attr  = w
                    data[nodetype,'intra_rev',nodetype].edge_index = ei.flip(0)
                    data[nodetype,'intra_rev',nodetype].edge_attr  = w.clone()
                else:
                    data[nodetype,'intra',nodetype].edge_index = torch.zeros((2,0),dtype=torch.long)
                    data[nodetype,'intra',nodetype].edge_attr  = torch.zeros((0,1),dtype=torch.float32)
                    data[nodetype,'intra_rev',nodetype].edge_index = torch.zeros((2,0),dtype=torch.long)
                    data[nodetype,'intra_rev',nodetype].edge_attr  = torch.zeros((0,1),dtype=torch.float32)
            # 跨模态边
            data['dose','cross','op'].edge_index = getattr(self,'edge_index_dose2op',torch.zeros((2,0),dtype=torch.long))
            data['dose','cross','op'].edge_attr  = getattr(self,'edge_attr_dose2op',torch.zeros((0,3),dtype=torch.float32))
            data['op','cross','dose'].edge_index = getattr(self,'edge_index_op2dose',torch.zeros((2,0),dtype=torch.long))
            data['op','cross','dose'].edge_attr  = getattr(self,'edge_attr_op2dose',torch.zeros((0,3),dtype=torch.float32))
            data.y = torch.tensor(self._win_labels[idx],dtype=torch.long)
            data.num_op   = self.num_op
            data.num_dose = self.num_dose
            data.op_steps   = self.op_steps
            data.dose_steps = self.dose_steps
            return data

        # —— TFF 单模态 ——
        X = self._ops[sid]
        t0 = st*self.ratio
        need = self.op_steps
        t1 = t0+need
        if t0>=X.shape[1]:
            last = X[:,-1:]
            op_win = np.repeat(last, need, axis=1)
        else:
            sl = X[:,t0:min(t1,X.shape[1])]
            if sl.shape[1]<need:
                pad = np.repeat(sl[:,-1:], need-sl.shape[1], axis=1)
                op_win = np.concatenate([sl,pad],axis=1)
            else:
                op_win = sl
        x_op = _zscore_per_row(op_win)
        x_op = np.clip(x_op, -self.clip_val, self.clip_val)
        data['op'].x = torch.tensor(x_op,dtype=torch.float32).float()
        ei_op = getattr(self,'edge_index_op',torch.zeros((2,0),dtype=torch.long))
        if ei_op.numel()>0:
            w = torch.ones((ei_op.shape[1],1),dtype=torch.float32)
            data['op','intra','op'].edge_index = ei_op
            data['op','intra','op'].edge_attr  = w
            data['op','intra_rev','op'].edge_index = ei_op.flip(0)
            data['op','intra_rev','op'].edge_attr  = w.clone()
        else:
            data['op','intra','op'].edge_index = torch.zeros((2,0),dtype=torch.long)
            data['op','intra','op'].edge_attr  = torch.zeros((0,1),dtype=torch.float32)
            data['op','intra_rev','op'].edge_index = torch.zeros((2,0),dtype=torch.long)
            data['op','intra_rev','op'].edge_attr  = torch.zeros((0,1),dtype=torch.float32)
        data.y = torch.tensor(self._win_labels[idx],dtype=torch.long)
        data.num_op   = self.num_op
        data.num_dose = 0
        data.op_steps   = self.op_steps
        data.dose_steps = self.dose_steps
        return data

    # ---------------- 跨模态滞后边 ----------------
    @staticmethod
    def _transpose_edges(e: torch.Tensor) -> torch.Tensor:
        if e.numel()==0: return e.clone()
        return torch.stack([e[1],e[0]],dim=0)

    def _build_lagged_cross_edges_with_attr(self,
                                            max_lag_dose: int = 5,
                                            topk_cross: int = 3,
                                            downsample_mode: str = "pick"):
        ratio = self.ratio
        op_all = getattr(self,'_op_all',None)
        dose_all = getattr(self,'_dose_all',None)
        if op_all is None or dose_all is None or op_all.shape[1]==0 or dose_all.shape[1]==0:
            return (torch.empty((2,0),dtype=torch.long),
                    torch.empty((0,3),dtype=torch.float32),
                    torch.empty((2,0),dtype=torch.long),
                    torch.empty((0,3),dtype=torch.float32))
        T60 = op_all.shape[1]//ratio
        if T60<=1:
            T = min(op_all.shape[1], dose_all.shape[1])
            op_down = op_all[:,:T]
            dose = dose_all[:,:T]
        else:
            if downsample_mode=="pick":
                op_down = op_all[:,:T60*ratio][:,::ratio]
            else:
                op_down = op_all[:,:T60*ratio].reshape(self.num_op, T60, ratio).mean(axis=2)
            T = min(op_down.shape[1], dose_all.shape[1])
            op_down = op_down[:,:T]
            dose = dose_all[:,:T]
        op_z = _zscore_per_row(op_down)
        dose_z = _zscore_per_row(dose)
        src, dst, attr = [], [], []
        for j in range(self.num_dose):
            dj = dose_z[j]
            cand = []
            for i in range(self.num_op):
                oi = op_z[i]
                best_abs, best_lag, best_signed = -1.0, 0, 0.0
                for lag in range(1, max_lag_dose+1):
                    if T-lag<=1: break
                    d_seg = dj[:T-lag]
                    o_seg = oi[lag:T]
                    corr = float((d_seg*o_seg).mean())
                    ac = abs(corr)
                    if ac>best_abs:
                        best_abs, best_lag, best_signed = ac, lag, corr
                if best_lag>0:
                    cand.append((best_abs,i,best_lag,best_signed))
            if not cand: continue
            cand.sort(key=lambda x: x[0], reverse=True)
            for ac,i,lag,signed in cand[:min(topk_cross,len(cand))]:
                src.append(j); dst.append(i)
                attr.append([lag/ max_lag_dose, ac, 1.0 if signed>=0 else -1.0])
        if not src:
            return (torch.empty((2,0),dtype=torch.long),
                    torch.empty((0,3),dtype=torch.float32),
                    torch.empty((2,0),dtype=torch.long),
                    torch.empty((0,3),dtype=torch.float32))
        e_d2o = torch.tensor([src,dst],dtype=torch.long)
        a_d2o = torch.tensor(attr,dtype=torch.float32)
        e_o2d = self._transpose_edges(e_d2o)
        a_o2d = a_d2o.clone()
        return e_d2o, a_d2o, e_o2d, a_o2d



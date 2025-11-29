# utils/train_eval.py
# -*- coding: utf-8 -*-
"""
训练与评估（NPS/TFF 通用，异构图 + 边属性 + 对比/一致性）
- SupCon 基于 z_fused
- XMD   基于 z_op / z_dose
- AUG   基于 z_fused 的轻量一致性（dropout 或加噪），受 --use_aug, --lmbd_aug 控制
"""

import os, inspect, random, sys
from typing import List, Optional, Tuple
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW, RMSprop, Adagrad
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
# ---- 解决直接运行时的导入路径 ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- 数据集与模型 ----
import dataset.data_sampling_lag_edge_attr as ds
from models.temporal_hetero_gnn_edge_attr_contrastive import TemporalPhysicalHeteroGNN_V2


# ==========================
# 工具函数
# ==========================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def class_balanced_weights(labels: List[int], num_classes: int, beta: float = 0.999) -> torch.Tensor:
    cnt = Counter(labels)
    weights = []
    for c in range(num_classes):
        n_c = cnt.get(c, 0)
        if n_c <= 0:
            weights.append(0.0); continue
        w = (1.0 - beta) / (1.0 - (beta ** n_c))
        weights.append(w)
    w = torch.tensor(weights, dtype=torch.float32)
    if w.sum() > 0:
        w = w * (len(w) / (w.sum() + 1e-12))  # 归一到均值=1
    return w


class FocalLoss(nn.Module):
    """多分类 Focal Loss（对 CE 加权）"""
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight_buf", weight)
        else:
            self.weight_buf = None  # type: ignore
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        ce = F.cross_entropy(logits, target, weight=self.weight_buf, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss


def supcon_loss(
    z: Optional[torch.Tensor],
    y: torch.Tensor,
    tau: float = 0.1,
    soft: bool = False,
    label_tau: float = 0.5,
) -> torch.Tensor:
    """
    Supervised contrastive loss with optional SoftCL weighting.

    默认（soft=False）行为与原实现完全一致：
    - z 归一化；
    - 相似度 sim = (z z^T) / tau；
    - 同标签为正样本（去对角线），按 log-softmax 后对正样本平均；

    当 soft=True：
    - 基于标签构造 dist (同类=0，异类=1)；
    - soft_pos = exp(-dist / label_tau) 并置零对角线；
    - 对每行 soft_pos 归一化；
    - 使用 soft_pos 对每行 log 概率加权求和（SoftCL）。

    鲁棒性：当 z 无效或 batch<2 时返回 0（与 y 在同一设备）。
    """
    # —— 无效输入与 batch 过小 ——
    if z is None or z.numel() == 0:
        return torch.tensor(0.0, device=y.device)
    if z.dim() == 0 or z.size(0) < 2:
        return torch.tensor(0.0, device=y.device)

    # 归一化与相似度
    z = F.normalize(z, dim=-1)
    sim = torch.mm(z, z.t()) / max(tau, 1e-6)

    # 为了与旧实现保持一致，logits/exp_logits 的处理保持不变
    logits = sim - torch.max(sim, dim=1, keepdim=True).values
    # 去除自身参与：通过 (1 - I) 屏蔽对角线
    I = torch.eye(z.size(0), device=z.device)
    exp_logits = torch.exp(logits) * (1 - I)
    log_den = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)
    log_prob = logits - log_den  # 每行 log-softmax（去掉对角线的归一化）

    # 硬对比（与现有实现完全一致）
    if not soft:
        yv = y.view(-1, 1)
        mask_pos = torch.eq(yv, yv.t()).float()
        mask_pos = mask_pos - I  # 去除对角线
        denom = (mask_pos.sum(1) + 1e-9)
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / denom
        return -mean_log_prob_pos.mean()

    # 软对比（SoftCL）
    # 距离矩阵：同类=0，异类=1
    yv = y.view(-1, 1)
    dist = (yv != yv.t()).float()
    lt = max(float(label_tau), 1e-6)
    soft_pos = torch.exp(-dist / lt)
    # 去除对角线，避免自身作“正样本”
    soft_pos = soft_pos * (1 - I)
    # 行归一化（若全零则保持为零行，分母加小常数防止除零告警）
    soft_pos = soft_pos / (soft_pos.sum(dim=1, keepdim=True) + 1e-12)
    # 用软权重加权每行的 log 概率
    weighted_log = (soft_pos * log_prob).sum(1)
    return -weighted_log.mean()


def cosine_align_loss(z_a: Optional[torch.Tensor], z_b: Optional[torch.Tensor]) -> torch.Tensor:
    if z_a is None or z_b is None:
        dev = z_a.device if z_a is not None else (z_b.device if z_b is not None else 'cpu')
        return torch.tensor(0.0, device=dev)
    za = F.normalize(z_a, dim=-1); zb = F.normalize(z_b, dim=-1)
    cos = (za * zb).sum(dim=-1)
    return (1.0 - cos).mean()


def ortho_penalty(z_shared: Optional[torch.Tensor],
                  z_private: Optional[torch.Tensor]) -> torch.Tensor:
    """
    正交约束损失：对每个样本的 shared 与 private 表示做余弦相似度，取平方再对 batch 平均。
    - 输入任一为 None 时返回 0（设备与已有张量一致）。
    - 不做缩放，权重由外部 lmbd_ortho 控制。
    """
    if z_shared is None and z_private is None:
        return torch.tensor(0.0)
    if z_shared is None:
        return torch.tensor(0.0, device=z_private.device)
    if z_private is None:
        return torch.tensor(0.0, device=z_shared.device)
    zs = F.normalize(z_shared, dim=-1)
    zp = F.normalize(z_private, dim=-1)
    # 若 batch 尺寸不一致，按最小对齐（理论上应一致）
    if zs.size(0) != zp.size(0):
        n = min(zs.size(0), zp.size(0))
        zs = zs[:n]
        zp = zp[:n]
    
    # 确保维度一致才能做点积
    if zs.size(1) != zp.size(1):
        min_d = min(zs.size(1), zp.size(1))
        zs = zs[:, :min_d]
        zp = zp[:, :min_d]
        
    cos = (zs * zp).sum(dim=-1)
    return (cos ** 2).mean()


# ==========================
# 评估
# ==========================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, use_amp: bool = True) -> float:
    model.eval(); correct=total=0
    amp_dev = 'cuda' if (str(device).startswith('cuda') and torch.cuda.is_available()) else 'cpu'
    use_amp_flag = bool(use_amp and amp_dev == 'cuda')
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        with autocast(device_type=amp_dev, enabled=use_amp_flag):
            out = model(batch)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            pred = logits.argmax(dim=1)
        correct += (pred == batch.y).sum().item(); total += batch.num_graphs
    return correct / max(1, total)


# ==========================
# 损失解析（兼容两套参数）
# ==========================
def resolve_cls_loss(
    args,
    labels_all: List[int],
    train_idx: np.ndarray,
    num_classes: int,
    device: str
) -> Tuple[nn.Module, str]:
    # 新版参数（与 train.py 对齐）
    loss_type = getattr(args, 'loss_type', None)  # {'ce','focal','cb_ce','cb_focal'}
    gamma = float(getattr(args, 'gamma', 2.0))
    beta  = float(getattr(args, 'beta', 0.999))

    # 兼容旧参数
    use_focal    = bool(getattr(args, 'use_focal', False))
    focal_gamma  = float(getattr(args, 'focal_gamma', gamma))
    cb_beta      = float(getattr(args, 'cb_beta', beta))

    # 以 loss_type 为准；若未提供则回退到旧参数风格
    if loss_type is None:
        if use_focal and cb_beta is not None:
            loss_type = 'cb_focal'
        elif use_focal:
            loss_type = 'focal'
        elif cb_beta is not None:
            loss_type = 'cb_ce'
        else:
            loss_type = 'ce'

    # 类平衡权重
    weights = None
    if loss_type in ('cb_ce', 'cb_focal'):
        weights = class_balanced_weights(
            [labels_all[i] for i in train_idx], num_classes, beta=cb_beta
        ).to(device)

    # 具体损失
    if loss_type in ('focal', 'cb_focal'):
        criterion = FocalLoss(gamma=focal_gamma, weight=weights)
        name = 'CB-Focal' if weights is not None else f'Focal(gamma={focal_gamma})'
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
        name = 'CB-CE' if weights is not None else 'CE'
    return criterion, name


# ==========================
# 训练主循环
# ==========================
def train_loop(args):
    # ---------- Early-Stopping ----------
    patience = int(getattr(args, "patience", 10))  # 默认 10
    no_improve = 0

    # [1] 种子 & 目录
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, f"{args.exp_name}_best.pt")
    last_path = os.path.join(args.save_dir, f"{args.exp_name}_last.pt")

    # [2] 数据集
    print("[2] 开始加载数据集 (HeteroGraph, 支持 NPS/TFF).")
    print(f"[Dataset] using module file: {ds.__file__}")
    sig_ds = inspect.signature(ds.MultiModalHeteroDatasetV2.__init__)
    allowed_ds = set(sig_ds.parameters.keys()) - {"self"}
    candidate_ds = dict(
        dataset=getattr(args, 'dataset', 'auto'),
        op_dir=args.op_dir, dose_dir=getattr(args, 'dose_dir', ''),
        op_interval=args.op_interval, dose_interval=args.dose_interval,
        dose_window_steps=args.dose_window_steps, dose_stride_steps=args.dose_stride_steps,
        task_type=args.task_type, clip_val=args.clip_val,
        op_topk=getattr(args,'op_topk',10), dose_topk=getattr(args,'dose_topk',10),
        min_abs_corr=getattr(args,'min_abs_corr',0.2),
        max_lag_dose=getattr(args,'max_lag_dose',5),
        topk_cross=getattr(args,'topk_cross',3),
        downsample_mode=getattr(args,'downsample_mode','pick'),
    )
    init_kwargs_ds = {k:v for k,v in candidate_ds.items() if k in allowed_ds}
    print(f"[Dataset] init kwargs: {sorted(init_kwargs_ds.keys())}")
    dataset = ds.MultiModalHeteroDatasetV2(**init_kwargs_ds)

    if len(dataset) == 0:
        raise RuntimeError(
            "[Dataset] 构造出的窗口数为 0。请检查：\n"
            "  - --op_dir 是否指向 NPS 的 Operation CSV 目录\n"
            "  - --dose_dir 是否有 CSV；若没有，当前版本会用 OP 下采样生成 pseudo-dose\n"
            "  - --dose_window_steps 是否过大（单条 DOSE 序列不足以滑窗）\n"
            "  - CSV 是否包含数值列（非数值会被过滤）\n"
        )

    # [3] num_classes 推断
    labels_all = getattr(dataset, "_win_labels", [])
    if args.task_type == 'binary':
        num_classes = 2
    elif getattr(dataset, "label2id", None):
        num_classes = len(dataset.label2id)
    elif labels_all:
        num_classes = int(max(labels_all)) + 1
    else:
        raise RuntimeError("[Dataset] 无法推断类别数。")

    # [4] 划分
    labels_all: List[int] = getattr(dataset, '_win_labels', [])
    print(f"[DatasetV2] type={getattr(dataset,'dataset_type','?')} | windows={len(dataset)} "
          f"| dose_steps={dataset.dose_steps} | op_steps={dataset.op_steps} "
          f"| ratio={dataset.dose_interval//dataset.op_interval} | N_classes={num_classes}")
    cnt_all = Counter(labels_all)
    print("全数据集分布：")
    for c in sorted(cnt_all.keys()): print(f"  类别 {c:2d}: {cnt_all[c]}")
    # =============================
    # 相邻样本不落同一子集（窗口级交错分配，分布一致）
    # =============================
    window2file = getattr(dataset, '_window2file', [])
    file_labels = getattr(dataset, '_labels', [])
    if not window2file or not file_labels:
        raise RuntimeError('[Split] 缺少文件级映射(_window2file/_labels)，无法进行无重叠划分。')

    # 交错分配：对每个文件的窗口按时间顺序交错到 train/val/test，确保相邻不在同一子集
    files = np.arange(len(file_labels))
    data_fraction = float(getattr(args, 'data_fraction', 1.0))
    base_files = files
    if 0.0 < data_fraction < 1.0:
        # 子集文件选择不分层，避免稀有类导致报错；分布一致通过窗口级交错实现
        base_files, _ = train_test_split(files, train_size=data_fraction, random_state=args.seed, stratify=None)
        print(f"[Subset-Files] 使用 {len(base_files)}/{len(files)} 个文件 (~{data_fraction*100:.1f}%).")

    # 模式：严格相邻不同子集 + 比例 7:1.5:1.5（约等于 0.7/0.15/0.15）
    ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    subsets = ['train', 'val', 'test']
    train_idx, val_idx, test_idx = [], [], []
    # 全局目标配额，避免“每文件四舍五入”造成累计误差
    all_wins = [w for w in range(len(window2file)) if window2file[w] in set(base_files)]
    total_n = len(all_wins)
    global_target = {
        'train': int(round(total_n * ratios['train'])),
        'val':   int(round(total_n * ratios['val'])),
        'test':  int(round(total_n * ratios['test']))
    }
    # 调整总和到 total_n
    gdiff = total_n - sum(global_target.values())
    if gdiff != 0:
        for s in sorted(subsets, key=lambda k: ratios[k], reverse=True):
            if gdiff == 0: break
            global_target[s] += 1 if gdiff > 0 else -1
            gdiff += -1 if gdiff > 0 else 1
    global_used = {s: 0 for s in subsets}
    # 遍历文件，内部仍保持相邻不同，优先满足全局剩余配额
    for f in base_files:
        wins = [w for w, fid in enumerate(window2file) if fid == f]
        prev = None
        for w in wins:
            # 候选按全局剩余配额排序，且不等于 prev
            candidates = sorted(subsets, key=lambda s: (global_target[s] - global_used[s], ratios[s]), reverse=True)
            chosen = None
            for s in candidates:
                if s == prev: continue
                if global_used[s] < global_target[s]:
                    chosen = s
                    break
            if chosen is None:
                # 若所有候选已满或与 prev 冲突，选择与 prev 不同的集合（允许略超配额）
                for s in subsets:
                    if s != prev:
                        chosen = s
                        break
            global_used[chosen] += 1
            prev = chosen
            if chosen == 'train':
                train_idx.append(w)
            elif chosen == 'val':
                val_idx.append(w)
            else:
                test_idx.append(w)

    print('[4] 窗口级交错划分 (相邻不同行, 近似均衡).')
    print(f"窗口数: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    cnt_tr = Counter([labels_all[i] for i in train_idx])
    cnt_val = Counter([labels_all[i] for i in val_idx])
    cnt_te  = Counter([labels_all[i] for i in test_idx])
    print('训练集分布：')
    for c in sorted(cnt_tr.keys()): print(f"  类别 {c:2d}: {cnt_tr[c]}")
    print('验证集分布：')
    for c in sorted(cnt_val.keys()): print(f"  类别 {c:2d}: {cnt_val[c]}")
    print('测试集分布：')
    for c in sorted(cnt_te.keys()): print(f"  类别 {c:2d}: {cnt_te[c]}")

    # [5] DataLoader
    print("[5] 构建 DataLoader.")
    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    test_set  = Subset(dataset, test_idx)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=getattr(args,'num_workers',8),
                              pin_memory=True, persistent_workers=False)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=getattr(args,'test_workers',0),
                              pin_memory=True, persistent_workers=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                              num_workers=getattr(args,'test_workers',0),
                              pin_memory=True, persistent_workers=False)
    print(f"[5] DataLoader 构建完成。Train nw={getattr(args,'num_workers',8)} | Val nw={getattr(args,'test_workers',0)} | Test nw={getattr(args,'test_workers',0)}")

    # [6] 初始化模型 —— 自动别名映射（含必需的 seq_len/通道数/edge_attr_dim）
    print("[6] 初始化模型 (HeteroGraph + edge_attr + Contrastive heads).")
    sig_m = inspect.signature(TemporalPhysicalHeteroGNN_V2.__init__)
    allowed_m = set(sig_m.parameters.keys()) - {"self"}
    cand_m = {
        # transformer/gnn 相关（与 train.py 对齐的常用命名）
        'trans_dim'   : getattr(args,'trans_dim',256),
        'trans_layers': getattr(args,'trans_layers',2),
        'nhead'       : getattr(args,'nhead',4),
        'gcn_hidden'  : getattr(args,'gnn_hidden',getattr(args,'gcn_hidden',512)),
        'gcn_layers'  : getattr(args,'gnn_layers',getattr(args,'gcn_layers',2)),
        'dropout'     : getattr(args,'dropout',0.1),
        'num_classes' : num_classes,
        # 模态维度/序列长度（来自数据集）
        'num_op'      : int(getattr(dataset,'num_op',0)),
        'num_dose'    : int(getattr(dataset,'num_dose',0)),
        'op_seq_len'  : int(getattr(dataset,'op_steps',0)),
        'dose_seq_len': int(getattr(dataset,'dose_steps',0)),
        # 跨模态边特征维度（我们构造为 3）
        'edge_attr_dim': 3,
    }
    alias = {
        'trans_dim'   : ['trans_dim','d_model','embed_dim','model_dim','hidden_dim'],
        'trans_layers': ['trans_layers','num_layers','encoder_layers','n_layers','layers'],
        'nhead'       : ['nhead','num_heads','heads','n_heads'],
        'gcn_hidden'  : ['gcn_hidden','gcn_hidden_dim','gcn_hidden_channels','hidden_channels','hidden_dim','gnn_hidden','gnn_dim'],
        'gcn_layers'  : ['gcn_layers','num_gcn_layers','gnn_layers','num_gnn_layers','num_conv_layers','num_layers_gnn'],
        'dropout'     : ['dropout','drop_rate','p_dropout'],
        'num_classes' : ['num_classes','n_classes','out_dim','out_channels','num_outputs'],
        'num_op'      : ['num_op','op_channels','in_dim_op','op_in_channels','op_nvars','n_op'],
        'num_dose'    : ['num_dose','dose_channels','in_dim_dose','dose_in_channels','dose_nvars','n_dose'],
        'op_seq_len'  : ['op_seq_len','seq_len_op','op_length','op_steps','len_op'],
        'dose_seq_len': ['dose_seq_len','seq_len_dose','dose_length','dose_steps','len_dose'],
        'edge_attr_dim': ['edge_attr_dim','edge_dim','edge_attr_size'],
    }
    used_map, init_kwargs_m = {}, {}
    for std_key, value in cand_m.items():
        for real_key in alias.get(std_key, [std_key]):
            if real_key in allowed_m:
                init_kwargs_m[real_key] = value
                used_map[std_key] = real_key
                break
    print(f"[Model] ctor allowed: {sorted(list(allowed_m))}")
    print(f"[Model] arg mapping: " + ", ".join([f"{k}->{v}" for k,v in used_map.items()]))
    missing_required = [p.name for p in sig_m.parameters.values()
                        if p.default is inspect._empty and p.name not in init_kwargs_m and p.name != 'self']
    if missing_required:
        raise TypeError(f"模型构造缺少必要参数: {missing_required}. 已可用映射: {used_map}.")

    model = TemporalPhysicalHeteroGNN_V2(**init_kwargs_m).to(args.device)
    print("[6] 模型初始化完成。")

    # [7] 主损失 & 优化器
    criterion_ce, ce_name = resolve_cls_loss(args, labels_all, train_idx, num_classes, args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # ---------- 对比/一致性超参 ----------
    lmbd_supcon = float(getattr(args,'lmbd_supcon',0.0))
    lmbd_xmod   = float(getattr(args,'lmbd_xmod',0.0))
    lmbd_aug    = float(getattr(args,'lmbd_aug',0.0))
    tau         = float(getattr(args,'tau',0.1))
    tau_aug     = float(getattr(args,'tau_aug',0.1))  # 预留

    # ---------- 新增：正交与 SoftCL 超参 ----------
    lmbd_ortho      = float(getattr(args, 'lmbd_ortho', 0.0))
    use_soft_supcon = bool(getattr(args, 'use_soft_supcon', False))
    soft_label_tau  = float(getattr(args, 'soft_label_tau', 0.5))
    shared_ratio    = float(getattr(args, 'shared_ratio', 0.5))

    # ---------- AMP / 梯度累积 ----------
    use_amp = bool(getattr(args,'use_amp',True))
    amp_dev = 'cuda' if (str(args.device).startswith('cuda') and torch.cuda.is_available()) else 'cpu'
    scaler = GradScaler(amp_dev, enabled=(use_amp and amp_dev=='cuda'))
    grad_accum = int(getattr(args,'grad_accum_steps', getattr(args,'grad_accum',1)))
    assert grad_accum >= 1

    # ---------- 可选：断点恢复 ----------
    start_epoch = 1  # 🔵 起始 epoch
    no_improve = 0  # 🔵 早停计数
    best_acc = -1.0

    if getattr(args, 'resume', '') and os.path.isfile(args.resume):
        try:
            # PyTorch 2.6 默认 weights_only=True 会阻止旧版 checkpoint（含优化器/调度器状态）恢复。
            # 这里显式禁用该限制，前提是我们信任本地保存的 ckpt 文件。
            state = torch.load(args.resume, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            if use_amp and state.get('scaler'):
                scaler.load_state_dict(state['scaler'])
            if 'scheduler' in locals() and state.get('scheduler'):
                scheduler.load_state_dict(state['scheduler'])
            best_acc = float(state.get('best_acc', -1.0))
            no_improve = int(state.get('no_improve', 0))
            start_epoch = int(state.get('epoch', 0)) + 1  # 🔵 新
            print(f"[*] 已从 {args.resume} 恢复：继续 epoch {start_epoch}，best_acc={best_acc:.4f}")

        except Exception as e:
            print(f"[WARN] 恢复失败：{e}")

    # [8] 训练
    print(f"[8] 开始训练主循环（{ce_name}+SupCon+XMod+Aug | AMP 与梯度累积）.")
    warned_no_embed = False
    warned_missing_xmod = False

    for epoch in range(start_epoch, args.epochs+1):
        model.train()
        epoch_loss=ce_sum=sup_sum=xmd_sum=aug_sum=ortho_sum=0.0; step_count=0
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch:03d}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            batch = batch.to(args.device)
            with autocast(device_type=amp_dev, enabled=(use_amp and amp_dev=='cuda')):
                out = model(batch)
                if isinstance(out, (tuple,list)):
                    logits = out[0]
                    z_fused = out[1] if len(out)>1 else None
                    z_op    = out[2] if len(out)>2 else None
                    z_dose  = out[3] if len(out)>3 else None
                else:
                    logits, z_fused, z_op, z_dose = out, None, None, None

                # 首个 batch 打印嵌入形状（便于排查）
                if epoch == 1 and step == 1:
                    def shp(x): return None if x is None else tuple(x.shape)
                    print(f"[Probe] use_aug={getattr(args,'use_aug',True)} | "
                          f"z_fused={shp(z_fused)}, z_op={shp(z_op)}, z_dose={shp(z_dose)}")

                # -------------- shared/private 分解 --------------
                z_shared = None
                z_op_sh = z_op_pr = None
                z_dose_sh = z_dose_pr = None

                can_split = (z_op is not None) and (z_dose is not None)
                if can_split:
                    d_op = z_op.size(-1)
                    d_ds = z_dose.size(-1)
                    if (d_op == d_ds):
                        # 共享:私有 按 shared_ratio 切分，落在 [1, d-1]
                        h = int(d_op * shared_ratio)
                        h = max(1, min(d_op - 1, h))
                        if h <= 0 or h >= d_op:
                            h = d_op // 2  # 兜底回退为对半分
                        z_op_sh, z_op_pr = z_op[:, :h], z_op[:, h:]
                        z_dose_sh, z_dose_pr = z_dose[:, :h], z_dose[:, h:]
                        z_shared = 0.5 * (z_op_sh + z_dose_sh)
                    else:
                        # 维度不合适，退化使用 z_fused 作为 shared
                        z_shared = z_fused if z_fused is not None else None
                else:
                    # 缺少某一模态，退化使用 z_fused 作为 shared
                    z_shared = z_fused if z_fused is not None else None

                ce = criterion_ce(logits, batch.y)

                # 三个附加损失
                sup = torch.tensor(0.0, device=args.device)
                xmd = torch.tensor(0.0, device=args.device)
                aug = torch.tensor(0.0, device=args.device)
                ortho = torch.tensor(0.0, device=args.device)

                # 嵌入缺失一次性告警
                if (lmbd_supcon>0 or lmbd_xmod>0 or lmbd_aug>0) and (z_fused is None and z_op is None and z_dose is None) and (not warned_no_embed):
                    print("[WARN] 模型 forward 未返回嵌入（z_fused/z_op/z_dose），对比/一致性损失将被跳过。")
                    warned_no_embed=True

                # SupCon（优先使用 shared，支持 SoftCL）
                if lmbd_supcon>0 and z_shared is not None:
                    sup = supcon_loss(z_shared, batch.y, tau=tau,
                                      soft=use_soft_supcon, label_tau=soft_label_tau) * lmbd_supcon

                # XMod（优先对 shared 子空间对齐，退化到原 z_op/z_dose）
                if lmbd_xmod>0:
                    if (z_op_sh is not None) and (z_dose_sh is not None):
                        xmd = cosine_align_loss(z_op_sh, z_dose_sh) * lmbd_xmod
                    elif (z_op is not None) and (z_dose is not None):
                        xmd = cosine_align_loss(z_op, z_dose) * lmbd_xmod
                    elif not warned_missing_xmod:
                        print("[WARN] XMD 启用但缺少可对齐的表示（shared 或原始 z_op/z_dose），已跳过跨模态一致性；请检查模型 forward 的返回。")
                        warned_missing_xmod = True

                # AUG（轻量一致性；默认 Dropout，也可切到噪声）
                if lmbd_aug>0 and getattr(args, 'use_aug', True) and (z_fused is not None):
                    mode = str(getattr(args, 'aug_mode', 'dropout')).lower()
                    if mode == 'noise':
                        sigma = float(getattr(args, 'aug_noise_std', 0.05))
                        z_aug = z_fused + torch.randn_like(z_fused) * sigma
                    else:
                        p = float(getattr(args, 'aug_dropout', 0.2))
                        z_aug = F.dropout(z_fused, p=p, training=True)

                    za = F.normalize(z_fused, dim=-1)
                    zb = F.normalize(z_aug,   dim=-1)
                    # 用同一风格的“1-cos”对齐；如需温度，可替换为 (1 - cos/τ_aug) 等自定义形式
                    aug = (1.0 - (za * zb).sum(dim=-1)).mean() * lmbd_aug

                # 正交损失（需要成功拆分出 private 子空间）
                if lmbd_ortho > 0:
                    o_sum = torch.tensor(0.0, device=args.device)
                    cnt = 0
                    if (z_op_sh is not None) and (z_op_pr is not None):
                        o_sum = o_sum + ortho_penalty(z_op_sh, z_op_pr)
                        cnt += 1
                    if (z_dose_sh is not None) and (z_dose_pr is not None):
                        o_sum = o_sum + ortho_penalty(z_dose_sh, z_dose_pr)
                        cnt += 1
                    if cnt > 0:
                        ortho = o_sum * lmbd_ortho

                loss = (ce + sup + xmd + aug + ortho) / grad_accum

            # 反传与优化
            if use_amp and amp_dev=='cuda':
                scaler.scale(loss).backward()
                if step % grad_accum == 0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if step % grad_accum == 0:
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)

            # 统计
            epoch_loss += float((loss * grad_accum).detach().cpu())
            ce_sum += float((ce).detach().cpu())
            sup_sum += float((sup).detach().cpu())
            xmd_sum += float((xmd).detach().cpu())
            aug_sum += float((aug).detach().cpu())
            ortho_sum += float((ortho).detach().cpu())
            step_count += 1
            pbar.set_postfix(loss=(loss.item()*grad_accum),
                             ce=ce.item(), sup=float(sup), xmd=float(xmd), aug=float(aug), ortho=float(ortho))

        # [9] 评估
        acc = evaluate(model, test_loader, args.device, use_amp=use_amp)
        print(f"[Epoch {epoch:03d}] TrainLoss:{epoch_loss/max(1,step_count):.4f} | CE:{ce_sum/max(1,step_count):.4f} "
              f"Sup:{sup_sum/max(1,step_count):.4f} XMD:{xmd_sum/max(1,step_count):.4f} AUG:{aug_sum/max(1,step_count):.4f} "
              f"ORTH:{ortho_sum/max(1,step_count):.4f} | "
              f"Test Acc:{acc:.4f} | Best:{(-1.0 if epoch==1 and best_acc<0 else best_acc):.4f}")

        # [10] 保存
        state = {'epoch': epoch,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'args': vars(args),
                 'train_idx': train_idx, 'test_idx': test_idx,
                 'best_acc': max(best_acc, acc)}
        torch.save(state, last_path)

        if acc > best_acc:  # 有提升 ➜ 记录 & 归零计数
            best_acc = acc
            no_improve = 0
            torch.save(state, best_path)
            print(f"[*] 新最佳模型已保存到: {best_path}")
            print(f"Test Acc improved -> {best_acc:.4f}")
        else:  # 无提升 ➜ 计数 +1
            no_improve += 1
            print(f"[EarlyStop] no_improve = {no_improve}/{patience}")

        # -------- 触发 Early-Stopping --------
        if no_improve >= patience:
            print(f"[EarlyStop] 连续 {patience} 个 epoch 未提升，提前终止训练。")
            break

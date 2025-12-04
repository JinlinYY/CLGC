# -*- coding: utf-8 -*-


import os
import inspect
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.amp import autocast
from matplotlib import cm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    roc_curve, auc as sk_auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 数据集与模型
import dataset.data_sampling_lag_edge_attr as ds
from models.temporal_hetero_gnn_edge_attr_contrastive import TemporalPhysicalHeteroGNN_V2

try:
    plt.style.use("seaborn-v0_8-white")
except OSError:
    plt.style.use("seaborn-whitegrid")

# 设置全局字体
# 优先尝试 Times New Roman，若无则回退到 DejaVu Serif (类似 Times 的衬线字体)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体兼容

# 增大全局字号
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14


# ---------------- 工具函数 ----------------
def _infer_best_ckpt(save_dir: str, exp_name: str) -> Optional[str]:
    cand = os.path.join(save_dir, f"{exp_name}_best.pt")
    if os.path.isfile(cand):
        return cand
    if os.path.isdir(save_dir):
        bests = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith("_best.pt")]
        if bests:
            bests.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return bests[0]
    return None


def _maybe_from_dir(path_or_dir: str) -> Optional[str]:
    if not path_or_dir:
        return None
    if os.path.isdir(path_or_dir):
        cand = [os.path.join(path_or_dir, f) for f in os.listdir(path_or_dir) if f.endswith("_best.pt")]
        if not cand:
            cand = [os.path.join(path_or_dir, f) for f in os.listdir(path_or_dir) if f.endswith(".pt")]
        if cand:
            cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return cand[0]
        return None
    return path_or_dir if os.path.isfile(path_or_dir) else None


def _build_dataset(args):
    print("[Eval] 使用数据集模块:", ds.__file__)
    sig = inspect.signature(ds.MultiModalHeteroDatasetV2.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}

    candidate = dict(
        dataset=getattr(args, 'dataset', 'auto'),
        op_dir=args.op_dir,
        dose_dir=getattr(args, 'dose_dir', ''),
        op_interval=args.op_interval,
        dose_interval=args.dose_interval,
        dose_window_steps=getattr(args, 'dose_window_steps', 60),
        dose_stride_steps=getattr(args, 'dose_stride_steps', 30),
        task_type=getattr(args, 'task_type', 'multiclass'),
        clip_val=getattr(args, 'clip_val', None),
        op_topk=getattr(args, 'op_topk', 10),
        dose_topk=getattr(args, 'dose_topk', 10),
        min_abs_corr=getattr(args, 'min_abs_corr', 0.2),
        max_lag_dose=getattr(args, 'max_lag_dose', 5),
        topk_cross=getattr(args, 'topk_cross', 3),
        downsample_mode=getattr(args, 'downsample_mode', 'pick'),
    )
    kwargs = {k: v for k, v in candidate.items() if k in allowed}
    print(f"[Eval] Dataset kwargs: {sorted(kwargs.keys())}")
    dataset = ds.MultiModalHeteroDatasetV2(**kwargs)
    if len(dataset) == 0:
        raise RuntimeError("[Eval] 数据集为空，请检查目录与窗口参数。")

    # 类别数
    if getattr(args, 'task_type', 'multiclass') == 'binary':
        num_classes = 2
    elif getattr(dataset, "label2id", None):
        num_classes = len(dataset.label2id)
    elif getattr(dataset, "_win_labels", None):
        num_classes = int(max(dataset._win_labels)) + 1
    else:
        raise RuntimeError("[Eval] 无法推断类别数。")

    print(f"[Eval] dataset_type={getattr(dataset,'dataset_type','?')} | windows={len(dataset)} "
          f"| dose_steps={dataset.dose_steps} | op_steps={dataset.op_steps} "
          f"| ratio={dataset.dose_interval//max(1, dataset.op_interval)} | N_classes={num_classes}")
    dist = Counter(getattr(dataset, "_win_labels", []))
    print("[Eval] 全集分布：", dict(sorted(dist.items())))
    return dataset, num_classes


def _map_model_args(args, dataset) -> Tuple[dict, dict]:
    sig_m = inspect.signature(TemporalPhysicalHeteroGNN_V2.__init__)
    allowed_m = set(sig_m.parameters.keys()) - {"self"}

    cand_m = {
        'trans_dim'   : getattr(args, 'trans_dim', 256),
        'trans_layers': getattr(args, 'trans_layers', 2),
        'nhead'       : getattr(args, 'nhead', 4),
        'gcn_hidden'  : getattr(args, 'gnn_hidden', getattr(args, 'gcn_hidden', 512)),
        'gcn_layers'  : getattr(args, 'gnn_layers', getattr(args, 'gcn_layers', 2)),
        'dropout'     : getattr(args, 'dropout', 0.1),
        'num_classes' : getattr(args, 'num_classes_override', None) or
                        (len(getattr(dataset, 'label2id', {})) or
                         int(max(getattr(dataset, '_win_labels', [1]))) + 1),
        'num_op'      : int(getattr(dataset, 'num_op', 0)),
        'num_dose'    : int(getattr(dataset, 'num_dose', 0)),
        'op_seq_len'  : int(getattr(dataset, 'op_steps', 0)),
        'dose_seq_len': int(getattr(dataset, 'dose_steps', 0)),
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
    missing_required = [p.name for p in sig_m.parameters.values()
                        if p.default is inspect._empty and p.name not in init_kwargs_m and p.name != 'self']
    if missing_required:
        raise TypeError(f"[Eval] 模型构造缺少必要参数: {missing_required}. 当前映射: {used_map}")
    return init_kwargs_m, used_map


@torch.no_grad()
def _evaluate_collect(model: nn.Module,
                      loader: DataLoader,
                      device: str,
                      use_amp: bool = True,
                      embedding_mode: str = "fused",
                      shared_ratio: float = 0.5):
    """收集 logits、选定嵌入(z_fused或shared)、y_true、y_pred。
    embedding_mode: 'fused' | 'shared'
    shared_ratio: 用于 'shared' 模式下的 shared/private 切分比例
    """
    model.eval()
    amp_dev = 'cuda' if (str(device).startswith('cuda') and torch.cuda.is_available()) else 'cpu'
    enabled = bool(use_amp and amp_dev == 'cuda')

    logits_list, zembed_list, y_list, yhat_list = [], [], [], []
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        with autocast(device_type=amp_dev, enabled=enabled):
            out = model(batch)
            if isinstance(out, (tuple, list)):
                logits = out[0]
                z_fused = out[1] if len(out) > 1 else None
                z_op    = out[2] if len(out) > 2 else None
                z_dose  = out[3] if len(out) > 3 else None
            else:
                logits = out
                z_fused = None
                z_op = None
                z_dose = None
            pred = logits.argmax(dim=1)

        logits_list.append(logits.detach().cpu())
        # 选择可视化用的嵌入
        z_sel = None
        if embedding_mode == "shared" and (z_op is not None) and (z_dose is not None) and (z_op.size(-1) == z_dose.size(-1)):
            d = z_op.size(-1)
            h = int(d * shared_ratio)
            h = max(1, min(d - 1, h))
            try:
                z_sel = 0.5 * (z_op[:, :h] + z_dose[:, :h])
            except Exception:
                z_sel = None
        if z_sel is None:
            z_sel = z_fused
        if z_sel is not None:
            zembed_list.append(z_sel.detach().cpu())
        y_list.append(batch.y.detach().cpu())
        yhat_list.append(pred.detach().cpu())

    logits_all = torch.cat(logits_list, dim=0).numpy() if logits_list else np.zeros((0, 1), dtype=np.float32)
    zfused_all = torch.cat(zembed_list, dim=0).numpy() if zembed_list else None
    y_true = torch.cat(y_list, dim=0).numpy() if y_list else np.zeros((0,), dtype=np.int64)
    y_pred = torch.cat(yhat_list, dim=0).numpy() if yhat_list else np.zeros((0,), dtype=np.int64)
    return logits_all, zfused_all, y_true, y_pred


def _softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def _calc_specificity_fnr(cm: np.ndarray):
    """基于混淆矩阵计算每类 specificity 与 FNR，以及 macro/weighted 平均。"""
    num_classes = cm.shape[0]
    total = cm.sum()
    spec_list, fnr_list, support = [], [], []
    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - TP - FN - FP
        spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
        fnr = FN / (FN + TP) if (FN + TP) > 0 else np.nan
        spec_list.append(spec)
        fnr_list.append(fnr)
        support.append(cm[i, :].sum())
    support = np.array(support, dtype=np.float64)
    w = support / max(1, support.sum())
    spec_arr = np.array(spec_list, dtype=np.float64)
    fnr_arr = np.array(fnr_list, dtype=np.float64)
    spec_macro = np.nanmean(spec_arr)
    fnr_macro = np.nanmean(fnr_arr)
    spec_weighted = np.nansum(spec_arr * w)
    fnr_weighted = np.nansum(fnr_arr * w)
    return (spec_macro, spec_weighted, fnr_macro, fnr_weighted), (spec_arr, fnr_arr)


def _plot_tsne(z: np.ndarray, y: np.ndarray, id2label: Optional[Dict[int, str]], out_png: str,
               perplexity: float = 30.0, lr: float = 200.0, n_iter: int = 1000, seed: int = 42):
    n = z.shape[0]
    if n < 3:
        print("[Eval][TSNE] 样本数过少，跳过 t-SNE。")
        return
    # 自动约束 perplexity
    perplexity = max(5.0, min(perplexity, (n - 1) / 3))
    
    emb2d = TSNE(n_components=2, perplexity=perplexity, learning_rate=lr,
                 init="random", random_state=seed,
                 metric="euclidean").fit_transform(z)

    classes = np.unique(y)
    labels  = [id2label.get(c, str(c)) for c in classes] if id2label else [str(c) for c in classes]
    cmap = cm.get_cmap("tab20", len(classes))

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    for idx_class, (c, name) in enumerate(zip(classes, labels)):
        mask = (y == c)
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                   s=10, alpha=0.75,
                   label=name,
                   color=cmap(idx_class))

    leg = ax.legend(loc="best", fontsize=8, markerscale=2, frameon=True)
    leg.get_frame().set_edgecolor("grey")
    leg.get_frame().set_linewidth(1.0)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("t-SNE of Fused Embeddings")
    for spine in ax.spines.values():
        spine.set_edgecolor("black"); spine.set_linewidth(1.0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


def _plot_cm(cm: np.ndarray, labels: List[str], out_png: str, title: str = "Confusion Matrix"):
    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap="Blues")
    ax.set_title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick = np.arange(len(labels))
    ax.set_xticks(tick)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
    ax.set_yticks(tick)
    ax.set_yticklabels(labels, fontsize=13.8)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         ax.text(j, i, str(cm[i, j]),
    #                 ha="center", va="center",
    #                 color="white" if cm[i, j] > thresh else "black", fontsize=5)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


def _plot_roc(prob: np.ndarray, y_true: np.ndarray, num_classes: int,
              id2label: Optional[Dict[int, str]], out_png: str):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cmap = cm.get_cmap("tab20", num_classes)

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, prob[:, 1])
        ax.plot(fpr, tpr, lw=4, color=cmap(1),
                label=f"AUC = {sk_auc(fpr, tpr):.3f}")
    else:
        y_onehot = label_binarize(y_true, classes=list(range(num_classes)))
        for c in range(num_classes):
            if y_onehot[:, c].sum() in {0, len(y_onehot)}:
                continue
            fpr, tpr, _ = roc_curve(y_onehot[:, c], prob[:, c])
            ax.plot(fpr, tpr, lw=4, color=cmap(c),
                    label=f"{id2label.get(c, c) if id2label else c} (AUC={sk_auc(fpr, tpr):.3f})")
        fpr_m, tpr_m, _ = roc_curve(y_onehot.ravel(), prob.ravel())
        ax.plot(fpr_m, tpr_m, lw=4.5, ls="--", color="k",
                label=f"micro (AUC={sk_auc(fpr_m, tpr_m):.3f})")

    ax.plot([0, 1], [0, 1], ls=":", lw=3, color="grey")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.set_title("ROC Curve", fontsize=18)
    ax.grid(True, ls="--", lw=3, alpha=0.4)

    for spine in ax.spines.values():
        spine.set_edgecolor("black"); spine.set_linewidth(2.0)

    # leg = ax.legend(fontsize=9, loc="lower right", frameon=True, ncol=2)
    # leg.get_frame().set_edgecolor("grey"); leg.get_frame().set_linewidth(1.0)
    # leg.get_frame().set_facecolor("white")
    
    # 获取图例句柄和标签，用于外部绘制
    handles, labels = ax.get_legend_handles_labels()
    
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return handles, labels


def _plot_pr(prob: np.ndarray, y_true: np.ndarray, num_classes: int,
             id2label: Optional[Dict[int, str]], out_png: str):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cmap = cm.get_cmap("tab20", num_classes)

    if num_classes == 2:
        prec, rec, _ = precision_recall_curve(y_true, prob[:, 1])
        ax.plot(rec, prec, lw=4, color=cmap(1), label=f"AP = {average_precision_score(y_true, prob[:, 1]):.3f}")
    else:
        y_onehot = label_binarize(y_true, classes=list(range(num_classes)))
        for c in range(num_classes):
            if y_onehot[:, c].sum() in {0, len(y_onehot)}:
                continue
            prec, rec, _ = precision_recall_curve(y_onehot[:, c], prob[:, c])
            ap = average_precision_score(y_onehot[:, c], prob[:, c])
            ax.plot(rec, prec, lw=4, color=cmap(c),
                    label=f"{id2label.get(c, c) if id2label else c} (AP={ap:.3f})")
        prec_m, rec_m, _ = precision_recall_curve(y_onehot.ravel(), prob.ravel())
        ap_m = average_precision_score(y_onehot, prob, average="micro")
        ax.plot(rec_m, prec_m, lw=4.5, ls="--", color="k",
                label=f"micro (AP={ap_m:.3f})")

    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    ax.set_title("Precision–Recall Curve", fontsize=18)
    ax.grid(True, ls="--", lw=3, alpha=0.4)

    for spine in ax.spines.values():
        spine.set_edgecolor("black"); spine.set_linewidth(2.0)

    # leg = ax.legend(fontsize=9, loc="lower left", frameon=True, ncol=2)
    # leg.get_frame().set_edgecolor("grey"); leg.get_frame().set_linewidth(1.0)
    # leg.get_frame().set_facecolor("white")
    
    # 获取图例句柄和标签
    handles, labels = ax.get_legend_handles_labels()

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return handles, labels

def _plot_shared_legend(handles, labels, out_png: str):
    """绘制一个单独的图例图片"""
    # 创建一个新的 figure，专门用来放 legend
    # 大小可以根据类别数量动态调整，这里给一个大概的尺寸
    n_cols = 4  # 列数
    n_rows = (len(labels) + n_cols - 1) // n_cols
    fig_leg = plt.figure(figsize=(n_cols * 2.5, n_rows * 0.5), dpi=300)
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis('off')
    
    leg = ax_leg.legend(handles, labels, loc='center', ncol=n_cols, frameon=True, fontsize=14)
    leg.get_frame().set_edgecolor("grey")
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_facecolor("white")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, bbox_inches='tight')
    plt.close(fig_leg)



# ---------- 读取 ckpt（带 allowlist + 回退） ----------
def _load_ckpt_state(ckpt_path: str, need_meta: bool) -> Tuple[dict, Optional[dict]]:
    """
    返回：(state_weights, state_meta)
    - state_weights: 用于提取 'model' 权重的对象（可能就是 state_dict）
    - state_meta: 若 need_meta=True，尽量以非安全模式读全量（含 test_idx 等）；否则返回 None
    """
    print(f"[Eval] 使用 checkpoint: {ckpt_path}")

    # 先配置 allowlist，提升 weights_only=True/False 的成功率
    try:
        import numpy as np
        import torch.serialization as ts
        try:
            from numpy.core.multiarray import _reconstruct as _np_reconstruct
            ts.add_safe_globals([_np_reconstruct])
        except Exception:
            pass
        try:
            ts.add_safe_globals([np.ndarray, np.dtype])
        except Exception:
            pass
    except Exception:
        pass

    # 1) 优先安全加载用于权重
    state_weights = None
    try:
        state_weights = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # PyTorch ≥ 2.4
        print("[Eval] 使用 weights_only=True 成功加载（权重）。")
    except TypeError:
        state_weights = torch.load(ckpt_path, map_location="cpu")
        print("[Eval] 当前 PyTorch 不支持 weights_only；已使用常规加载（权重）。")
    except Exception as e:
        print(f"[Eval][WARN] weights_only=True 加载失败：{e}")
        print("[Eval][WARN] 将回退到 weights_only=False 加载（需信任该 checkpoint）。")
        state_weights = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 2) 若需要 meta（为了读取 test_idx 等），尽量再读一次“全量”状态
    state_meta = None
    if need_meta:
        try:
            state_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # 可能含 test_idx/splits
            print("[Eval] 已加载 checkpoint 全量元数据（含 test_idx 等，若存在）。")
        except Exception as e:
            print(f"[Eval][WARN] 读取 checkpoint 元数据失败：{e}，将无法使用 ckpt 内置测试划分。")
            state_meta = None

    return state_weights, state_meta


def _extract_ckpt_test_indices(state_meta: Optional[dict]) -> Optional[List[int]]:
    """从 ckpt 元数据里尽力挖掘 test_idx。支持常见键名：test_idx / test_indices / splits / split。"""
    if not isinstance(state_meta, dict):
        return None
    keys_candidates = ["test_idx", "test_indices", "test_index", "test_ids", "test"]
    for k in keys_candidates:
        if k in state_meta and isinstance(state_meta[k], (list, np.ndarray)):
            arr = np.array(state_meta[k]).astype(np.int64).tolist()
            return arr

    # 兼容某些保存结构，比如：state_meta['splits'] = {'train': [...], 'val': [...], 'test': [...]}
    for k in ["splits", "split"]:
        if k in state_meta and isinstance(state_meta[k], dict):
            for tk in ["test", "test_idx", "test_indices"]:
                if tk in state_meta[k]:
                    arr = np.array(state_meta[k][tk]).astype(np.int64).tolist()
                    return arr
    return None


# ---------------- 主流程 ----------------
def eval_only(args):
    """
    仅评估：构建数据与模型，加载 checkpoint，在测试集上评估。
    输出：
      - 指标报告（--save_report）
      - 预测 CSV（--save_pred）
      - 混淆矩阵 .npy（--save_cm）与 .png（--save_cm_png）
      - t-SNE 图（--save_tsne，基于 z_fused）
      - ROC 曲线（--save_roc_png）
      - PR  曲线（--save_pr_png）
    """
    device = args.device if (torch.cuda.is_available() or "cpu" in str(args.device).lower()) else "cpu"

    # 1) checkpoint
    ckpt_path = None
    if getattr(args, 'resume', None):
        ckpt_path = _maybe_from_dir(args.resume)
    if ckpt_path is None:
        ckpt_path = _infer_best_ckpt(args.save_dir, args.exp_name)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[Eval] 找不到有效的 checkpoint：{ckpt_path or '(None)'}")

    # 是否需要从 ckpt 读取测试集划分
    force_resplit = bool(getattr(args, 'force_resplit', True))
    need_meta = not force_resplit

    # 更健壮地加载 ckpt（权重 + 可选元数据）
    state_weights, state_meta = _load_ckpt_state(ckpt_path, need_meta=need_meta)

    # 2) 数据
    dataset, num_classes = _build_dataset(args)
    labels_all_list: List[int] = getattr(dataset, '_win_labels', [])
    id2label = None
    if getattr(dataset, "label2id", None):
        id2label = {v: k for k, v in dataset.label2id.items()}

    # 3) 测试划分
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(dataset))
    labels_all = np.asarray(labels_all_list, dtype=np.int64) if labels_all_list else np.array([], dtype=np.int64)
    test_ratio = float(getattr(args, 'test_ratio', 0.2))
    seed = int(getattr(args, 'seed', 42))

    test_idx: Optional[List[int]] = None
    if not force_resplit:
        # 优先尝试 ckpt 内置划分
        test_idx_candidate = _extract_ckpt_test_indices(state_meta)
        if test_idx_candidate:
            # 过滤掉越界索引（如果训练/评估窗口生成方式不一致，可能越界）
            test_idx_inrange = [int(i) for i in test_idx_candidate if 0 <= int(i) < len(dataset)]
            if len(test_idx_inrange) >= max(1, int(0.05 * len(test_idx_candidate))):
                test_idx = test_idx_inrange
                print(f"[Eval] 使用 ckpt 内置测试集划分（有效索引 {len(test_idx_inrange)}/{len(test_idx_candidate)}）。")
            else:
                print("[Eval][WARN] ckpt 内置 test_idx 与当前数据不匹配（越界过多），将改为重新划分。")
        else:
            print("[Eval] ckpt 未提供 test_idx，改为重新划分。")

    if test_idx is None:
        # 分层/非分层 自动回退
        stratify_arg = None
        if labels_all.size > 0:
            cls_counts = np.bincount(labels_all, minlength=int(labels_all.max()) + 1)
            if (cls_counts < 2).any():
                print("[Eval] 警告：存在样本数 < 2 的类别，分层抽样不可用，将改为非分层随机划分。")
                stratify_arg = None
            else:
                stratify_arg = labels_all

        _, test_idx_np = train_test_split(
            idx,
            test_size=test_ratio,
            random_state=seed,
            stratify=stratify_arg
        )
        test_idx = list(map(int, test_idx_np.tolist()))
        print(f"[Eval] 已重新划分测试集（test_ratio={test_ratio}, seed={seed}, "
              f"stratify={'yes' if stratify_arg is not None else 'no'}）；样本数: {len(test_idx)}")

    test_set = Subset(dataset, test_idx)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,  # 评估不打乱，确保可复现
        num_workers=getattr(args, 'num_workers', 0),
        pin_memory=True, persistent_workers=False
    )

    # 4) 模型
    model_kwargs, used_map = _map_model_args(args, dataset)
    print("[Eval] 模型参数映射：", ", ".join(f"{k}->{v}" for k, v in used_map.items()))
    model = TemporalPhysicalHeteroGNN_V2(**model_kwargs).to(device)

    # —— 仅加载“形状匹配”的权重，避免 size mismatch 报错（典型为分类头）
    if isinstance(state_weights, dict) and 'model' in state_weights and isinstance(state_weights['model'], dict):
        ckpt_sd = state_weights['model']
    else:
        ckpt_sd = state_weights  # weights_only=True 时通常直接是 state_dict

    msd = model.state_dict()
    compatible_sd = {k: v for k, v in ckpt_sd.items() if (k in msd and hasattr(v, "shape") and v.shape == msd[k].shape)}
    ignored = sorted(set(ckpt_sd.keys()) - set(compatible_sd.keys()))
    print(f"[Eval] 忽略形状不匹配参数（未加载）: {ignored[:12]}{' ...' if len(ignored) > 12 else ''}")

    missing = model.load_state_dict(compatible_sd, strict=False)
    if getattr(missing, 'missing_keys', None):
        print("[Eval] state_dict 缺失键：", missing.missing_keys)
    if getattr(missing, 'unexpected_keys', None):
        print("[Eval] state_dict 多余键：", missing.unexpected_keys)

    # 5) 推理收集
    embedding_mode = 'shared' if bool(getattr(args, 'tsne_use_shared', False)) else 'fused'
    logits_all, zfused_all, y_true, y_pred = _evaluate_collect(
        model, test_loader, device=device, use_amp=getattr(args, 'use_amp', True),
        embedding_mode=embedding_mode, shared_ratio=float(getattr(args, 'shared_ratio', 0.5))
    )

    # 6) 指标
    acc = accuracy_score(y_true, y_pred) if y_true.size > 0 else 0.0
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if y_true.size > 0 else 0.0

    auc_val = float("nan")
    prob = None
    if logits_all.size > 0:
        prob = _softmax_np(logits_all, axis=1)
        try:
            if model_kwargs.get('num_classes', num_classes) == 2:
                auc_val = roc_auc_score(y_true, prob[:, 1])
            else:
                y_true_oh = label_binarize(y_true, classes=list(range(model_kwargs.get('num_classes', num_classes))))
                auc_val = roc_auc_score(y_true_oh, prob, average="macro", multi_class="ovr")
        except Exception as e:
            print(f"[Eval][WARN] 计算 AUC 失败：{e}; 置为 NaN")

    real_num_classes = model_kwargs.get('num_classes', num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(real_num_classes)))
    (spec_macro, spec_weighted, fnr_macro, fnr_weighted), (spec_arr, fnr_arr) = _calc_specificity_fnr(cm)

    # 7) 保存：报告、预测、混淆矩阵、t-SNE、ROC、PR
    if getattr(args, 'save_report', ''):
        os.makedirs(os.path.dirname(args.save_report) or ".", exist_ok=True)
        with open(args.save_report, "w", encoding="utf-8") as f:
            f.write(f"Accuracy: {acc:.6f}\n")
            f.write(f"Precision (macro):    {prec_m:.6f}\n")
            f.write(f"Precision (weighted): {prec_w:.6f}\n")
            f.write(f"Recall    (macro):    {rec_m:.6f}\n")
            f.write(f"Recall    (weighted): {rec_w:.6f}\n")
            f.write(f"F1        (macro):    {f1_m:.6f}\n")
            f.write(f"F1        (weighted): {f1_w:.6f}\n")
            f.write(f"AUC: {auc_val:.6f}\n")
            f.write(f"MCC: {mcc:.6f}\n")
            f.write(f"Specificity (macro):    {spec_macro:.6f}\n")
            f.write(f"Specificity (weighted): {spec_weighted:.6f}\n")
            f.write(f"FNR         (macro):    {fnr_macro:.6f}\n")
            f.write(f"FNR         (weighted): {fnr_weighted:.6f}\n")
        print(f"[Eval] 指标报告已保存到: {args.save_report}")

    if getattr(args, 'save_pred', ''):
        out_csv = args.save_pred
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("idx,y_true,y_pred,true_name,pred_name\n")
            id2label = {v: k for k, v in dataset.label2id.items()} if getattr(dataset, "label2id", None) else None
            for i, (t, p) in enumerate(zip(y_true, y_pred)):
                tn = id2label.get(int(t), "") if id2label else ""
                pn = id2label.get(int(p), "") if id2label else ""
                f.write(f"{i},{int(t)},{int(p)},{tn},{pn}\n")
        print(f"[Eval] 预测结果已保存到: {out_csv}")

    if getattr(args, 'save_cm', ''):
        os.makedirs(os.path.dirname(args.save_cm) or ".", exist_ok=True)
        np.save(args.save_cm, cm)
        print(f"[Eval] 混淆矩阵 (numpy) 已保存到: {args.save_cm}")

    if getattr(args, 'save_cm_png', ''):
        cls_names = [id2label.get(i, str(i)) for i in range(real_num_classes)] if id2label else [str(i) for i in range(real_num_classes)]
        _plot_cm(cm, cls_names, args.save_cm_png, title="Confusion Matrix")
        print(f"[Eval] 混淆矩阵图已保存到: {args.save_cm_png}")

    if getattr(args, 'save_tsne', '') and (zfused_all is not None):
        _plot_tsne(
            zfused_all, y_true, id2label, args.save_tsne,
            perplexity=getattr(args, 'tsne_perplexity', 30.0),
            lr=getattr(args, 'tsne_lr', 200.0),
            n_iter=getattr(args, 'tsne_iter', 1000),
            seed=getattr(args, 'seed', 42)
        )
        print(f"[Eval] t-SNE 图已保存到: {args.save_tsne}")
    elif getattr(args, 'save_tsne', '') and (zfused_all is None):
        print("[Eval][WARN] 模型未返回 z_fused，无法绘制 t-SNE。")

    if prob is not None:
        handles, labels = None, None
        if getattr(args, 'save_roc_png', ''):
            h, l = _plot_roc(prob, y_true, real_num_classes, id2label, args.save_roc_png)
            if handles is None: handles, labels = h, l
            print(f"[Eval] ROC 曲线已保存到: {args.save_roc_png}")
        if getattr(args, 'save_pr_png', ''):
            h, l = _plot_pr(prob, y_true, real_num_classes, id2label, args.save_pr_png)
            if handles is None: handles, labels = h, l
            print(f"[Eval] PR 曲线已保存到: {args.save_pr_png}")
        
        # 绘制共享图例
        if handles is not None and (getattr(args, 'save_roc_png', '') or getattr(args, 'save_pr_png', '')):
            # 假设图例保存在 ROC 或 PR 同级目录下，命名为 legend.png
            base_dir = os.path.dirname(args.save_roc_png or args.save_pr_png)
            legend_path = os.path.join(base_dir, "legend.png")
            _plot_shared_legend(handles, labels, legend_path)
            print(f"[Eval] 共享图例已保存到: {legend_path}")
    else:
        if getattr(args, 'save_roc_png', '') or getattr(args, 'save_pr_png', ''):
            print("[Eval][WARN] 未得到概率分布(probabilities)，无法绘制 ROC/PR。")

    print(
        "ACC={:.4f} | Pm={:.4f} Pw={:.4f} | Rm={:.4f} Rw={:.4f} | "
        "F1m={:.4f} F1w={:.4f} | AUC={:.4f} | MCC={:.4f} | "
        "Spec_m={:.4f} Spec_w={:.4f} | FNR_m={:.4f} FNR_w={:.4f}".format(
            acc, prec_m, rec_w, rec_m, rec_w, f1_m, f1_w, auc_val, mcc, spec_macro, spec_weighted, fnr_macro, fnr_weighted
        )
    )

    return {
        "acc": acc,
        "precision_macro": prec_m, "precision_weighted": prec_w,
        "recall_macro": rec_m, "recall_weighted": rec_w,
        "f1_macro": f1_m, "f1_weighted": f1_w,
        "auc": auc_val, "mcc": mcc,
        "spec_macro": spec_macro, "spec_weighted": spec_weighted,
        "fnr_macro": fnr_macro, "fnr_weighted": fnr_weighted,
        "cm": cm
    }


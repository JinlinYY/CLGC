# utils/robust_eval.py
# -*- coding: utf-8 -*-

import os, inspect, argparse, warnings
from typing import Optional, List, Dict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# 数据集 & 模型
import dataset.data_sampling_lag_edge_attr as ds
from models.temporal_hetero_gnn_edge_attr_contrastive import TemporalPhysicalHeteroGNN_V2

torch.backends.cudnn.benchmark = True


# ========= CLI =========
def get_args():
    p = argparse.ArgumentParser("Robustness evaluation (noise/fault + tqdm + full metrics + optional sweep)")

    # 基本路径
    p.add_argument("--dataset", type=str, default="NPS", choices=["NPS", "TFF"])
    p.add_argument("--op_dir", type=str, default="/tmp/pycharm_project_856/Data/Operation_csv_data")
    p.add_argument("--dose_dir", type=str, default="/tmp/pycharm_project_856/Data/Dose_csv_data")
    p.add_argument("--ckpt", type=str, default="/tmp/pycharm_project_856/checkpoints/NPS18-(残差MLP-Adam)-消融BOTH_best.pt")

    # 图 & 滑窗
    p.add_argument("--op_interval", type=int, default=10)
    p.add_argument("--dose_interval", type=int, default=60)
    p.add_argument("--dose_window_steps", type=int, default=10)
    p.add_argument("--dose_stride_steps", type=int, default=1)
    p.add_argument("--op_topk", type=int, default=10)
    p.add_argument("--dose_topk", type=int, default=10)
    p.add_argument("--topk_cross", type=int, default=3)
    p.add_argument("--min_abs_corr", type=float, default=0.2)
    p.add_argument("--max_lag_dose", type=int, default=5)

    # 划分（分层 8:2）
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.2)

    # 模态 / 设备 / DataLoader
    p.add_argument("--modal", type=str, default="both", choices=["both", "op", "dose"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--use_amp", type=lambda x: str(x).lower() in ["1", "true", "yes"], default=True)

    # 失真（单次）
    p.add_argument("--robust_mode", type=str, default="noise", choices=["noise", "fault"])
    p.add_argument("--noise_std", type=float, default=0.05)
    p.add_argument("--fault_ratio", type=float, default=0.1)
    p.add_argument("--fault_seed", type=int, default=0)
    p.add_argument("--fault_level", type=str, default="sensor", choices=["time", "sensor"],
                   help="time=按列/时间片置零；sensor=按节点/传感器置零")

    # Sweep（可选）
    p.add_argument("--sweep", type=str, default="fault", choices=["none", "noise", "fault", "both"])
    p.add_argument("--noise_grid", type=str, default="0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30", help='如 "0,0.01,0.02,0.05,0.1"')
    p.add_argument("--fault_grid", type=str, default="0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30", help='如 "0,0.05,0.1,0.2,0.4"')

    # 输出
    p.add_argument("--save_dir", type=str, default="/tmp/pycharm_project_856/eval_outputs")

    # 进度条开关
    p.add_argument("--no_tqdm", action="store_true", help="关闭 tqdm 进度条显示")

    return p.parse_args()


# ========= 小工具 =========
def parse_grid(s: str) -> List[float]:
    if not s:
        return []
    for sep in [";", " "]:
        s = s.replace(sep, ",")
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    return sorted(set(vals))


def tag_prefix(args, mode: str) -> str:
    base = os.path.splitext(os.path.basename(args.ckpt))[0]
    return f"{args.dataset}_{args.modal}_{mode}_{base}"


def ensure_dir(d: str):
    if d:
        os.makedirs(d, exist_ok=True)


# ========= 扰动（batch 就地矢量化） =========
def corrupt_batch_inplace(batch,
                          mode: str,
                          noise_std: float = 0.05,
                          fault_ratio: float = 0.1,
                          fault_level: str = "time",
                          rng: Optional[torch.Generator] = None):
    node_types = getattr(batch, "node_types", [])
    if not node_types:
        return batch

    dev = None
    if 'op' in node_types and hasattr(batch['op'], 'x'):
        dev = batch['op'].x.device
    elif 'dose' in node_types and hasattr(batch['dose'], 'x'):
        dev = batch['dose'].x.device
    else:
        return batch
    gen = rng if rng is not None else torch.Generator(device=dev).manual_seed(0)

    if mode == "noise":
        if 'op' in node_types and hasattr(batch['op'], 'x'):
            batch['op'].x.add_(torch.randn_like(batch['op'].x, device=dev) * noise_std)
        if 'dose' in node_types and hasattr(batch['dose'], 'x'):
            batch['dose'].x.add_(torch.randn_like(batch['dose'].x, device=dev) * noise_std)
        return batch

    if mode == "fault":
        if fault_level == "time":
            if 'op' in node_types and hasattr(batch['op'], 'x'):
                x = batch['op'].x
                C = x.size(1)
                G = int(batch['op'].batch.max().item() + 1) if hasattr(batch['op'], 'batch') else 1
                keep = (torch.rand(G, C, generator=gen, device=dev) > fault_ratio).float()
                x.mul_(keep[batch['op'].batch.long()])
            if 'dose' in node_types and hasattr(batch['dose'], 'x'):
                x = batch['dose'].x
                C = x.size(1)
                G = int(batch['dose'].batch.max().item() + 1) if hasattr(batch['dose'], 'batch') else 1
                keep = (torch.rand(G, C, generator=gen, device=dev) > fault_ratio).float()
                x.mul_(keep[batch['dose'].batch.long()])
            return batch

        if fault_level == "sensor":
            if 'op' in node_types and hasattr(batch['op'], 'x'):
                x = batch['op'].x
                N = x.size(0)
                drop = (torch.rand(N, 1, generator=gen, device=dev) < fault_ratio).float()
                x.mul_(1.0 - drop)
            if 'dose' in node_types and hasattr(batch['dose'], 'x'):
                x = batch['dose'].x
                N = x.size(0)
                drop = (torch.rand(N, 1, generator=gen, device=dev) < fault_ratio).float()
                x.mul_(1.0 - drop)
            return batch

        raise ValueError(f"未知 fault_level: {fault_level}")

    raise ValueError(f"未知失真模式: {mode}")


# ========= 单次评测（含 tqdm + 全指标） =========
@torch.no_grad()
def eval_once(loader, model, dev, use_amp: bool,
              robust_mode: str,
              noise_std: float,
              fault_ratio: float,
              fault_level: str,
              fault_seed: int,
              show_tqdm: bool = True,
              tqdm_desc: str = "Robust testing") -> Dict[str, float]:
    y_true, y_pred, y_prob = [], [], []
    rng = torch.Generator(device=dev.type).manual_seed(fault_seed)

    iterable = tqdm(loader, desc=tqdm_desc, dynamic_ncols=True, leave=False) if show_tqdm else loader
    for batch in iterable:
        batch = batch.to(dev, non_blocking=True)
        corrupt_batch_inplace(
            batch,
            mode=robust_mode,
            noise_std=noise_std,
            fault_ratio=fault_ratio,
            fault_level=fault_level,
            rng=rng
        )
        with torch.autocast("cuda", enabled=use_amp):
            out = model(batch)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            prob = F.softmax(logits, dim=1)

        y_true.append(batch.y.detach().cpu())
        y_pred.append(logits.argmax(dim=1).detach().cpu())
        y_prob.append(prob.detach().cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    y_prob = torch.cat(y_prob).numpy()

    # 基本指标（macro/weighted）
    acc   = accuracy_score(y_true, y_pred)
    prec_m = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_m  = recall_score(y_true, y_pred,   average="macro",     zero_division=0)
    rec_w  = recall_score(y_true, y_pred,   average="weighted",  zero_division=0)
    f1_m   = f1_score(y_true, y_pred,       average="macro",     zero_division=0)
    f1_w   = f1_score(y_true, y_pred,       average="weighted",  zero_division=0)
    mcc    = matthews_corrcoef(y_true, y_pred)

    # AUC（仅对出现的类）
    present = np.unique(y_true)
    if len(present) >= 2:
        y_prob_present = y_prob[:, present]
        try:
            auc_val = roc_auc_score(y_true, y_prob_present, labels=present, multi_class="ovo")
        except ValueError:
            auc_val = float('nan')
    else:
        auc_val = float('nan')

    # Specificity / FNR
    cm = confusion_matrix(y_true, y_pred, labels=present)
    spec_arr, fnr_arr = [], []
    total = cm.sum()
    class_counts = cm.sum(axis=1).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(len(present)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = total - tp - fn - fp
            spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            fnr  = fn / (fn + tp) if (fn + tp) > 0 else np.nan
            spec_arr.append(spec)
            fnr_arr.append(fnr)
    spec_arr = np.array(spec_arr, dtype=float)
    fnr_arr  = np.array(fnr_arr, dtype=float)

    spec_macro = float(np.nanmean(spec_arr)) if spec_arr.size > 0 else float('nan')
    fnr_macro  = float(np.nanmean(fnr_arr))  if fnr_arr.size  > 0 else float('nan')

    weights = class_counts / class_counts.sum() if class_counts.sum() > 0 else np.ones_like(class_counts) / max(1, len(class_counts))

    def nan_weighted_mean(x, w):
        m = ~np.isnan(x)
        if m.sum() == 0:
            return float('nan')
        w_sub = w[m]
        if w_sub.sum() == 0:
            return float('nan')
        return float(np.dot(x[m], w_sub) / w_sub.sum())

    spec_weighted = nan_weighted_mean(spec_arr, weights)
    fnr_weighted  = nan_weighted_mean(fnr_arr,  weights)

    return dict(
        Accuracy=acc,
        Precision_macro=prec_m, Precision_weighted=prec_w,
        Recall_macro=rec_m,    Recall_weighted=rec_w,
        F1_macro=f1_m,         F1_weighted=f1_w,
        AUC=auc_val, MCC=mcc,
        Specificity_macro=spec_macro, Specificity_weighted=spec_weighted,
        FNR_macro=fnr_macro,          FNR_weighted=fnr_weighted
    )


def format_detail_lines(metrics: Dict[str, float]) -> List[str]:
    # 按你指定的逐行格式
    return [
        f"Accuracy: {metrics['Accuracy']:.6f}",
        f"Precision (macro):    {metrics['Precision_macro']:.6f}",
        f"Precision (weighted): {metrics['Precision_weighted']:.6f}",
        f"Recall    (macro):    {metrics['Recall_macro']:.6f}",
        f"Recall    (weighted): {metrics['Recall_weighted']:.6f}",
        f"F1        (macro):    {metrics['F1_macro']:.6f}",
        f"F1        (weighted): {metrics['F1_weighted']:.6f}",
        f"AUC: {metrics['AUC']:.6f}",
        f"MCC: {metrics['MCC']:.6f}",
        f"Specificity (macro):    {metrics['Specificity_macro']:.6f}",
        f"Specificity (weighted): {metrics['Specificity_weighted']:.6f}",
        f"FNR         (macro):    {metrics['FNR_macro']:.6f}",
        f"FNR         (weighted): {metrics['FNR_weighted']:.6f}",
    ]


# ========= 主流程 =========
@torch.no_grad()
def main():
    args = get_args()
    dev = torch.device(args.device)
    ensure_dir(args.save_dir)

    # 1) 数据集
    ds_kwargs = dict(
        dataset=args.dataset, op_dir=args.op_dir, dose_dir=args.dose_dir,
        op_interval=args.op_interval, dose_interval=args.dose_interval,
        dose_window_steps=args.dose_window_steps, dose_stride_steps=args.dose_stride_steps,
        op_topk=args.op_topk, dose_topk=args.dose_topk,
        topk_cross=args.topk_cross, min_abs_corr=args.min_abs_corr,
        max_lag_dose=args.max_lag_dose
    )
    full_set = ds.MultiModalHeteroDatasetV2(**ds_kwargs)
    if len(full_set) == 0:
        raise RuntimeError("数据集为空，请检查路径与参数。")

    labels_all = getattr(full_set, "_window_labels", None)
    if labels_all is None:
        labels_all = getattr(full_set, "_win_labels", None)
    if labels_all is None:
        raise RuntimeError("数据集中未找到窗口级标签 _window_labels。")

    idx = np.arange(len(full_set))
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=np.array(labels_all)
    )
    test_set = torch.utils.data.Subset(full_set, test_idx.tolist())

    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=(args.num_workers > 0)
    )

    # 2) 模型
    sig = inspect.signature(TemporalPhysicalHeteroGNN_V2.__init__)
    need = set(sig.parameters) - {"self"}
    model_kwargs = dict(
        num_op=getattr(full_set, "num_op", 0),
        num_dose=getattr(full_set, "num_dose", 0),
        op_seq_len=getattr(full_set, "op_steps", 0),
        dose_seq_len=getattr(full_set, "dose_steps", 0),
        num_classes=len(getattr(full_set, "label2id", {0: 0})),
    )
    if "modal" in need:
        model_kwargs["modal"] = args.modal
    for k, p in sig.parameters.items():
        if k not in model_kwargs and k != "self":
            model_kwargs[k] = p.default
    model = TemporalPhysicalHeteroGNN_V2(**{k: model_kwargs[k] for k in need}).to(dev).eval()

    # 权重
    state = torch.load(args.ckpt, map_location="cpu")
    state_dict = None
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            state_dict = state["model"]
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            state_dict = state["state_dict"]
    if state_dict is None:
        state_dict = state
    model.load_state_dict(state_dict, strict=True)
    print(f"[✓] 已加载权重: {args.ckpt}")

    use_amp = (dev.type == "cuda") and bool(args.use_amp)

    # A) 不做 sweep：单次评测（进度条 + 全指标打印 + 保存）
    if args.sweep == "none":
        metrics = eval_once(
            loader, model, dev, use_amp,
            robust_mode=args.robust_mode,
            noise_std=args.noise_std,
            fault_ratio=args.fault_ratio,
            fault_level=args.fault_level,
            fault_seed=args.fault_seed,
            show_tqdm=not args.no_tqdm,
            tqdm_desc="Robust testing"
        )

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        prefix = tag_prefix(args, args.robust_mode)
        tag = (f"{prefix}_{ts}_std{args.noise_std}" if args.robust_mode == "noise"
               else f"{prefix}_{ts}_r{args.fault_ratio}_lvl{args.fault_level}")
        save_path = os.path.join(args.save_dir, f"{tag}.txt")
        ensure_dir(os.path.dirname(save_path))

        lines = format_detail_lines(metrics)
        with open(save_path, "w", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln + "\n")

        # 控制台完整打印
        print("\n======== Robustness Metrics (full) ========")
        for ln in lines:
            print(ln)
        print("===========================================")
        print(f"[Eval] 指标报告已保存到: {save_path}")
        return

    # B) 做 sweep（每个参数点：进度条 + 全指标打印）
    print(f"[Info] 执行参数扫评：{args.sweep}")
    results_rows = []
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix_noise = tag_prefix(args, "noise")
    prefix_fault = tag_prefix(args, "fault")

    def run_noise_sweep():
        grid = parse_grid(args.noise_grid)
        if not grid:
            raise ValueError("sweep=noise 但未提供 --noise_grid")
        rows = []
        for std in grid:
            m = eval_once(loader, model, dev, use_amp,
                          robust_mode="noise",
                          noise_std=std,
                          fault_ratio=0.0,
                          fault_level=args.fault_level,
                          fault_seed=args.fault_seed,
                          show_tqdm=not args.no_tqdm,
                          tqdm_desc=f"noise={std:.4f}")
            rows.append({"mode": "noise", "level": std, **m})

            print("\n===== Noise Sweep | noise_std={:.6f} | 全部指标 =====".format(std))
            for ln in format_detail_lines(m):
                print(ln)
            print("=====================================================")

        csv_path = os.path.join(args.save_dir, f"{prefix_noise}_SWEEP_{ts}.csv")
        save_rows_to_csv(rows, csv_path)
        txt_path = os.path.join(args.save_dir, f"{prefix_noise}_SWEEP_{ts}.txt")
        save_rows_to_txt(rows, txt_path)
        print(f"[Noise] 结果已保存：\n  CSV: {csv_path}\n  TXT: {txt_path}")
        return rows

    def run_fault_sweep():
        grid = parse_grid(args.fault_grid)
        if not grid:
            raise ValueError("sweep=fault 但未提供 --fault_grid")
        rows = []
        for r in grid:
            m = eval_once(loader, model, dev, use_amp,
                          robust_mode="fault",
                          noise_std=0.0,
                          fault_ratio=r,
                          fault_level=args.fault_level,
                          fault_seed=args.fault_seed,
                          show_tqdm=not args.no_tqdm,
                          tqdm_desc=f"fault={r:.4f} ({args.fault_level})")
            rows.append({"mode": "fault", "level": r, "fault_level": args.fault_level, **m})

            print("\n===== Fault Sweep | fault_ratio={:.6f} | level={} | 全部指标 =====".format(r, args.fault_level))
            for ln in format_detail_lines(m):
                print(ln)
            print("===================================================================")

        csv_path = os.path.join(args.save_dir, f"{prefix_fault}_SWEEP_{ts}.csv")
        save_rows_to_csv(rows, csv_path)
        txt_path = os.path.join(args.save_dir, f"{prefix_fault}_SWEEP_{ts}.txt")
        save_rows_to_txt(rows, txt_path)
        print(f"[Fault] 结果已保存：\n  CSV: {csv_path}\n  TXT: {txt_path}")
        return rows

    if args.sweep in ("noise", "both"):
        results_rows.extend(run_noise_sweep())
    if args.sweep in ("fault", "both"):
        results_rows.extend(run_fault_sweep())

    # 控制台简要汇总
    print("\n================ Sweep Summary (console) ================")
    if results_rows:
        header = ["mode", "level", "Accuracy", "F1_macro", "AUC", "MCC"]
        print("\t".join(header))
        for r in results_rows:
            print("\t".join([
                str(r.get("mode")),
                f"{r.get('level'):.4f}",
                f"{r.get('Accuracy', float('nan')):.4f}",
                f"{r.get('F1_macro', float('nan')):.4f}",
                f"{r.get('AUC', float('nan')):.4f}",
                f"{r.get('MCC', float('nan')):.4f}",
            ]))
    print("=========================================================")


def save_rows_to_csv(rows: List[Dict[str, float]], path: str):
    ensure_dir(os.path.dirname(path))
    fields = [
        "mode", "level", "fault_level",
        "Accuracy", "Precision_macro", "Precision_weighted",
        "Recall_macro", "Recall_weighted",
        "F1_macro", "F1_weighted",
        "AUC", "MCC",
        "Specificity_macro", "Specificity_weighted",
        "FNR_macro", "FNR_weighted",
    ]
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_rows_to_txt(rows: List[Dict[str, float]], path: str):
    ensure_dir(os.path.dirname(path))
    def fmt(x):
        try:
            return f"{float(x):.6f}"
        except Exception:
            return str(x)
    with open(path, "w", encoding="utf-8") as f:
        f.write("mode\tlevel\tfault_level\tAccuracy\tF1_macro\tAUC\tMCC\n")
        for r in rows:
            f.write("\t".join([
                str(r.get("mode", "")),
                fmt(r.get("level", "")),
                str(r.get("fault_level", "")),
                fmt(r.get("Accuracy", float('nan'))),
                fmt(r.get("F1_macro", float('nan'))),
                fmt(r.get("AUC", float('nan'))),
                fmt(r.get("MCC", float('nan'))),
            ]) + "\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

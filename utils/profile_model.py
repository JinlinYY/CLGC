# utils/profile_model.py
# -*- coding: utf-8 -*-
"""
模型规模（Params/FLOPs）与推理耗时评估脚本
------------------------------------------------
用法示例：
python utils/profile_model.py \
    --dataset NPS \
    --op_dir Data/54/Operation_csv_data \
    --dose_dir Data/54/Dose_csv_data \
    --ckpt checkpoints/NPS54_best.pt \
    --device cuda:0
"""

import os, inspect, time, argparse, warnings
import torch
from torch_geometric.loader import DataLoader
from thop import profile, clever_format     # pip install thop
from tqdm import tqdm

# ---------- 数据集与模型 ----------
import dataset.data_sampling_lag_edge_attr as ds
from models.temporal_hetero_gnn_edge_attr_contrastive import TemporalPhysicalHeteroGNN_V2

# ----------------------- 参数 -----------------------
def get_args():
    p = argparse.ArgumentParser("Profile HeteroGNN 模型大小与推理速度")
    # 数据路径
    p.add_argument("--dataset", type=str, default="NPS", choices=["NPS", "TFF"])
    p.add_argument("--op_dir", type=str, default="Data/Operation_csv_data")
    p.add_argument("--dose_dir", type=str, default="Data/Dose_csv_data")
    # 滑窗 & 图参数（保持与训练脚本一致即可）
    p.add_argument("--op_interval", type=int, default=10)
    p.add_argument("--dose_interval", type=int, default=60)
    p.add_argument("--dose_window_steps", type=int, default=10)
    p.add_argument("--dose_stride_steps", type=int, default=1)
    p.add_argument("--op_topk", type=int, default=10)
    p.add_argument("--dose_topk", type=int, default=10)
    p.add_argument("--topk_cross", type=int, default=3)
    p.add_argument("--min_abs_corr", type=float, default=0.2)
    p.add_argument("--max_lag_dose", type=int, default=5)
    # 设备 & 批大小
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=64, help="单次推理批大小")
    # 计时参数
    p.add_argument("--warmup", type=int, default=10, help="warm-up 次数")
    p.add_argument("--repeat", type=int, default=50, help="计时循环次数")
    # checkpoint
    p.add_argument("--ckpt", type=str, default="/tmp/pycharm_project_261/checkpoints/NPS54_best.pt", help="模型权重路径")
    return p.parse_args()

# ----------------------- 主流程 -----------------------
@torch.no_grad()
def main():
    args = get_args()
    device = torch.device(args.device)

    # 1. 构建数据集（与 utils/eval_only.py 中逻辑保持一致，省略检查）
    ds_kwargs = dict(
        dataset=args.dataset, op_dir=args.op_dir, dose_dir=args.dose_dir,
        op_interval=args.op_interval, dose_interval=args.dose_interval,
        dose_window_steps=args.dose_window_steps, dose_stride_steps=args.dose_stride_steps,
        op_topk=args.op_topk, dose_topk=args.dose_topk, topk_cross=args.topk_cross,
        min_abs_corr=args.min_abs_corr, max_lag_dose=args.max_lag_dose
    )
    dataset = ds.MultiModalHeteroDatasetV2(**ds_kwargs)

    # 2. 模型参数自动映射
    sig = inspect.signature(TemporalPhysicalHeteroGNN_V2.__init__)
    need = set(sig.parameters) - {"self"}
    m_kwargs = dict(
        num_op=dataset.num_op, num_dose=dataset.num_dose,
        op_seq_len=dataset.op_steps, dose_seq_len=dataset.dose_steps,
        num_classes=len(getattr(dataset, "label2id", {0:0}))
    )
    # 填默认值
    for k, v in sig.parameters.items():
        if k not in m_kwargs and k != "self":
            m_kwargs[k] = v.default
    model = TemporalPhysicalHeteroGNN_V2(**{k: m_kwargs[k] for k in need})

    # 3. 加载权重（可选）
    if args.ckpt and os.path.isfile(args.ckpt):
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", state))
        print(f"[Info] 权重已加载: {args.ckpt}")

    model.to(device).eval()

    # 4. 构造 dummy batch（真实样本更精确）
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dummy_batch = next(iter(loader)).to(device)

    # 5. 模型规模评估
    print("\n========= 模型规模评估 (Params / FLOPs) =========")
    macs, params = profile(model, inputs=(dummy_batch,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print(f"FLOPs: {macs} | Params: {params}")

    # 6. 推理耗时评估
    print("\n========= 推理耗时评估 =========")
    # warm-up
    for _ in range(args.warmup):
        _ = model(dummy_batch)
    torch.cuda.synchronize() if device.type == "cuda" else None

    # timing
    start = time.time()
    for _ in tqdm(range(args.repeat), desc="Profiling"):
        _ = model(dummy_batch)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()

    avg_ms = (end - start) / args.repeat * 1000
    fps = 1000.0 / avg_ms if avg_ms > 0 else float("inf")
    print(f"平均单次延迟: {avg_ms:.2f} ms | FPS: {fps:.2f}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

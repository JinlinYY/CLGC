# eval.py
# -*- coding: utf-8 -*-
"""
命令行评估入口（含指标报告、混淆矩阵、t-SNE、ROC、PR）
"""

import argparse
import torch
from utils.eval_only import eval_only


def str2bool(x):
    return str(x).lower() in {"1", "true", "yes", "y", "t"}


def build_argparser():
    p = argparse.ArgumentParser("Eval-only for NPS/TFF HeteroGNN")

    # ================= 基本路径与数据集 =================
    p.add_argument("--dataset", type=str, default="NPS", help="NPS / TFF / auto")
    p.add_argument('--op_dir', type=str, default='/tmp/pycharm_project_156/Data/Operation_csv_data',
                   help='Operation 数据或 TFF 根目录（a..g 子文件夹）')
    p.add_argument('--dose_dir', type=str, default='/tmp/pycharm_project_156/Data/Dose_csv_data',
                   help='Dose 数据目录（TFF 单模态可忽略）')

    # ================= 滑窗 / 采样参数 =================
    p.add_argument("--op_interval", type=int, default=10)
    p.add_argument("--dose_interval", type=int, default=60)
    p.add_argument("--dose_window_steps", type=int, default=10)
    p.add_argument("--dose_stride_steps", type=int, default=1)
    p.add_argument("--downsample_mode", type=str, default="pick",
                   choices=["pick", "mean", "sum", "none"])

    # ================= 图构建超参 =================
    p.add_argument("--op_topk", type=int, default=10)
    p.add_argument("--dose_topk", type=int, default=10)
    p.add_argument("--topk_cross", type=int, default=3)
    p.add_argument("--min_abs_corr", type=float, default=0.2)
    p.add_argument("--max_lag_dose", type=int, default=5)

    # ================= 任务与模型公共设置 =================
    p.add_argument("--task_type", type=str, default="multiclass",
                   choices=["binary", "multiclass"])
    p.add_argument("--clip_val", type=float, default=6.0)

    # ---------- 模型结构（与训练保持一致） ----------
    p.add_argument("--trans_dim", type=int, default=256)
    p.add_argument("--trans_layers", type=int, default=2)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--gnn_hidden", type=int, default=512)
    p.add_argument("--gnn_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_classes_override", type=int, default=None)

    # ================= 运行时参数 =================
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_amp", type=str2bool, default=True)

    # ================= 评估行为开关 =================
    p.add_argument("--force_resplit", type=str2bool, default=True,
                   help="是否强制忽略 ckpt 内置 test_idx、每次随机重划分测试集（默认 True）")
    # t-SNE 嵌入选择（可选用 shared 表示用于可视化；默认用 z_fused）
    p.add_argument("--tsne_use_shared", type=str2bool, default=False,
                   help="若为 True，则在可视化中使用 shared 表示（基于 z_op/z_dose 按 shared_ratio 切分并平均）")
    p.add_argument("--shared_ratio", type=float, default=0.15,
                   help="shared/private 维度切分比例（0-1，默认0.5，对半分；仅影响 tsne_use_shared=True 时）")

    # ================= checkpoint 相关 =================
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--exp_name", type=str, default="")
    p.add_argument("--resume", type=str, default="/tmp/pycharm_project_156/checkpoints/NPS18-(改进测试中)_2025-11-26_01-50-39/NPS18-(改进测试中)_best_09401.pt",
                   help="checkpoint 路径或目录（可不填，将尝试从 save_dir/exp_name 推断 *_best.pt）")

    # ================= 各类输出文件 =================
    OUT_DIR = "/tmp/pycharm_project_156/eval_outputs/NPS18-test6"
    p.add_argument("--save_report", type=str,
                   default=f"{OUT_DIR}/report.txt", help="保存指标报告 (.txt)")
    p.add_argument("--save_pred", type=str,
                   default=f"{OUT_DIR}/pred.csv", help="导出预测 CSV")
    p.add_argument("--save_cm", type=str,
                   default=f"{OUT_DIR}/cm.npy", help="保存混淆矩阵 (.npy)")
    p.add_argument("--save_cm_png", type=str,
                   default=f"{OUT_DIR}/cm.png", help="保存混淆矩阵图 (.png)")
    p.add_argument("--save_tsne", type=str,
                   default=f"{OUT_DIR}/tsne.png", help="保存 t-SNE 图 (.png)")
    p.add_argument("--save_roc_png", type=str,
                   default=f"{OUT_DIR}/roc.png", help="保存 ROC 曲线 (.png)")
    p.add_argument("--save_pr_png", type=str,
                   default=f"{OUT_DIR}/pr.png", help="保存 PR 曲线 (.png)")

    # ================= t-SNE 可调参数 =================
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_lr", type=float, default=200.0)
    p.add_argument("--tsne_iter", type=int, default=1000)

    # 当 ckpt 里没有 test_idx 或 force_resplit 时，兜底随机划分
    p.add_argument("--test_ratio", type=float, default=0.3)

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()
    eval_only(args)


if __name__ == "__main__":
    main()

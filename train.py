# -*- coding: utf-8 -*-
"""
训练入口：支持 NPS / TFF
- NPS：op + dose 双模态，含跨模态滞后边与同模态相关性边
- TFF：建议单模态（仅 op 的相关性边），启动方式：
    --dataset TFF --op_dir Data/TFF_csv --topk_cross 0 --dose_topk 0
    （这样数据集中会自动不构造 dose 与跨模态边，模型前向也走单模态分支）
"""
import os, argparse, warnings, torch, datetime, json
from utils.train_eval import train_loop

def get_args():
    p = argparse.ArgumentParser(description="Train Hetero Temporal-Physical GNN on NPS/TFF")

    # ---------------- data ----------------
    p.add_argument('--dataset', type=str, default='NPS', choices=['NPS', 'TFF'], help='选择数据集：NPS 或 TFF')
    p.add_argument('--modal', type=str, default='both', choices=['both', 'op', 'dose'],help="选择模态：'both'、'op' 或 'dose'")
    p.add_argument('--op_dir', type=str, default='/tmp/pycharm_project_156/Data/54/Operation_csv_data', help='Operation 数据或 TFF 根目录（a..g 子文件夹）')
    p.add_argument('--dose_dir', type=str, default='/tmp/pycharm_project_156/Data/Dose_csv_data', help='Dose 数据目录（TFF 单模态可忽略）')
    p.add_argument('--op_interval', type=int, default=10, help='Operation 采样间隔（秒）')
    p.add_argument('--dose_interval', type=int, default=60, help='Dose 采样间隔（秒）或 TFF 中的“等效慢速间隔”')
    p.add_argument('--dose_window_steps', type=int, default=10, help='Dose 窗口步数（锚点窗口长度）')
    p.add_argument('--dose_stride_steps', type=int, default=1, help='Dose 窗口滑动步长（步）')
    p.add_argument('--task_type', type=str, default='multiclass', choices=['binary', 'multiclass'], help='任务类型')
    p.add_argument('--clip_val', type=float, default=4.0, help='节点时间序列标准化后的截断阈值（防极端值）')

    # -------------- graph build --------------
    p.add_argument('--op_topk', type=int, default=10, help='op 内部相关性建边的 top-k')
    p.add_argument('--dose_topk', type=int, default=10, help='dose 内部相关性建边的 top-k（TFF 单模态请设为 0）')
    p.add_argument('--min_abs_corr', type=float, default=0.2, help='相关性建边的最小绝对相关阈值')
    p.add_argument('--topk_cross', type=int, default=3, help='跨模态滞后边的 top-k（TFF 单模态请设为 0）')
    p.add_argument('--max_lag_dose', type=int, default=5, help='跨模态滞后最大步数（以 dose 步为单位）')

    # ---------------- model ----------------
    p.add_argument('--trans_dim', type=int, default=256, help='时间编码通道数 d_model')
    p.add_argument('--trans_layers', type=int, default=2, help='时间编码 Transformer 层数')
    p.add_argument('--nhead', type=int, default=8, help='Transformer 多头数')
    p.add_argument('--gnn_hidden', type=int, default=512, help='GNN 隐藏维度')
    p.add_argument('--gnn_layers', type=int, default=2, help='HeteroConv 层数')
    p.add_argument('--dropout', type=float, default=0.1, help='dropout 比例')
    p.add_argument('--edge_attr_dim', type=int, default=3, help='跨模态边属性维度（lag_norm, |corr|, sign）')

    # --------------- training ---------------
    p.add_argument('--epochs', type=int, default=2000, help='训练轮数')
    p.add_argument('--batch_size', type=int, default=256, help='批大小（越大越能提高 GPU 利用率，显存允许请调大）')
    p.add_argument('--lr', type=float, default=1e-6, help='学习率')
    p.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减（L2 正则）')
    p.add_argument('--use_amp', type=lambda x: str(x).lower() in ['1','true','yes'], default=True, help='是否启用混合精度 AMP')
    p.add_argument('--grad_accum_steps', type=int, default=1, help='梯度累积步数（小显存可>1）')
    p.add_argument('--patience', type=int, default=50, help='早停耐心值')
    p.add_argument('--delta', type=float, default=1e-4, help='早停最小提升阈值')
    p.add_argument('--sampler', type=str, default='classaware', choices=['none','classaware'], help='采样策略（classaware 对长尾更友好）')

    # ------- classification loss (CB/Focal) -------
    p.add_argument('--loss_type', type=str, default='ce', choices=['ce','focal','cb_ce','cb_focal'], help='主损失类型：交叉熵/焦点损失/类平衡变体')
    p.add_argument('--gamma', type=float, default=2.0, help='FocalLoss 的 gamma')
    p.add_argument('--beta', type=float, default=0.999, help='Class-Balanced loss 的 beta')

    # ---------- contrastive / consistency ----------
    p.add_argument('--lmbd_supcon', type=float, default=0, help='监督对比损失权重（返回 z_fused 才会生效）')
    p.add_argument('--lmbd_xmod', type=float, default=0, help='跨模态一致性权重（返回 z_op/z_dose 才会生效）')
    p.add_argument('--lmbd_aug', type=float, default=0, help='数据增强一致性权重（返回 z_fused 才会生效）')
    p.add_argument('--tau', type=float, default=0, help='SupCon 温度参数')
    p.add_argument('--use_aug', type=lambda x: str(x).lower() in ['1','true','yes'], default=True, help='是否启用简单时域增强（翻转/抖动等）')
    # ---------- FOCAL factorization + SoftCL ----------
    p.add_argument('--lmbd_ortho', type=float, default=0,
                   help='FOCAL 式因子化潜空间的正交 regularization 权重（shared/private 正交约束）')
    p.add_argument('--use_soft_supcon', type=lambda x: str(x).lower() in ['1','true','yes'], default=False,
                   help='是否启用 Soft Contrastive Learning 风格的软监督对比损失')
    p.add_argument('--soft_label_tau', type=float, default=0,
                   help='SoftCL 标签相似度温度（越大越平滑）')
    p.add_argument('--shared_ratio', type=float, default=0.15,
                   help='shared/private 维度切分比例（0-1，默认0.5，对半分）')
    # ---------- 调试子集比例 ----------
    p.add_argument('--data_fraction', type=float, default=1,
                   help='使用数据比例进行训练（0-1 之间；例如 0.1 表示仅用 10% 窗口调试）')

    # ---------------- misc ----------------
    p.add_argument('--seed', type=int, default=42, help='随机种子')
    p.add_argument('--device', type=str, default='cuda', help='cuda / cpu')
    p.add_argument('--num_workers', type=int, default=12, help='DataLoader 的工作进程数（Windows 上测试集将自动使用 0）')
    p.add_argument('--test_workers', type=int, default=12, help='评测 DataLoader 的 worker 数（Windows 建议 0；Linux 可设 2-4）')  # test workers
    p.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    p.add_argument('--exp_name', type=str, default='/tmp/pycharm_project_156/checkpoints/NPS54-(改进测试中)', help='实验名（用于保存 ckpt 文件名）')
    p.add_argument('--resume', type=str, default='/tmp/pycharm_project_156/checkpoints/NPS54-(改进测试中)_2025-11-28_05-25-04/NPS54-(改进测试中)_best.pt', help='断点恢复：ckpt 路径（可选）')

    return p.parse_args()

def main():
    args = get_args()

    # 小提示：TFF 单模态建议参数
    if args.dataset.upper() == 'TFF' and (args.topk_cross > 0 or args.dose_topk > 0):
        warnings.warn("TFF 建议单模态：请考虑设置 --topk_cross 0 --dose_topk 0 以只保留 op 的相关性边。", UserWarning)

    # 设备可用性提示
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        warnings.warn("CUDA 不可用，已自动回退到 CPU。", RuntimeError)
        args.device = 'cpu'

    # 友好提示：TFF 路径
    if args.dataset.upper() == 'TFF':
        # 现在我们读取的是“已拆分的 CSV 根目录（含 a..g 子目录）”
        if os.path.splitext(args.op_dir)[1].lower() == '.xlsx':
            warnings.warn(f"你选择了 TFF，且 --op_dir 看起来是 Excel 文件：{args.op_dir}。本版本期望的是 CSV 根目录（含 a..g 子文件夹）。", UserWarning)
        elif not os.path.isdir(args.op_dir):
            raise FileNotFoundError(f"TFF 模式下 --op_dir 应指向 CSV 根目录（含 a..g 子文件夹），但未找到目录：{args.op_dir}")

    # 打印关键配置
    print(f"[1] 设置随机种子..."); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    print(f"[2] 开始加载数据集 (HeteroGraph, 支持 NPS/TFF)...")

    # --- 修改：创建独立运行目录并保存配置 ---
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_basename = os.path.basename(args.exp_name)
    # 如果 exp_name 是路径，这里只取文件名部分作为前缀，避免 os.path.join 出错
    # 新的保存目录： args.save_dir / {exp_basename}_{timestamp}

    # 确保使用绝对路径：如果是相对路径，则基于当前脚本所在目录（项目根目录）
    if not os.path.isabs(args.save_dir):
        project_root = os.path.dirname(os.path.abspath(__file__))
        args.save_dir = os.path.join(project_root, args.save_dir)

    run_save_dir = os.path.join(args.save_dir, f"{exp_basename}_{timestamp}")
    os.makedirs(run_save_dir, exist_ok=True)

    # 更新 args
    args.save_dir = run_save_dir
    args.exp_name = exp_basename  # 确保后续生成文件名时只用文件名部分

    # 保存超参数
    config_path = os.path.join(run_save_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        # 将 args 转为 dict，过滤掉不可序列化的对象（如函数）
        args_dict = {k: v for k, v in vars(args).items() if not callable(v)}
        json.dump(args_dict, f, indent=4, ensure_ascii=False)
    print(f"[Info] 本次训练结果将保存至: {run_save_dir}")
    print(f"[Info] 超参数配置已保存至: {config_path}")
    # ------------------------------------

    # 启动训练/评测主流程（其余逻辑、模型构建、评测与保存均在 train_eval.py 内处理）
    train_loop(args)

if __name__ == '__main__':
    main()

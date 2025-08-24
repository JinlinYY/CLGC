import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GraphConv, GATv2Conv, global_mean_pool

class _ResMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = None, p_drop: float = 0.1):
        super().__init__()
        hidden = hidden or dim
        self.pre  = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, hidden)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.fc2  = nn.Linear(hidden, dim)
        self.post = nn.LayerNorm(dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.drop(self.act(self.fc1(self.pre(x)))))
        return self.post(x + h)

class TemporalPhysicalHeteroGNN_V2(nn.Module):
    def __init__(self,
                 trans_dim: int = 256,
                 trans_layers: int = 2,
                 nhead: int = 4,
                 dropout: float = 0.1,
                 gnn_hidden: int = 512,
                 gnn_layers: int = 2,
                 num_classes: int = 18,
                 num_op: int = 100,
                 num_dose: int = 53,
                 op_seq_len: int = 65,
                 dose_seq_len: int = 10,
                 edge_attr_dim: int = 3,
                 modal: str = "both"):  # 新增 modal 参数
        super().__init__()
        self.modal = modal.lower().strip()  # 保存模态设置
        d_model = trans_dim
        ff = 4 * d_model

        # ---------------- 时序编码 ----------------
        enc_op = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                            batch_first=True, dropout=dropout,
                                            dim_feedforward=ff, norm_first=True)
        enc_ds = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                            batch_first=True, dropout=dropout,
                                            dim_feedforward=ff, norm_first=True)
        self.time_encoder_op   = nn.TransformerEncoder(enc_op, num_layers=trans_layers,
                                                       enable_nested_tensor=False)
        self.time_encoder_dose = nn.TransformerEncoder(enc_ds, num_layers=trans_layers,
                                                       enable_nested_tensor=False)
        self.in_proj_op   = nn.Linear(1, d_model)
        self.in_proj_dose = nn.Linear(1, d_model)
        self.readout = nn.AdaptiveAvgPool1d(1)

        # ---------------- Hetero-GNN 主干 ----------------
        convs = []
        in_dim = d_model
        for _ in range(gnn_layers):
            conv_dict = {
                ('op',   'intra',      'op'):   GraphConv(in_dim, gnn_hidden),
                ('op',   'intra_rev',  'op'):   GraphConv(in_dim, gnn_hidden),
                ('dose', 'intra',      'dose'): GraphConv(in_dim, gnn_hidden),
                ('dose', 'intra_rev',  'dose'): GraphConv(in_dim, gnn_hidden),
                ('dose', 'cross',      'op'):   GATv2Conv((in_dim, in_dim), gnn_hidden,
                                                          heads=1, add_self_loops=False,
                                                          edge_dim=edge_attr_dim),
                ('op',   'cross',      'dose'): GATv2Conv((in_dim, in_dim), gnn_hidden,
                                                          heads=1, add_self_loops=False,
                                                          edge_dim=edge_attr_dim),
            }
            convs.append(HeteroConv(conv_dict, aggr='sum'))
            in_dim = gnn_hidden
        self.hetero_convs = nn.ModuleList(convs)


        # ---------------- 分类头（加深为 3 层 MLP） ----------------
        self.head_op   = nn.Linear(in_dim, in_dim)
        self.head_dose = nn.Linear(in_dim, in_dim)

        hidden_dim = in_dim  # 若想减小中间维度，可用 in_dim // 2

        # 双模态分类器：残差块后接线性分类
        self.cls = nn.Sequential(
            _ResMLP(in_dim * 2, hidden=hidden_dim, p_drop=dropout),
            nn.Linear(in_dim * 2, num_classes)
        )

        # 单模态分类器：注意输入维度为 in_dim（不是 2*in_dim）
        self.cls_single = nn.Sequential(
            _ResMLP(in_dim, hidden=hidden_dim, p_drop=dropout),
            nn.Linear(in_dim, num_classes)
        )
        # ---------------- 对比投影头 ----------------
        self.proj_op   = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim))
        self.proj_dose = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim))
        self.proj_fuse = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim))
        self.dropout = nn.Dropout(dropout)

    # ======== 辅助函数 ========
    def _encode_series(self, x_seq: torch.Tensor, lin: nn.Linear,
                       enc: nn.TransformerEncoder) -> torch.Tensor:
        h = lin(x_seq.unsqueeze(-1))          # [B,T,1] → [B,T,d_model]
        h = enc(h)                            # Transformer 编码
        h = h.transpose(1, 2)                 # [B,d_model,T]
        h = self.readout(h).squeeze(-1)       # Global-AvgPool → [B,d_model]
        return h

    # ======== 前向 ========
    def forward(self, data):
        mono = ('dose' not in data.node_types) or \
               (not hasattr(data['dose'], 'x')) or \
               (data['dose'].x.numel() == 0)

        # --- 编码 op ---
        x_op = self._encode_series(data['op'].x, self.in_proj_op, self.time_encoder_op)
        x_dict = {'op': x_op}

        edge_index_dict = {
            ('op', 'intra',      'op'): data['op', 'intra',     'op'].edge_index,
            ('op', 'intra_rev',  'op'): data['op', 'intra_rev', 'op'].edge_index,
        }
        edge_weight_dict = {}
        if data['op', 'intra', 'op'].edge_index.numel() > 0:
            edge_weight_dict[('op', 'intra', 'op')] = data['op', 'intra', 'op'].edge_attr.squeeze(-1)
        if data['op', 'intra_rev', 'op'].edge_index.numel() > 0:
            edge_weight_dict[('op', 'intra_rev', 'op')] = data['op', 'intra_rev', 'op'].edge_attr.squeeze(-1)
        edge_attr_dict = {}

        # --- 若存在 dose 模态 ---
        if not mono and self.modal != "op":  # 仅在 modal 不是 "op" 时使用 dose
            x_dose = self._encode_series(data['dose'].x, self.in_proj_dose, self.time_encoder_dose)
            x_dict['dose'] = x_dose

            edge_index_dict.update({
                ('dose', 'intra',      'dose'): data['dose', 'intra',     'dose'].edge_index,
                ('dose', 'intra_rev',  'dose'): data['dose', 'intra_rev', 'dose'].edge_index,
                ('dose', 'cross',      'op'):   data['dose', 'cross',     'op'].edge_index,
                ('op',   'cross',      'dose'): data['op',   'cross',     'dose'].edge_index,
            })
            if data['dose', 'intra', 'dose'].edge_index.numel() > 0:
                edge_weight_dict[('dose', 'intra', 'dose')] = data['dose', 'intra', 'dose'].edge_attr.squeeze(-1)
            if data['dose', 'intra_rev', 'dose'].edge_index.numel() > 0:
                edge_weight_dict[('dose', 'intra_rev', 'dose')] = data['dose', 'intra_rev', 'dose'].edge_attr.squeeze(-1)
            if data['dose', 'cross', 'op'].edge_index.numel() > 0:
                edge_attr_dict[('dose', 'cross', 'op')] = data['dose', 'cross', 'op'].edge_attr
            if data['op', 'cross', 'dose'].edge_index.numel() > 0:
                edge_attr_dict[('op', 'cross', 'dose')] = data['op', 'cross', 'dose'].edge_attr

        # --- Hetero-GNN 传播 ---
        for conv in self.hetero_convs:
            x_dict = conv(x_dict, edge_index_dict,
                          edge_weight_dict=edge_weight_dict,
                          edge_attr_dict=edge_attr_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {k: self.dropout(v) for k, v in x_dict.items()}

        # --- 节点 → 图 ---
        z_op_nodes = self.head_op(x_dict['op'])
        b_op = getattr(data['op'], 'batch',
                       torch.zeros(z_op_nodes.size(0), dtype=torch.long, device=z_op_nodes.device))
        g_op = global_mean_pool(z_op_nodes, b_op)   # [B, D]

        # ------- 单模态 -------
        if mono or self.modal == "op":
            logits = self.cls_single(g_op)
            z_op = self.proj_op(g_op)
            z_dose = z_op
            z_fused = z_op
            return logits, z_fused, z_op, z_dose

        # ------- 双模态 -------
        z_ds_nodes = self.head_dose(x_dict['dose'])
        b_ds = getattr(data['dose'], 'batch',
                       torch.zeros(z_ds_nodes.size(0), dtype=torch.long, device=z_ds_nodes.device))
        g_ds = global_mean_pool(z_ds_nodes, b_ds)   # [B, D]

        g_cat = torch.cat([g_op, g_ds], dim=-1)     # [B, 2D]
        logits = self.cls(g_cat)

        z_fused = self.proj_fuse(g_cat)
        z_op    = self.proj_op(g_op)
        z_dose  = self.proj_dose(g_ds)
        return logits, z_fused, z_op, z_dose



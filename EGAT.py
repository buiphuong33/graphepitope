#EGAT.py
import torch
import torch.nn as nn
import torch.nn.functional as F
class AE(nn.Module):
    def __init__(self, dim_in, dim_out, hidden, dropout = 0., bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden, bias=bias),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim_out, bias=bias),
            nn.LayerNorm(dim_out),
        )
    def forward(self, x):
        return self.net(x)
class EGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_attr):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        e = e*edge_attr
        zero_vec = -9e15*torch.ones_like(e)
        e = torch.where(edge_attr > 0, e, zero_vec)
        e = F.softmax(e, dim=1)
        e = F.dropout(e, self.dropout, training=self.training)
        
        h_prime=[]
        for i in range(edge_attr.shape[0]):
            h_prime.append(torch.matmul(e[i],Wh))
        
        if self.concat:
            h_prime = torch.cat(h_prime,dim=1)
        else:
            h_prime = torch.stack(h_prime,dim=0).mean(0)
        return F.elu(h_prime),e

    #compute attention coefficient
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class EGAT(nn.Module):
    def __init__(self, nfeat, nhid, efeat, dropout=0.2, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.in_att = EGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
        self.out_att = EGraphAttentionLayer(nhid*efeat, nfeat, dropout=dropout, alpha=alpha, concat=False)
    def forward(self, x, edge_attr):
        x_cut=x
        x = F.dropout(x, self.dropout, training=self.training)
        x, edge_attr=self.in_att(x, edge_attr)
        x, edge_attr=self.out_att(x, edge_attr)
        return x+x_cut, edge_attr


class HierarchicalPooling(nn.Module):
    def __init__(self, hidden_dim, pool_ratio=8, max_nodes=1024, num_levels=2):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.max_k = max_nodes // pool_ratio
        self.num_levels = num_levels

        # Mỗi cấp có assignment matrix riêng
        self.assign_mats = nn.ModuleList([
            nn.Linear(hidden_dim, max_nodes // (pool_ratio ** (i+1)))
            for i in range(num_levels)
        ])

        # GCN ở mỗi cấp macro — dùng 2 lớp thay vì 1
        self.macro_gcns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_levels)
        ])

        # Gated fusion để đưa thông tin macro trở lại micro
        # Thay vì chỉ cộng, dùng gate học được
        self.fusion_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
            for _ in range(num_levels)
        ])

        self.layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_levels)
        ])

    def forward(self, x, adj):
        L = x.shape[0]
        
        # Lưu lại x và S ở mỗi cấp để dùng khi top-down
        level_x   = [x]      # level_x[0] = residue-level
        level_adj = [adj]
        S_list    = []

        # Bottom-up
        for level in range(self.num_levels):
            x_curr   = level_x[-1]
            adj_curr = level_adj[-1]
            L_curr   = x_curr.shape[0]
            K = max(2, L_curr // self.pool_ratio)

            S = self.assign_mats[level](x_curr)[:, :K]
            S = torch.softmax(S, dim=-1)
            S_list.append(S)

            X_macro = S.T @ x_curr
            A_macro = S.T @ adj_curr @ S
            rowsum  = A_macro.sum(-1, keepdim=True).clamp(min=1e-6)
            A_macro = A_macro / rowsum
            X_macro = self.macro_gcns[level](A_macro @ X_macro)

            level_x.append(X_macro)
            level_adj.append(A_macro)

        # Top-down: từ cấp cao nhất xuống cấp residue
        x_top = level_x[-1]
        for level in reversed(range(len(S_list))):
            S = S_list[level]
            x_low  = level_x[level]       # đặc trưng cấp thấp hơn
            x_down = S @ x_top            # broadcast xuống

            gate = self.fusion_gates[level](
                torch.cat([x_low, x_down], dim=-1)
            )
            x_top = self.layernorms[level](x_low + gate * x_down)

        return x_top  # shape (L, hidden_dim) — về lại cấp residue
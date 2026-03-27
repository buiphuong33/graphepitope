import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_cluster import radius_graph, knn_graph


class HGNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        
        self.gcn = GCNConv(in_channels, out_channels)
        self.gat = GATConv(in_channels, out_channels // 4, heads=4)
        self.sage = SAGEConv(in_channels, out_channels)

        self.lin = nn.Linear(out_channels * 3, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_shallow, edge_medium, edge_deep):
        z1 = F.elu(self.gcn(x, edge_shallow))
        z2 = F.elu(self.gat(x, edge_medium))
        z3 = F.elu(self.sage(x, edge_deep))

        z = torch.cat([z1, z2, z3], dim=-1)
        z = self.lin(z)
        z = self.norm(z)
        z = self.dropout(z)

        return z, [z1, z2, z3]


class HierarchicalGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.2):
        super().__init__()

        self.num_layers = num_layers

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList([
            HGNNBlock(hidden_channels, hidden_channels, dropout)
            for _ in range(num_layers)
        ])

        # Deep gating (attention-like)
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels * num_layers, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_layers),
            nn.Softmax(dim=-1)
        )

        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, pos, batch_index=None):
        # Graph construction (shared across layers)
        edge_shallow = radius_graph(pos, r=6.0, batch=batch_index, loop=False)
        edge_medium = radius_graph(pos, r=10.0, batch=batch_index, loop=False)
        edge_deep = knn_graph(pos, k=20, batch=batch_index, loop=False)

        x = self.input_proj(x)

        layer_outputs = []
        all_scale_outputs = []

        for layer in self.layers:
            res = x
            x, scale_out = layer(x, edge_shallow, edge_medium, edge_deep)
            
            # Residual connection
            x = x + res

            layer_outputs.append(x)
            all_scale_outputs.extend(scale_out)

        # ===== Multi-layer fusion =====
        stacked = torch.cat(layer_outputs, dim=-1)  # [N, L*hidden]
        gate_weights = self.gate(stacked)           # [N, L]

        z_final = 0
        for i in range(self.num_layers):
            z_final += gate_weights[:, i:i+1] * layer_outputs[i]

        return all_scale_outputs, self.norm(z_final)
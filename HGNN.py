#HGNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_cluster import radius_graph, knn_graph

class HierarchicalGNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shallow_conv = GCNConv(in_channels, out_channels)
        self.medium_conv = GATConv(in_channels, out_channels // 4, heads=4)
        self.deep_conv = SAGEConv(in_channels, out_channels)

        # Gate nhận vào concat của 3 đặc trưng (3 * out_channels)
        self.gate = nn.Linear(out_channels * 3, 3)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, pos, batch_index=None):
        edge_index_shallow = radius_graph(pos, r=6.0, batch=batch_index, loop=False)
        edge_index_medium = radius_graph(pos, r=10.0, batch=batch_index, loop=False)
        edge_index_deep = knn_graph(pos, k=20, batch=batch_index, loop=False)

        z1 = F.elu(self.shallow_conv(x, edge_index_shallow))
        z2 = F.elu(self.medium_conv(x, edge_index_medium))
        z3 = F.elu(self.deep_conv(x, edge_index_deep))

        combined = torch.cat([z1, z2, z3], dim=-1)
        gate_weights = F.softmax(self.gate(combined), dim=-1)
        
        # Cộng có trọng số
        z_final = (gate_weights[:, 0:1] * z1 + 
                   gate_weights[:, 1:2] * z2 + 
                   gate_weights[:, 2:3] * z3)
        
        return [z1, z2, z3], self.norm(z_final)
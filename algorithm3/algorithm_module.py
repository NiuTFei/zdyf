## 定义模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, ChebConv
from torch_geometric.nn import EdgePooling, global_mean_pool

class GraphSage(torch.nn.Module):
    def __init__(self, feature, out_channel):
        super(GraphSage, self).__init__()
        self.pool1 = EdgePooling(1024)

        self.GConv1 = ChebConv(feature, 1024, K=1)
        self.bn1 = BatchNorm(1024)

        self.GConv2 = ChebConv(1024, 512, K=1)
        self.bn2 = BatchNorm(512)

        self.fc = nn.Sequential(nn.Linear(512, 200), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Sequential(nn.Linear(200, out_channel))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.GConv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x, edge_index, batch, _ = self.pool1(x=x, edge_index=edge_index, batch=batch)
        x = self.GConv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        self.fc_feature = x
        x = self.dropout(x)
        x = self.fc1(x)
        return x



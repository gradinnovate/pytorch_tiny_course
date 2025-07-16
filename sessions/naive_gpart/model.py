import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv,BatchNorm


class GraphPartitionNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=2, aggr='mean'):
        super().__init__()
        decoder_dim = 32
        self.bn1 = BatchNorm(hidden_dim)
        
        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr=aggr)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
        self.conv4 = SAGEConv(hidden_dim, decoder_dim, aggr=aggr)
        
        self.mask_token = nn.Parameter(torch.randn(input_dim))
        self.linear1 = nn.Linear(decoder_dim, decoder_dim)
        self.linear2 = nn.Linear(decoder_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        # Xavier initialization for all SAGEConv layers
        for m in [self.conv1, self.conv2, self.conv3, self.conv4]:
            if hasattr(m, 'lin_l') and hasattr(m.lin_l, 'weight'):
                nn.init.xavier_uniform_(m.lin_l.weight)
            if hasattr(m, 'lin_r') and hasattr(m.lin_r, 'weight'):
                nn.init.xavier_uniform_(m.lin_r.weight)
        for m in [self.linear1, self.linear2]:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias'):
                nn.init.zeros_(m.bias)

    def masking(self, x):
        # x: [N, F]
        N = x.size(0)
        num_mask = max(1, int(N * 0.01))
        mask_idx = torch.randperm(N)[:num_mask]
        x_masked = x.clone()
        x_masked[mask_idx]=self.mask_token
        return x_masked

    def forward(self, x, edge_index):
        if self.training:
            if torch.rand(1) < 0.2:
                x = self.masking(x)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
         
        x = self.conv2(x, edge_index)
        x = F.relu(x)
       
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
       
        x = F.softmax(x, dim=-1)  # 每個 node 屬於 partition 0/1 的機率
        return x



if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data = torch.load(os.path.join(base_dir, "data/tiny.pt"), weights_only=False)
    input_dim = data.x.shape[1]
    
    model = GraphPartitionNet(input_dim=input_dim, hidden_dim=16, output_dim=2)
    print('model structure:')
    print(model)
   
    out = model(data.x, data.edge_index)
    print('output:')
    print(out)
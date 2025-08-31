import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, LayerNorm, BatchNorm

class HypergraphModel(nn.Module):
    """
    一個為學習超圖頻譜嵌入而優化的、更穩健的 GNN 架構。

    - 採用 2 層結構以避免過度平滑。
    - 在每個隱藏層後使用 LayerNorm, LeakyReLU, 和 Dropout，確保訓練穩定性。
    - 輸出層為線性，不加激活函數，適用於嵌入任務。
    - 包含可學習的 mask token，訓練時隨機遮蔽節點特徵以增強泛化能力。
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5, mask_prob: float = 0.1):
        super().__init__()
        
        # 第一層：輸入維度 -> 隱藏維度
        self.conv1 = HypergraphConv(input_dim, hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)
        
       
        self.conv2 = HypergraphConv(hidden_dim, hidden_dim)
        self.conv3 = HypergraphConv(hidden_dim, hidden_dim)
        self.conv4 = HypergraphConv(hidden_dim, hidden_dim)
        self.conv5 = HypergraphConv(hidden_dim, output_dim)
       
        self.dropout = dropout
        self.mask_prob = mask_prob
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(input_dim))

        # 初始化權重
        self._init_weights()

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        # Apply masking during training
        if self.training and self.mask_prob > 0:
            # Create random mask
            mask = torch.rand(x.size(0), device=x.device) < self.mask_prob
            
            # Replace masked nodes with mask token
            x = x.clone()  # Create copy to avoid modifying input
            x[mask] = self.mask_token.to(x.device)
        
        # 第一層
        x = self.conv1(x, hyperedge_index)
        
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
       
        x = self.conv2(x, hyperedge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, hyperedge_index)
        x = F.relu(x)
        
        
        x = self.conv4(x, hyperedge_index)
        x = F.relu(x)
        z = self.conv5(x, hyperedge_index)
        
        
        
        return z

    def _init_weights(self):
        """使用 Xavier/Glorot 初始化權重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                # 對於權重矩陣
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # 對於偏置項
                nn.init.constant_(param, 0)

if __name__ == "__main__":
    from hgr2indices import parse_hgr_file
    
    # Load hypergraph data
    hyperedge_index, num_vertices, num_hyperedges = parse_hgr_file("ibm01.hgr")
    
    # Create model
    model = HypergraphModel(input_dim=1, hidden_dim=32, output_dim=32)
    
    # Create dummy node features (all ones)
    x = torch.ones(num_vertices, 1)
    
    # Forward pass
    print(f"Input shape: {x.shape}")
    print(f"Hyperedge index shape: {hyperedge_index.shape}")
    
    output = model(x, hyperedge_index)
    print(f"Output shape: {output.shape}")
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
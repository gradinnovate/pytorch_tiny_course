import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_geometric.nn import HypergraphConv, LayerNorm, BatchNorm

class HypergraphModel(nn.Module):
    """
    超圖譜嵌入的 GNN 模型，專為超圖二分割任務設計。

    架構特點：
    - 3層 HypergraphConv with multi-head attention：input_dim -> hidden_dim -> hidden_dim -> output_dim
    - 第一層後使用 LayerNorm 穩定訓練
    - 使用 LeakyReLU 激活函數避免死神經元
    - Dropout 正則化防止過擬合  
    - 可學習的 mask token 增強泛化能力
    - Single-head attention for all layers
    - 最終層無激活函數，直接輸出嵌入向量
    
    當前實現：
    - 採用 Variational 架構，輸出均值和方差
    - 訓練時使用重參數化技巧進行隨機採樣
    - 推理時返回均值，但保留 mu 和 logvar 供手動採樣
    - 支援推理時多次採樣以獲得多個解
    - 適合小規模隱藏維度 (如 hidden_dim=8)
    
    用途：
    - 學習超圖的低維嵌入表示
    - 配合瑞利商 loss function 進行譜聚類
    - 生成適合平衡二分割的節點嵌入
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5, mask_prob: float = 0.0):
        """
        初始化超圖模型。
        
        Args:
            input_dim (int): 輸入節點特徵維度
            hidden_dim (int): 隱藏層維度
            output_dim (int): 輸出嵌入維度 (通常為1用於二分割)
            dropout (float): Dropout 比例，預設0.5
            mask_prob (float): 訓練時節點遮蔽比例，預設0.5
        """
        super().__init__()
        
        # 超圖卷積層 with attention
        self.conv1 = HypergraphConv(input_dim, hidden_dim, use_attention=False, heads=1)      # 第一層：input -> hidden
        self.norm1 = LayerNorm(hidden_dim)                     # 層正規化穩定訓練
        
        self.conv2 = HypergraphConv(hidden_dim, hidden_dim, use_attention=False, heads=1)     # 第二層：hidden -> hidden
        
        # Variational layers: 輸出均值和對數方差
        self.conv_mu = HypergraphConv(hidden_dim, hidden_dim)    # 均值分支
        self.conv_logvar = HypergraphConv(hidden_dim, hidden_dim) # 對數方差分支
        
        # 3-layer MLP decoder for final embedding refinement
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 8),
            nn.LeakyReLU(), 
            nn.Dropout(dropout),
            nn.Linear(8, output_dim)
        )
        
        # 訓練參數
        self.dropout = dropout
        self.mask_prob = mask_prob
        
        # 可學習的遮蔽標記，用於增強模型泛化能力
        self.mask_token = nn.Parameter(torch.randn(input_dim))

        # 初始化權重
        self._init_weights()

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。
        
        Args:
            x (torch.Tensor): 節點特徵矩陣 [num_nodes, input_dim]
            hyperedge_index (torch.Tensor): 超邊索引 [2, num_edges]
            
        Returns:
            torch.Tensor: 節點嵌入 [num_nodes, output_dim]
        """
        # 訓練時應用遮蔽機制增強泛化能力
        if self.training and self.mask_prob > 0:
            # 隨機選擇要遮蔽的節點
            mask = torch.rand(x.size(0), device=x.device) < self.mask_prob
            
            # 用可學習的遮蔽標記替換選中的節點
            x = x.clone()  # 創建副本避免修改原始輸入
            x[mask] = self.mask_token.to(x.device)
        
        # 第一層：超圖卷積 + 正規化 + 激活 + Dropout  
        x = self.conv1(x, hyperedge_index)
        x = self.norm1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二層：超圖卷積 + 激活 + Dropout
        x = self.conv2(x, hyperedge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Variational layers: 計算均值和對數方差
        mu = self.conv_mu(x, hyperedge_index)
        logvar = self.conv_logvar(x, hyperedge_index)
        
        # 重參數化技巧 (reparameterization trick)
        if self.training:
            # 訓練時進行隨機採樣
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # 推理時也返回 mu 作為預設嵌入，但保留採樣能力
            z = mu
        
        # 3-layer MLP decoder for final refinement
        z_decoded = self.decoder(z)
        mu_decoded = self.decoder(mu)
        
        # QR Decomposition for orthogonal embeddings
        z_orth = self._apply_qr_orthogonalization(z_decoded)
        mu_orth = self._apply_qr_orthogonalization(mu_decoded)
        
        # 始終返回 (z, mu, logvar) 以支持推理時的手動採樣
        return z_orth, mu_orth, logvar

    def _apply_qr_orthogonalization(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        對embedding應用QR分解以獲得正交化的向量。
        
        Args:
            embeddings: [num_nodes, output_dim]
        
        Returns:
            正交化後的embeddings
        """
        if embeddings.shape[1] == 1:
            # 對於單維輸出，直接標準化
            return F.normalize(embeddings, dim=0, eps=1e-8)
        else:
            # 對於多維輸出，使用QR分解
            Q, R = torch.linalg.qr(embeddings.T)  # 轉置後分解
            
            # 確保R對角線為正（標準化）
            signs = torch.sign(torch.diag(R))
            signs[signs == 0] = 1
            Q = Q * signs.unsqueeze(0)
            
            return Q.T  # 轉回原始形狀

    def _init_weights(self):
        """
        初始化模型權重。
        
        使用 Xavier/Glorot 初始化權重矩陣，零初始化偏置項。
        這有助於穩定訓練並避免梯度爆炸/消失問題。
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                # 對於多維權重矩陣使用 Xavier 初始化
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # 偏置項初始化為零
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
import torch
import torch.nn as nn
import torch.nn.functional as F

class HypergraphRayleighQuotientLoss(nn.Module):
    """
    重新設計的超圖瑞利商損失函數，拆分為兩個項。
    
    實現瑞利商: R_j = (expansion_term - cut_term) / (Z_j^T Z_j)
    其中:
    - expansion_term = Z_j^T D_v Z_j (鼓勵節點分散)
    - cut_term = Z_j^T H W D_e^{-1} H^T Z_j (懲罰割邊)
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Args:
            alpha: expansion term 的權重
            beta: cut term 的權重
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def _calculate_expansion_term(self, z_col: torch.Tensor, Dv: torch.Tensor) -> torch.Tensor:
        """
        計算 expansion term: Z_j^T D_v Z_j
        鼓勵節點在不同分割中分散
        """
        # Z_j^T D_v Z_j = sum_i D_v[i] * z_i^2
        expansion = torch.sum(Dv * z_col.squeeze()**2)
        return expansion
    
    def _calculate_cut_term(self, z_col: torch.Tensor, 
                           hyperedge_index: torch.Tensor,
                           hyperedge_weight: torch.Tensor,
                           De: torch.Tensor) -> torch.Tensor:
        """
        計算 cut term: Z_j^T H W D_e^{-1} H^T Z_j
        等價於公式 (152) 中的 0.5 * sum_{e} (w(e)/delta(e)) * sum_{u,v in e} ||z_u - z_v||^2
        """
        node_idx, edge_idx = hyperedge_index
        num_hyperedges = De.shape[0]
        
        # 計算每個超邊內節點對的差值平方和
        # sum_{u,v in e} ||z_u - z_v||^2 = 2 * |e| * sum_{u in e} ||z_u||^2 - 2 * ||sum_{u in e} z_u||^2
        
        # 計算每個超邊內節點嵌入的和: sum_{u in e} z_u
        z_sum_per_edge = torch.zeros((num_hyperedges, 1), device=z_col.device)
        z_sum_per_edge.scatter_add_(0, edge_idx.unsqueeze(1), z_col[node_idx])

        # 計算每個超邊內節點嵌入平方的和: sum_{u in e} ||z_u||^2
        z_sq_sum_per_edge = torch.zeros((num_hyperedges, 1), device=z_col.device)
        z_sq_sum_per_edge.scatter_add_(0, edge_idx.unsqueeze(1), z_col[node_idx]**2)
        
        # 組合公式: delta(e) * sum ||z_u||^2 - ||sum z_u||^2
        term_per_edge = De.unsqueeze(1) * z_sq_sum_per_edge - z_sum_per_edge**2
        
        # 乘以權重 w(e)/delta(e) 並求和，再乘以 0.5
        cut_term = 0.5 * torch.sum(hyperedge_weight.unsqueeze(1) * term_per_edge / De.unsqueeze(1))
        
        return cut_term

    def forward(self, Z: torch.Tensor, hyperedge_index: torch.Tensor, 
                num_nodes: int, hyperedge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        計算重新設計的瑞利商損失。

        Args:
            Z: GNN 輸出的節點嵌入, shape (num_nodes, k)
            hyperedge_index: 超圖的關聯索引
            num_nodes: 圖中的節點數量
            hyperedge_weight: 超邊的權重 (可選)

        Returns:
            torch.Tensor: 標量損失值
        """
        k = Z.shape[1]
        node_idx, edge_idx = hyperedge_index
        num_hyperedges = (edge_idx.max() + 1).item()

        # 設置默認權重
        if hyperedge_weight is None:
            hyperedge_weight = torch.ones(num_hyperedges, device=Z.device)
        
        # 計算節點度 D_v
        edge_weights_expanded = hyperedge_weight[edge_idx]
        Dv = torch.zeros(num_nodes, device=Z.device)
        Dv.scatter_add_(0, node_idx, edge_weights_expanded)
        
        # 計算超邊度 D_e
        De = torch.zeros(num_hyperedges, device=Z.device)
        De.scatter_add_(0, edge_idx, torch.ones_like(edge_idx, dtype=torch.float))

        # 處理度為0的情況
        Dv[Dv == 0] = 1.0
        De[De == 0] = 1.0
        
        # 正規化 Z 使每一列的分母為 1 (Z_j^T Z_j = 1)
        Z_normalized = F.normalize(Z, p=2, dim=0)  # 按列正規化，使每列的L2範數為1
        
        # 分別計算兩個損失項
        expansion_terms = []
        cut_terms = []
        
        for j in range(k):
            z_col = Z_normalized[:, j:j+1]  # shape: (num_nodes, 1)
            
            # 計算 expansion term: Z_j^T D_v Z_j
            expansion = self._calculate_expansion_term(z_col, Dv)
            expansion_terms.append(expansion)
            
            # 計算 cut term: Z_j^T H W D_e^{-1} H^T Z_j
            cut = self._calculate_cut_term(z_col, hyperedge_index, hyperedge_weight, De)
            cut_terms.append(cut)
        
        expansion_terms = torch.stack(expansion_terms)
        cut_terms = torch.stack(cut_terms)
        
        # 組合損失: alpha * expansion - beta * cut
        # 我們要最小化 expansion 但最大化 cut (即最小化 -cut)
        loss = self.alpha * torch.mean(expansion_terms) - self.beta * torch.mean(cut_terms)
        
        return loss
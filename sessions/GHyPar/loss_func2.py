import torch
import torch.nn as nn

class HypergraphRayleighQuotientLossDirect(nn.Module):
    """
    基於 NIPS 論文的歸一化超圖 Laplacian 廣義瑞利商 Loss Function。

    實現歸一化 Laplacian: ∆ = I - Θ, 其中 Θ = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
    廣義瑞利商: f^T ∆ f / f^T D_v f = (f^T D_v f - f^T Θ f) / f^T D_v f
    Loss = mean(Quotients)
    """
    def __init__(self):
        """
        直接優化超圖瑞利商的 Loss Function，不包含正交性懲罰項。
        """
        super().__init__()

    def _calculate_theta_quadratic_form(self, f: torch.Tensor, 
                                        hyperedge_index: torch.Tensor,
                                        hyperedge_weight: torch.Tensor,
                                        De: torch.Tensor,
                                        Dv_inv_sqrt: torch.Tensor) -> torch.Tensor:
        """
        計算 f^T Θ f，其中 Θ = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
        
        使用變分形式高效計算:
        f^T Θ f = sum_{e} w(e)/δ(e) * (sum_{u∈e} D_v^{-1/2}[u] * f[u])^2
        """
        node_idx, edge_idx = hyperedge_index
        num_hyperedges = De.shape[0]
        
        # 計算歸一化後的節點值: D_v^{-1/2} * f
        normalized_f = Dv_inv_sqrt * f
        
        # 對每個超邊 e，計算 sum_{u∈e} D_v^{-1/2}[u] * f[u]
        weighted_sum_per_edge = torch.zeros((num_hyperedges, 1), device=f.device)
        weighted_sum_per_edge.scatter_add_(0, edge_idx.unsqueeze(1), normalized_f[node_idx])
        
        # 計算 f^T Θ f = sum_{e} w(e)/δ(e) * (sum_{u∈e} D_v^{-1/2}[u] * f[u])^2
        theta_quadratic_form = torch.sum(
            hyperedge_weight.unsqueeze(1) * weighted_sum_per_edge.pow(2) / De.unsqueeze(1)
        )
        
        return theta_quadratic_form


    def forward(self, Z: torch.Tensor, hyperedge_index: torch.Tensor, 
                num_nodes: int, hyperedge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        計算 Loss。

        Args:
            Z (torch.Tensor): GNN 輸出的節點嵌入, shape 為 (num_nodes, k)。
            hyperedge_index (torch.Tensor): 超圖的關聯索引。
            num_nodes (int): 圖中的節點數量。
            hyperedge_weight (torch.Tensor, optional): 超邊的權重。

        Returns:
            torch.Tensor: 一個標量 (scalar) 的 Loss 值。
        """
        k = Z.shape[1]
        node_idx, edge_idx = hyperedge_index
        num_hyperedges = (edge_idx.max() + 1).item()

        # --- 步驟 1: 預計算節點度和超邊度 ---
        if hyperedge_weight is None:
            hyperedge_weight = torch.ones(num_hyperedges, device=Z.device)
        
        edge_weights_expanded = hyperedge_weight[edge_idx]
        
        Dv = torch.zeros(num_nodes, device=Z.device)
        Dv.scatter_add_(0, node_idx, edge_weights_expanded)
        
        De = torch.zeros(num_hyperedges, device=Z.device)
        De.scatter_add_(0, edge_idx, torch.ones_like(edge_idx, dtype=torch.float))

        # 處理度為0的孤立點/邊，避免除以零
        Dv[Dv == 0] = 1.0
        De[De == 0] = 1.0
        
        # --- 步驟 2: 計算歸一化超圖 Laplacian 廣義瑞利商 ---
        # 歸一化 Laplacian: ∆ = I - Θ, 其中 Θ = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
        # 廣義瑞利商: f^T ∆ f / f^T D_v f = (f^T D_v f - f^T Θ f) / f^T D_v f
        
        # 預計算 D_v^{-1/2}
        Dv_inv_sqrt = Dv.pow(-0.5).unsqueeze(1)
        
        rayleigh_quotients = []
        
        for j in range(k):
            f = Z[:, j:j+1]  # 當前列向量
            
            # 計算 f^T D_v f (廣義內積)
            f_Dv_f = torch.sum(f.pow(2) * Dv.unsqueeze(1))
            
            # 計算 f^T Θ f
            theta_quad_form = self._calculate_theta_quadratic_form(
                f, hyperedge_index, hyperedge_weight, De, Dv_inv_sqrt
            )
            
            # 廣義瑞利商: f^T ∆ f / f^T D_v f = (f^T D_v f - f^T Θ f) / f^T D_v f
            # = 1 - (f^T Θ f / f^T D_v f)
            epsilon = 1e-8
            rayleigh_quotient = 1.0 - theta_quad_form / (f_Dv_f + epsilon)
            rayleigh_quotients.append(rayleigh_quotient)
        
        rayleigh_quotients = torch.stack(rayleigh_quotients)
        
        # 損失是所有瑞利商的均值
        loss = torch.mean(rayleigh_quotients)
        
        return loss
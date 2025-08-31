import torch
import torch.nn as nn

class HypergraphRayleighQuotientLossDirect(nn.Module):
    """
    直接優化超圖瑞利商的 Loss Function。

    這個 Loss Function 最小化 GNN 輸出 Z 的每一列對應的瑞利商之和，
    同時加入一個正交性懲罰項，以確保嵌入維度之間的多樣性。
    Loss = mean(Quotients) + lambda * Ortho_Penalty
    """
    def __init__(self, lambda_ortho: float = 1.0):
        """
        Args:
            lambda_ortho (float, optional): 控制正交性懲罰項強度的超參數。 Defaults to 1.0.
        """
        super().__init__()
        self.lambda_ortho = lambda_ortho

    def _calculate_numerator_per_col(self, y_col: torch.Tensor, 
                                     hyperedge_index: torch.Tensor,
                                     hyperedge_weight: torch.Tensor,
                                     De: torch.Tensor) -> torch.Tensor:
        """
        為單個嵌入向量 y_col (對應論文中的 f) 計算瑞利商的分子。
        分子 = 0.5 * sum_{e} w(e)/delta(e) * sum_{u,v in e} ||y_u - y_v||^2
        """
        node_idx, edge_idx = hyperedge_index
        num_hyperedges = De.shape[0]
        
        # 這是論文公式 (152) 的直接實現
        # 為了效率，這裡使用向量化操作而非 Python for 迴圈
        
        # 1. 計算每個節點在超邊內的差值平方和
        # 對於每個超邊e, sum_{u,v in e} ||y_u - y_v||^2 可以被高效計算為:
        # 2 * |e| * sum_{u in e} ||y_u||^2 - 2 * ||sum_{u in e} y_u||^2
        
        # sum_{u in e} y_u
        y_sum_per_edge = torch.zeros((num_hyperedges, 1), device=y_col.device)
        y_sum_per_edge.scatter_add_(0, edge_idx.unsqueeze(1), y_col[node_idx])

        # sum_{u in e} ||y_u||^2
        y_sq_sum_per_edge = torch.zeros((num_hyperedges, 1), device=y_col.device)
        y_sq_sum_per_edge.scatter_add_(0, edge_idx.unsqueeze(1), y_col[node_idx]**2)
        
        # 2. 組合起來
        # delta(e) * sum ||y_u||^2 - ||sum y_u||^2
        term_per_edge = De.unsqueeze(1) * y_sq_sum_per_edge - y_sum_per_edge**2
        
        # 3. 乘以權重 w(e)/delta(e) 並加總
        numerator = torch.sum(hyperedge_weight.unsqueeze(1) * term_per_edge / De.unsqueeze(1))
        
        return numerator


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
        
        # --- 步驟 2: 計算瑞利商 ---
        # 根據論文 f 和 r 的關係，我們需要對 Z 進行 D_v 加權
        # 完整的瑞利商是 r^T(D_v - HWD_e^{-1}H^T)r / (r^T D_v r)
        # GNN的輸出 Z 對應於 r。
        
        # 分子: Numerator = Z^T * (D_v - HWD_e^{-1}H^T) * Z
        # 我們使用論文公式 (152) 的等價形式來計算分子，這樣更高效
        # 注意需要對Z的每一列單獨計算
        
        numerators = []
        # y_col 對應論文中的 f 向量
        Dv_inv_sqrt = Dv.pow(-0.5).unsqueeze(1)
        Y = Dv_inv_sqrt * Z
        
        for j in range(k):
            # 為Z的每一列計算分子
            numerators.append(self._calculate_numerator_per_col(
                Y[:, j:j+1], hyperedge_index, hyperedge_weight, De
            ))
        numerators = torch.stack(numerators)

        # 分母: Denominator = Z^T * D_v * Z
        # 我們需要的是對角線上的元素，即每個 z_j^T * D_v * z_j
        denominators = torch.sum(Z.pow(2) * Dv.unsqueeze(1), dim=0)

        # 為避免除以零，加入一個極小值
        epsilon = 1e-8
        rayleigh_quotients = numerators / (denominators + epsilon)
        
        # 主要損失是所有瑞利商的均值 (或總和)
        main_loss = torch.mean(rayleigh_quotients)
        
        variance_penalty = -torch.var(Z) 


        # --- 步驟 3: 計算正交性懲罰 ---
        # 目標是讓 Z^T * D_v * Z 接近單位矩陣 I
        I = torch.eye(k, device=Z.device)
        
        # 計算 Z^T * D_v * Z
        Z_T_Dv_Z = Z.T @ (Z * Dv.unsqueeze(1))
        
        ortho_loss = torch.norm(Z_T_Dv_Z - I, p='fro')

        # --- 步驟 4: 組合 Loss ---
        total_loss = main_loss
        
        return total_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class HypergraphRayleighQuotientLoss(nn.Module):
    """
    計算超圖瑞利商(Rayleigh Quotient)的Loss Function。
    目標是最小化 Tr(Z^T * Delta * Z)，其中 Delta 是超圖拉普拉斯算子。
    這會驅使 GNN 的輸出 Z 的列向量收斂到 Delta 的最小特徵向量所張成的空間。
    """
    def __init__(self):
        super(HypergraphRayleighQuotientLoss, self).__init__()

    def forward(self, Z: torch.Tensor, hyperedge_index: torch.Tensor, 
                num_nodes: int, hyperedge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        計算 Loss。

        Args:
            Z (torch.Tensor): GNN 輸出的節點嵌入, shape 為 (num_nodes, k)。
            hyperedge_index (torch.Tensor): 超圖的關聯索引, shape 為 (2, num_edges_in_incidence_matrix)。
            num_nodes (int): 圖中的節點數量。
            hyperedge_weight (torch.Tensor, optional): 超邊的權重, shape 為 (num_hyperedges,)。 Defaults to None.

        Returns:
            torch.Tensor: 一個標量 (scalar) 的 Loss 值。
        """
        num_hyperedges = hyperedge_index[1].max() + 1
        
        # 步驟 1: 計算正規化的節點嵌入 Y = D_v^(-1/2) * Z
        # 根據論文 [cite: 152, 156]，優化是在 f/sqrt(d(u)) 上進行的
        # 這等價於用 D_v^(-1/2)Z 替換 Z，其中 Z=D_v^(1/2)f
        
        # 計算節點度 D_v
        # d(v) = sum_{e | v in e} w(e)
        node_idx, edge_idx = hyperedge_index
        if hyperedge_weight is None:
            hyperedge_weight = torch.ones(num_hyperedges, device=Z.device)
            
        # 擴展超邊權重到每個 (node, hyperedge) 的關聯上
        edge_weights_expanded = hyperedge_weight[edge_idx]
        
        # 計算 D_v
        Dv = torch.zeros(num_nodes, device=Z.device)
        Dv.scatter_add_(0, node_idx, edge_weights_expanded)
        Dv[Dv == 0] = 1.0 # 避免除以零
        
        # 計算 D_v^(-1/2)
        Dv_inv_sqrt = Dv.pow(-0.5).unsqueeze(1) # shape: (num_nodes, 1)
        
        # 正規化節點嵌入
        Y = Dv_inv_sqrt * Z

        # 步驟 2: 計算瑞利商 Tr(Y^T * (I - Theta) * Y) = Tr(Y^T*Y) - Tr(Y^T*Theta*Y)
        # 其中 Theta = H * W * D_e^(-1) * H^T
        # 我們直接計算論文公式 (157) 中 2f^T * Delta * f 的形式，更為直接
        # Loss = 0.5 * sum_{e} sum_{u,v in e} (w(e)/delta(e)) * ||Y[u] - Y[v]||^2
        
        # 計算超邊度 D_e (delta(e))
        De = torch.zeros(num_hyperedges, device=Z.device)
        # delta(e) = |e|, 即每個超邊包含的節點數
        De.scatter_add_(0, edge_idx, torch.ones_like(edge_idx, dtype=torch.float))
        De[De == 0] = 1.0 # 避免除以零

        # 遍歷所有超邊，計算內部節點對的差的平方和
        total_loss = 0.0
        # 為了效率，避免 python for 循環，我們使用 tensor 操作
        # 將超邊權重和度廣播到每個關聯
        w_e = hyperedge_weight[edge_idx] # shape: (num_incidences,)
        delta_e = De[edge_idx]           # shape: (num_incidences,)
        
        # 計算每個節點在每次關聯中的加權嵌入
        # Y_weighted shape: (num_incidences, k)
        Y_weighted = Y[node_idx] * (w_e / delta_e).sqrt().unsqueeze(1)

        # 對於每個超邊，我們需要計算其節點嵌入的 "質心"
        # sum_{u in e} Y_weighted[u]
        hyperedge_centroids = torch.zeros(num_hyperedges, Z.shape[1], device=Z.device)
        hyperedge_centroids.scatter_add_(0, edge_idx.unsqueeze(1).expand(-1, Z.shape[1]), Y_weighted)
        
        # 對於每個超邊e, sum_{u,v in e} ||a_u - a_v||^2 = 2 * delta(e) * sum_{u in e} ||a_u||^2 - 2 * ||sum_{u in e} a_u||^2
        # Loss = 0.5 * sum_e (1/delta(e)) * [2*delta(e)*sum_{u in e}||Y'_u||^2 - 2*||sum_{u in e}Y'_u||^2]
        #      = sum_e sum_{u in e}||Y'_u||^2 - sum_e (1/delta(e)) * ||sum_{u in e}Y'_u||^2
        # where Y'_u = Y[u] * sqrt(w(e))
        
        # 這裡我們用一個更易於理解和實現的循環版本，實際應用中可向量化
        # 根據公式(152)來實現
        loss = 0.0
        for e_idx in range(num_hyperedges):
            # 找到屬於這個超邊的所有節點
            nodes_in_edge = node_idx[edge_idx == e_idx]
            
            if len(nodes_in_edge) > 1:
                y_e = Y[nodes_in_edge] # (num_nodes_in_edge, k)
                
                # 計算所有節點對的差的平方和
                # sum_{u,v in e} ||y_u - y_v||^2
                # 使用 torch.cdist 計算 pairwise distance
                pair_diff_sq = torch.cdist(y_e, y_e, p=2).pow(2)
                
                # 乘以權重 w(e)/delta(e)
                term = hyperedge_weight[e_idx] / De[e_idx] * pair_diff_sq.sum()
                loss += term

        # 根據公式(157)，loss是 2 * f^T * Delta * f，所以我們要除以2
        loss = loss * 0.5

        # 步驟 3: 添加正交性約束
        # 為了避免所有嵌入向量都收斂到同一個最小特徵向量（或0），
        # 我們需要讓 Z 的列向量相互正交 Z^T * D_v * Z = I
        # 論文中的約束是 f^T D_v f = 1, f_i^T D_v f_j = 0
        # 在我們的 Y 中就是 Y^T Y = I
        k = Z.shape[1]
        I = torch.eye(k, device=Z.device)
        ortho_loss = torch.norm(torch.matmul(Y.T, Y) - I, p='fro')
        
        # 最終的 Loss 是瑞利商加上正交性懲罰項
        # lambda 是一個超參數，用於平衡兩者
        lambda_ortho = 0.1
        return loss + lambda_ortho * ortho_loss
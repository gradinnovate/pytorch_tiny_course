import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HypergraphRayleighQuotientLossGeneralized(nn.Module):
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

    def _generate_partition_samples(self, Z: torch.Tensor, hyperedge_index: torch.Tensor, num_samples: int = 8) -> torch.Tensor:
        """生成多個 partition 樣本用於對比學習"""
        num_nodes = Z.shape[0]
        partitions = []
        
        for _ in range(num_samples):
            # 使用 embedding 生成 partition
            partition_probs = torch.sigmoid(Z.squeeze())
            partition = (partition_probs > 0.5).long()
            
            # 確保 balanced (50-50 split 附近)
            if torch.sum(partition) > 0.6 * num_nodes:
                # 太多 1，隨機設置一些為 0
                ones_indices = torch.where(partition == 1)[0]
                num_to_flip = int(0.1 * num_nodes)
                flip_indices = ones_indices[torch.randperm(len(ones_indices))[:num_to_flip]]
                partition[flip_indices] = 0
            elif torch.sum(partition) < 0.4 * num_nodes:
                # 太多 0，隨機設置一些為 1
                zeros_indices = torch.where(partition == 0)[0]
                num_to_flip = int(0.1 * num_nodes)
                flip_indices = zeros_indices[torch.randperm(len(zeros_indices))[:num_to_flip]]
                partition[flip_indices] = 1
                
            partitions.append(partition)
            
        return torch.stack(partitions)

    def _contrastive_partition_loss(self, Z: torch.Tensor, hyperedge_index: torch.Tensor, 
                                   num_nodes: int, hint_partition: torch.Tensor = None, 
                                   cut_size_func=None, temperature: float = 0.1) -> torch.Tensor:
        """對比學習：驗證後的 hint partition (好的) vs 隨機 partition (壞的)"""
        if hint_partition is None or cut_size_func is None:
            return torch.tensor(0.0, device=Z.device)
        
        # 生成多個隨機 balanced partitions
        random_partitions = []
        for _ in range(5):  # 生成 5 個隨機樣本
            random_partition = self._generate_random_balanced_partition(num_nodes, Z.device)
            random_partitions.append(random_partition.squeeze())
        
        # 計算所有 partition 的 cut size (包括 hint)
        hyperedge_index_cpu = hyperedge_index.cpu()
        hint_cut_size = cut_size_func(hint_partition.cpu(), hyperedge_index_cpu, num_nodes)
        
        random_cut_sizes = []
        for rp in random_partitions:
            cut_size = cut_size_func(rp.cpu(), hyperedge_index_cpu, num_nodes)
            random_cut_sizes.append(cut_size)
        
        # 找到比 hint 更差的隨機 partition
        worse_partitions = []
        for i, cut_size in enumerate(random_cut_sizes):
            if cut_size > hint_cut_size:  # 只用確實更差的作為負樣本
                worse_partitions.append(random_partitions[i])
        
        if not worse_partitions:
            # 如果沒有更差的，回退到 0 loss
            return torch.tensor(0.0, device=Z.device)
        
        # 方法1：通過 Z 生成 partition，再與 hint 和隨機 partition 比較
        
        # 用 embedding 的第一列（Fiedler vector）生成 predicted partition
        fiedler_vector = Z[:, 0] if Z.dim() > 1 and Z.shape[1] > 1 else Z.squeeze()
        predicted_partition = (torch.sigmoid(fiedler_vector) > 0.5).float()
        
        # 好的樣本：hint partition  
        good_partition = hint_partition.float()
        
        # 使用最差的隨機 partition 作為負樣本
        worst_idx = max(range(len(random_cut_sizes)), key=lambda i: random_cut_sizes[i])
        bad_partition = random_partitions[worst_idx].float()
        
        # 計算 partition 之間的相似度
        good_sim = F.cosine_similarity(predicted_partition.unsqueeze(0), good_partition.unsqueeze(0))
        bad_sim = F.cosine_similarity(predicted_partition.unsqueeze(0), bad_partition.unsqueeze(0))
        
        # 對比損失：predicted partition 應該更接近 hint partition
        contrastive_loss = -torch.log(torch.exp(good_sim / temperature) / 
                                     (torch.exp(good_sim / temperature) + torch.exp(bad_sim / temperature) + 1e-8))
        
        return contrastive_loss
    
    def _generate_random_balanced_partition(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        """生成隨機的 balanced partition 作為負樣本"""
        # 隨機分配，但保持 balanced (50-50 split)
        partition = torch.randperm(num_nodes, device=device)
        partition = (partition < num_nodes // 2).float().unsqueeze(1)
        return partition

    def forward(self, Z: torch.Tensor, hyperedge_index: torch.Tensor, 
                num_nodes: int, mu: torch.Tensor = None, logvar: torch.Tensor = None, 
                hint_partition: torch.Tensor = None, cut_size_func=None) -> torch.Tensor:
        """
        計算 Loss。

        Args:
            Z (torch.Tensor): GNN 輸出的節點嵌入, shape 為 (num_nodes, k)。
            hyperedge_index (torch.Tensor): 超圖的關聯索引。
            num_nodes (int): 圖中的節點數量。

        Returns:
            torch.Tensor: 一個標量 (scalar) 的 Loss 值。
        """
        k = Z.shape[1]
        node_idx, edge_idx = hyperedge_index
        num_hyperedges = (edge_idx.max() + 1).item()

        # --- 步驟 1: 預計算節點度和超邊度 ---
        # 使用單位權重 w(e) = 1 for all hyperedges
        hyperedge_weight = torch.ones(num_hyperedges, device=Z.device)
        
        # 計算節點度數: d(v) = 節點 v 連接的超邊數量
        Dv = torch.zeros(num_nodes, device=Z.device)
        Dv.scatter_add_(0, node_idx, torch.ones_like(node_idx, dtype=torch.float))
        
        # 計算超邊度數: δ(e) = 超邊 e 包含的節點數量  
        De = torch.zeros(num_hyperedges, device=Z.device)
        De.scatter_add_(0, edge_idx, torch.ones_like(edge_idx, dtype=torch.float))

        # 處理度為0的孤立點/邊，避免除以零
        epsilon_deg = 1e-6
        Dv = torch.clamp(Dv, min=epsilon_deg)
        De = torch.clamp(De, min=epsilon_deg)
        
        # --- 步驟 2: 計算歸一化超圖 Laplacian 廣義瑞利商 ---
        # 歸一化 Laplacian: ∆ = I - Θ, 其中 Θ = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
        # 廣義瑞利商: f^T ∆ f / f^T D_v f = (f^T D_v f - f^T Θ f) / f^T D_v f
        
        # 預計算 D_v^{-1/2}
        Dv_inv_sqrt = Dv.pow(-0.5).unsqueeze(1)
        
        rayleigh_quotients = []
        
        for j in range(k):
            f = Z[:, j:j+1]  # 當前列向量
            
            # 計算 f^T D_v f (廣義內積)  
            # f: [num_nodes, 1], Dv: [num_nodes] -> 需要正確廣播
            f_Dv_f = torch.sum(f.squeeze() * f.squeeze() * Dv)
            
            # 計算 f^T Θ f
            theta_quad_form = self._calculate_theta_quadratic_form(
                f, hyperedge_index, hyperedge_weight, De, Dv_inv_sqrt
            )
            
            # 廣義瑞利商: f^T ∆ f / f^T D_v f = (f^T D_v f - f^T Θ f) / f^T D_v f
            # = 1 - (f^T Θ f / f^T D_v f)
            epsilon = 1e-6
            
            # 確保分母不為零且數值穩定
            f_Dv_f_safe = torch.clamp(f_Dv_f, min=epsilon)
            theta_ratio = theta_quad_form / f_Dv_f_safe
            rayleigh_quotient = 1.0 - theta_ratio
            
            # 檢查數值穩定性
            if torch.isnan(rayleigh_quotient) or torch.isinf(rayleigh_quotient):
                rayleigh_quotient = torch.tensor(0.0, device=Z.device)
            
            rayleigh_quotients.append(rayleigh_quotient)
        
        rayleigh_quotients = torch.stack(rayleigh_quotients)
        
        # 主要損失：所有瑞利商的均值
        rayleigh_loss = torch.mean(rayleigh_quotients)
        
        # 添加 Fiedler vector 分離約束，防止退化為單峰
        separation_loss = self._fiedler_separation_penalty(Z)
        
        # 對比學習損失：驗證 hint 確實比隨機好
        contrastive_loss = self._contrastive_partition_loss(Z, hyperedge_index, num_nodes, hint_partition, cut_size_func)
        
        # VAE 的 KL 散度項（如果提供了 mu 和 logvar）
        kl_loss = 0.0
        if mu is not None and logvar is not None:
            # KL(q(z|x) || p(z)) 其中 p(z) = N(0,I)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / (num_nodes * Z.shape[1])  # 正確標準化
            
        
        # 組合損失 - 自監督學習
        lambda_contrastive = 0.01 # 對比學習權重
        lambda_kl = 0.001         # KL 散度權重 (降低)
        lambda_separation = 0.1   # 分離約束權重
        
        loss = (rayleigh_loss + 
                lambda_contrastive * contrastive_loss +
                lambda_kl * kl_loss +
                lambda_separation * separation_loss)
        
        
        
        return loss
    
    def _fiedler_separation_penalty(self, Z: torch.Tensor) -> torch.Tensor:
        """
        計算 Fiedler vector 分離約束，防止退化為單峰分佈。
        鼓勵第一個維度（Fiedler vector）維持雙峰分佈。
        """
        fiedler_vector = Z[:, 0]  # 第一列作為 Fiedler vector
        
        # 計算正負值的均值分離度
        pos_mask = fiedler_vector > 0
        neg_mask = fiedler_vector < 0
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            # 如果全為正或全為負，給予懲罰
            return torch.tensor(1.0, device=Z.device)
        
        pos_mean = fiedler_vector[pos_mask].mean()
        neg_mean = fiedler_vector[neg_mask].mean()
        
        # 分離度：正負均值的差距，越大越好
        separation = torch.abs(pos_mean - neg_mean)
        
        # 懲罰項：分離度越小懲罰越大
        penalty = 1.0 / (separation + 1e-6)
        
        return penalty
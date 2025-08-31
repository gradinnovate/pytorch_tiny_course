# 基於神經網路的超圖分割

本專案使用圖神經網路(GNNs)與基於歸一化超圖 Laplacian 廣義瑞利商的損失函數實現超圖分割。

## 損失函數數學公式

`loss_func2.py` 中的損失函數實現了**歸一化超圖 Laplacian 廣義瑞利商損失**，結合對比學習增強性能。

### 主要組件

#### 1. 歸一化超圖 Laplacian
實現歸一化 Laplacian: $\Delta = I - \Theta$，其中：

$$\Theta = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}$$

#### 2. 廣義瑞利商
對於每個嵌入維度 $j$，廣義瑞利商為：

$$R_j = \frac{f^T \Delta f}{f^T D_v f} = \frac{f^T D_v f - f^T \Theta f}{f^T D_v f} = 1 - \frac{f^T \Theta f}{f^T D_v f}$$

其中 $f = Z_j$ 是第 $j$ 列嵌入向量。

#### 3. $f^T \Theta f$ 的高效計算
使用變分形式高效計算：

$$f^T \Theta f = \sum_{e} \frac{w(e)}{\delta(e)} \times \left(\sum_{u \in e} D_v^{-1/2}[u] \cdot f[u]\right)^2$$

#### 4. 對比學習損失
當提供 hint partition 時，使用對比學習驗證 partition 品質：
- **正樣本**: hint partition (已知的好分割)
- **負樣本**: 隨機生成的 balanced partition
- **相似度**: cosine similarity between predicted 和 reference partitions
- **溫度縮放**: 使用溫度參數 $\tau = 0.1$ 進行 softmax 標準化

#### 5. 最終損失函數

$$\text{損失} = \text{瑞利商損失} + \lambda_c \cdot \text{對比損失} + \lambda_{KL} \cdot \text{KL散度}$$

其中：
- 瑞利商損失 = $\frac{1}{k}\sum_{j=1}^k R_j$
- $\lambda_c = 0.01$ (對比學習權重)
- $\lambda_{KL} = 0.001$ (VAE KL 散度權重)

### 關鍵數學定義

- **節點度數矩陣**: $D_v[i] = \sum_{e: i \in e} w(e)$ 對所有包含節點 $i$ 的超邊 $e$
- **超邊度數矩陣**: $D_e[e] = |e|$ (超邊 $e$ 的基數)  
- **關聯矩陣**: $H[i,e] = 1$ 若節點 $i \in$ 超邊 $e$，否則為 0
- **權重矩陣**: $W = \text{diag}(w(e))$ 對所有超邊 $e$

### 目標
最小化此損失函數會驅動 GNN 嵌入朝向正規化超圖拉普拉斯算子的最小特徵值對應的特徵向量，根據譜圖理論，這些向量對圖分割是最優的。

## 模型架構
- 3 層 HypergraphConv 網路
- 輸入：節點度數作為初始特徵
- 輸出：用於二元分割的 1 維嵌入
- 激活函數：LeakyReLU
- 正規化：LayerNorm

## 分割方法
訓練後，透過對嵌入進行排序並在中位數處分割來創建平衡分割。

### How to run
```bash
python3 train.py
```
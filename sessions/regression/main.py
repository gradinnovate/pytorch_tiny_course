import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib as mpl

# Unknown target function
def f(x1, x2):
    return 4 * x1 ** 2 + 3 * x2 - 6 * x1 - 0.2

# Surrogate function: y = a*x1^2 + b*x2^2 + c*x1 + d*x2 + e
def surrogate(params, x1, x2):
    a, b, c, d, e = params
    return a * x1 ** 2 + b * x2 ** 2 + c * x1 + d * x2 + e

# 1. 取樣 100 個 sample
torch.manual_seed(42)
x1 = torch.rand(100) * 10 - 5  # [-5, 5]
x2 = torch.rand(100) * 10 - 5
X = torch.stack([x1, x2], dim=1)
y = f(x1, x2)

# 2. surrogate function 參數
params = torch.nn.Parameter(torch.randn(5, requires_grad=True))
optimizer = optim.SGD([params], lr=1.2e-4)

# 3. batch 設定
batch_size = 10
epochs = 100

# 4. 用於畫圖的 meshgrid
grid_x1, grid_x2 = torch.meshgrid(
    torch.linspace(-5, 5, 100), torch.linspace(-5, 5, 100), indexing="ij"
)
with torch.no_grad():
    target_heatmap = f(grid_x1, grid_x2)

# 5. 記錄每個 epoch 的預測 heatmap
pred_heatmaps = []
losses = []

# 6. 訓練過程
def get_pred_heatmap(params):
    with torch.no_grad():
        return surrogate(params, grid_x1, grid_x2).cpu().numpy()

pred_heatmaps.append(get_pred_heatmap(params))  # epoch 0 (init)

for epoch in range(epochs):
    idx = torch.randperm(100)
    for i in range(0, 100, batch_size):
        batch_idx = idx[i:i+batch_size]
        x1b, x2b = x1[batch_idx], x2[batch_idx]
        yb = y[batch_idx]
        pred = surrogate(params, x1b, x2b)
        loss = nn.functional.mse_loss(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        pred_heatmaps.append(get_pred_heatmap(params))
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 7. 畫圖
fig, axes = plt.subplots(4, 3, figsize=(12, 12))
axes = axes.flatten()
ims = []
for i, ax in enumerate(axes):
    if i == 11:
        # loss curve
        ax.plot(range(1, epochs+1), losses, marker='o')
        ax.set_title('Loss Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        continue
    if i >= len(pred_heatmaps) + 1:
        ax.axis('off')
        continue
    if i == 0:
        im = ax.imshow(target_heatmap.cpu().numpy(), origin="lower", extent=[-5,5,-5,5], cmap='coolwarm')
        ax.set_title("Target f(x1,x2)")
    else:
        im = ax.imshow(pred_heatmaps[i-1], origin="lower", extent=[-5,5,-5,5], cmap='coolwarm')
        ax.set_title(f"Epoch {(i-1)*10}")
    ax.axis('off')
    ims.append(im)
plt.tight_layout()
plt.show()

# 印出學到的 surrogate function 參數
print('Learned parameters:')
print('a, b, c, d, e =', [f"{v:.2f}" for v in params.detach().cpu().numpy()])
print('Ideal parameters:')
print('a, b, c, d, e =', [4, 0, -6, 3, -0.2])
import torch

# 建立一個 tensor，並設定 requires_grad=True 以追蹤其梯度
x = torch.tensor([2.0, 3.0], requires_grad=True)

# 定義一個簡單的函數 y = x1^2 + 3*x2
# y 是一個 scalar
# x[0] = x1, x[1] = x2
y = x[0] ** 2 + 3 * x[1]

print(f"y = {y.item()}")

# 執行 backward，計算梯度
y.backward()

# 觀察 x 的梯度
grad = x.grad
print(f"x.grad = {grad}")

# 也可以觀察每個步驟的 requires_grad 屬性
print(f"x.requires_grad = {x.requires_grad}")
print(f"y.requires_grad = {y.requires_grad}") 
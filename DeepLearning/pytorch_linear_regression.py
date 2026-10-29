import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成线性数据（y = 3x + 2 + 噪声）
np.random.seed(0)
torch.manual_seed(0) # 设置PyTorch的随机种子

x = np.linspace(-10, 10, 100) # 生成100个点
y = 3 * x + 2 + np.random.normal(0, 5, size=x.shape) # 添加噪声，标准差为5

x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1) # 转换为PyTorch张量并调整形状为 (100, 1), 100行1列的向量
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 定义模型、损失与优化器
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练
epochs = 1000
for epoch in range(epochs):
    model.train()
    preds = model(x_tensor)
    loss = loss_fn(preds, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 输出参数并保存模型
w = model.weight.item()
b = model.bias.item()
print(f"Learned parameters: w = {w:.4f}, b = {b:.4f}")
torch.save(model.state_dict(), 'linear_model.pth')

# 可视化拟合结果
pred_y = model(x_tensor).detach().numpy().flatten()
plt.figure(figsize=(6,4))
plt.scatter(x, y, label='data', alpha=0.6)
plt.plot(x, pred_y, color='red', label='fitted')
plt.legend()
plt.title(f"Linear fit: y = {w:.2f}x + {b:.2f}")
plt.tight_layout()
plt.savefig('pytorch_linear_regression.png', dpi=150)
plt.show()



import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 生成数据：输入 (a, b)，输出 1 表示 a > b，否则 0
def generate_data(n_samples=1000):
    a = torch.rand(n_samples, 1)
    b = torch.rand(n_samples, 1)
    x = torch.cat([a, b], dim=1)
    y = (a > b).float().squeeze()  # 变成 [1000]
    return x, y

# 定义一个简单神经网络
class CompareNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# 准备训练数据
x_train, y_train = generate_data(1000)

# 构建模型
model = CompareNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

# 训练网络
losses = []
for epoch in range(240):
    y_pred = model(x_train).squeeze()
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# 画出训练损失
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.title("Training Loss: Compare if a > b")
plt.grid(True)
plt.show()

# 测试模型
test_samples = torch.tensor([[0.9, 0.2], [0.1, 0.8], [0.5, 0.5]])
preds = model(test_samples).detach().squeeze()
for pair, result in zip(test_samples.tolist(), preds.tolist()):
    print(f"Compare {pair[0]:.2f} > {pair[1]:.2f} → Prediction: {result:.2f}")

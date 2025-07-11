import numpy as np

# 模拟数据
X = np.random.rand(100, 1)
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.1

# 初始化参数
w, b = 0.0, 0.0
lr = 0.1
epochs = 10

# SGD 训练
for epoch in range(epochs):
    for i in range(len(X)):
        xi = X[i]
        yi = y[i]
        pred = w * xi + b
        error = pred - yi
        w -= lr * error * xi
        b -= lr * error

print(f"Learned w: {w.item():.2f}, b: {b.item():.2f}")


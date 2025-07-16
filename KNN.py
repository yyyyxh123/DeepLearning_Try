import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# 构造数据
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, size=X.shape[0])

# 拟合 KNN 回归模型
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X, y)

# 预测
X_test = np.linspace(0, 10, 200).reshape(-1, 1)
y_pred = knn.predict(X_test)

# 可视化
plt.scatter(X, y, label="Train data", alpha=0.5)
plt.plot(X_test, y_pred, label="KNN Regression", color='red')
plt.legend()
plt.title("k-NN Regression (k=5)")
plt.grid(True)
plt.show()

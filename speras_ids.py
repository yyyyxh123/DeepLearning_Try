import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression

# 构造一个稀疏线性模型：只前5个特征有效
X, y, coef = make_regression(
    n_samples=100, 
    n_features=20, 
    n_informative=5,   # 只有前5个特征影响 y
    noise=0.1, 
    coef=True, 
    random_state=42
)

# 用 Lasso（带 L1 正则）进行回归
lasso = Lasso(alpha=0.05)  # alpha 越大，越稀疏
lasso.fit(X, y)

# 可视化真实 vs 拟合的权重
plt.figure(figsize=(10,5))
plt.plot(coef, 'o-', label='True Coefficients')
plt.plot(lasso.coef_, 'x--', label='Lasso Estimated Coefficients')
plt.axhline(0, color='gray', linestyle='--')
plt.legend()
plt.title("Lasso for Sparse Modeling")
plt.xlabel("Feature Index")
plt.ylabel("Weight")
plt.grid(True)
plt.show()

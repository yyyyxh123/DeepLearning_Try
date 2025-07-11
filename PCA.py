from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 载入示例数据
data = load_iris()
X = data.data
y = data.target

# 降维到 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 可视化
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title("Iris dataset projected to 2D by PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

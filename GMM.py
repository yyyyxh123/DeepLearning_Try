import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 混合两个正态分布
np.random.seed(0)
samples = np.concatenate([
    np.random.normal(loc=-2, scale=1.0, size=300),
    np.random.normal(loc=2, scale=0.5, size=700)
])

# 可视化
plt.hist(samples, bins=50, density=True, alpha=0.5, label="Mixture samples")

# 理论混合分布密度
x = np.linspace(-6, 6, 1000)
pdf = 0.3 * norm.pdf(x, -2, 1.0) + 0.7 * norm.pdf(x, 2, 0.5)
plt.plot(x, pdf, label="True mixture PDF", color='red')
plt.title("Mixture of Two Gaussians")
plt.legend()
plt.show()

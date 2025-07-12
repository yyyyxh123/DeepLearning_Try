import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)

# 指数分布（仅正方向）
from scipy.stats import expon
plt.plot(x[x >= 0], expon.pdf(x[x >= 0], scale=1), label='Exponential(λ=1)')

# 拉普拉斯分布
from scipy.stats import laplace
plt.plot(x, laplace.pdf(x, loc=0, scale=1), label='Laplace(μ=0, b=1)')

plt.legend()
plt.title("Exponential vs Laplace Distribution")
plt.grid(True)
plt.show()

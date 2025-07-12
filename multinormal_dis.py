import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal

mu = [0, 0]
cov = [[1, 0.8], [0.8, 1]]

samples = multivariate_normal(mu, cov, size=500)

plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.axis('equal')
plt.title("Multivariate Normal Distribution (2D)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()

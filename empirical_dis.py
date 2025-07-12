import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

data = np.random.normal(0, 1, size=100)
ecdf = ECDF(data)

plt.plot(ecdf.x, ecdf.y)
plt.title("Empirical CDF")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.grid(True)
plt.show()

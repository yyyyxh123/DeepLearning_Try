import numpy as np
import matplotlib.pyplot as plt

def softplus(x):
    return np.log(1 + np.exp(x))

x = np.linspace(-10, 10, 200)
y = softplus(x)

plt.plot(x, y, label="Softplus")
plt.plot(x, np.maximum(0, x), '--', label="ReLU")
plt.legend()
plt.title("Softplus vs ReLU")
plt.grid(True)
plt.show()

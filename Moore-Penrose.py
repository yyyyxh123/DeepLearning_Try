import numpy as np

A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

A_pinv = np.linalg.pinv(A)
print(A_pinv)

import numpy as np

a = np.array([
    [1, 2],
    [3, 4]
])

b = np.array([1, 2])

print(a*b)
print()
print(np.sum(a*b, axis=0))


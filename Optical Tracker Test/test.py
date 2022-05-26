import numpy as np

og_basis = [[1,0], [0,1]]

R = np.array([[np.cos(-0.785), -np.sin(-0.785)],
           [np.sin(-0.785), np.cos(-0.785)]])

vector = [-5, 5]
new_basis = np.matmul(R, og_basis)
print(np.dot(new_basis, vector))

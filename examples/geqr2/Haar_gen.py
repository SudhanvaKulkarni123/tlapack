import numpy as np
def get_norm_mat(n):
    a = []
    for j in range(n):
        b = []
        for i in range(n):
            b = b + [np.random.normal()]
        a = a + [b]
    return np.matrix(a)

print(get_norm_mat(3))
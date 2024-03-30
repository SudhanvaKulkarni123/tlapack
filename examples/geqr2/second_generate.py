import numpy as np
import scipy
from scipy.stats import ortho_group


a_values = []
for i in range(255):
    if i != 128 and i != 127:
        if not ((i >> 3) & 0b01111) == 0:
            if i >> 7 == 1:
                val1 = (2.0 ** (((i >> 3) & 0b01111) - 8)) * (1.0 + (1.0/2.0) * ((i >> 2) & 0b000001) + (1.0/4.0) * ((i >> 1) & 0b0000001) + (1.0/8.0) * ((i >> 0) & 0b00000001))
            else :
                val1 = -(2.0 ** (((i >> 3) & 0b01111) - 8)) * (1.0 + (1.0/2.0) * ((i >> 2) & 0b000001) + (1.0/4.0) * ((i >> 1) & 0b0000001) + (1.0/8.0) * ((i >> 0) & 0b00000001))

        else:
            if i >> 7 == 1:
                val1 = (2.0 ** -7) * (0.0 + (1.0/2.0) * ((i >> 2) & 0b000001) + (1.0/4.0) * ((i >> 1) & 0b0000001) + (1.0/8.0) * ((i >> 0) & 0b00000001))
            else :
                val1 = -(2.0 ** -7) * (0.0 + (1.0/2.0) * ((i >> 2) & 0b000001) + (1.0/4.0) * ((i >> 1) & 0b0000001) + (1.0/8.0) * ((i >> 0) & 0b00000001))
        
        a_values = [val1] + a_values




def find_closest_value(b, lst):
    min_diff = float('inf')  # Initialize minimum difference to infinity
    
    for a in lst:
        diff = abs(b - a)
        if diff < min_diff:
            min_diff = diff
            closest = a
    
    return closest

def init_matrices(dim, cond, n):
    #this function generates a matrix of given condition number in float32 using SVD.
    #We then round that matrix down to fp8
    to_ret = []
    for _ in range(n):
        sigma = np.diag([np.power(cond,float(-i)/float(n - 1)) for i in range(dim)])
        U = ortho_group.rvs(dim)
        V = ortho_group.rvs(dim)
        A = np.matmul(np.matmul(U,sigma),V)
        m,n = np.shape(A)
        for i in range(m):
            for j in range(n):
                A[i,j] = find_closest_value(A[i,j], a_values)       ##rounds to desired precision
        to_ret = to_ret + [A]
    return to_ret


def x_step(x,v):
    x = x + v
    return x, v

def v_step()




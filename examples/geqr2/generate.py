import random
import numpy as np
import scipy as sc


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

def perturb(X, epsilon):
    rows,cols = np.shape(X)
    Y = np.eye(rows)
    for i in range(rows):
        for j in range(cols):
            Y[i,j] = find_closest_value(X[i,j]*(1 + np.random.normal()*epsilon),a_values)
    while abs(np.linalg.det(Y)) <= 0.01 :
        for i in range(rows):
            for j in range(cols):
                Y[i,j] = find_closest_value(X[i,j]*(1 + np.random.normal()*epsilon),a_values)
    flip = random.randint(0, 1)
    if flip == 1:
        for i in range(rows):
            for j in range(cols):
                Y[i,j] = -Y[i,j]
    return Y
    

def Energy(X):
    a = np.linalg.cond(X)
    if a == float('inf'):
        a = 9999999999999999            #to avoid infs tho we won't run into them anyway since we exclude singular matrices
    return abs(a - 500.0)

def annealing_step(X,T,gamma, lowest, iter):
    energy1 = Energy(X)
    if energy1 < lowest:
        lowest = energy1
    energy1 = 1 - np.exp(-gamma*(energy1 - lowest))
    Y = perturb(X, 0.25)
    energy2 = Energy(Y)
    if energy2 < lowest:
        lowest = energy2
    energy2 = 1 - np.exp(-gamma*(energy2 - lowest))
    if energy2 < energy1:
        T = T/(1 + np.log(1 + iter))
        return Y , lowest, T
    else:
        ran = np.random.uniform()
        delta = energy2 - energy1
        prob = np.exp(-delta/T)
        if ran < prob:
            T = T/(1 + np.log(1 + iter))
            return Y , lowest , T
        else :
            T = T/(1 + np.log(1 + iter))
            return X , lowest , T



def cond_annealing(n):
    A_orig = np.identity(n)
    for i in range(n):
        for j in range(n):
            A_orig[i,j] = random.choice(a_values)
    count = 0
    while abs(np.linalg.det(A_orig)) <= 0.01 :
         for i in range(n):
            for j in range(n):
                A_orig[i,j] = random.choice(a_values)
    
    A = A_orig
    lowest = float('inf')
    T = 250
    while count < 500 :
        A , lowest, T = annealing_step(A, T, 0.002, lowest, count)
        count = count + 1
        if abs(np.linalg.cond(A) - 500.0) < 1.0:
            return np.linalg.cond(A)
    
    return (np.linalg.cond(A))


print(cond_annealing(100))

    

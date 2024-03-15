import random
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

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
a_values.sort()



def find_closest_value(b, lst):
    min_diff = float('inf')  # Initialize minimum difference to infinity
    
    for a in lst:
        diff = abs(b - a)
        if diff < min_diff:
            min_diff = diff
            closest = a
    
    return closest


def random_orthog(n):
    H = np.random.randn(n, n)
    Q, R = np.linalg.qr(H)
    return Q

def cond_estim(A):
    #estimates condition nummber of A = LU
    return


def init_matrix(dim, cond, is_geom):
    #this function generates a matrix of given condition number in float32 using SVD.
    #We then round that matrix down to fp8
    if is_geom:
        sigma = np.diag([np.power(cond,float(-i)/float(dim - 1)) for i in range(dim)])
    else :
        sigma = np.diag([float(1.0 - float(i)*(1.0 - 1.0/cond)/float(dim-1)) for i in range(dim)] )
    U = random_orthog(dim)
    V = random_orthog(dim)
    A = np.matmul(np.matmul(U,sigma),V)
    m,n = np.shape(A)
    for i in range(m):
        for j in range(n):
            A[i,j] = find_closest_value(A[i,j], a_values)       ##rounds to desired precision

    return A




def perturb(X, epsilon, is_LU, m):
    rows,cols = np.shape(X)
    Y = np.eye(rows)
    if not is_LU:
        I = range(rows)
        J = range(cols)
    else :
        I = range(rows- m,rows)
        J = range(cols- m, cols)
    for i in I:
        for j in J:
            Y[i,j] = find_closest_value(X[i,j]*(1 + np.random.uniform(-1.5,1.5)*epsilon),a_values)
    while abs(np.linalg.det(Y)) <= 0.000000001 :               #regenerate if matrix is close to singular
        for i in I:
            for j in J:
                Y[i,j] = find_closest_value(X[i,j]*(1 + np.random.uniform(-1.5,1.5)*epsilon),a_values)
    P,L,U = sc.linalg.lu(Y[-m:,-m:])
    Y[-m:,-m:] = np.matmul(np.transpose(P),Y[-m:,-m:])
    # flip = random.randint(0, 1)
    # if flip == 1:
    #     for i in I:
    #         for j in J:
    #             Y[i,j] = -Y[i,j]
    return Y, 0

    

def Energy(X, cond):
    a = np.linalg.cond(X)
    if a == float('inf'):
        a = 9999999999999999            #to avoid infs tho we won't run into them anyway since we exclude singular matrices
    return abs(a - cond)

def annealing_step(X,T,gamma, lowest, iter, cond, is_LU, m):
    energy1 = Energy(X, cond)
    if energy1 < lowest:
        lowest = energy1
    energy1 = 1 - np.exp(-gamma*(energy1 - lowest))
    Y , num = perturb(X, 0.25, is_LU, m)
    energy2 = Energy(Y, cond)
    if energy2 < lowest:
        lowest = energy2
    energy2 = 1 - np.exp(-gamma*(energy2 - lowest))
    if energy2 < energy1:
        T = T/(1 + np.log(iter)) 
        return Y , lowest, T
    else:
        ran = np.random.uniform()
        delta = energy2 - energy1
        prob = np.exp(-delta/T)
        if ran < prob:
            T = T/(1 + np.log(iter))
            return Y , lowest , T
        else :
            T = T/(1 + (np.log(iter)))
            return X , lowest , T



def cond_annealing(n, cond):
    A_orig = init_matrix(n, cond, False)
    count = 0
    while abs(np.linalg.det(A_orig)) <= 0.0000001 :          #exclude nearly singular matrices
         for i in range(n):
            for j in range(n):
                A_orig[i,j] = random.choice(a_values)
    
    A = A_orig
    lowest = float('inf')
    T = 250
    while count < 500 :
        A , lowest, T = annealing_step(A, T, 0.15 , lowest, count, cond, False, 0)
        count = count + 1
        if abs(np.linalg.cond(A) - cond) < cond/10.0:
            return np.linalg.cond(A), count
    
    return np.linalg.cond(A), count        #return only the condition number since I don't want to print out the bigger matrices

def vanilla_LU_gen(A, n, cond, new_val):
     
     last = A[n-1,n-1]
     A[n-1,n-1] = new_val
     to_ret = np.linalg.cond(A)
     A[n-1,n-1] = last
     return [to_ret, new_val - last]

def LU_gen(n,cond,m, mode):
    #m is dimension of trailing submatrix that we will optimize on
    A_orig = init_matrix(n, cond, mode)
    for i in range(n):
        for j in range(n):
            A_orig[i,j] = random.choice(a_values)
    while abs(np.linalg.det(A_orig)) <= 0.0000001 :          #exclude nearly singular matrices
         for i in range(n):
            for j in range(n):
                A_orig[i,j] = random.choice(a_values)
    
    P,L,U = sc.linalg.lu(A_orig)
    A = np.matmul(np.transpose(P), A_orig)
    lowest = float('inf')
    T = 250
    count = 1
    while count < 500:
        A , lowest, T = annealing_step(A, T, 5, lowest, count, cond, True, m)
        count = count + 1
        if abs(np.linalg.cond(A)/cond) < 1.5 and abs(np.linalg.cond(A)/cond) > 0.66:
            return np.linalg.cond(A), count 
    
    return np.linalg.cond(A), count       #return only the condition number since I don't want to print out the bigger matrices



# print(cond_annealing(100, 100.0))
#print(vanilla_LU_gen(100,100.0))
#print(LU_gen(100,100.0,20, True))
A_orig = init_matrix(10, 100, True)
print("original condition number")
print(np.linalg.cond(A_orig))
P,L,U = sc.linalg.lu(A_orig)
A_send = np.matmul(np.transpose(P),A_orig)
k = a_values.index(A_send[4,4])
to_use = a_values[k-10:k+11]
print(to_use[-1])
z = [vanilla_LU_gen(A_send,5, 10, val) for val in to_use]
y = [z[j][0] for j in range(len(z))]
x = [z[j][1] for j in range(len(z))]
for k in range(len(x)):
    if x[k] == 0:
        print(y[k], x[k], k)
print(x)
plt.plot(x,y)
plt.show()



    

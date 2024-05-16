import random
import numpy as np
import scipy as sc
#import matplotlib.pyplot as plt

def set_vals(p):
    a_values = []
    count = 0
    
    for i in range(255):
            exp = ((i >> (p - 1)) & (2**(9 - p) - 2**(8 - p) - 1))
            if i != 128 and i != 127:   
                if not exp == 0:
                    count = count + 1
                    if i >> 7 == 1:
                        val1 = 1.0
                        for j in range(1,p): 
                            val1 = val1 + (2**(-p+j))*((i >> (j-1)) & 1)
                    else :
                        val1 = 1.0
                        for j in range(1,p): 
                            val1 = val1 + (2**(-p+j))*((i >> (j-1)) & 1)   
                        val1 = -val1 
                else:
                    if i >> 7 == 1:
                        val1 = 0.0
                        for j in range(1,p): 
                            val1 = val1 + (2**(-p+j))*((i >> (j-1)) & 1)
                    else :
                        val1 = 0.0
                        for j in range(1,p): 
                            val1 = val1 + (2**(-p+j))*((i >> (j-1)) & 1)
                val1 = (2**(exp - 2**(7-p)))*val1
                a_values = [val1] + a_values
    a_values.sort()
    return a_values


    

# def write_to_file(A):
#     cond = np.linalg.cond(A)
#     f = open("mat/" + int(cond).__str__() + ".txt", "w")
#     f.write(A.shape[0].__str__())
#     for i in range(A.shape[0]):
#         f.write("\n")
#         for j in range(A.shape[1]):
#             f.write(str(A[i,j]) + ",")


def find_closest_value(b, lst):
    min_diff = float('inf')  # Initialize minimum difference to infinity
    closest = float('inf')
    for a in lst:
        diff = abs(b - a)
        if diff < min_diff:
            min_diff = diff
            closest = a
        elif diff == min_diff:
            if lst.index(a) % 2 == 0:
                closest = a
    
    return closest


def random_orthog(n):
  
    H = np.random.randn(n, n)
    Q, R = np.linalg.qr(H)
    return Q

def get_cond(lst):
    return np.linalg.cond(A)


def cond_estim(A):
    #estimates condition nummber of A = LU
    return



def init_matrix(dim1, dim2, cond, is_geom, lst):
    #this function generates a matrix of given condition number in float32 using SVD.
    #We then round that matrix down to fp8

    a_values = lst
    dim = min(dim1,dim2)
    sigma = np.matrix([[0.0 for _ in range(dim2)] for _ in range(dim1)])
    if is_geom:
        for i in range(dim):
            sigma[i,i] = np.power(cond,float(-i)/float(dim - 1))
    else :
        for i in range(dim):
            sigma[i,i] = float(1.0 - float(i)*(1.0 - 1.0/cond)/float(dim-1))  
    U = random_orthog(dim1)
    V = random_orthog(dim2)
    A = np.matmul(np.matmul(U,sigma),V)
    m,n = np.shape(A)
    for i in range(m):
        for j in range(n):
            A[i,j] = find_closest_value(A[i,j], a_values)       ##rounds to desired precision

    return A





def perturb(X, epsilon, is_LU, m, lst):
    a_values = lst
    rows,cols = np.shape(X)
    Y = np.eye(rows)
    if not is_LU:
        I = range(rows)
        J = range(cols)
    else :
        I = range(rows- m - (rows - cols),rows)
        J = range(cols- m, cols)
    for i in I:
        for j in J:
            X[i,j] = find_closest_value(X[i,j]*(1 + np.random.uniform(-1.5,1.5)*epsilon),a_values)
    P,L,U = sc.linalg.lu(X[-m - (rows - cols):,-m:])
    Y[-m - (rows - cols):,-m:] = np.matmul(np.transpose(P),X[-m - (rows - cols):,-m:])
    X[-m - (rows - cols):,-m:] = Y[-m - (rows - cols):,-m:]
    return X, 0

    

def Energy(X, cond):
  
    a = np.linalg.cond(X)
    if a == float('inf'):
        a = 9999999999999999            #to avoid infs tho we won't run into them anyway since we exclude singular matrices
    return abs(a - cond) 

def tunneling_step(X,T,gamma, lowest, iter, cond, is_LU, m, lst):
   
    energy1 = Energy(X, cond)
    if energy1 < lowest:
        lowest = energy1
    energy1 = 1 - np.exp(-gamma*(energy1 - lowest))
    Y , num = perturb(X, 0.125, is_LU, m, lst)
    energy2 = Energy(Y, cond)
    if energy2 < lowest:
        lowest = energy2
    energy2 = 1 - np.exp(-gamma*(energy2 - lowest))
    if energy2 < energy1:
        T = T/(1 + np.log(np.sqrt(iter))) 
        return Y , lowest, T
    else:
        ran = np.random.uniform()
        delta = energy2 - energy1
        prob = np.exp(-delta/T)
        if ran < prob:
            T = T/(1 + np.log(np.sqrt(iter)))
            return Y , lowest , T
        else :
            T = T/(1 + (np.log(np.sqrt(iter))))
            return X , lowest , T


def annealing_step():
    return
def cond_annealing(n, cond, p):

    a_values = set_vals(p)
    A_orig = init_matrix(n, cond, False, a_values)
    for i in range(n):
        for j in range(n):
            A[i,j] = random.choice(a_values)
    count = 0
    A = A_orig
    lowest = float('inf')
    T = 2500
    while count < 5000 :
        A , lowest, T = tunneling_step(A, T, 0.15 , lowest, count, cond, False, 0, a_values)
        count = count + 1
        if abs(np.linalg.cond(A) - cond) < cond/10.0:
            return np.linalg.cond(A), count
    
    return np.linalg.cond(A), count        #return only the condition number since I don't want to print out the bigger matrices

def vanilla_LU_gen(A, n, m,cond, new_val):   
    last = A[m-1,n-1]
    A[m-1,n-1] = new_val
    to_ret = np.linalg.cond(A)
    A[m-1,n-1] = last
    return [to_ret, new_val - last]

def LU_gen(n1, n2, cond,m, mode, p):
    #m is dimension of trailing submatrix that we will optimize on
    a_values = set_vals(p)
    A_orig = init_matrix(n1, n2, cond, mode, a_values)
    P,L,U = sc.linalg.lu(A_orig)
    A = np.matmul(np.transpose(P), A_orig)
    lowest = float('inf')
    T = 2500
    count = 1
    while count < 500:
        A , lowest, T = tunneling_step(A, T, 0.3, lowest, count, cond, True, m, a_values)
        count = count + 1
        if abs(np.linalg.cond(A) - cond) < 0.1*cond:
            Ai = np.linalg.pinv(A)
            ATA = np.linalg.inv(np.matmul(np.transpose(A), A))
            to_ret = A.flatten('F').tolist()[0]
            to_ret.append(np.linalg.cond(A))
            # Q : Do we make a new copy of to_ret in the following line? If yes, can we change the line so that we modify the existing copy of to_ret?
            # A : Yes, we make a new copy of to_ret in the following line. We can change the line so that we modify the existing copy of to_ret.
            #
            to_ret += Ai.flatten('F').tolist()[0] + ATA.flatten('F').tolist()[0]     
            return to_ret
    
    Ai = np.linalg.pinv(A)
    ATA = np.linalg.inv(np.matmul(np.transpose(A), A))
    to_ret = A.flatten('F').tolist()[0]
    to_ret.append(np.linalg.cond(A))
    to_ret += Ai.flatten('F').tolist()[0] + ATA.flatten('F').tolist()[0]  

    return to_ret





    

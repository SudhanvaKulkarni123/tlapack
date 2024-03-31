import numpy as np
import scipy as sc 
from random import randrange


def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1J / N )
    W = np.power( omega, i * j ) / np.sqrt(N)
    return W

def Hartley_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    W = np.cos(2*np.pi * i * j / N) + np.sin(2*np.pi * i * j / N)
    return W

def Helmert_matrix(N):
    return sc.linalg.helmert(N)



def generate_random_matrix(n):
    random_matrix = np.random.rand(n, n).astype(np.float32)
    return random_matrix

def round(matrix):
    return np.float16(matrix) - (np.finfo(np.float16).eps/2)*matrix -  (np.finfo(np.float16).eps/4)*matrix 


def FourOne(cond, n, choice):
    if choice == "Fourier":
        Q = DFT_matrix(n)
    elif choice == "Helmert":
        Q = Helmert_matrix(n)
    elif choice == "Hartley":
        Q = Hartley_matrix(n)
    alpha = -2
    l = randrange(0,n)
    y = np.zeros(n)
    for i in range(n):
        if i == l:
            y[i] = Q[n-1,i]*Q[n-1,l]*(1.0/cond - 1.0) + 1.0
        else : 
            y[i] = Q[n-1,i]*Q[n-1,l]*(1.0/cond - 1.0)
    for i in range(n):
        Q[n-1,i]  = (1.0/cond)*Q[n-1,i]
    return Q + alpha*np.outer(Q[l,:],y)




a = FourOne(100.0,100,"Hartley")
print(np.linalg.cond(a))
print(np.linalg.cond(np.float32(round(a))))
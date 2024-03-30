import numpy as np
n = 1
random_matrix = np.random.rand(n, n)
import numpy as np

import numpy as np

def givens_matrix(a,b):

   if b == 0:
       c = 1
       s = 0
   else :
       if abs(b) <= abs(a):
           t = -a/b
           s = 1/np.sqrt(1 + t**2)
           c = s*t
       else :
           t = -b/a
           c = 1/np.sqrt(1 + t**2)
           s = c*t
   return c,s
            



def qr_factorization(A):
    m, n = A.shape
    Q = np.eye(m)  
    R = np.copy(A)
    I = np.eye(m)   
    
    for j in range(n):
      
        normx = np.linalg.norm(R[j:, j])
        s = -np.sign(R[j, j])
        u1 = R[j, j] - s * normx
        w = np.copy(R[j:, j])
        w = w / u1
        w[0] = 1
        tau = -s * u1 / normx
        
        # Update R and Q
        R[j:, :] -= np.matmul((tau * np.outer(w, w)),R[j:, :])
        Q[:, j:] -= np.matmul(Q[:, j:], (tau * np.outer(w, w)))
        
    return Q, R


def generate_random_matrix(n):
    # Generate n-by-n random matrix of floats with float16 datatype
    random_matrix = np.random.rand(n, n).astype(np.float32)
    return random_matrix

def delete_row(matrix, k):
    # Delete the k-th row from the matrix
    new_matrix = np.delete(matrix, k, axis=0)
    return new_matrix

def relative_error_QR(matrix):
    Q, R = np.linalg.qr(matrix)
    QR = np.dot(Q, R)
    print(type(Q[0,0]))
    error = np.linalg.norm(QR - matrix, ord=2) / np.linalg.norm(matrix, ord=2)
    return error

import numpy as np

def insert_random_row(matrix, i):

    # Determine the shape of the original matrix
    num_rows, num_cols = matrix.shape
    
    # Generate a new row of random elements
    new_row = np.random.rand(1, num_cols)
    
    # Split the original matrix into two parts
    upper_half = matrix[:i+1, :]
    lower_half = matrix[i+1:, :]
    
    # Stack the new row between the two parts
    new_matrix = np.vstack([upper_half, new_row, lower_half])
    
    return new_matrix


def matmul(A,B):
    m, n_A = A.shape
    n_B, p = B.shape
    result = np.zeros((m, p))

    for i in range(m):
        for j in range(p):
            for k in range(n_A):
                result[i, j] += A[i, k] * B[k, j]

    return result


def R_update(Q, R, k):
    #finds QR of matrix with kth row deleted
    q = Q[k,:]
    
    q = np.transpose(q)
    m, n = R.shape
   
    for j in range(m - 2, -1, -1):
        c,s = givens_matrix(q[j], q[j+1])
       
        # Update q
        q[j] = c * q[j] - s * q[j+1]
        
        # Update R if there is a nonzero row
        G = np.matrix([[c,s],[-s,c]])
        if j < n:
            R[j:j+2, j:] = matmul(np.transpose(G),R[j:j+2,j:])
    return R[1:,:]


def Q_update(Q, k):
    m, n = Q.shape
    q = Q[k,:]
    if k > 0:
        Q[1:k+1,:] = Q[:k,:]
    for j in range(m-2,0,-1):
        c,s = givens_matrix(q[j], q[j+1])
        G = np.matrix([[c,s],[-s,c]])
        Q[1:,j:j+2] = matmul(Q[1:,j:j+2],G)
    c,s = givens_matrix(q[0], q[1])
    Q[1:,1] = s*Q[1:,0] + c*Q[1:,1]
    return Q[1:,1:]



# Example: Calculate relative error for a random 5-by-5 matrix
n = 4
random_matrix = generate_random_matrix(n)
higher_random_matrix = np.float64(random_matrix)
Q1, R1 = np.linalg.qr(higher_random_matrix, 'complete')


A = insert_random_row(random_matrix, 2)
A2 = insert_random_row(random_matrix, 3)
A4 = insert_random_row(random_matrix, 4)
Q2, R2 = qr_factorization(np.float16(A))
Q3, R3 = qr_factorization(np.float16(A2))
Q4, R4 = qr_factorization(np.float16(A4))
B = A.copy()
final_R = R_update(np.float16(Q2),np.float16(R2),3)
final_R2 = R_update(np.float16(Q3),np.float16(R3),4)
final_R3 = R_update(np.float16(Q4),np.float16(R4),5)
Q3, R3 = qr_factorization(np.float16(higher_random_matrix))

#print(Q2)

# Q2, R2 = np.linalg.qr(B, 'complete')f



print(np.linalg.norm(R1 - (-np.float32(final_R) + 2.0*np.float32(final_R2) - np.float32(final_R3))))
print(np.linalg.norm(R1 - R3))











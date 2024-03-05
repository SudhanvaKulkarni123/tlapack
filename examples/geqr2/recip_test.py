import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
import random
a_values = []
b_values = []
c_values = []
d_values = []
import math

for i in range(1, 255):
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
 
def LU_approx(n,a):
    A_orig = np.identity(n)
    for i in range(n):
        for j in range(n):
            A_orig[i,j] = random.choice(a_values)

    P, L,U = scipy.linalg.lu(np.matrix(A_orig))

    U_copy = U.copy()
    U_copy[n-1,n-1] = a
    A_new = np.matmul(np.transpose(P), A_orig)
    A_new[n-1,n-1] = a

    return np.linalg.cond(A_new)


x = np.array(a_values)
y = [LU_approx(5,a) for a in x]
plt.loglog(x,y)
plt.xlabel("U[n,n]")
plt.ylabel("condition number")
plt.text(-5, 60, '$n = 100$', fontsize = 22)
plt.show()









# for i in range(1, 255):
#     if i != 128 and i != 127:
#         sign = -1.0 if i >> 7 else 1.0
#         if (i >> 2) & 0b011111:
#             bal1 = sign * (2.0 ** (((i >> 2) & 0b011111) - 16)) * (1.0 + (1.0/2.0) * ((i >> 1) & 0b000001) + (1.0/4.0) * ((i >> 0) & 0b0000001))
#         else:
#             bal1 = sign * (2.0 ** -15) * (0.0 + (1.0/2.0) * ((i >> 1) & 0b000001) + (1.0/4.0) * ((i >> 0) & 0b0000001))
#         b_values = [bal1] + b_values

# for i in range(1, 255):
#     if i != 128 and i != 127:
#         sign = -1.0 if i >> 7 else 1.0
#         if (i >> 1) & 0b0111111:
#             cal1 = sign * (2.0 ** (((i >> 1) & 0b011111) - 32)) * (1.0 + (1.0/2.0) * ((i >> 0) & 0b000001))
#         else:
#             cal1 = sign * (2.0 ** -31) * (0.0 + (1.0/2.0) * ((i >> 0) & 0b000001))
#         c_values = [cal1] + c_values

# for i in range(1, 255):
#     if i != 128 and i != 127:
#         sign = -1.0 if i >> 7 else 1.0
#         if (i >> 4) & 0b0111:
#             dal1 = sign * (2.0 ** (((i >> 2) & 0b011111) - 16)) * (1.0 + (1.0/2.0) * ((i >> 1) & 0b000001) + (1.0/4.0) * ((i >> 0) & 0b0000001))
#         else:
#             cal1 = sign * (2.0 ** -31) * (0.0 + (1.0/2.0) * ((i >> 0) & 0b000001))
#         c_values = [cal1] + c_values




# def find_closest_value(b, lst):
#     min_diff = float('inf')  # Initialize minimum difference to infinity
    
#     for a in lst:
#         diff = abs(b - a)
#         if diff < min_diff:
#             min_diff = diff
#             closest = a
    
#     return closest

# def isinf(x, lst):
#     return abs(x) > np.max(lst)

# for a in a_values:
#     recip_a = find_closest_value(1.0/a, a_values)
#     if not isinf(1.0/a, a_values):
#         a_x_recip_a = find_closest_value(a*recip_a, a_values)
#         if a_x_recip_a - 1.0 != 0.0: 
#             print(a.__str__() + "," + recip_a.__str__() + "," + a_x_recip_a.__str__())

# for b in b_values:
#     recip_b = find_closest_value(1.0/b, b_values)
#     if not isinf(1.0/b, b_values):
#         b_x_recip_b = find_closest_value(b*recip_b, b_values)
#         if b_x_recip_b - 1.0 != 0.0: 
#             print(b.__str__() + "," + recip_b.__str__() + "," + b_x_recip_b.__str__())


# print("single")
# max_float = 2.0**(127)
# sum = 1.0
# for i in range(10):
#     sum = sum + ((1.0/2.0)**float(i+1))
# max_float = (max_float)*(sum)
# print(np.float32(max_float))
# recip = (1.0/max_float)
# print(recip)
# print(np.float32(np.float32(max_float)*(np.float32(1.0)/np.float32(max_float))))
# print((1.0/max_float))
# print("double")
# max_float = 2.0**(1023)
# sum = 1.0
# for i in range(52):
#     sum = sum + ((1.0/2.0)**float(i+1))
# max_float = (max_float)*(sum)
# print((max_float))
# print((max_float)*((1.0)/(max_float)))
# print((1.0/max_float))
# print("bfloat")
# max_float = 2.0**(127)
# sum = 1.0
# for i in range(7):
#     sum = sum + ((1.0/2.0)**float(i+1))
# max_float = (max_float)*(sum)
# print(tf.cast(max_float, dtype=tf.bfloat16))
# print(tf.cast(max_float, dtype=tf.bfloat16)*(tf.cast(1.0, dtype=tf.bfloat16)/tf.cast(max_float, dtype=tf.bfloat16)))
# print((tf.cast(1.0, dtype=tf.bfloat16)/tf.cast(max_float, dtype=tf.bfloat16)))
# print("half")
# max_float = 2.0**(15)
# sum = 1.0
# for i in range(10):
#     sum = sum + ((1.0/2.0)**float(i+1))
# max_float = (max_float)*(sum)
# print(np.float16(max_float))
# print(np.float16(np.float32(max_float)*(np.float16(1.0)/np.float16(max_float))))
# print((1.0/max_float))
# print(1.0 - 2**(-24))






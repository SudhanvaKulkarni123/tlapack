import numpy as np
import os
import sys
import matplotlib.pyplot as plt

matrices = os.listdir("./mat")

to_plot = []
for matrix in matrices :
    f = open("./mat/" + matrix, 'r')
    f_prime = open("./fmat/" + matrix, 'r')
    n = int(f.readline())
    f_prime.readline()
    a = np.zeros((n,n))
    b = np.zeros((n,n))
    e = np.zeros((n,n))
    for i in range(n) :         ##rows of matrix
        entries = f.readline().split(",")     ##entry of each row
        other_entries = f_prime.readline().split(",")
        for j in range(len(entries) - 1) :
            a[i,j] = float(entries[j])
            b[i,j] = float(other_entries[j])
            e[i,j] = a[i,j] - b[i,j]
    cond = np.linalg.cond(a)
    
    z = np.linalg.norm(e,2.0)
    if cond != float('inf') :
        to_plot = to_plot + [z]
    csv = open("./cond.csv", 'a')
    expected = matrix.__str__().replace(".txt","")
    csv.write(expected + "," + cond.__str__() + "\n")


# gamma = np.linspace(50,100,1000)
# res = np.linspace(0,100,1000)
# for i in range(len(gamma)):
#     res[i] = abs(gamma[i]*(2**int(np.log2(51500/gamma[i]))))
print("mean")
print(np.mean(to_plot).__str__())
print("std dev ")
print(np.sqrt(np.var(to_plot)))
print("min")
print(np.min(to_plot))
print("max")
print(np.max(to_plot))
plt.hist(to_plot, bins = 40,color='skyblue', edgecolor='black')
plt.title("condition number")
plt.show()
# plt.plot(gamma, res)
# plt.show()


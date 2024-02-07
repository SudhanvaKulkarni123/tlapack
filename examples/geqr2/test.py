import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
from scipy import stats
matrices = os.listdir("./examples/geqr2/mat")
to_plot = []
for matrix in matrices :
    f = open("./examples/geqr2/mat/" + matrix, 'r')
    n = int(f.readline())
    a = np.zeros((n,n))
    for i in range(n) :         ##rows of matrix
        entries = f.readline().split(",")     ##entry of each row
        for j in range(len(entries) - 1) :
            a[i,j] = float(entries[j])
    cond = np.linalg.cond(a)
    if cond < 20000:
        to_plot = to_plot + [cond]
    csv = open("./examples/geqr2/cond.csv", 'a')
    expected = matrix.__str__().replace(".txt","")
    csv.write(expected + "," + cond.__str__() + "\n")

print(np.mean(to_plot).__str__())
print(np.sqrt(np.var(to_plot)))
plt.hist(to_plot, bins = 40,color='skyblue', edgecolor='black')
plt.show()


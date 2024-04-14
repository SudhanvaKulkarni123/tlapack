import matplotlib.pyplot as plt
import pandas as pd


e5m2 = pd.read_csv("~/Documents/tlapack/examples/lu/e5m2_error_f_cond.csv",header=None)


plt.yscale('log')
plt.plot(e5m2[0],e5m2[1],label="r vs iter")
plt.xlabel("iter")
plt.ylabel("r (inf Norm)")
plt.legend()
plt.grid()
plt.show()

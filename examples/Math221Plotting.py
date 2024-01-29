#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 08:53:41 2023

@author: dlioce
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

e4m3fn = pd.read_csv("~/Documents/tlapack/examples/e4m3_error_d_cond.csv",header=None)
n_e4m3fn = e4m3fn[0]
err1_e4m3fn = e4m3fn[1]
err2_e4m3fn = e4m3fn[2]
err3_e4m3fn = e4m3fn[3]

e5m2= pd.read_csv("~/Documents/tlapack/examples/e5m2_error_d_cond.csv",header=None)
n_e5m2 = e5m2[0]
err1_e5m2 = e5m2[1]
err2_e5m2 = e5m2[2]
err3_e5m2 = e5m2[3]





plt.yscale('log')
plt.scatter(n_e4m3fn,err1_e4m3fn,label="e4m3 : ||QR - A||/||A||")
plt.scatter(n_e5m2,err1_e5m2,label="e5m2 : ||QR - A||/||A||")
plt.title("Error vs. condition number for Various 8-bit Data Types: \n QR Decomposition w/ Hybrid Rounding")
plt.xlabel("condition number")
plt.ylabel("Relative Error (inf Norm)")
plt.legend()
plt.grid()
plt.show()

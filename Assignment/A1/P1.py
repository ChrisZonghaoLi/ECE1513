"""
Created on Mon Jan 13 19:50:23 2020

This is the script for assignment 1

@author: Zonghao Li
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# input data
N = 7
X = np.array([1,2,3,4,5,6,7])
Y = np.array([6,4,2,1,3,6,10])
# shape the vector
Y.shape = (1,7)
X.shape = (1,7)

# draw the X Y scatter plot
plt.plot(X, Y, 'o', color='black')

# w's numerator
w_num = np.sum(np.matmul(np.transpose(X),Y)) - N * np.sum(X*Y)                 
# w's denominator
w_den = (np.sum(X))^2 - N * np.sum(np.square(X))                               
w = float(w_num) / float(w_den)                                                # w = 0.122

# b's numerator 
b_num = (np.sum(X))^2 * np.sum(X*Y) - np.sum(np.matmul(np.transpose(X),Y)) * np.sum(np.square(X))  
# b's denominator                                                  
b_den = (np.sum(X))^3 - N * np.sum(np.square(X)) * (np.sum(X))                 
b = float(b_num) / float(b_den)                                                # b = 4.56

# draw the picture of bot X Y scatter plot and linear approx.
x = np.arange(8)
y = np.arange(10)
Y_hat = w * x + b
plt.plot(X, Y, 'bo', x, Y_hat, 'r--')
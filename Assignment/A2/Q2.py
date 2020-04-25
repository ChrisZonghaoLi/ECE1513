"""
Created on Tue Jan 28 21:21:36 2020
Assignment 2 Q2
@author: Zonghao Li
"""

from sklearn.datasets import load_breast_cancer
import numpy as np

# import data
breast_cancer = load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target
train_X = X[:450]
test_X = X[450:] 
# Convert "0" to "-1"
train_Y = 2 * Y[:450] - np.ones(len(Y[:450]))
test_Y = 2 * Y[450:] - np.ones(len(Y[450:]))

# perceptron
epoch = 100
alpha = 0.01
w = np.zeros(np.shape(train_X)[1])
z = np.matmul(train_X, w)
for j in range(epoch):
    for i in range(450):
        if z[i] * train_Y[i] <= 0:
            #print z[i]
            w = w + train_X[i] * train_Y[i] * alpha
            #print w
            z[i] = np.dot(train_X[i], w) 
            #print z[i]

# normalize the data
for k in range(450):
    if z[k] > 0:
        z[k] = 1
    else:
        z[k] = -1
# calculate training accuracy
p_train = np.shape(np.nonzero(z - train_Y))[1]
precision_train = 1 - float(p_train)/450
print precision_train # 100%

z_prime = np.matmul(test_X, w)

for n in range(119):
    if z_prime[n] > 0:
        z_prime[n] = 1
    else:
        z_prime[n] = -1
#calculate testing accuracy
p = np.shape(np.nonzero(z_prime - test_Y))[1]
precision = 1 - float(p)/119
print precision # 77.3%
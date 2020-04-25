class Perceptron(object):
    """Implements a perceptron network"""
    def __init__(self, input_size, lr=1, epochs=100):
        #self.W = np.zeros(input_size+1)
        self.W = np.zeros(input_size)
        # add one for bias
        self.epochs = epochs
        self.lr = lr
    
    def activation_fn(self, x):
        #return (x >= 0).astype(np.float32)
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                #x = np.insert(X[i], 0, 1)
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x
                
                
from sklearn.datasets import load_breast_cancer
import numpy as np
breast_cancer = load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target
train_X = X[:450]
test_X = X[450:] 
train_Y = Y[:450]
test_Y = Y[450:]

if __name__ == '__main__':
    X = train_X
    d = train_Y
 
    perceptron = Perceptron(input_size=30)
    perceptron.fit(X, d)
    print(perceptron.W)

w = perceptron.W

z_prime = np.matmul(test_X, w)

for n in range(119):
    if z_prime[n] > 0:
        z_prime[n] = 1
    else:
        z_prime[n] = -1
    
p = np.shape(np.nonzero(z_prime - test_Y))[1]
precision = 1 - float(p)/119
print precision # 77.3%
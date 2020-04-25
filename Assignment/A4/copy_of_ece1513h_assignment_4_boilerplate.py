# -*- coding: utf-8 -*-
"""Copy of ECE1513H - Assignment 4 boilerplate

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lmsYRDtum0ot25W11w8UI6g0SmkGJ0ei

Let's first get the imports out of the way.
"""

import array
import gzip
import itertools
import numpy
import numpy.random as npr
import os
import struct
import time
from os import path
import urllib.request
import matplotlib.pyplot as plt

import jax.numpy as np
from jax.api import jit, grad
from jax.config import config
from jax.scipy.special import logsumexp
from jax import random

"""The following cell contains boilerplate code to download and load MNIST data."""

_DATA = "/tmp/"

def _download(url, filename):
  """Download a url to a file in the JAX data temp directory."""
  if not path.exists(_DATA):
    os.makedirs(_DATA)
  out_file = path.join(_DATA, filename)
  if not path.isfile(out_file):
    urllib.request.urlretrieve(url, out_file)
    print("downloaded {} to {}".format(url, _DATA))


def _partial_flatten(x):
  """Flatten all but the first dimension of an ndarray."""
  return numpy.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=numpy.float32):
  """Create a one-hot encoding of x of size k."""
  return numpy.array(x[:, None] == numpy.arange(k), dtype)


def mnist_raw():
  """Download and parse the raw MNIST dataset."""
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

  def parse_labels(filename):
    with gzip.open(filename, "rb") as fh:
      _ = struct.unpack(">II", fh.read(8))
      return numpy.array(array.array("B", fh.read()), dtype=numpy.uint8)

  def parse_images(filename):
    with gzip.open(filename, "rb") as fh:
      _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
      return numpy.array(array.array("B", fh.read()),
                      dtype=numpy.uint8).reshape(num_data, rows, cols)

  for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
    _download(base_url + filename, filename)

  train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
  train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
  test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
  test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

  return train_images, train_labels, test_images, test_labels


#def mnist(create_outliers=False):
def mnist(create_outliers=True):
  """Download, parse and process MNIST data to unit scale and one-hot labels."""
  train_images, train_labels, test_images, test_labels = mnist_raw()

  train_images = _partial_flatten(train_images) / numpy.float32(255.)
  test_images = _partial_flatten(test_images) / numpy.float32(255.)
  train_labels = _one_hot(train_labels, 10)
  test_labels = _one_hot(test_labels, 10)

  if create_outliers:
    mum_outliers = 30000
    perm = numpy.random.RandomState(0).permutation(mum_outliers)
    train_images[:mum_outliers] = train_images[:mum_outliers][perm]

  return train_images, train_labels, test_images, test_labels

def shape_as_image(images, labels, dummy_dim=False):
  target_shape = (-1, 1, 28, 28, 1) if dummy_dim else (-1, 28, 28, 1)
  return np.reshape(images, target_shape), labels

#train_images, train_labels, test_images, test_labels = mnist(create_outliers=False)
train_images, train_labels, test_images, test_labels = mnist(create_outliers=True)
num_train = train_images.shape[0]

"""# **Problem 1**

This function computes the output of a fully-connected neural network (i.e., multilayer perceptron) by iterating over all of its layers and:

1. taking the `activations` of the previous layer (or the input itself for the first hidden layer) to compute the `outputs` of a linear classifier. Recall the lectures: `outputs` is what we wrote $z=w\cdot x + b$ where $x$ is the input to the linear classifier. 
2. applying a non-linear activation. Here we will use $tanh$.

Complete the following cell to compute `outputs` and `activations`.
"""

def predict(params, inputs):
  activations = inputs
  for w, b in params[:-1]:
    outputs = np.dot(activations, w) + b 
    activations = np.tanh(outputs) 

  final_w, final_b = params[-1]
  logits = np.dot(activations, final_w) + final_b
  return logits - logsumexp(logits, axis=1, keepdims=True)

"""The following cell computes the loss of our model. Here we are using cross-entropy combined with a softmax but the implementation uses the `LogSumExp` trick for numerical stability. This is why our previous function `predict` returns the logits to which we substract the `logsumexp` of logits. We discussed this in class but you can read more about it [here](https://blog.feedly.com/tricks-of-the-trade-logsumexp/).

Complete the return line. Recall that the loss is defined as :
$$ l(X, Y) = -\frac{1}{n} \sum_{i\in 1..n}  \sum_{j\in 1.. K}y_j^{(i)} \log(f_j(x^{(i)})) = -\frac{1}{n} \sum_{i\in 1..n}  \sum_{j\in 1.. K}y_j^{(i)} \log\left(\frac{z_j^{(i)}}{\sum_{k\in 1..K}z_k^{(i)}}\right) $$
where $X$ is a matrix containing a batch of $n$ training inputs, and $Y$ a matrix containing a batch of one-hot encoded labels defined over $K$ labels. Here $z_j^{(i)}$ is the logits (i.e., input to the softmax) of the model on the example $i$ of our batch of training examples $X$.
"""

def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  ce = -np.mean(np.sum(targets*preds, axis=1))
  print(ce)
  return ce

"""The following cell defines the accuracy of our model and how to initialize its parameters."""

def accuracy(params, batch):
  inputs, targets = batch
  target_class = np.argmax(targets, axis=1)
  predicted_class = np.argmax(predict(params, inputs), axis=1)
  return np.mean(predicted_class == target_class)

def init_random_params(layer_sizes, rng=npr.RandomState(0)):
  scale = 0.1
  return [(scale * rng.randn(m, n), scale * rng.randn(n))
          for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

"""The following line defines our architecture with the number of neurons contained in each fully-connected layer (the first layer has 784 neurons because MNIST images are 28*28=784 pixels and the last layer has 10 neurons because MNIST has 10 classes)"""

layer_sizes = [784, 1024, 1024, 10]
# [784, 1024, 128, 10]

"""The following cell creates a Python generator for our dataset. It outputs one batch of $n$ training examples at a time."""

batch_size = 32
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)
def data_stream():
  rng = npr.RandomState(0)
  while True:
    perm = rng.permutation(num_train)
    for i in range(num_batches):
      batch_idx = perm[i * batch_size:(i + 1) * batch_size]
      yield train_images[batch_idx], train_labels[batch_idx]
batches = data_stream()

"""We are now ready to define our optimizer. Here we use mini-batch stochastic gradient descent. Complete `<w UPDATE RULE>` and `<b UPDATE RULE>` using the update rule we saw in class. Recall that `dw` is the partial derivative of the `loss` with respect to `w` and `learning_rate` is the learning rate of gradient descent."""

learning_rate = 0.1
# 0.01: slow
# 1: oscillate but converge
# 2: oscillate but non-converge

@jit
def update(params, batch):
  grads = grad(loss)(params, batch)
  return [(w - learning_rate * dw, b - learning_rate * db)
          for (w, b), (dw, db) in zip(params, grads)]

"""This is now the proper training loop for our fully-connected neural network."""

num_epochs = 50
#num_epochs = 10
params = init_random_params(layer_sizes)
for epoch in range(num_epochs):
  start_time = time.time()
  for _ in range(num_batches):
    params = update(params, next(batches))
  epoch_time = time.time() - start_time

  train_acc = accuracy(params, (train_images, train_labels))
  test_acc = accuracy(params, (test_images, test_labels))

  print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
  print("Training set accuracy {}".format(train_acc))
  print("Test set accuracy {}".format(test_acc))

#plt.figure()
#standard = plt.plot([0.9398333430290222, 0.9593333601951599, 0.9687666893005371, 0.9751499891281128, 
#     0.9800000190734863, 0.984333336353302, 0.9868333339691162, 0.9889833331108093,
#     0.9921333193778992, 0.9933500289916992])

#plt.figure()
#slow = plt.plot([0.8699666857719421, 0.8939833641052246, 0.9076666831970215, 0.9162333607673645, 
#     0.9237666726112366, 0.9287999868392944, 0.9334666728973389, 0.9371833205223083,
#     0.9408666491508484, 0.9433333277702332])

#plt.figure()
#oscillation = plt.plot([0.9065666794776917, 0.9262833595275879, 0.9513500332832336, 
#                        0.9539833664894104, 0.9590166807174683, 0.9721166491508484,
#                        0.970550000667572, 0.9745666980743408, 0.9773666858673096, 0.984083354473114])

#plt.figure()
#oscillation_non = plt.plot([0.09751667082309723, 0.09035000205039978, 0.11236666887998581, 
#                        0.10441666841506958, 0.09930000454187393, 0.09863333404064178,
#                        0.11236666887998581, 0.09751667082309723, 0.10441666841506958, 0.09863333404064178])

"""# **Problem 2**

Before we get started, we need to import two small libraries that contain boilerplate code for common neural network layer types and for optimizers like mini-batch SGD.
"""

from jax.experimental import optimizers
from jax.experimental import stax

"""Here is a fully-connected neural network architecture, like the one of Problem 1, but this time defined with `stax`"""

init_random_params, predict = stax.serial(

    stax.Conv(32, (3, 3), strides=(1, 1)),
    stax.Relu,
    stax.MaxPool((2, 2), strides=(2, 2)),
   
    stax.Conv(64, (3, 3), strides=(1, 1)),
    stax.Relu,
    stax.Conv(64, (3, 3), strides=(1, 1)),
    stax.Relu,
    stax.MaxPool((2, 2), strides=(2, 2)),

    stax.Flatten,
    stax.Dense(100),
    stax.Relu,

    stax.Dense(10),
)

"""We redefine the cross-entropy loss for this model. As done in Problem 1, complete the return line below (it's identical)."""

def loss(params, batch):
  inputs, targets = batch
  logits = predict(params, inputs)
  preds  = stax.logsoftmax(logits)
  return -np.mean(np.sum(targets*preds, axis=1))

"""Next, we define the mini-batch SGD optimizer, this time with the optimizers library in JAX."""

learning_rate = 0.01
opt_init, opt_update, get_params = optimizers.momentum(learning_rate, 0.9)

@jit
def update(_, i, opt_state, batch):
  params = get_params(opt_state)
  return opt_update(i, grad(loss)(params, batch), opt_state)

"""The next cell contains our training loop, very similar to Problem 1."""

num_epochs = 10

key = random.PRNGKey(123)
_, init_params = init_random_params(key, (-1, 28, 28, 1))
opt_state = opt_init(init_params)
itercount = itertools.count()

for epoch in range(1, num_epochs + 1):
  for _ in range(num_batches):
    opt_state = update(key, next(itercount), opt_state, shape_as_image(*next(batches)))

  params = get_params(opt_state)
  test_acc = accuracy(params, shape_as_image(test_images, test_labels))
  test_loss = loss(params, shape_as_image(test_images, test_labels))
  print('Test set loss, accuracy (%): ({:.2f}, {:.2f})'.format(test_loss, 100 * test_acc))
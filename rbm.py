# Using Contrastive Divergence for unsupervised feature learning
import numpy as np


data = np.array([
    [1,1,0,0],
    [1,0,1,0],
    [0,1,1,1],
    [0,0,1,1]
])

n_visible = data.shape[1]

n_hidden = 2

learning_rate = 0.1

epochs = 1000
weights = np.random.randn(n_visible, n_hidden) * 0.1

visible_bias = np.zeros(n_visible)

hidden_bias = np.zeros(n_hidden)
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
def hidden_probability(v):
  return sigmoid(np.dot(h, weights.T) + visible_bias)

for epoch in range(epochs):
 visible = data
 hidden_prob = hidden_probability(visible)
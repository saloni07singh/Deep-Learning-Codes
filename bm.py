import numpy as np

data = np.array([
    [1,1,0,0],
    [1,0,0,1],
    [0,0,1,1],
    [0,1,1,0]
])

np.random.seed(1)
weights = np.random.rand(4,4)
weights

np.fill_diagonal(weights,0)

#calculate probablity of each pattern
def probablity(x):
  energy = -0.5*np.dot(x, np.dot(weights, x))
  return np.exp(-energy)/(1+np.exp(energy))
#display probablyiy of training patters
for x in data:
  print(x, "probablity:", round(probablity(x), 3))
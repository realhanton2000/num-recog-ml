import numpy as np
from sigmoid import sigmoid

def lrCostFunction(theta, X, y, lambda_t):
  m = len(y)

  theta = np.reshape(theta, (theta.size, 1))

  h = sigmoid( np.matmul(X, theta) )
  
  J = 1 / m * ( np.matmul(np.transpose(np.log(h)), -y) - np.matmul(np.transpose(np.log(1-h)), (1-y)) ) \
       + lambda_t / (2 * m)  * np.matmul(np.transpose(theta[1:]), theta[1:])

  grad = 1 / m * np.matmul(np.transpose(X), (h - y))

  grad[1:] = grad[1:] + lambda_t / m * theta[1:]

  return J, grad

def costf(theta, X, y, lambda_t):
  J, grad = lrCostFunction(theta, X, y, lambda_t)
  return J

def gradf(theta, X, y, lambda_t):
  J, grad = lrCostFunction(theta, X, y, lambda_t)
  grad = grad.flatten()
  return grad

import numpy as np
from sigmoid import sigmoid
from scipy import optimize

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

def lrTrain(X, y, num_labels, lambda_):
    m, n = np.shape(X)
    all_theta = np.zeros((num_labels, n+1))
    X = np.concatenate((np.ones((m,1)), X), axis=1)

    for c in range(1, num_labels + 1):
        initial_theta = all_theta[c - 1]
        xopt, fopt, iter, funcalls, warnflag = optimize.fmin_cg(costf, x0=initial_theta, fprime=gradf, \
            maxiter=100, \
            full_output=True, disp=True, args=(X, (y == c)*1, lambda_))
        all_theta[c - 1] = xopt

    return all_theta

def predictToExpectation(all_theta, X):
    m, n = np.shape(X)
    num_labels, number_features = np.shape(all_theta)
    p = np.zeros((m, 1))
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    p = np.argmax(np.matmul(X, np.transpose(all_theta)), axis=1) + 1
    return p
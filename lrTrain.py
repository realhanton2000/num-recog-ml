import numpy as np
from scipy import optimize
from lrCostFunction import costf
from lrCostFunction import gradf

def lrTrain(X, y, num_labels, lambda_):
    m, n = np.shape(X)
    all_theta = np.zeros((num_labels, n+1))
    X = np.concatenate((np.ones((m,1)), X), axis=1)

    for c in range(1, num_labels + 1):
        initial_theta = all_theta[c - 1]
        xopt, fopt, iter, funcalls, warnflag = optimize.fmin_cg(costf, x0=initial_theta, fprime=gradf, \
            maxiter=50, \
            full_output=True, disp=True, args=(X, (y == c)*1, lambda_))
        all_theta[c - 1] = xopt

    return all_theta

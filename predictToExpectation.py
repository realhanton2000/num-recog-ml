import numpy as np

def predictToExpectation(all_theta, X):
    m, n = np.shape(X)
    num_labels, number_features = np.shape(all_theta)
    p = np.zeros((m, 1))
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    p = np.argmax(np.matmul(X, np.transpose(all_theta)), axis=1) + 1
    return p

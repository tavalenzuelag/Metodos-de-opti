import numpy as np


def prediction(X, alpha, beta):
    "return vector of predictions"
    return np.power(X, alpha.T).dot(beta)


def dbeta(X, y, alpha, beta):
    "return beta gradient vector"
    error = prediction(X, alpha, beta) - y
    return np.power(X, alpha.T).T.dot(error)


def dalpha(X, y, alpha, beta):
    "return alpha gradient vector"
    error = prediction(X, alpha, beta) - y
    return (np.power(X, alpha.T) * np.log(X)).T.dot(error) * beta
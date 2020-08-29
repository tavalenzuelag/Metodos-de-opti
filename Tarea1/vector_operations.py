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
    x_power_alpha = np.power(X, alpha.T)
    return (np.power(X, alpha.T) * np.log(X)).T.dot(error) * beta


def calcular_gradiente(X, y, alpha, beta):
    return np.concatenate((dbeta(X, y, alpha, beta), dalpha(X, y, alpha, beta)), axis=0)
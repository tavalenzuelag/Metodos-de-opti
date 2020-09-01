import numpy as np
from parametros import * 

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

def dbj_bk(X, alpha, j, k):
    "Retorna la derivada de segundo orden con respecto a b_j i b_k"
    col_j = X[:, j].reshape((X.shape[0], 1))
    col_k = X[:, k].reshape((X.shape[0], 1))
    col_j_power = np.power(col_j, alpha[j])
    col_k_power = np.power(col_k, alpha[k])

    return (col_j_power * col_k_power).sum()


def dbj_bj(X, alpha, j):
    "Retorna la derivada de segundo orden con respecto a b_j" 
    col_j = X[:, j].reshape((X.shape[0], 1))
    col_j_power = np.power(X, 2 * alpha[j])


    return col_j_power.sum()


def dbj_aj(X, alpha, beta, j):
    "Retorna la derivada de segundo orden con respecto a b_j y a_j"
    col_j = X[:, j].reshape((X.shape[0], 1))
    col_j_power = np.power(col_j, alpha[j])
    col_j_log = np.log(col_j)
    error = prediction(X, alpha, beta) - y

    return (col_j_power * col_j_log * (beta[j] + error)).sum()


def dbj_ak(X, alpha, beta, j, k):
    "Retorna la derivada de segundo orden con respecto a b_j y a_k"
    col_j = X[:, j].reshape((X.shape[0], 1))
    col_k = X[:, k].reshape((X.shape[0], 1))
    col_j_power = np.power(col_j, alpha[j])
    col_k_power = np.power(col_k, alpha[k])
    col_k_log = np.log(col_k)

    return (col_j_power * col_k_power * col_k_log * beta[k]).sum()


def daj_aj():
    "Retorna la derivada de segundo orden con respecto a a_j"
    pass


def daj_ak(X, alpha, beta, j, k):
    "retorna la derivada de segundo orden con respecto a a_j y a_k"
    col_j = X[:, j].reshape((X.shape[0], 1))
    col_k = X[:, k].reshape((X.shape[0], 1))
    col_j_power = np.power(col_j, alpha[j])
    col_k_power = np.power(col_k, alpha[k])
    col_j_log = np.log(col_j)
    col_k_log = np.log(col_k)

    return  (col_j_power * col_k_power * col_j_log * col_k_log * beta[j] * beta[k]).sum()


def get_hessiano(X, y, alpha, beta):
    pass




 

df = load_data('data.xlsx')
df = drop_row_and_column(0, 0, df)
X, y = get_design_matrix_and_co2_vector(df)

z = np.ones((10,1))

alpha = z[5:]
beta = z[:5] 

#print(np.empty((2,2)))







    
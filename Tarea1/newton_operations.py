import numpy as np
from gradient_operations import prediction


def d_bj_bj(j, X, alpha):
    "Retorna la derivada de segundo orden con respecto a b_j" 

    col_j = X[:, j].reshape((X.shape[0], 1))
    col_j_power = np.power(col_j, 2 * alpha[j])
    
    return col_j_power.sum()


def d_aj_aj(j, X, y, alpha, beta):
    "Retorna la derivada de segundo orden con respecto a a_j"

    col_j = X[:, j].reshape((X.shape[0], 1))
    col_j_power = np.power(col_j, alpha[j])
    col_j_log_power = np.power(np.log(col_j), 2)
    error = prediction(X, alpha, beta) - y
    
    return (beta[j] * col_j_power * col_j_log_power * (beta[j] * col_j_power +  error)).sum()

def d_bj_bk(j, k, X, alpha):
    "Retorna la derivada de segundo orden con respecto a b_j y b_k"

    col_j = X[:, j].reshape((X.shape[0], 1))
    col_k = X[:, k].reshape((X.shape[0], 1))
    col_j_power = np.power(col_j, alpha[j])
    col_k_power = np.power(col_k, alpha[k])

    return (col_j_power * col_k_power).sum()


def d_aj_ak(j, k, X, alpha, beta):
    "retorna la derivada de segundo orden con respecto a a_j y a_k"

    col_j = X[:, j].reshape((X.shape[0], 1))
    col_k = X[:, k].reshape((X.shape[0], 1))
    col_j_power = np.power(col_j, alpha[j])
    col_k_power = np.power(col_k, alpha[k])
    col_j_log = np.log(col_j)
    col_k_log = np.log(col_k)
    
    return  (beta[j] * beta[k] * col_j_power * col_k_power * col_j_log * col_k_log).sum()


def d_bj_ak(j, k, X, alpha, beta):
    "Retorna la derivada de segundo orden con respecto a b_j y a_k"

    col_j = X[:, j].reshape((X.shape[0], 1))
    col_k = X[:, k].reshape((X.shape[0], 1))
    col_j_power = np.power(col_j, alpha[j])
    col_k_power = np.power(col_k, alpha[k])
    col_k_log = np.log(col_k)

    return  (beta[k] * col_j_power * col_k_power * col_k_log).sum()



def d_bj_aj(j, X, y, alpha, beta):
    "Retorna la derivada de segundo orden con respecto a b_j y a_j"

    col_j = X[:, j].reshape((X.shape[0], 1))
    col_j_power = np.power(col_j, alpha[j])
    col_j_log = np.log(col_j)
    error = prediction(X, alpha, beta) - y

    return (col_j_power * col_j_log * (error + beta[j] * (1 + error * col_j_log))).sum()

    

def diagonal_derivate(idx, X, y, alpha, beta):
    "Retorna derivada de segundo orden para la diagonal en el índice idx"

    if idx <= 4:
        # derivada con respecto a beta
        return  d_bj_bj(idx, X, alpha) 
    
    # Sino, derivada con respecto a alpha
    return d_aj_aj(idx % 5, X, y, alpha, beta)


def non_diagonal_derivate(j, k, X, y, alpha, beta):
    "Retorna derivada de segundo orden en índices no diagonales"

    if j <= 4:
        # En este caso k también será <= 4
        return d_bj_bk(j, k, X, alpha)
    
    
    elif j > 4 and k <= 4:
        # k corresponderá a db y j a da
        
        if j % 5 == k:
            # La derivada está en el mismo índice para alpha y beta
            return d_bj_aj(k, X, y, alpha, beta)

        else:
            # Los índices son distintos
            return d_bj_ak(k, j % 5, X, alpha, beta)

    
    elif j > 4 and k > 4:
        # ambos corresponden a da
        return d_aj_ak(j % 5, k % 5, X, alpha, beta)



def get_hessian(X, y, alpha, beta):
    "Obtenemos hessiano aprovechándonos de la simetría de la matriz"

    hessian = np.zeros((10,10), dtype = float)
    diagonal_index = [tupla for tupla in zip(range(10), range(10))]

    for tupla in diagonal_index:
        # Tupla es un elemento de la diagonal
        j, k = tupla
        hessian[j, k] = diagonal_derivate(j, X, y, alpha, beta)

        # Efectuamos barrido para elementos a la izquierda de la diagonal
        while k > 0:
            k -= 1
            hessian[j, k] = non_diagonal_derivate(j, k, X, y, alpha, beta)
            hessian[k, j] = hessian[j, k]

    return hessian




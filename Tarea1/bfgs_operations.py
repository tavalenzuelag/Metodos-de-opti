from vector_operations import *
import numpy as np
from parametros import * 

def y_k(X, y, alpha_k, beta_k, alpha_k1, beta_k1):
    # Función que retorna el y de paso k
    grad_k = np.concatenate((dbeta(X, y, alpha_k, beta_k), dalpha(X, y, alpha_k, beta_k)), axis=0)
    grad_k1 = np.concatenate((dbeta(X, y, alpha_k1, beta_k1), dalpha(X, y, alpha_k1, beta_k1)), axis=0)

    return grad_k1 - grad_k


def p_k(y_k, s_k):
    # Función que retorna el escalar p (rho) en el paso k
    return 1/(y_k.T.dot(s_k))


def H_k1(H_k, p_k, s_k, y_k):
    # Función que retorna la matriz H para el paso k+1

    # Matriz que multiplica por la izquierda
    left_dot_argument = 1 - p_k * np.dot(y_k, s_k.T)


    u = H_k.dot(s_k)

    # Matriz que multiplica por la derecha
    right_dot_argument = H_k - p_k * u.dot(y_k.T)  


    # Matriz que es sumada al producto matricial
    add_argument = p_k * np.dot(s_k, s_k.T)

    return left_dot_argument.dot(right_dot_argument) + add_argument





    

df = load_data('data.xlsx')
df = drop_row_and_column(0, 0, df)
X, y = get_design_matrix_and_co2_vector(df)

z = np.ones((10,1))

alpha = z[5:]
beta = z[:5] 

alpha2 = alpha + 1
beta2 = alpha + 2
H = np.ones((10, 10))

print(H_k1(H, 5, z/2, z+1).shape)
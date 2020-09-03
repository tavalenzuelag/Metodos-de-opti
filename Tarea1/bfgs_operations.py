from vector_operations import *
import numpy as np


def p_k(y_k, s_k):
    # Función que retorna el escalar p (rho) en el paso k
    return 1/(y_k.T.dot(s_k))


def H_k1(H, p_k, s_k, y_k):
    # Función que retorna la matriz H para el paso k+1

    # Matriz que multiplica por la izquierda
    left_dot_argument = np.identity(10, dtype = float) - p_k * np.dot(y_k, s_k.T)


    u = H.dot(s_k)

    # Matriz que multiplica por la derecha
    right_dot_argument = H - p_k * u.dot(y_k.T)  


    # Matriz que es sumada al producto matricial
    add_argument = p_k * np.dot(s_k, s_k.T)

    return left_dot_argument.dot(right_dot_argument) + add_argument








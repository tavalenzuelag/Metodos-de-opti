from bfgs_operations import *
from gradient_operations import prediction, dalpha, dbeta
from parametros import *
import numpy as np
import time
import scipy.optimize



def timer(funcion):
    """
    Se crea un decorador (googlear) del tipo timer para testear el tiempo
    de ejecucion del programa
    """
    def inner(*args, **kwargs):

        inicio = time.time()
        resultado = funcion(*args, **kwargs)
        final = round(time.time() - inicio, 3)
        print("\nTiempo de ejecucion total: {}[s]".format(final))

        return resultado
    return inner


def subrutina(X, y, z, z_k1, gradiente, H):
    """
    Esta funcion va creando el paso de cada iteracion. Ocupando la teoría
    estudiada. Retorna el valor de la funcion, su gradiente y su hessiano segun
    la iteracion estudiada.
    """

    # Calculamos error y la función objetivo
    alpha = z[5:].reshape((5,1))
    beta = z[:5].reshape((5,1))
    error = prediction(X, alpha, beta) - y
    funcion_objetivo = 0.5 * (error.T.dot(error))


    # gradiente en el paso k + 1
    alpha_k1 = z_k1[5:].reshape((5,1))
    beta_k1 = z_k1[:5].reshape((5,1))
    gradiente_k1 = np.concatenate((dbeta(X, y, alpha_k1, beta_k1), dalpha(X, y, alpha_k1, beta_k1)), axis=0)
    


    # Parámetros para calcular la matriz H del paso k+1
    yk = gradiente_k1 - gradiente
    sk = z_k1 - z
    pk = p_k(yk, sk)
    H = H_k1(H, pk, sk, yk)

    return funcion_objetivo, gradiente_k1, H
    
    

def funcion_enunciado(lambda_, X, y, z, direccion_descenso):
    """
    Funcion original evaluada en: x + lambda*direccion_descenso
    """

    # Se actualiza el valor de z
    z = z + lambda_ * direccion_descenso
    
    alpha = z[5:].reshape((5,1))
    beta = z[:5].reshape((5,1))

    error = prediction(X, alpha, beta) - y

    return 0.5 * (error.T.dot(error))


@timer
def bfgs(X, y, z0, H0, epsilon, iteracion_maxima):
    """
    Esta funcion es una aplicacion del metodo bfgs.
    Su entrada posee:
    - X : matriz de diseño
    - z0 : punto inicial de prueba
    - H0 : Matriz H simétrica inicial
    - epsilon : error/ tolerancia deseada
    - iteracion_maxima : numero maximo de iteraciones
    """


    # 1º paso del algoritmo: Se definen los parametros iniciales
    iteracion = 0
    stop = False
    z = z0
    H = H0
    alpha = z[5:].reshape((5,1))
    beta = z[:5].reshape((5,1))
    gradiente = np.concatenate((dbeta(X, y, alpha, beta), dalpha(X, y, alpha, beta)), axis=0)
    error_list = []

    # Se prepara el output del codigo para en cada iteracion
    print("\n\n*********       METODO DE NEWTON      **********\n")
    print("ITERACION     VALOR OBJ      NORMA        LAMBDA")

    # Se inicia el ciclo para las iteraciones maximas seteadas por el usuario
    while (stop == False) and (iteracion <= iteracion_maxima):
        
        # 1º paso: se obtiene la direccion de descenso
        direccion_descenso = np.dot(-np.linalg.inv(H), gradiente)


        # 2º paso: Se analiza el criterio de parada
        norma = np.linalg.norm(gradiente, ord=2)

        if norma <= epsilon:
            stop = True
        
        else:
        # 3º paso: Se busca el peso (lambda) optimo
            lambda_ = scipy.optimize.fminbound(funcion_enunciado, 0, 10, args=(X, y, z, direccion_descenso))
        

        # 4º paso del algoritmo: Se busca obtiene z_{k+1} 
            z_k1 = z + lambda_ * direccion_descenso

            [valor, gradiente_k1, H] = subrutina(X, y, z, z_k1, gradiente, H)

            if iteracion > 0:
                error_list.append(valor[0])

        # Se imprimen en pantalla los resultados de interés
        retorno_en_pantalla = [iteracion, valor, norma, lambda_]
        print("%12.6f %12.6f %12.6f %12.6f" % (retorno_en_pantalla[0],retorno_en_pantalla[1][0][0],retorno_en_pantalla[2],retorno_en_pantalla[3]))

        
        # Actualizamos los valores 
        gradiente = gradiente_k1
        z = z_k1

        iteracion += 1

    return retorno_en_pantalla, error_list

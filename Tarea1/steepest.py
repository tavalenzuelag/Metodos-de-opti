__author__ = "Moises Saavedra Caceres"
__email__ = "mmsaavedra1@ing.puc.cl"


# Modulos nativos de python
import numpy as np
import time
import scipy.optimize
from numpy import linalg as la
from parametros import *
from gradient_operations import *


# Se crea un decorador (googlear) del tipo timer para testear el tiempo
# de ejecucion del programa
def timer(funcion):
    def inner(*args, **kwargs):

        inicio = time.time()
        resultado = funcion(*args, **kwargs)
        final = round(time.time() - inicio, 3)
        print("\nTiempo de ejecucion total: {}[s]".format(final))

        return resultado
    return inner

# Se define la evaluacion de valores dentro de cada iteracion
# de la rutina del gradiente
def subrutina(X, y, z):
    """
    Esta funcion va creando el paso de cada iteracion. Ocupando la teoría
    estudiada. Retorna el valor de la funcion, su gradiente segun
    la iteracion estudiada.
    """
    
    alpha = z[5:].reshape((5,1))
    beta = z[:5].reshape((5,1))
    
    
    # Funcion a optimizar, gradiente y hessiano
    error = prediction(X, alpha, beta) - y
    funcion_objetivo =  0.5 * (error.T.dot(error))
    gradiente = np.concatenate((dbeta(X, y, alpha, beta), dalpha(X, y, alpha, beta)), axis=0)

    return funcion_objetivo, gradiente


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
def gradiente(X, y, z0, epsilon, iteracion_maxima):
    """
    Esta funcion es una aplicacion del metodo del gradiente, la que
    va a ir devolviendo valor objetivo, gradiente actual.

    Su entrada posee:
    - Q : matriz cuadrada que constituye la funcion definida
    - c : vector asociado que constituye la funcion definida
    - z0 : punto inicial de prueba
    - epsilon : error/ tolerancia deseada
    - iteracion_maxima : numero maximo de iteraciones

    Su retorno (salida) es:
    - valor : valor de la funcion evaluada en x en la iteracion actual
    - x : solucion en la que se alcanza el valor objetivo
    - R : matriz con la informacion de cada iteracion. Es una fila por iteracion
          y esta constituida por:
          - Numero de iteracion
          - valor
          - norma del gradiente
          - paso (lambda)
    """
    # 1º paso del algoritmo: Se definen los parametros iniciales
    iteracion = 0
    stop = False
    z = z0
    error_list = []

    # Se prepara el output del codigo para en cada iteracion
    # entregar la informacion correspondiente
    print("\n\n*********      METODO DE GRADIENTE      **********\n")
    print("ITERACION     VALOR OBJ      NORMA        LAMBDA")

    # Se inicia el ciclo para las iteraciones maximas seteadas por el usuario
    while (stop == False) and (iteracion <= iteracion_maxima):

        # 2º paso del algoritmo: Se obtiene la informacion para determinar
        # el valor de la direccion de descenso
        [valor, gradiente] = subrutina(X, y, z)
        if iteracion > 0:
            error_list.append(valor[0])
        
        direccion_descenso = -1 * gradiente


        # 3º paso del algoritmo: Se analiza el criterio de parada
        norma = np.linalg.norm(gradiente, 2)

        if norma <= epsilon:
            stop = True

        else:
        # 4º paso del algoritmo: Se busca el peso (lambda) optimo

            # Se resuelve el subproblema de lambda
            lambda_ = scipy.optimize.fminbound(funcion_enunciado, 0, 10, args=(X, y, z, direccion_descenso))

        # La rutina del gradiente muestra en pantalla para cada iteracion:
        # nº de iteracion, valor de la funcion evaluada en el x de la iteracion,
        # la norma del gradiente y el valor de peso de lambda
        retorno_en_pantalla = [iteracion, valor, norma, lambda_]

        print("%12.6f %12.6f %12.6f %12.6f" % (retorno_en_pantalla[0],retorno_en_pantalla[1][0][0],retorno_en_pantalla[2],retorno_en_pantalla[3]))


        # 5º paso del algoritmo: Se actualiza el valor de x para la siguiente
        # iteracion del algoritmo
        z = z + lambda_ * direccion_descenso
        iteracion += 1

    return retorno_en_pantalla, error_list





__author__ = "Moises Saavedra Caceres"
__email__ = "mmsaavedra1@ing.puc.cl"


# Modulos nativos de python
import numpy as np
import time
import scipy.optimize

# Modulo creado por nosotros (parametros.py)
from parametros import *
from newton_operations import get_hessian
from gradient_operations import prediction, dbeta, dalpha


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

def subrutina(X, y, z):
    """
    Esta funcion va creando el paso de cada iteracion. Ocupando la teoría
    estudiada. Retorna el valor de la funcion, su gradiente y su hessiano segun
    la iteracion estudiada.
    """

    alpha = z[5:].reshape((5,1))
    beta = z[:5].reshape((5,1))
    
    
    # Funcion a optimizar, gradiente y hessiano
    error = prediction(X, alpha, beta) - y
    funcion_objetivo =  0.5 * (error.T.dot(error))
    gradiente = np.concatenate((dbeta(X, y, alpha, beta), dalpha(X, y, alpha, beta)), axis=0)
    hessiano = get_hessian(X, y, alpha, beta)

    return funcion_objetivo, gradiente, hessiano


def funcion_enunciado(lambda_, X, y, z, direccion_descenso):
    """
    Funcion original evaluada en: x + lambda*direccion_descenso
    """
    # Se actualiza el valor de z
    z = z + lambda_ * direccion_descenso
    
    alpha = z[5: ].reshape((5,1))
    beta = z[ :5].reshape((5,1))

    error = prediction(X, alpha, beta) - y

    return 0.5 * (error.T.dot(error))

@timer
def newton(X, y, z0, epsilon, iteracion_maxima):
    # 1º paso del algoritmo: Se definen los parametros iniciales
    iteracion = 0
    stop = False
    z = z0
    error_list = []

    # Se prepara el output del codigo para en cada iteracion
    print("\n\n*********       METODO DE NEWTON      **********\n")
    print("ITERACION     VALOR OBJ      NORMA        LAMBDA")

    # Se inicia el ciclo para las iteraciones maximas seteadas por el usuario
    while (stop == False) and (iteracion <= iteracion_maxima):

        # 2º paso del algoritmo: Se obtiene la informacion para determinar
        # el valor de la direccion de descenso
        [valor, gradiente, hessiano] = subrutina(X, y, z)
        direccion_descenso = np.dot(-np.linalg.inv(hessiano), gradiente)

        # 3º paso del algoritmo: Se analiza el criterio de parada
        norma = np.linalg.norm(gradiente, ord = 2)

        if norma <= epsilon:
            stop = True
        else:
        # 4º paso del algoritmo: Se busca el peso (lambda) optimo
            if iteracion > 1:
               error_list.append(valor[0])
            lambda_ = scipy.optimize.fminbound(funcion_enunciado, 0, 0.1, args=(X, y, z, direccion_descenso))

        # La rutina de Newton muestra en pantalla para cada iteracion:
        # nº de iteracion, valor de la funcion evaluada en el x de la iteracion,
        # la norma del gradiente y el valor de peso de lambda
        retorno_en_pantalla = [iteracion, valor, norma, lambda_]
#       Nota de J. Vera: Esta forma de "print" requiere Python 3.6 
#        print(f"{retorno_en_pantalla[0]: ^12d}{retorno_en_pantalla[1][0][0]: ^12f} {retorno_en_pantalla[2]: ^12f} {retorno_en_pantalla[3]: ^12f}")

        print("%12.6f %12.6f %12.6f %12.6f" % (retorno_en_pantalla[0],retorno_en_pantalla[1][0][0],retorno_en_pantalla[2],retorno_en_pantalla[3]))


        # 5º paso del algoritmo: Se actualiza el valor de x para la siguiente
        # iteracion del algoritmo
        z = z + lambda_ * direccion_descenso
        iteracion += 1

    return retorno_en_pantalla, error_list

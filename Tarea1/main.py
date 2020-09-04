__author__ = "Moises Saavedra Caceres"
__email__ = "mmsaavedra1@ing.puc.cl"


# Modulos creados por usuario
from newton import *
from steepest import *
from parametros import *
from bfgs import *
from newton_operations import *
from visualization import visualization


# Primero se generan datos para la funcion
# esta informacion es la que pueden cambiar a su antojo
iteracion_maxima_newton = 700
iteracion_maxima_gradiente = 800000
iteracion_maxima_bfgs = 80000
epsilon = 0.005


# ******* NO TOCAR NADA DE AQUI PARA ABAJO ********
# (si deseas probar un metodo y no ambos comenta la linea 31 o 34, respectivamente)


# Para que siempre genere los mismos datos al azar
# En base a su informacion entregada como input de aqui en adelante el programa
# se corre solo
#Q, c = generar_datos(dimension_matriz_Q)

df = load_data('data.xlsx')
df = drop_row_and_column(0, 0, df)
X, y = get_design_matrix_and_co2_vector(df)


# Se ocupa el vector de "unos" como punto de inicio
# (notar el salto que pega) de la iteracion 1 a la 2 el valor objetivo
# -- Queda a tu eleccion que vector ingresar como solucion para la iteracion 1 --
z0 = np.ones((10,1), dtype = float)
H = np.identity(10, dtype = float)


#_, error_list = newton(X, y, z0, epsilon, iteracion_maxima_newton)
#visualization(error_list, "Convergencia Método de Newton")


#_, error_list = gradiente(X, y, z0, epsilon, iteracion_maxima_gradiente)
#visualization(error_list, "Convergencia Método del gradiente")


#_, error_list = bfgs(X, y, z0, H, epsilon, iteracion_maxima_bfgs)
#visualization(error_list, "Convergencia Método Bfgs")






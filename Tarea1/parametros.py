__author__ = "Moises Saavedra Caceres"
__email__ = "mmsaavedra1@ing.puc.cl"


# Modulos nativos de python
import numpy as np
import scipy.linalg
import random
import pandas as pd

def generar_datos(dimension):
    """
    Esta funcion crea una matriz cuadrada semidefinida positiva para
    ocuparla en los programas presentados.
    """
    
    # Se crea un vector aleatorio segun la dimension entregada
#    random.seed(16)
    vector = np.random.rand(dimension, dimension)
    # Se obtienen las dimensiones del vector
    m, n = vector.shape

    # Se crea una matriz que contenga en su diagonal
    # el vector ingresado como argumento
    D = np.diag(np.diag(vector))
    B = scipy.linalg.orth(np.random.rand(n,n))
    Q = np.transpose(B)*D*B
    b = np.random.rand(n, 1)

    # Retorna la matriz y vector listo para ejecutar
    return Q, b


def load_data(file_name):
    return pd.read_excel(file_name)

def drop_row_and_column(n_row, n_column, df):
    'delete row of column names and index column'
    return df.drop(df.columns[n_column], axis=1).drop([n_row])

def get_design_matrix_and_co2_vector(df):
    array = df.to_numpy()
    y = array[:,0].reshape((55, 1))
    X = np.delete(array, 0, axis=1)
    return X, y


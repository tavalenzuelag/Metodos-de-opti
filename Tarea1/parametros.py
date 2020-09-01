__author__ = "Moises Saavedra Caceres"
__email__ = "mmsaavedra1@ing.puc.cl"


# Modulos nativos de python
import numpy as np
import scipy.linalg
import random
import pandas as pd


def load_data(file_name):
    return pd.read_excel(file_name)

def drop_row_and_column(n_row, n_column, df):
    'delete row of column names and index column'
    return df.drop(df.columns[n_column], axis=1).drop([n_row])

def get_design_matrix_and_co2_vector(df):
    array = df.to_numpy()
    y = array[:,0].reshape((55, 1))/1000
    X = np.delete(array, 0, axis=1)
    return X.astype(float), y.astype(float)



#df = load_data('data.xlsx')
#df = drop_row_and_column(0, 0, df)
#X, y = get_design_matrix_and_co2_vector(df)





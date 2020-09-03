import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(file_name):
    return pd.read_excel(file_name)

def drop_row_and_column(n_row, n_column, df):
    'delete row of column names and index column'
    return df.drop(df.columns[n_column], axis=1).drop([n_row])

def get_design_matrix_and_co2_vector(df):
    scaler = MinMaxScaler(feature_range=(1, 2))
    array = scaler.fit_transform((df.to_numpy().astype(float)))
    y = array[:,0].reshape((55, 1))
    X = np.delete(array, 0, axis=1)
    return X, y




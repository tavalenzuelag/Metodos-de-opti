import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(file_name):
    return pd.read_excel(file_name)

def get_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    array = scaler.fit_transform((df.to_numpy().astype(float)))    
    b = array[:,0]
    A = np.delete(array, 0, axis=1)
    return A, b.reshape(b.shape[0], 1)
    
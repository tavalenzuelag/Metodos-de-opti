" Archivo que contiene las funciones que permiten generar una grilla de subplots "


import matplotlib.pyplot as plt


def get_index_params(n_sub_plots, n_columns):
    "Función que permite obtener los índices en los cuales se pondrán los subplots"
    
    n_rows = n_sub_plots// n_columns
    
    if n_sub_plots % n_columns == 0:
        index = [(i, j) for i in range(n_rows) for j in range(n_columns)]
        return n_rows, n_columns, index, []
    
    n_delete = n_columns - (n_sub_plots % n_columns)
    index = [(i, j) for i in range(n_rows + 1) for j in range(n_columns)]
    return n_rows + 1, n_columns, index[:-n_delete], index[-n_delete:]


def initialize_plot(y_size, x_size, n_sub_plots, n_columns):
    "Función que inicializa el gráfico de subplots vacíos a partir de los índices"
    
    n_rows, n_columns, index, index_delete = get_index_params(n_sub_plots, n_columns)
    fig, axs = plt.subplots(n_rows, n_columns, figsize=(y_size, x_size))
    remove_subplot(axs, index_delete)
    return fig, axs, index


def add_sub_plot(axs, x, y, i, j, title, x_label, y_label):
    "Función que agrega el gráfico a cada subplot"
    
    axs[i, j].plot(x, y)
    axs[i, j].set_title(title)
    axs[i, j].set(xlabel=x_label, ylabel=y_label)
    

def remove_subplot(axs, index_list):
    " Función que remueve subplots vacíos "
    
    for tupla in index_list:
        axs[tupla[0], tupla[1]].remove() 

        

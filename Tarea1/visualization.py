from matplotlib import pyplot as plt

def visualization(error_list, name = None):
    iterations = [x for x in range(len(error_list))]
    plt.plot(iterations, error_list, '.')
    if name is not None:
        plt.title(name)
    plt.xlabel('n° Iteración')
    plt.ylabel('Error')
    plt.show()


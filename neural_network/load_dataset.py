import numpy as np
import pandas as pd

def load_data_k_plus(u_file, y_file, i):
    x ,y = load_data(u_file, y_file)
    x = x[:len(x)-i]
    y = y[i:]
    return x, y

def load_data(u_file, y_file):
    u = pd.read_csv(u_file)
    y = pd.read_csv(y_file)

    u = u.values
    y = y.values

    y.transpose()
    u = np.reshape(u, (1, -1)).transpose()
    y = y.transpose()

    y = np.reshape(y, (4, -1)).transpose()
    x = np.concatenate((u, y), axis=1)

    x = np.delete(x, len(x) - 1, axis=0)
    y = np.delete(y, 0, axis=0)

    return x, y

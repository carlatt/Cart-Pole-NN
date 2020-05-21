from numpy import loadtxt
import pandas as pd
import numpy as np

def import_data():
    u = pd.read_csv("./simulation_data/U.csv")
    y = pd.read_csv("./simulation_data/Y.csv")

    u=u.values
    y=y.values
    y.transpose()
    u=np.reshape(u,(10,-1)).transpose()
    y=y.transpose()
    y=np.reshape(y,(40,-1)).transpose()

    return u, y

def load_data():
    u, y = import_data()

    split = 3/4

    x_train = u[:int(split*len(u))]
    y_train = y[:int(split*len(y))]
    x_test = u[int(split*len(u)):]
    y_test = y[int(split*len(y)):]

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    import_data()
    a=[]
    for i in range(0,201):
        a.append(i)
    print(a)
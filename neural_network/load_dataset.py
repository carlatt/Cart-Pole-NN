from numpy import loadtxt
import pandas as pd
import numpy as np

def import_data(u_file, y_file):
    u = pd.read_csv(u_file)
    y = pd.read_csv(y_file)

    u=u.values
    y=y.values
    y.transpose()
    u=np.reshape(u,(1,-1)).transpose()
    y=y.transpose()
    y=np.reshape(y,(4,-1)).transpose()
    u=np.concatenate((u, y), axis=1)
    u = np.delete(u,len(u)-1, axis=0)
    y = np.delete(y,0,axis=0)

    return u, y

def load_data(u_file, y_file):
    u, y = import_data(u_file, y_file)

    x_train = u
    y_train = y
    x_test = u
    y_test = y
    # print("la mia x_train è questa: ")
    # print(x_train)
    # print("la mia y_train è questa: ")
    # print(y_train)
    # print("fine.")
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    import_data()
    a=[]
    for i in range(0,201):
        a.append(i)
    print(a)
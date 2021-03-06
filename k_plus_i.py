import os
import pathlib

import numpy as np

from neural_network.physics_neural_network import physical_nn
from plot import plot_results, plot_errors, plot_NRMSE

if __name__ == "__main__":

    NRMSEs = []
    for I in range(1, 6):
        model_name = "k_plus_" + str(I)
        print(model_name)

        # if it exists it will be loaded without training
        if os.path.isdir(model_name):
            model = physical_nn(model_name)
        else:
            model = physical_nn()
            model.compile()
            # get number of tests
            count = 0
            for path in pathlib.Path("./simulation_data").iterdir():
                if path.is_file():
                    count += 1
            n = int(count / 2)

            for i in range(1, n):
                print("Train on test " + str(i))
                model.load_data_k_plus("./simulation_data/U" + str(i) + ".csv",
                                       "./simulation_data/Y" + str(i) + ".csv", I)
                model.fit(epochs=100)
                model.evaluate()
                model.save_model(model_name)

        # here is where the prediction happens
        i = 10

        model.load_data_k_plus("./simulation_data/U" + str(i) + ".csv",
                               "./simulation_data/Y" + str(i) + ".csv", I)

        Y = model.y
        pred = model.predict()

        realcartpos = []
        predcartpos = []
        realcartvel = []
        predcartvel = []
        realtheta = []
        predtheta = []
        realomega = []
        predomega = []
        for n in range(len(model.x)):
            realcartpos.append(Y[n][0])
            predcartpos.append(pred[n][0])
            realcartvel.append(Y[n][1])
            predcartvel.append(pred[n][1])
            realtheta.append(Y[n][2])
            predtheta.append(pred[n][2])
            realomega.append(Y[n][3])
            predomega.append(pred[n][3])

        plot_results(realcartpos, predcartpos, realcartvel, predcartvel, realtheta, predtheta, realomega, predomega)
        plot_errors(Y, pred)
        RMSE = np.sum((pred - Y) ** 2) / len(Y)
        NRMSE = RMSE / (np.max(Y) - np.min(Y))
        NRMSEs.append(NRMSE)
    plot_NRMSE(NRMSEs)

import numpy as np
from plot import plot_errors
from plot import plot_results
from neural_network.physics_neural_network import physical_nn



if __name__ == "__main__":
    model = physical_nn("./neural_network/cart_pole_nn_saved")
    model.model.summary()

    i = 10
    model.load_data("./simulation_data/U" + str(i) + ".csv",
                    "./simulation_data/Y" + str(i) + ".csv")

    # tempreal = []
    # temppred = []
    # for n in range(len(model.x)):
    #     img = np.array([model.x[n]]) 
    #     tempreal.append(np.array([model.y[n]]))

    #     prediction = model.predict(img)
    #     temppred.append(prediction)

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
    
    error = Y - pred
    
    plot_errors(error)
    
import matplotlib.pyplot as plt
import numpy as np

from neural_network.physics_neural_network import physical_nn


def plot_results(realcartpos, predcartpos, realcartvel, predcartvel, realtheta, predtheta, realomega, predomega):
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    ax1.plot(realcartpos, 'b', label='Real pos')
    ax1.plot(predcartpos, 'r:', label='Pred pos')
    ax1.legend(loc="upper right")
    ax2.plot(realcartvel, 'b', label='Real vel')
    ax2.plot(predcartvel, 'r:', label='Pred vel')
    ax2.legend(loc="upper right")
    ax3.plot(realtheta, 'b', label='Real theta')
    ax3.plot(predtheta, 'r:', label='Pred theta')
    ax3.legend(loc="upper right")
    ax4.plot(realomega, 'b', label='Real omega')
    ax4.plot(predomega, 'r:', label='Pred omega')
    ax4.legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    model = physical_nn("./neural_network/cart_pole_nn_saved")
    model.model.summary()

    i = 10
    model.load_data("./simulation_data/U" + str(i) + ".csv",
                    "./simulation_data/Y" + str(i) + ".csv")

    tempreal = []
    temppred = []
    for n in range(len(model.x)):
        img = np.array([model.x[n]])
        tempreal.append(np.array([model.y[n]]))

        prediction = model.predict(img)
        temppred.append(prediction)



    realcartpos = []
    predcartpos = []
    realcartvel = []
    predcartvel = []
    realtheta = []
    predtheta = []
    realomega = []
    predomega = []
    for n in range(len(model.x)):
        realcartpos.append(tempreal[n][0, 0])
        predcartpos.append(temppred[n][0, 0])
        realcartvel.append(tempreal[n][0, 1])
        predcartvel.append(temppred[n][0, 1])
        realtheta.append(tempreal[n][0, 2])
        predtheta.append(temppred[n][0, 2])
        realomega.append(tempreal[n][0, 3])
        predomega.append(temppred[n][0, 3])

    plot_results(realcartpos, predcartpos, realcartvel, predcartvel, realtheta, predtheta, realomega, predomega)
    Y = model.y
    T = model.predict()
    nMSE = np.power(T - Y, 2) / np.power(T, 2)
    #nMSE = ((T - Y) ** 2) / (T ** 2)
    plt.plot(nMSE)

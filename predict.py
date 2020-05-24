import random
import numpy as np
import matplotlib.pyplot as plt

from neural_network.physics_neural_network import physical_nn

if __name__ == "__main__":
    model = physical_nn("neural_network/cart_pole_nn_saved")
    
    i = 10
    model.load_data("./simulation_data/U" + str(i) + ".csv",
                    "./simulation_data/Y" + str(i) + ".csv")
    
    # print("il test ha dimensione")
    # print(model.x_test[19])
    #print(model.x_test)
    tempreal = []
    temppred = []
    for n in range(len(model.x_test)):
        img = np.array([model.x_test[n]])
        # print("testiamo l'input")
        # print(img)
        tempreal.append(img)

        prediction = model.predict(img)

        temppred.append(prediction)

        #print('Original: {} \nPredicted: {}'.format(img, prediction))
    realvalues = []
    predvalues = []
    for n in range(len(model.x_test)):
        realvalues.append(tempreal[n][0,3])
        predvalues.append(temppred[n][0,2])
    # plt.plot(realvalues)
    # plt.ylabel('values')
    # plt.show()

    # plt.plot(predvalues)
    # plt.ylabel('values')
    # plt.show()

    plt.figure()

    plt.subplot(211)
    plt.plot(realvalues)

    plt.subplot(212)
    plt.plot(predvalues)
    plt.show()


    # fig, ax = plt.subplots()
    # ax.plot(realvalues)
    # ax.set_title('A single plot')

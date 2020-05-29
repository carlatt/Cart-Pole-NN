import os
import pathlib

from neural_network.physics_neural_network import physical_nn

if __name__ == "__main__":

    #input an integer to obtain k+i model
    a = "Hello"
    try:
        I = int(input("Enter a number: "))
    except ValueError:
        print("Not an integer value...")
    model_name = "k_plus_"+str(I)
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
                            "./simulation_data/Y" + str(i) + ".csv",I)
            model.fit(epochs=100)
            model.evaluate()
            model.save_model(model_name)

    # here is where the prediction happens
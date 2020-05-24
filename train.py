import pathlib

from neural_network.physics_neural_network import physical_nn

if __name__ == "__main__":
    model = physical_nn("neural_network/cart_pole_nn_saved")

    # get number of tests
    count = 0
    for path in pathlib.Path("./simulation_data").iterdir():
        if path.is_file():
            count += 1
    n = int(count / 2)

    for i in range(1, n):
        print("Train on test " + str(i))
        model.load_data("./simulation_data/U" + str(i) + ".csv",
                        "./simulation_data/Y" + str(i) + ".csv")
        model.fit(epochs=20)
        model.evaluate()
        model.save_model("neural_network/cart_pole_nn_saved")
        print()


import random
import numpy as np

from neural_network.physics_neural_network import physical_nn

if __name__ == "__main__":
    model = physical_nn("neural_network/cart_pole_nn_saved")
    i = random.randint(1, 10)
    model.load_data("./simulation_data/U" + str(i) + ".csv",
                    "./simulation_data/Y" + str(i) + ".csv")
    img = np.array([model.x_test[0]])
    predictions = model.predict(img)
    predicted_class = predictions[0]
    original_class = model.y_test[0]
    print('Original: {} \nPredicted: {}'.format(original_class, predicted_class))
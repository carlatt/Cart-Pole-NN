import pathlib
import random
import numpy as np
import tensorflow as tf
from neural_network.load_dataset import load_data

class physical_nn(object):
    def __init__(self, model_restore_path=None):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(5,)),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Embedding(input_dim=1024, output_dim=512),
            tf.keras.layers.Dense(300, activation='sigmoid'),
            tf.keras.layers.Dense(300, activation='tanh'),
            tf.keras.layers.LSTM(512),
            #tf.keras.layers.Dense(300, activation='relu'),
            #tf.keras.layers.Dense(300, activation='sigmoid'),
            tf.keras.layers.Dense(4, activation='linear')
        ])
        if model_restore_path is None:
            pass
        else:
            self.model = tf.keras.models.load_model(model_restore_path)

    def load_data(self,u_file, y_file):
        (x_train, y_train), (x_test, y_test) = load_data(u_file, y_file)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def compile(self):
        self.model.compile(optimizer='adam',
                      loss='MSE',
                      metrics=['accuracy'])

    def fit(self, epochs=50):
        self.model.fit(self.x_train, self.y_train, epochs=epochs)


    def save_model(self, model_file_path):
        self.model.save(model_file_path)

    def evaluate(self):
        self.model.evaluate(self.x_test, self.y_test, verbose=2)

    def predict(self, input):
        return self.model.predict(input)


if __name__ == "__main__":

    #(x_train, y_train), (x_test, y_test) = load_data()

    #model = physical_nn("neural_network/cart_pole_nn_saved")
    model = physical_nn()

    model.compile()

    # #get number of tests
    # count = 0
    # for path in pathlib.Path("./simulation_data").iterdir():
    #     if path.is_file():
    #         count += 1
    # n=int(count/2)

    for i in range(1,9):

        print("Train on simulation "+str(i))
        model.load_data("./simulation_data/U"+str(i)+".csv",
                        "./simulation_data/Y"+str(i)+".csv")
        model.fit(epochs=100)
        #model.evaluate()
        model.save_model("cart_pole_nn_saved")
        print()


    i = 10
    model.load_data("./simulation_data/U" + str(i) + ".csv",
                    "./simulation_data/Y" + str(i) + ".csv")
    print("la mia x_test è la seguente: ")
    print(model.x_test[0])
    img = np.array([model.x_test[0]])
    print("cerco di predictare l'evoluzione relativa a questo input: ")
    print(img)
    predictions = model.predict(img)
    predicted_class = predictions[0]
    original_class = model.y_test[0]
    print('Original: {} \nPredicted: {}'.format(original_class, predicted_class))

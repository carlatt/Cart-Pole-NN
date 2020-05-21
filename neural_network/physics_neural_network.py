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
            #tf.keras.layers.Flatten(input_shape=(1,)),
            tf.keras.layers.Embedding(input_dim=1000, output_dim=201),
            #tf.keras.layers.GRU(512, return_sequences=True),
            tf.keras.layers.LSTM(256),
            tf.keras.layers.Dense(40, activation='linear')
        ])
        if model_restore_path is None:
            pass
        else:
            self.model = tf.keras.models.load_model(model_restore_path)

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = load_data()
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
        self.model.evaluate(x_test, y_test, verbose=2)

    def predict(self, input):
        return self.model.predict(input)


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_data()

    model = physical_nn()
    model.load_data()
    model.compile()
    model.fit(epochs=100)
    model.evaluate()

    img = np.array([x_test[0]])
    predictions = model.predict(img)
    predicted_class = predictions[0]
    original_class = y_test[0]
    print('Original: {} \nPredicted: {}'.format(original_class, predicted_class))

import numpy as np
import tensorflow as tf

from load_dataset import load_data


class physical_nn(object):
    def __init__(self, model_restore_path=None):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        if model_restore_path is None:
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(5,)),
                tf.keras.layers.Dense(50, activation='tanh'),
                tf.keras.layers.Dense(50, activation='tanh'),
                tf.keras.layers.Dense(40, activation='tanh'),
                tf.keras.layers.Dense(4, activation='tanh')
            ])
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

    def fit(self, epochs=50, model_file_path=None):
        self.model.fit(x_train, y_train, epochs=epochs)
        if model_file_path is not None:
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
    model.fit(model_file_path="ciao")
    model.evaluate()

    img = np.array([x_test[0]])
    predictions = model.predict(img)
    predicted_class = predictions[0]
    original_class = y_test[0]
    print('Original: {} \nPredicted: {}'.format(original_class, predicted_class))

import tensorflow as tf

from neural_network.load_dataset import load_data, load_data_k_plus


class physical_nn(object):
    def __init__(self, model_restore_path=None):
        self.x = None
        self.y = None
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(5,)),
            tf.keras.layers.Dense(500, activation='sigmoid'),
            tf.keras.layers.Dense(400, activation='sigmoid'),
            # tf.keras.layers.Dense(128, activation='sigmoid'),
            # tf.keras.layers.Dense(128, activation='sigmoid'),
            tf.keras.layers.Dense(4)
        ])
        if model_restore_path is None:
            pass
        else:
            self.model = tf.keras.models.load_model(model_restore_path)

    def load_data(self, u_file, y_file):
        (x, y) = load_data(u_file, y_file)
        self.x = x
        self.y = y

    def load_data_k_plus(self, u_file, y_file, i):
        (x, y) = load_data_k_plus(u_file, y_file, i)
        self.x = x
        self.y = y

    def compile(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
                           loss='MSE',
                           metrics=['accuracy'])

    def fit(self, epochs=50):
        self.model.fit(self.x, self.y, epochs=epochs)

    def save_model(self, model_file_path):
        self.model.save(model_file_path)

    def evaluate(self):
        self.model.evaluate(self.x, self.y, verbose=2)

    def predict(self, input=None):
        if input is None:
            return self.model.predict(self.x)
        return self.model.predict(input)


if __name__ == "__main__":
    # reset weights
    model = physical_nn()
    model.compile()
    model.save_model("./neural_network/cart_pole_nn_saved")

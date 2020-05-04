import tensorflow as tf
from tensorflow.keras import layers


class cart_pole_nn(object):
    def __init__(self, model_struct, input_size, output_size, learning_rate=0.01):
        '''

        :param model_struct: array n where each row contains a dict with:
         number of nodes (node:int) and activation function (activation:callable, it should
         be from tensorflow.nn)
        '''
        self.layers = None
        self.model_structure = model_struct
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.loss = None
        self.train_op = None

    def set_loss(self, loss):
        self.loss = loss

    def construct_model(self, x_train):
        prev_size = self.input_size
        prev_layer = x_train
        layers = []
        for layer_model in self.model_structure:
            w_init = tf.random_normal_initializer()
            weights = tf.Variable(initial_value=w_init(shape=(prev_size, layer_model['node']),
                                                       dtype='float32'),
                                  trainable=True)
            b_init = tf.zeros_initializer()
            bias = tf.Variable(initial_value=b_init(shape=(layer_model['node'],),
                                                    dtype='float32'),
                               trainable=True)

            layer = tf.add(tf.matmul(prev_layer, weights), bias)
            layer = layer_model['activation'](layer)

            prev_size = layer_model['node']
            prev_layer = layer

            layers.append(layer)

        self.layers = layers

    def train(self, x_train, epochs, batch_size, x_test):
        pass


if __name__ == "__main__":
    model_struct = [{'node': 50, 'activation': tf.nn.relu},
                    {'node': 50, 'activation': tf.nn.relu},
                    {'node': 50, 'activation': tf.nn.relu},
                    {'node': 50, 'activation': tf.nn.relu}]

    neural = cart_pole_nn(model_struct, 4, 4, learning_rate=0.001)
    v = tf.Variable(1., shape=tf.TensorShape(None))
    v.assign([[1.,2.,3.,6.]])
    neural.construct_model(v)

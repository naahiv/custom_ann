import numpy as np

class Initializer:
    def init_biases(prev_layer_size, layer_size):
        raise NotImplementedError

    def init_weights(prev_layer_size, layer_size):
        raise NotImplementedError

class XavierInitializer(Initializer):
    def init_biases(prev_layer_size, layer_size):
        return np.zeros(layer_size)

    def init_weights(prev_layer_size, layer_size):
        rng = np.random.default_rng()
        b = np.sqrt(6.0) / np.sqrt(prev_layer_size + layer_size)
        a = -b
        return (b - a) * rng.random(size=(layer_size, prev_layer_size)) + a

class HeInitializer(Initializer):
    def init_biases(prev_layer_size, layer_size):
        return 0.001 + np.zeros(layer_size)

    def init_weights(prev_layer_size, layer_size):
        rng = np.random.default_rng()
        std = np.sqrt(2.0 / prev_layer_size)
        return rng.normal(loc=0.0, scale=std, size=(layer_size, prev_layer_size))

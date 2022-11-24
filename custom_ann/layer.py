import numpy as np

from nonlinearity import ActivationFunction, PredefinedActivationFunctions
from initialization import Initializer, XavierInitializer, HeInitializer

class Layer:
    """
    A class representing the base unit of an ANN: a layer. This layer is a base class,
    and it can be either a HiddenLayer, InputLayer, or OutputLayer. This class is created
    just for hierarchy purposes.
    """
    def __init__(self, *, size):
        assert type(size) == int

        self.size = size
        self.activations = None

    def set_neurons(self, activations):
        assert isinstance(activations, np.ndarray) and neurons.size == self.size
        self.activations = np.copy(activations) # necessary?

class InputLayer(Layer):
    """
    An input layer, which has simply no additional structure than a layer
    """
    pass

class HiddenLayer(Layer):
    """
    A full hidden layer, which has the ability to be initalized, and computed
    """
    def __init__(self, *, size, prev_layer, initializer=HeInitializer, act_func=PredefinedActivationFunctions.ReLU):
        assert isinstance(prev_layer, Layer)
        assert issubclass(initializer, Initializer)
        assert isinstance(act_func, ActivationFunction)
        super().__init__(size)

        self.prev_layer = prev_layer
        self.initializer = initializer
        self.weight_matrix, self.bias_vector = None, None

    def initialize(self):
        self.bias_vector = self.initializer.init_biases(self.prev_layer.size, self.size)
        self.weight_matrix = self.initializer.init_weights(self.prev_layer.size, self.size)

    def compute(self):
        if prev_layer.activations:
            z_vector = np.matmul(self.weight_matrix, self.prev_layer.activations) + self.bias_vector
            self.activations = act_func.apply(z_vector)
        else:
            raise Exception("Previous layer has not been computed yet!")

class OutputLayer(HiddenLayer):
    """
    An output layer needs similar initialization as a Hidden layer, except that its activation function is defaulted
    to an output activation function, and it initalizes its values using the standard Xavier method.
    """
    def __init__(self, *, **kwargs):
        if not 'act_func' in kwargs:
            kwargs['act_func'] = PredefinedActivationFunctions.output_linear
        if not 'initializer' in kwargs:
            kwargs['initializer'] = XavierInitializer
        super().__init__(**kwargs)

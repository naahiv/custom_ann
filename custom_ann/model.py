import numpy as np

from layer import Layer, InputLayer, HiddenLayer, OutputLayer

class Network:
    """
    The main component of the project is the (Artifical Neural) Network,
    which houses and connects all the layers. Any testing operations will
    also reside here.
    """
    def __init__(self):
        """
        The status codes below stand for the following:
        0 : model empty
        1 : input layer added
        2 : output layer added
        3 : layers initialized
        """
        self.status = 0

        self.layers = []
        self.input_size = 0
        self.output_size = 0

    def add_layer(self, layer):
        assert isinstance(layer, Layer)
        if self.status == 0:
            assert isinstance(layer, InputLayer)
            self.layers.append(layer)
            self.input_size = self.layers[0].size
            self.status = 1
        elif self.status == 3:
            raise Exception("Cannot add layer; model is fully built!")
        elif isinstance(layer, InputLayer):
            raise Exception("Cannot add input layer; model already has an input layer")
        else:
            if isinstance(layer, HiddenLayer):
                layer.set_prev_layer(self.layers[-1]) # set proper previous layer
                self.layers.append(layer) # add hidden/output layer to stack
                if isinstance(layer, OutputLayer):
                    self.output_size = self.layers[-1].size
                    self.status = 2 
            else:
                raise Exception("Layer must be either HiddenLayer or OutputLayer")

    def initialize(self):
        for layer in self.layers:
            layer.initialize()
        self.status = 3

    def output(self, inp_vec):
        """
        The output (possibly a vector) for a given set of input neurons. This can be
        computed once the network is initialized.
        """
        assert self.status == 3
        assert inp_vec.shape == (self.input_size,)

        self.layers[0].set_neurons(inp_vec)
        for layer in self.layers[1:]:
            layer.compute()
        return self.layers[-1].get_neurons()

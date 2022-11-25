import numpy as np

from layer import Layer, InputLayer, HiddenLayer, OutputLayer
from training import SGDBackpropogation

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

        NOTE: this is a 'destructive' function - the model's neurons are all activated
        afterwards.
        """
        assert self.status == 3
        assert inp_vec.shape == (self.input_size,)

        self.layers[0].set_neurons(inp_vec)
        for layer in self.layers[1:]:
            layer.compute()
        return self.layers[-1].get_neurons()

    def adjust_parameters(self, adjustment_list):
        """
        Takes in a list of adjustments to the weights and biases in each layer
        """
        assert len(adjustment_list) == len(self.layers) - 1
        for (i, layer) in enumerate(self.layers)[1:]:
            layer.adjust_weights(adjustment_list[i][0])
            layer.adjust_biases(adjustment_list[i][1])

    def compute_loss(self, inp_vec, target_out):
        """
        Computes the loss function (SE) for one training example. No regularization is
        included in this (L1 or L2).
        """
        assert target_out.shape == (self.output_size,)
        return ( target_out - self.output(inp_vec) )**2

    def compute_batch_MSE(self, inputs, targets):
        """
        Computes the overall model loss for a set of training examples.
        It computes the average MSE for each example.
        """
        total = 0.0
        for (x, y) in zip(inputs, targets):
            total += self.compute_loss(x, y) / self.output_size
        return total / len(inputs)

    def train(self, *args, **kwargs):
        SGDBackpropogation.train(self, *args, **kwargs)


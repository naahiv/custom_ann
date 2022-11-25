import numpy as np

class SGDBackpropogation:
    def train(model, train_inputs, train_targets, *,
              epochs=100,
              learn_rate=0.03,
              batch_size=100):
        # in v1, forget about batch size
        for epoch in range(epochs):

            gradient_matrix = self.compute_mean_grad(model, train_inputs, train_targets, learn_rate)
            model.adjust_parameters(-1 * learning_rate * adjustment_matrix)
            # gather stats here
            print('training loss: ' + model.compute_loss(train_inputs))

    def compute_mean_grad(model, train_inputs, train_targets):
        """
        This method will output a list of tuples of np.arrays; one tuple
        for each layer, for each tuple one entry for weights and one for
        biases. A class structure will be used for the output to make it
        more code-readable to average the matrices.
        """
        sum_grad_matrix = 0
        for (x, y) in zip(train_inputs, train_targets):
            sum_grad_matrix += AdjustmentMatrix(self.compute_indiv_grad(model, x, y))
        return sum_grad_matrix / len(train_inputs)

    def compute_indiv_grads(model, x, y):
        """
        Backpropogates through layers to determine gradient for weights
        and biases.
        """
        grad_array = []
        tmp_delta_a = 2 * (model.output(x) - y) # activates neurons
        for l in range(len(model.layers)-1, 0, -1):
            delta_b, delta_w, tmp_delta_a = self.get_layer_adjustments(model.layers[l-1], model.layers[l], tmp_delta_a)
            grad_array.insert(0, (delta_b, delta_w))
        return grad_array

    def get_layer_adjustments(prev_layer, layer, delta_a):
        delta_b = delta_a * layer.act_func.apply_d(layer.activations)
        delta_w = np.outer(delta_b, prev_layer.activations)
        delta_prev_a = np.matmul(delta_b, layer.weight_matrix)
        return delta_b, delta_w, delta_prev_a


class AdjustmentMatrix:
    """
    An organizational structure which can manipulate a list of pairs of
    numpy.ndarray objects
    """
    def __init__(self, matr):
        assert isinstance(matr, list)
        assert isinstance(matr[0], tuple)
        assert isinstance(matr[0][0], np.ndarray)
        self.matr = matr

    def __add__(self, other):
        assert isinstance(other, AdjustmentMatrix)
        assert len(self.matr) == len(other.matr)
        return AdjustmentMatrix([(l1[0]+l2[0], l1[1]+l2[1]) for (l1, l2) in zip(self.matr, other.matr)])

    def __radd__(self, number): # ideally for zero case
        return AdjustmentMatrix([(number + layer[0], number + layer[1]) for layer in self.matr])

    def __truediv__(self, number):
        return AdjustmentMatrix([(layer[0] / number, layer[1] / number) for layer in self.matr)

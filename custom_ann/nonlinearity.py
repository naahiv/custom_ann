import numpy as np

class ActivationFunction:
    def __init__(self, *, func, d_func):
        """
        The func parameter simply takes in a regular numerical function of one variable.
        (The exception is possibly a softmax function). d_func should be set to the 
        derivative of func, but in terms of the original function. For example, if 
        func is a sigmoid, then d_func would be set to lambda f: f * (1 - f).
        """
        self.func = np.vectorize(func)
        self.d_func = np.vectorize(d_func)

    def apply(self, arr):
        assert isinstance(arr, np.ndarray)
        return self.func(arr)

    def apply_d(self, arr_val, *, pure=False):
        """
        Note that the 'arr_val' parameter must be given as some previous output of the
        original activation function. If the direct derivative is necessary, then one
        can supply pure=True
        """
        assert isinstance(arr_val, np.ndarray)
        if pure: # remove if this check is slow and delegate to other function
            return self.d_func(self.func(arr_val))
        else:
            # most of the time
            return self.d_func(arr_val)

class OutputActivationFunction(ActivationFunction):
    """
    A subclass of ActivationFunction meant for output layer activation functions. This
    is separated for identification purposes, readability, and pre-defined examples.
    """
    pass


class PredefinedActivationFunctions:
    def ReLU_func(x):
        return max(0.0, x)
    def d_ReLU_func(y):
        return max(0.0, np.sign(y))
    ReLU = ActivationFunction(func=ReLU_func, d_func=d_ReLU_func)

    def sigmoid_func(x):
        return 1 / (1 + np.exp(x))
    def d_sigmoid_func(y):
        return y * (1 - y)
    sigmoid = ActivationFunction(func=sigmoid_func, d_func=d_sigmoid_func)
    output_sigmoid = OutputActivationFunction(func=sigmoid_func, d_func=d_sigmoid_func)

    def linear_func(x):
        return x
    def d_linear_func(x):
        return 1
    linear = ActivationFunction(func=linear_func, d_func=d_linear_func)
    output_linear = OutputActivationFunction(func=linear_func, d_func=d_linear_func)

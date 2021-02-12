# TODO: 
# See if I can still do __call__ this with staticmethod. Cant use class name because it has child classes
# Support for non-numpy array inputs: ints, floats, lists. 
# If not error for bad input types

import numpy as np

# TODO: How to stop users from using this base activation class directly. Make it unimportable/callable
class Activation:
    @classmethod
    def __call__(cls, x):  
        return cls.evaluate(x)

    @staticmethod
    def evaluate(x):
        return x

    @staticmethod
    def delta(x):
        return x


class Identity(Activation):
    @staticmethod
    def delta(x):
        return np.ones(x.shape)


class Relu(Activation):
    @staticmethod
    def evaluate(x):
        x[x < 0] = 0
        return x

    @staticmethod
    def delta(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


class Sigmoid(Activation):
    @staticmethod  # TODO: Pending
    def evaluate(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def delta(x):
        return 1


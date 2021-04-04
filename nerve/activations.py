# TODO: See if I can still do __call__ this with staticmethod. Cant use class name because it has child classes
# TODO: I feel like this class inheritance can be further optimized. too many @staticmethod statements and 
# function defs are there. Maybe add a decorator that makes them all static and ads call function

import numpy as np

# TODO: How to stop users from using this base activation class directly. Make it unimportable/callable
class Activation:
    @classmethod
    def __call__(cls, x):  
        return cls.evaluate(x)

    @staticmethod  # TODO: Make this abstract method and see
    def evaluate(x):
        pass

    @staticmethod
    def delta(x):
        pass


class Linear(Activation):
    def __init__(self, slope: float = 1.0):
        self.slope = slope
        
    @staticmethod
    def evaluate(x):
        return self.slope * x

    @staticmethod
    def delta(x):
        return self.slope * np.ones(x.shape)


class Relu(Activation):
    def __init__(self, threshold: float = 0.0):
        self.thresh = threshold
    
    @staticmethod
    def evaluate(x):
        x[x < self.thresh] = 0
        return x

    @staticmethod
    def delta(x):
        x[x <= self.thresh] = 0
        x[x > self.thresh] = 1
        return x


class LeakyRelu(Activation):
    def __init__(self, threshold: float = 0.0, damper: float = 0.01):
        self.thresh = threshold
    
    @staticmethod
    def evaluate(x):
        x[x < self.thresh] *= damper
        return x

    @staticmethod
    def delta(x):
        x[x < self.thresh] = damper
        x[x >= self.thresh] = 1
        return x

    
class Elu(Activation):
    def __init__(self, threshold: float = 0.0, damper: float = 0.01):
        self.thresh = threshold
    
    @staticmethod
    def evaluate(x):
        x[x < self.thresh] = damper * (np.exp(x[x < self.thresh]) - 1)
        return x

    @staticmethod
    def delta(x):
        x[x < self.thresh] = damper * (np.exp(x[x < self.thresh]))
        x[x >= self.thresh] = 1
        return x


class Sigmoid(Activation):
    @staticmethod
    def evaluate(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def delta(x):
        s = self.evaluate(x)
        return s(1-s)


class Step(Activation):
    def __init__(self, threshold: float = 0.0):
        self.thresh = threshold

    @staticmethod
    def evaluate(x):
        x[x < self.thresh] = 0
        x[x >= self.thresh] = 1
        return x

    @staticmethod
    def delta(x):
        return 0


class Tanh(Activation):
    @staticmethod
    def evaluate(x):
        return np.tanh(x)

    @staticmethod
    def delta(x):
        return 1 - np.tanh(x)**2


class Softmax(Activation):
    # TODO: Pending
    @staticmethod
    def evaluate(x):
        z = np.exp(x)
        return z/z.sum()

    @staticmethod
    def delta(x):
        # return 1 - np.tanh(x)**2
        raise NotImplementedError
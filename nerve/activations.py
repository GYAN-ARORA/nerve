# TODO: See if I can still do __call__ this with staticmethod. Cant use class name because it has child classes
# TODO: I feel like this class inheritance can be further optimized. too many @staticmethod statements and 
# function defs are there. Maybe add a decorator that makes them all static and ads call function

import numpy as np
from abc import abstractmethod

# TODO: How to stop users from using this base activation class directly. Make it unimportable/callable
class Activation:
    def __call__(self, x):  
        return self.evaluate(x)

    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def delta(self, x):
        pass


class Linear(Activation):
    def __init__(self, slope: float = 1.0):
        self.slope = slope
        
    def evaluate(self, x):
        return self.slope * x

    def delta(self, x):
        return self.slope * np.ones(x.shape)


class Relu(Activation):
    def __init__(self, threshold: float = 0.0):
        self.thresh = threshold
    
    def evaluate(self, x):
        x[x < self.thresh] = 0
        return x

    def delta(self, x):
        x[x <= self.thresh] = 0
        x[x > self.thresh] = 1
        return x


class LeakyRelu(Activation):
    def __init__(self, threshold: float = 0.0, damper: float = 0.01):
        self.thresh = threshold
    
    def evaluate(self, x):
        x[x < self.thresh] *= damper
        return x

    def delta(self, x):
        x[x < self.thresh] = damper
        x[x >= self.thresh] = 1
        return x

    
class Elu(Activation):
    def __init__(self, threshold: float = 0.0, damper: float = 0.01):
        self.thresh = threshold
    
    def evaluate(self, x):
        x[x < self.thresh] = damper * (np.exp(x[x < self.thresh]) - 1)
        return x

    def delta(self, x):
        x[x < self.thresh] = damper * (np.exp(x[x < self.thresh]))
        x[x >= self.thresh] = 1
        return x


class Sigmoid(Activation):
    def evaluate(self, x):
        return 1/(1 + np.exp(-x))

    def delta(self, x):
        s = self.evaluate(x)
        return s(1-s)


class Step(Activation):
    def __init__(self, threshold: float = 0.0):
        self.thresh = threshold

    def evaluate(self, x):
        x[x < self.thresh] = 0
        x[x >= self.thresh] = 1
        return x

    def delta(self, x):
        return 0


class Tanh(Activation):
    def evaluate(self, x):
        return np.tanh(x)

    def delta(self, x):
        return 1 - np.tanh(x)**2


class Softmax(Activation):
    # TODO: Pending
    def evaluate(self, x):
        z = np.exp(x)
        return z/z.sum()

    def delta(self, x):
        # return 1 - np.tanh(x)**2
        raise NotImplementedError
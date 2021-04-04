from collections import deque
import numpy as np

class GradientDescentOptimizer:
    def __init__(self, learning_rate):
        self.l_rate = learning_rate

    def step(self, inp):
        return self.l_rate * inp


class AdaDelta:
    def __init__(self, learning_rate=0.1, window=10):
        self.l_rate = learning_rate
        self._past_avg = deque(maxlen=window)
        self._multiplier = 0

    def step(self, inp):
        self._past_avg.append(abs(inp).mean())
        self._multiplier = max(len(self._past_avg)/sum(self._past_avg), self._multiplier)
        return self.l_rate * inp * self._multiplier


class ForceFit:
    def __init__(self, learning_rate=0.1, multiplier=2, tolerance=1e-4):
        self.l_rate = learning_rate
        self.multiplier = multiplier
        self.tolerance = tolerance
        self._past_error = 0
    
    def step(self, inp):
        print(inp.mean())
        if np.isclose(inp.mean(), self._past_error, rtol=self.tolerance):
            self.l_rate *= self.multiplier
        self._past_error = abs(inp).mean()
        return self.l_rate * inp
        
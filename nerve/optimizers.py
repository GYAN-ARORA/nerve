class GradientDescentOptimizer:
    def __init__(self, learning_rate):
        self.l_rate = learning_rate

    def step(self, inp):
        return self.l_rate * inp
import numpy as np


class Network:
    # TODO: Add easy selection of a subset of the network
    def __init__(self, layers):
        self.layers = layers
        self._init_params()

    def __call__(self, inp):
        return self.evaluate(inp)

    def _init_params(self):
        for layer in self.layers:
            layer._init_params(self)

    def get_params(self):  # TODO: If Params in layer is made redundant change this API to params()
        for layer in self.layers:
            layer.get_params()

    def evaluate(self, inp):
        step = inp
        for layer in self.layers:
            step = layer(step)
        return step

    def backpropogate(self, error):
        for layer in reversed(self.layers):
            error = layer.backpropogate(error)

    def update_params(self):
        # TODO: Here I do have the opportunity to have a different optimizer for each layer if that makes sense
        for layer in self.layers:
            layer.update_params(self.optimizer)

    def epoch(self, X, y):
        # TODO: Make a similar function for a batch
        # TODO: Remove this loop and pass in the entire matrix
        # TODO: Find out how to report loss, rmse of individual neuron errors, or first just add errors and then rmse?
        # epoch_error = np.zeros(self.layers[-1].shape)
        activation = self.evaluate(X)
        error = activation - y
        self.backpropogate(error)
        
        # for sample_X, sample_y in zip(X, y):
        #     activation = self.evaluate(sample_X)
        #     error = activation - sample_y
        #     self.backpropogate(error)
        #     epoch_error += error**2  # TODO: Hard coded for RMSE
            # self.network.backpropogate(error)
            # loss = self.loss(activation, y)
            # losses.append(loss)
        self.update_params()  # TODO: This should be in a default callback setup
        return self.loss(error)

    def prepare(self, loss, optimizer, epochs):
        # TODO: Rename this function
        # TODO: Have these things also come in init?
        self.loss = loss  # TODO: Currently unused, find a use
        self.optimizer = optimizer
        self.epochs = epochs

    def train(self, X, y):
        # TODO: Implement callbacks here for various things
        # TODO: Make this class extendible and for people to be able to write custom training loops
        losses = [self.epoch(X,y) for i in range(self.epochs)]
        return losses

    # TODO: The backprop formula has been derived for rmse whose diff is just error (act-pred).
    # Need to recalc for any other metric and a way to put that into the formula.
    # Need to check if changing loss metric will make a difference, if yes how?

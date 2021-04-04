import numpy as np
from copy import deepcopy
from collections import Sequence
from typing import Optional, Union

from .data import Dataset, Batch
from .utils import empty_copy

def event(a: str):
    # TODO: Maybe have events as a context manager
    pass

class Network(Sequence):
    # TODO: Add easy selection of a subset of the network
    # TODO: Add strict type check for __init__, 'layers', check with babse Layer class
    def __init__(self, layers):
        self.layers = layers
        self._init_params()

    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, sliced):
        if isinstance(sliced, slice):
            slyce = self.__copy__()
            slyce.layers = slyce.layers[sliced]
            return slyce
        else:
            return self.layers[sliced]
    
    def __copy__(self):
        copy = empty_copy(self)
        copy.layers = self.layers
        return copy

    def __add__(self, network):
        # TODO: Do a compaitibility check. You can try to evaluate a sample, if if fails, give the traceback
        # NOTE: Observation! Because most of the state is held by the layers, this was so simple to do.
        new = empty_copy(self)
        new.layers = [deepcopy(layer) for layer in self.layers + network.layers]
        return new
    
    def __call__(self, inp):
        return self.evaluate(inp)

    def __repr__(self):
        return '\n'.join([str(layer) for layer in self.layers])

    def _init_params(self):
        for layer in self.layers:
            layer.init_params(self)

    def copy(self):
        return deepcopy(self)

    def show(self):
        for layer in self.layers:
            print(layer)

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

    def prepare(self, loss, optimizer, epochs):
        # TODO: Rename this function
        # TODO: Have these things also come in init?
        self.loss = loss  # TODO: Currently unused, find a use
        self.optimizer = optimizer
        self.epochs = epochs

    def _learn(self, X, y):
        # NOTE: Transpose at this level because the network sees the data vertically and the users like to see horizontally.
        activation = self.evaluate(X.T)
        error = activation - y.T
        self.backpropogate(error)
        return error

    def batch(self, X: np.ndarray, y: np.ndarray):
        # TODO: Make batched dataset API which works as a generator.
        event('batch_start')
        error = self._learn(X, y)
        self.update_params()  # TODO: This should be in a default callback setup
        event('batch_end')
        return error
    
    def epoch(
        self,
        data: Optional[Union[Dataset, Batch]]  = None,
        X: Optional[Union[np.ndarray, Batch]] = None,
        y: Optional[Union[np.ndarray, Batch]] = None
    ):
        # TODO: Find out how to report loss, rmse of individual neuron errors, or first just add errors and then rmse?
        # TODO: Clean this if-else
        if X is not None and y is not None and isinstance(X, Batch) and isinstance(y, Batch):
            event('epoch_start')
            errors = [self.batch(xb, yb) for xb, yb in zip(x, y)]
        else:
            if X is not None and y is not None and isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                batches = Batch(Dataset(X, y), -1) 
            elif isinstance(data, Dataset):
                batches = Batch(data, -1)
            elif isinstance(data, Batch):
                batches = data
            else:
                raise TypeError('Invalid input datatype')
            event('epoch_start')
            errors = [self.batch(batch.X, batch.y) for batch in batches]
        error = np.concatenate(errors, axis=-1)  # errors matrices from each batch concatenated
        event('epoch_end')
        l = self.loss(error)
        return l

    def train(
        self,
        data: Optional[Union[Dataset, Batch]]  = None,
        X: Optional[Union[np.ndarray, Batch]] = None,
        y: Optional[Union[np.ndarray, Batch]] = None
    ):
        # TODO: Implement callbacks here for various things
        # TODO: Make this class extendible and for people to be able to write custom training loops
        event('train_start')
        losses = [self.epoch(data, X, y) for i in range(self.epochs)]
        event('train_end')        
        return losses

    # TODO: The backprop formula has been derived for rmse whose diff is just error (act-pred).
    # Need to recalc for any other metric and a way to put that into the formula.
    # Need to check if changing loss metric will make a difference, if yes how?



import numpy as np
from copy import deepcopy
from abc import ABC, abstractclassmethod

from .activations import Linear

class Params:
    def __init__(self, frozen=False, **kwargs):
        self._frozen = frozen
        for arg in kwargs:
            self.__setattr__(arg, kwargs[arg])

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def __str__(self):
        return str(self.__dict__)

    def len(self):
        # TODO: Find number of params
        raise NotImplementedError


# TODO: Decide between keeping a base class or just treating Input layer as base without making it abstract
class Base(ABC):
    count = 0

    def __init__(self, name=None):
        Base.count += 1
        self.id = Base.count
        self._name = name

    def __repr__(self):
        return self.name

    def __call__(self, inp):
        return self.evaluate(inp)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        Base.count += 1
        result.id = Base.count
        return result

    # def __copy__(self): Not implimented here, will be needed when building the neuron level API.
    
    def _init_params(self, network):
        pass

    @property
    def name(self):
        return f"[{self.id}]{self._name or self.__class__.__name__}"

    def get_params(self):
        print(self.name, [])

    def copy(self):
        return deepcopy(self)

    @abstractclassmethod
    def evaluate(self, inp):
        # TODO: Chnage to abstract method. So that extending classes are required to implement it.
        raise NotImplementedError('BaseLayer cannot be evaluated')

    @abstractclassmethod
    def backpropogate(self, error):
        raise NotImplementedError('BaseLayer cannot be propgated')

    @abstractclassmethod
    def update_params(self, optimizer):
        raise NotImplementedError('BaseLayer has no params')


class Input(Base):
    def __init__(self, shape, name=None):
        super().__init__(name)
        self.shape = shape

    def __repr__(self):
        return f"{self.name}({self.shape})"

    def evaluate(self, inp):
        return inp

    def backpropogate(self, error):
        return error

    def update_params(self, optimizer):
        pass


class Dense(Base):
    def __init__(self, shape, activation=Linear(), bias=True, initialization='random', name=None):
        super().__init__(name)
        self.shape = shape
        self.activation = activation
        self.bias = bias
        self.initiaize = self._init_method(initialization)
        self.params = Params(weights=None)
        # TODO: Find wether initializing protected values is a best practice
        # self._input = None ?
        # TODO: Find what is the logic behind protected and private vars.
        # self.input OR self._input OR self.__input

    def __repr__(self):
        return f"{self.name}({self.shape})"

    def __init_delta(self):
        # TODO: THis batch count will not be necessary if weights updated
        # per batch, unless the user wants to right own callback for control.
        # if that is never going to be the case then remove this.
        self.__batch_count = 0
        self.__delta = np.zeros(self.params.weights.shape)

    def __update_delta(self, _delta):
        self.__batch_count += 1
        self.__delta += _delta

    def delta(self):
        return self.__delta / self.__batch_count

    def _init_method(self, initialization):
        # TODO: Add decorators for passing other params to rand or zeros.
        # TODO: Add checks to the signature and return type of the callable
        if initialization == 'random':
            return np.random.rand
        elif initialization == 'zeros':
            return np.zeros
        elif callable(initialization):
            return initialization

    def _init_params(self, network):
        # TODO: Network object is passed to each of its layers, and layers were passed to the network when created.
        # See if this can be done more elegantly, or if network needs to be an attribute of a layer.
        index = network.layers.index(self)
        self.params.weights = self.initiaize(self.shape, network.layers[index-1].shape + int(self.bias))
        self.__init_delta()  # TODO: Find a way to avoid for frozen.

    def get_params(self):
        print(self, self.params)

    def evaluate(self, inp):
        if self.bias:
            inp = np.append(inp, np.ones((1, inp.shape[-1])), axis=0)  # NOTE: Adds '1' for bias
        self._input = inp  # MEM: Immidiate computation is persisted in memory
        self._value = self.params.weights @ inp  # MEM
        return self.activation(self._value)

    def backpropogate(self, error):
        # feedback = np.diag(error) @ self.activation.delta(self._value)  # NOTE: Observation! Diag was just equal to element wise multiplication
        feedback = error * self.activation.delta(self._value)  # MEM
        self._delta = feedback @ self._input.T
        self.__update_delta(self._delta)  # MEM-MEM - # TODO: if frozen I can avoid having this variable
        if self.bias:
            next_error = self.params.weights.T[:-1] @ feedback  # NOTE: Drops the biases from weights
        else:
            next_error = self.params.weights.T @ feedback
        return next_error

    def update_params(self, optimizer):
        # TODO: Since this layer now has just one kind of param, param class seems uneccesary. Remove if redundant
        if not self.params._frozen:
            self.params.weights -= optimizer.step(self.delta())
        self.__init_delta()

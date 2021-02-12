import numpy as np
from abc import ABC, abstractclassmethod


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
        self._get_name(name)

    def __str__(self):
        return self.name

    # TODO: Write repr for the classes
    def __call__(self, inp):
        return self.evaluate(inp)

    def _get_name(self, name):
        self.name = name or f"{self.__class__.__name__}_{self.id}"

    def _init_params(self, network):
        pass

    def get_params(self):
        print(self.name, [])

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

    def __str__(self):
        return f"{self.name}({self.shape})"

    def evaluate(self, inp):
        return inp

    def backpropogate(self, error):
        return error

    def update_params(self, optimizer):
        pass


class Dense(Base):
    def __init__(self, shape, activation, bias=True, initialization='random', name=None):
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

    def __str__(self):
        return f"{self.name}({self.shape})"

    def __init_delta(self):
        self.__sample_count = 0
        self.__delta = np.zeros(self.params.weights.shape)

    def __update_delta(self, _delta):
        self.__sample_count += 1
        self.__delta += _delta

    def delta(self):
        return self.__delta / self.__sample_count

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
            # inp = np.append(inp, np.ones((1, inp.shape[-1])), axis=0)  # TODO: Correct this append for entire batch compute
            inp = np.append(inp, np.ones(1))  # Adds a '1' input backward disconnected neuron. This brings in the bias naturally
        self._input = inp  # MEM: Immidiate computation is persisted in memory
        self._value = self.params.weights @ inp  # MEM
        # print('Input:', self._input)
        # print('Value:', self._value)
        return self.activation(self._value)

    def backpropogate(self, error):
        # print('Error:', error)
        feedback = np.diag(error) @ self.activation.delta(self._value)  # MEM
        # print('Fdbck:', feedback)
        self._delta = np.outer(feedback, self._input.T)  # feedback @ self._input.T # TODO: Replace for bacth compute
        # print('Delta:', self._delta)
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

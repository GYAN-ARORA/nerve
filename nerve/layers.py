import numpy as np
from copy import deepcopy
from abc import ABC, abstractclassmethod

from . import activations

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
    
    def init_params(self, network):
        if not hasattr(self, 'shape'):
            self_index = network.index(self)
            if self_index == 0:
                raise TypeError('Network should start with "Input" layer')  # TODO: Make custom error here.
            self.shape = network[self_index - 1].shape  # Transfering shape for shapeless layers.

    # @abstractclassmethod
    # TODO: Decide wether this should be abstract. The delima is that this is called by default callback, 
    # so if someone has implemented a new layer and forgot to impliment this function. Otherwise I have to 
    # implment this unnecssarily in so many layers which dont have params.
    def update_params(self, optimizer):
        # raise NotImplementedError('BaseLayer has no params')
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


class Dense(Base):
    def __init__(self, shape, bias=True, initialization='random', name=None):
        super().__init__(name)
        self.shape = shape
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

    def init_params(self, network):
        # TODO: Network object is passed to each of its layers, and layers were passed to the network when created.
        # See if this can be done more elegantly, or if network needs to be an attribute of a layer.
        index = network.index(self)
        self.params.weights = self.initiaize(self.shape, network.layers[index-1].shape + int(self.bias))
        self.__init_delta()  # TODO: Find a way to avoid for frozen.

    def get_params(self):
        print(self, self.params)

    def evaluate(self, inp):
        if self.bias:
            inp = np.append(inp, np.ones((1, inp.shape[-1])), axis=0)  # NOTE: Adds '1' for bias
        self._input = inp  # MEM: Immidiate computation is persisted in memory
        return self.params.weights @ inp  # MEM

    def backpropogate(self, error):
        # feedback = np.diag(error) @ self._value  # NOTE: Observation! Diag was just equal to element wise multiplication
        # feedback = error * self._value  # MEM
        self._delta = error @ self._input.T
        self.__update_delta(self._delta)  # MEM-MEM - # TODO: if frozen I can avoid having this variable
        # NOTE: Here _delta is the sum from all samples, it should be mean, because if all samples 
        # said we want to move +1, it will move +n which is wrong. I wonder how NN are then trained in
        # parallel, because async addition is probably done. and the current method also trains by the way.
        # maybe the sensitivty to learning rate was so high due to this addition. check that out.
        if self.bias:
            next_error = self.params.weights.T[:-1] @ error  # NOTE: Drops the biases from weights
        else:
            next_error = self.params.weights.T @ error
        return next_error

    def update_params(self, optimizer):
        # TODO: Since this layer now has just one kind of param, param class seems uneccesary. Remove if redundant
        if not self.params._frozen:
            self.params.weights -= optimizer.step(self.delta())
        self.__init_delta()


class Conv(Dense):
    '''
    Convolution is implimented as a Dense Layer
    '''
    def __init__(self, kernel_size, n_kernels):
        # super('Base', self).__init__()
        raise NotImplementedError


class Pool(Base):
    def __init__(self):
        raise NotImplementedError


class Dropout(Base):
    def __init__(self):
        raise NotImplementedError


class Flatten(Base):
    def __init__(self):
        raise NotImplementedError


class Reshape(Base):
    def __init__(self):
        raise NotImplementedError


class Cropping(Base):
    def __init__(self):
        raise NotImplementedError


class Padding(Base):
    def __init__(self):
        raise NotImplementedError


class BatchNormalization(Base):
    def __init__(self):
        raise NotImplementedError


class LayerNormalization(Base):
    def __init__(self):
        raise NotImplementedError


class Relu(Base):
    def __init__(self, threshold=0.0, name=None):
        super().__init__(name)
        self.threshold = threshold
        self._activation = activations.Relu(threshold=threshold)

    def evaluate(self, inp):
        self._inp = inp  # MEM
        return self._activation(inp)

    def backpropogate(self, error):
        return error * self._activation.delta(self._inp)


class Softmax(Base):
    def __init__(self):
        raise NotImplementedError


class Add(Base):
    def __init__(self):
        raise NotImplementedError


class Multiply(Base):
    def __init__(self):
        raise NotImplementedError


class MaxMin(Base):
    def __init__(self):
        raise NotImplementedError


class Avg(Base):
    def __init__(self):
        raise NotImplementedError


class Dot(Base):
    def __init__(self):
        raise NotImplementedError

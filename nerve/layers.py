import numpy as np
from copy import deepcopy
from abc import ABC, abstractclassmethod

from . import utils
from . import activations
# TODO: Correct usage of pivate and protected members across the codebase


class Params:
    def __init__(self, freeze=False, **kwargs):
        self._freeze = freeze
        for arg in kwargs:
            self.__setattr__(arg, kwargs[arg])

    def freeze(self):
        self._freeze = True

    def unfreeze(self):
        self._freeze = False

    def __str__(self):
        return str(self.__dict__)

    def len(self):
        # TODO: Find number of params
        raise NotImplementedError


# TODO: Decide between keeping a base class or just treating Input layer as base without making it abstract
# TODO: Add all compulsary params as None in base class like shape, input_shape, etc.
class Base(ABC):
    count = 0

    def __init__(self, name=None):
        Base.count += 1
        self.id = Base.count
        self._name = name
        self._freeze = False

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
    
    def _get_input_shape(self, network):
        '''
        Most layers will need this
        '''
        self_index = network.index(self)
        if self_index == 0: 
            if isinstance(self, Input):
                self.input_shape = self.shape
            else:
                raise TypeError('Network should start with "Input" layer')  # TODO: Make custom error here.
        else:
            self.input_shape = network[self_index - 1].shape

    def init_params(self, network):
        self._get_input_shape(network)
        if not hasattr(self, 'shape'):
            self.shape = self.input_shape  # Transfering shape for shapeless layers.

    def freeze(self):
        self._freeze = True

    def unfreeze(self):
        self._freeze = False


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
        self._get_input_shape(network)
        self.params.weights = self.initiaize(self.shape, self.input_shape + int(self.bias))
        self.__init_delta()  # TODO: Find a way to avoid for frozen.

    def get_params(self):
        print(self, self.params)

    def evaluate(self, inp):
        if self.bias:
            inp = np.append(inp, np.ones((1, inp.shape[-1])), axis=0)  # NOTE: Adds '1' for bias
        self._input = inp  # MEM: Immidiate computation is persisted in memory
        # TODO: RuntimeWarning: overflow encountered in matmul numpy error
        return self.params.weights @ inp

    def backpropogate(self, error):
        # feedback = np.diag(error) @ self._value  # NOTE: Observation! Diag was just equal to element wise multiplication
        # feedback = error * self._value  # MEM
        self._delta = error @ self._input.T
        self.__update_delta(self._delta)  # MEM-MEM - # TODO: if frozen I can avoid having this variable
        # NOTE: Here _delta is the sum from all samples, it should be mean, because if all samples 
        # said we want to move +1, it will move +n which is wrong. I wonder how NN are then trained in
        # parallel, because async addition is probably done. and the current method also trains by the way.
        # maybe the sensitivty to learning rate was so high due to this addition. check that out. (Fix in RNN layer also)
        if self.bias:
            next_error = self.params.weights.T[:-1] @ error  # NOTE: Drops the biases from weights
        else:
            next_error = self.params.weights.T @ error
        return next_error

    def update_params(self, optimizer):
        # TODO: Since this layer now has just one kind of param, param class seems uneccesary. Remove if redundant
        if not self.params._freeze:
            self.params.weights -= optimizer.step(self.delta())
        self.__init_delta()


class Conv(Base):
    '''
    Convolution is implimented as a Dense Layer
    '''
    def __init__(self, kernel_shape=(3,3), n_kernels=3, name=None):
        super().__init__(self, name)
        self.kernel_shape = kernel_shape
        self.n_kernels = n_kernels
        self.shape = (None, '*', '*', n_kernels)

    def __repr__(self):
        return f"{self.name}{self.shape}"

    @staticmethod
    def __compute_shape(input_shape, kernel_shape):
        image_shape = input_shape[1:3]
        if image_shape[0] != image_shape[1]:
            raise Exception("Input Image must be square")  # TODO: Add custom exception
        img_len = image_shape[0] - kernel_shape[0] + 1  # TODO: Improve for non-squares and slide params
        img_width = image_shape[1] - kernel_shape[1] + 1
        return (img_len, img_width)
    
    def init_params(self, network):
        self._get_input_shape(network)  # (batch, len, width, channels)
        mask_kernel = np.arange(1, self.kernel_shape[0]**2 + 1).reshape(self.kernel_shape)
        self.n_inp_channels = self.input_shape[3]
        self._mask_weights = conv_mask(self.input_shape[1], mask_kernel)  # Assuming square images
        self.shape = (None, *self.__compute_shape(self.input_shape, self.kernel_shape), self.n_kernels)
        self.dense_layers = [
                [Dense(self.shape[1] * self.shape[2], False, 'zeros', f'{self.name}_dense_[K{j}][C{i}]')
                for i in range(self.n_inp_channels)] 
                for j in range(self.n_kernels)
            ]
        for kernel in range(self.n_kernels):
            for channel in range(self.n_inp_channels):
                layer = self.dense_layers[kernel][channel]
                layer.params.kernel = np.random.rand(self.kernel_shape)
                layer.params.weights = conv_mask(self.input_shape[1], layer.params.kernel)
                layer.__init_delta()

    def get_params(self):
        # TODO: Convert weights to kernels and print
        for kernel in range(self.n_kernels):
            for channel in range(self.n_inp_channels):
                self.dense_layers[kernel][channel].get_params()

    def evaluate(self, inp):
        # NOTE: Input shape at this level is (batch, len, width, channels)
        out = np.zeros(self.shape)
        for kernel in range(self.n_kernels):
            for channel in range(self.n_inp_channels):
                out[:, :, :, kernel] += self.dense_layers[kernel][channel](
                    inp[:, :, :, channel].reshape(None, inp.shape[1] * inp.shape[2])
                ).reshape(None, out.shape[1], out.shape[2], 1) 
                # TODO: Try to use bias for each dense layer, so ul learn n_channel biases instead of one bias = sum(n_channel biases) which is okay i guess
        return out

    def backpropogate(self, error):
        next_error = np.zeros(self.input_shape)
        for kernel in range(self.n_kernels):
            for channel in range(self.n_inp_channels):
                next_error[:, :, :, channel] += self.dense_layers[kernel][channel].backpropogate(
                    error[:, :, :, kernel].reshape(None, error.shape[1] * error.shape[2])
                ).reshape(None, next_error.shape[1], next_error.shape[2], 1)
        return next_error

    def update_params(self, optimizer):
        for kernel in range(self.n_kernels):
            for channel in range(self.n_inp_channels):
                layer = self.dense_layers[kernel][channel]
                for i in range(1, self.kernel_shape[0] * self.kernel_shape[1] + 1):
                    mask = self._mask_weights == i
                    layer.__delta[mask] = layer.__delta[mask].mean()
                layer.update_params(optimizer)


class Pool(Base):
    def __init__(self, operation='max', window_size=(2,2), name=None):
        super().__init__(name)
        if operation not in ['max', 'avg']:
            raise ValueError("operation argument must be one of 'max' or 'avg'")
        self.operation = operation
        self.window = window_size

    def __repr__(self):
        return f"{self.name}{self.window}"

    def init_params(self, network):
        self._get_input_shape(network)
        self.shape = (-1, -(self.input_shape[1] // -self.window[0]), -(self.input_shape[2] // -self.window[1]), self.input_shape[3])
        # NOTE: This is ceil division. I will be padding appropriately so as to loose no information for indivisible pool size
        self.padding = (self.window[0] - self.input_shape[1] % self.window[0], self.window[0] - self.input_shape[2] % self.window[1])

    def _pad(self, inp)  # TODO: Test and check this padding operation, make padding a util if also used elsewhere
        pad_mode = 'minimum' if self.operation == 'max' else 'mean'
        pad_width = ((0,0), (0, self.padding[0]), (0, self.padding[1]), (0,0))
        stat_len = ((1, 1), (1, self.window[0] - self.padding[0]), (1, self.window[1] - self.padding[1]), (1, 1))  # (1,1) does nothing, just to avoid error
        return np.pad(inp, pad_width=pad_width, mode=pad_mode, stat_length=stat_len)
    
    def _crop(self, inp):
        return inp[:, :self.input_shape[1], :self.input_shape[2], :]

    def evaluate(self, inp):
        inp = self._pad(inp)
        s = {
            'bs':inp.shape[0],
            'cx':int(inp.shape[1]/self.window[0]),
            'cy':int(inp.shape[2]/self.window[1]),
            'wx':self.window[0],
            'wy':self.window[1],
            'ch':inp.shape[3]
        }
        inp = inp \
            .reshape(s['bs'],  s['cx'], s['wx'], s['cy'], s['wy'], s['ch']) \
            .transpose(0,1,3,2,4,5) \
            .reshape(s['bs'], -1, s['wx']*s['wy'], s['ch']) \
            .transpose(0,1,3,2) \
            .reshape(-1, s['wx']*s['wy'])
        s['s'] = inp.shape
        self._s = s

        if self.operation == 'max':
            self._mask = np.argmax(inp, axis=-1)  # MEM
            out = np.max(inp, axis=-1).reshape(self.shape)
        elif self.operation == 'avg':
            out = np.mean(inp, axis=-1).reshape(self.shape)
        return out

    def backpropogate(self, error):
        s = self._s
        error = out.reshape(-1)
        if self.operation == 'max':
            next_error = np.zeros(s['s'])
            next_error[list(range(next_error.shape[0])), self._mask] = error
        elif self.operation == 'avg':
            next_error = np.ones(s['s']) * error[:, np.newaxis]/len(error)
        next_error = next_error \
            .reshape(s['bs'], -1, s['ch'], s['wx']*s['wy']) \
            .transpose(0, 1, 3, 2) \
            .reshape(s['bs'],  s['cx'], s['cy'], s['wx'], s['wy'], s['ch']) \
            .transpose(0,1,3,2,4,5) \
            .reshape(self.input_shape)
        next_error = self._crop(next_error)
        return next_error


class Dropout(Base):
    def __init__(self, drop_prob):
        self.drop_prob = drop_prob

    def evaluate(self, inp):
        self.mask = np.random.binomial(1, 1 - self.drop_prob, inp.shape) if not self._freeze else 1 - self.drop_prob
        return inp * self.mask

    def backpropogate(self, error):
        return error * self.mask


class RNN(Base):
    def __init__(self, network, name='None'):
        '''
        Note 'network' argument here is for the internal network of the RNN layer.
        '''
        # TODO: Add options for init method and bias like in dense layer. Add option for hidden_activation
        super().__init__(name)
        self.network = network
        self.shape = self.network[-1].shape
        self.init_state = np.zeros(self.shape)
        self.hidden_activation = activations.Tanh  # Hardcode for now

    def __repr__(self):
        return f"Recurrent: {self.name}({self.shape})"
    
    def __init_params(self, network):
        # TODO: Decide on using params class here.
        self._get_input_shape(network)
        self.recurrent_weights = np.random.rand(self.shape, self.shape)
        self._recurrent_delta = np.zeros(self.recurrent_weights.shape)
    
    def __update_delta(self, _delta):
        self._recurrent_delta += _delta

    def delta(self):
        return self._recurrent_delta

    def evaluate(self, inp):
        # NOTE: Refer here https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN#examples_2
        # TODO: Option to return entire sequence as numpy array
        self.output_sequence =  {}  # Output for RNN layer is equal to its state.
        self._input_sequence_length = len(inp)  # can be different for each sample. This is cached by the layer
        self.output_sequence[-1] = np.copy(self.state)
        # For single sample right now: (This will prbably work for batch aswell, test once.)
        for t in range(self._input_sequence_length): #This is 10xself.input_shape(8) -> a sequence of 10 items of size 8
            self.output_sequence[t] = self.network(inp[t]) + self.output_sequence[t-1] @ self.recurrent_weights
        return self.output_sequence[t]

    def backpropogate(self, error):
        for t in reversed(range(self._input_sequence_length)):
            self.network.backprop(error)
            self._recurrent_delta += error @ self.output_sequence[t-1].T
            error = self.reccurrent_weights.T @ error  # Same as dense layer
            
    def update_params(self, optimizer):
        self.network.optimizer = optimizer
        self.network.update_params()
        self.recurrent_weights += optimizer.step(self.delta())


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

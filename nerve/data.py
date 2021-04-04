from collections import Sequence
import numpy as np

from .utils import empty_copy

# NOTE: All shuffle etc functions to be added to this API, 
# then its X and y property to be accesses by network. If 
# there is any preprocessing to be done that nerve does not
# provide, then .data attribute can be used as a numpy array
# and X and y will automatically work out. Hence .data is 
# exposed to the user.

# TODO: Inherit from numpy and make Dataset awesome


class Dataset(Sequence):
    '''
    Merge X and y matrices.
    (For easy shuffling, batching, etc.)
    '''
    # TODO: Work this (X and y properties) out for n-dimensional arrays.
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self._y = y.shape[-1]
        self.data = self.merge(X, y)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return 'nerve.Dataset(' + repr(self.data) +')'

    def __copy__(self):
        newcopy = empty_copy(self)
        newcopy._y = self._y
        newcopy.data = self.data
        return newcopy

    def __getitem__(self, sliced):
        slyce = self.__copy__()
        slyce.data = slyce.data[sliced]
        return slyce

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def merge(X, y):
        return np.concatenate((X, y), axis=-1)

    @property
    def X(self):
        return self.data[:,:-self._y]

    @property
    def y(self):
        return self.data[:,-self._y:]


class Batch:
    '''
    Create batches from np.ndarray/ nerve.Dataset dataset as a generator.
    '''
    def __init__(self, dataset, batch_size: int = 64, rollover: bool = False):
        self.batch_num = 0
        self.data = dataset
        self.batch_size = batch_size
        self.rollover = rollover
        if self.batch_size == -1:
            self.batch_size = len(dataset)
        self.num_batches = int(np.ceil(len(self.data)/self.batch_size))
    
    def __next__(self):
        bn, bl = self.batch_num, self.batch_size
        self.batch_num += 1
        if self.batch_num > self.num_batches:
            self.batch_num = 0  
            if not self.rollover:
                raise StopIteration  # allows for looping again
        return self.data[bn * bl : (bn + 1) * bl]
    
    def __iter__(self):
        return self

    def copy(self):
        return copy.deepcopy(self)



'''
import numpy as np
from nerve.data import Dataset, Batch
X = np.array([list(range(20))]).reshape(20, 1)
y = np.array([[0]*10 + [1]*10]).T
def one_hot(a):
    # TODO: Make this function better and keep class mapping
    num_classes = len(np.unique(a))
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

y = one_hot(y)
z = Dataset(X,y)
z.data.shape
zb = Batch(z, 6)
zb1 = next(zb)
zb1.data
zb1.y
'''
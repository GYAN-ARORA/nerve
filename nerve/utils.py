import numpy as np

def empty_copy(obj):
    '''
    Create a blank copy of an object without calling the __init__ method.
    # https://www.oreilly.com/library/view/python-cookbook/0596001673/ch05s12.html
    '''
    class Empty(obj.__class__):
        def __init__(self): 
            pass
    newcopy = Empty()
    newcopy.__class__ = obj.__class__
    return newcopy


def conv_mask(image_size, kernel):
    # only square images and kernels, slide=1
    isz = image_size
    fsz = kernel.shape[0]
    tvs = isz-fsz+1
    w = np.zeros((tvs**2, isz**2))  # weights
    cnt = 0
    for r in range(tvs):
        for c in range(tvs):
            for fr in range(fsz):
                for fc in range(fsz):
                    w[cnt, (r * isz + c) + (fr * isz + fc)] = kernel[fr, fc]
            cnt += 1
    return w


def scale(X):
    min, max = X.min(), X.max()
    X = X - min
    X = X / (max - min)
    return X


def one_hot(a):
    # TODO: Make this function better and keep class mapping
    num_classes = len(np.unique(a))
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
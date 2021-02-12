import nerve
import numpy as np


def scale(X):
    min, max = X.min(), X.max()
    X = X - min
    X = X / (max - min)
    return X


# '''
# Simple Linear Regressor
activation = nerve.activations.Identity()
network = nerve.Network(
    layers=[
        nerve.layers.Input(1),
        nerve.layers.Dense(1, activation, bias=True)
    ]
)
X = np.array(list(range(20))).reshape(20, 1)
X = scale(X)  # Experiment without scale
y = X * 2.5 + 0.3

loss = nerve.loss.rmse
optimizer = nerve.optimizers.GradientDescentOptimizer(0.01)
network.prepare(loss, optimizer, 5000)
print(network(X[4]), y[4])
network.get_params()
# network.epoch(X,y)
losses = network.train(X, y)
network.get_params()
# print(losses)
print(network(X[4]), y[4])
# '''


# '''
# 1 X 3 X 4 X 2 -- binary classifier
X = np.array([list(range(20))]).reshape(20, 1)
y = np.array([[0]*10 + [1]*10])


def one_hot(a):
    num_classes = len(np.unique(a))
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

y = one_hot(y)
X = scale(X)

activation = nerve.activations.Relu()
network = nerve.Network(layers=[
    nerve.layers.Input(1),
    nerve.layers.Dense(3, activation, bias=True),  
    nerve.layers.Dense(4, activation, bias=True),
    nerve.layers.Dense(2, activation, bias=True)
])

loss = nerve.loss.rmse
optimizer = nerve.optimizers.GradientDescentOptimizer(0.01)
network.prepare(loss, optimizer, 100)
network.evaluate(X[16])
network.get_params()
network.epoch(X, y)
network.train(X, y)
# '''
import nerve
import numpy as np

# Simple Linear Regressor
network = nerve.Network(
    layers=[
        nerve.layers.Input(1),
        nerve.layers.Dense(1, bias=True),
        nerve.layers.Relu()
    ]
)
X = np.array(list(range(20))).reshape(20, 1)
X = scale(X)  # Experiment without scale
y = X * 2.5 + 0.3

loss = nerve.loss.mse
optimizer = nerve.optimizers.GradientDescentOptimizer(0.01)
network.prepare(loss, optimizer, 1000)
print(network(X[4].reshape(1,1)), y[4])
network.get_params()
# network.epoch(X,y)
import time
t = time.time()
losses = network.train(X=X, y=y)  # 8.9s
# network.epoch(X=X, y=y)
print(time.time() - t)
network.get_params()
# print(losses)
print(network(X[4].reshape(1,1)), y[4])

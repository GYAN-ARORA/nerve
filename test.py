import nerve
import numpy as np





# '''
# 1 X 3 X 4 X 2 -- binary classifier
X = np.array([list(range(20))]).reshape(20, 1)
y = np.array([[0]*10 + [1]*10])



y = one_hot(y)
X = scale(X)

# activation = nerve.activations.Relu()
network = nerve.Network(layers=[
    nerve.layers.Input(1),
    nerve.layers.Dense(3, bias=True),
    # nerve.layers.Relu(),  
    nerve.layers.Dense(4, bias=True),
    # nerve.layers.Relu(),
    nerve.layers.Dense(2, bias=True),
    # nerve.layers.Relu(),
])

loss = nerve.loss.rmse
optimizer = nerve.optimizers.GradientDescentOptimizer(0.01)
network.prepare(loss, optimizer, 1000)
print(network(X[4].reshape(1,1)), y[4])
print(network(X[16].reshape(1,1)), y[16])
# network.get_params()
# network.epoch(X=X, y=y)
network.train(X=X, y=y)
# network.get_params()
print(network(X[4].reshape(1,1)), y[4])
print(network(X[16].reshape(1,1)), y[16])
# '''

'''
# Batch API Test
data = nerve.data.Dataset(X, y)
data = nerve.data.Batch(data, 6)
# network.epoch(data)
network.train(data)
# network.get_params()
print(network(X[4].reshape(1,1)), y[4])
print(network(X[16].reshape(1,1)), y[16])

'''


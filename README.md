# NERVE

**Neural Networks in Python.**

Nerve is a raw implementation of neural networks written fully in python. It is written with an objective to better understand neural nets and hopes to provide a more intuitive api than torch/keras for building and training them, especially for people who wish to deep dive into the mechanics and customize operations as they learn fundamentals of these function fitters.

### Installation

This repository is not installable. I do not intend for people to copy/clone any code from here. This is made public as a read-only codebase for me to be able to present my work to my own network.


### Overview

> Do checkout /examples directory for full examples of working networks.

#### Build and train simple Neural Networks

```python
network = nerve.Network(layers=[
    nerve.layers.Input(784),
    nerve.layers.Dense(48),
    nerve.layers.Sigmoid(scale='auto'),
    nerve.layers.Dense(10),
    nerve.layers.Softmax()
])

loss = nerve.loss.rmse
optimizer = nerve.optimizers.GradientDescentOptimizer(learning_rate=0.2)
network.prepare(loss, optimizer, epochs=1000)

dataset = nerve.data.Dataset(data, labels)
batches = nerve.data.Batch(dataset, batch_size=512)

loss = network.train(batches)
plt.plot(loss)
```

#### Play with networks
``` python
layer = network[2]
head_net = network[:-1]
new_net = head_net + network[3:4]
```

#### Organize your data
```python
dataset = nerve.data.Dataset(X=independent_vars, y=dependent_vars)
print(data.shape)
subset = dataset[:100]
print(subset.X, subset.y)


batches = nerve.data.Batch(dataset, batch_size=64, rollover=True)
batch = next(batches)
print(batch.X, batch.y)
```

#### Convolutional Networks
```python
cnn = nerve.Network(layers=[
    nerve.layers.Input((-1, ims, ims, chn)),
    nerve.layers.Conv(kernel_shape=(5,5), n_kernels=1),
    nerve.layers.Pool(window_size=(2,2), operation='max'),
    nerve.layers.Flatten(),
    nerve.layers.Dense(48, bias=True),
    nerve.layers.Sigmoid(scale='auto'),
    nerve.layers.Dense(10, bias=True),
    nerve.layers.Softmax()
])
```

#### Recurrent Networks
```python
inner_network = nerve.Network(layers = [
    nerve.layers.Input(1),
    nerve.layers.Dense(5, bias=False),
])

rnn = nerve.Network(layers=[
    nerve.layers.Input(2),
    nerve.layers.RNN(inner_network),
    nerve.layers.Dense(1, bias=True),
    nerve.layers.Relu()
])
```

#### Roadmap

- Distributed training.
- Callbacks.
- More loss functions, optimizers and layers.
- Cool experiments.


#### Reach Me
You can reach out to me at gyansworld@gmail.com


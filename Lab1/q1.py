import numpy as np

z = np.linspace(-10,10,100)

#Sigmoid
sig = 1/(1+(np.exp(-z)))

#Tanh
tanh = (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))

#ReLU
ReLU = np.maximum(0,z)

#Leaky ReLU
LReLU = np.maximum(-z,z)

#Softmax
sftmx = np.exp(z)/sum(np.exp(z))
print(sftmx)
import numpy as np

#Using Sigmoid function
x = np.random.randn(4,1)
w = np.random.randn(1,4)
b = np.zeros((1,1))
z = np.dot(w,x) + b
print("Z:",z)
a = np.maximum(0,z)
print("a:",a)
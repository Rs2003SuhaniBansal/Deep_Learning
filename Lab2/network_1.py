import numpy as np

x = np.random.randn(4,1)
w = np.random.randn(1,4)
b = np.zeros((1,1))

Z_11 = []
for j in range(len(w)):
    s = 0
    for i in range(len(x)):
        s += w[j][i]*x[i][0]
    Z_11.append(s)

Z_11 = Z_11 + b
print("Z_11:",Z_11)

print("----------------------")
print("Using Numpy")

a = np.maximum(0,Z_11)
print("a:",a)

# using numpy
z = np.dot(w,x) + b
print("Z:",z)
a = np.maximum(0,z)
print("a:",a)
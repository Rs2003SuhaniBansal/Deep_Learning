import numpy as np

x = np.random.randn(4,1)
w_1 = np.random.randn(3,4)
w_2 = np.random.randn(2,3)
w_3 = np.random.randn(1,2)
b1 = np.zeros((3,1))
b2 = np.zeros((2,1))
b3 = np.zeros((1,1))


# using for loops
Z = []
for i in range(len(w_1)):
    s = 0
    for j in range(len(x)):
        s += w_1[i][j] * x[j][0]
    Z.append(s)

print("Z:",Z)

# Using Relu
print("layer1")
z = np.dot(w_1,x) + b1
print("z:", z)
a = np.maximum(0,z)
#for sigmoid function
# a = 1/(1+(np.exp(-z)))
print("a:",a)
x = a
print("---------------------")
print("layer2")
z = np.dot(w_2,x) + b2
a = np.maximum(0,z)
# a = 1/(1+(np.exp(-z)))
print("a:",a)
x = a
print("---------------------")
print("output layer")
z = np.dot(w_3,x) + b3
a = np.maximum(0,z)
# a = 1/(1+(np.exp(-z)))
print("y:",a)


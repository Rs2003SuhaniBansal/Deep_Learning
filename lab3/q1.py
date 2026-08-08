import numpy as np


x = np.random.randn(4,1)
w = np.random.randn(1,4)
# print("a:",a)
# x = [[0.05],[0.1]]
# w = [[0.15, 0.25], [0.2, 0.3]]
z = np.dot(w,x)
print("Z:",z)
a = np.maximum(0,z)
t=1 #=y

"""loss function = summation of 0.5x(Target - Output)^2"""
def loss(t,o):
    return (1/2)*((t-o)**2) # no summation since only one layer is present

"""Partial differentiation of loss function
dL/do = (t - o)*(-1) = (o-t)
where o = output, t = target"""
def deriv_loss(t,o):
    return o-t

def deriv_relu(z):
    return np.where(z > 0, 1 , 0)

""" output = ReLU(z)
do/dz = derivative of ReLU (deriv_relu)
dL/dz = (dL/do)*(do/dz) = (o-t) * deriv_relu"""
deriv_loss_wrt_z = deriv_loss(1,a)*deriv_relu(z)
print("deriv_loss_wrt_z:",deriv_loss_wrt_z)

"""dL/dw = (dL/dz) * (dz/dw) = (o-t) * deriv_relu * x"""
deriv_loss_wrt_w = deriv_loss_wrt_z*x
print("deriv_loss_wrt_w:",deriv_loss_wrt_w)

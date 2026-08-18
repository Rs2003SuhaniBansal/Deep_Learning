import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
y = np.array([[0],[1],[1],[0]])

class NeuralNetwork:
    def __init__(self, input_size,output_size ):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(self.input_size, self.output_size)
        self.bias= np.zeros((1, self.output_size))

    # def relu(self, x):
    #     return np.maximum(x, 0)
    #
    # def relu_deriv(self,x):
    #     return np.where(x > 0, 1, 0)

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_deriv(self,x):
        return x*(1-x)

    # def feedforward_relu(self, x):
    #     self.z = np.dot(x, self.weights) + self.bias
    #     self.pred_output = self.relu(self.z)
    #
    #     return self.pred_output

    def feedforward_sigmoid(self, x):
        self.z = np.dot(x, self.weights) + self.bias
        self.pred_output = self.sigmoid(self.z)

        return self.pred_output

    def backward_pass(self, X, y, learning_rate):
        out_error = self.pred_output - y
        output_delta = out_error * self.sigmoid_deriv(self.pred_output)

        # dL/dW
        weight_gradient = np.dot(X.T, output_delta)

        # dL/db
        bias_gradient = np.sum(output_delta,axis=0,keepdims=True)

        # gradient descent
        self.weights -= learning_rate * weight_gradient
        self.bias -= learning_rate * bias_gradient

    def train(self, X, y, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            output = self.feedforward_sigmoid(X)
            loss = 0.5 * np.mean((y - output) ** 2)
            losses.append(loss)
            self.backward_pass(X, y, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss:{loss}")
        return losses

nn = NeuralNetwork(input_size=3, output_size=1)
losses = nn.train(X,y,epochs=1000,learning_rate=0.1)

output_2 = nn.feedforward_sigmoid(X)
print(output_2)

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
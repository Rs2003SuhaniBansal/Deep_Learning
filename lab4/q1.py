import numpy as np

X = [[0,0,1],[1,1,1],[1,0,1],[0,1,1]]
y = [[0],[1],[1],[0]]

class NeuralNetwork:
    def __init__(self, input_size, hidden_size,output_size ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_to_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))

        self.weights_hidden_to_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def relu(self, x):
        return np.maximum(x, 0)

    def relu_deriv(self,x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_deriv(self,x):
        return x*(1-x)

    def feedforward_relu(self, x):
        self.hidden_layer = np.dot(x, self.weights_input_to_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_layer)

        self.output_layer = np.dot(self.hidden_output, self.weights_hidden_to_output) + self.bias_output
        self.output_output_pred_relu = self.relu(self.output_layer)

        return self.output_output_pred_relu

    def feedforward_sigmoid(self, x):
        self.hidden_layer = np.dot(x,self.weights_input_to_hidden + self.bias_hidden)
        self.hidden_output = self.sigmoid(self.hidden_layer)

        self.output_layer = np.dot(self.hidden_output, self.weights_hidden_to_output) + self.bias_output
        self.output_output_pred_sigmoid = self.sigmoid(self.output_layer)

        return self.output_output_pred_sigmoid

nn = NeuralNetwork(input_size=3, hidden_size=4, output_size=1)
output = nn.feedforward_relu(X)
print(output)
output_2 = nn.feedforward_sigmoid(X)
print(output_2)


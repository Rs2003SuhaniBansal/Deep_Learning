import numpy as np

X = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
y = np.array([[0],[1],[1],[0]])

class NeuralNetwork:
    def __init__(self, input_size, hidden_size,output_size ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_to_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))

        self.weights_hidden_to_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    # def relu(self, x):
    #     return np.maximum(x, 0)
    #
    # def relu_deriv(self,x):
    #     return np.where(x > 0, 1, 0)

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_deriv(self,x):
        return x*(1-x)
    #
    # def feedforward_relu(self, x):
    #     self.hidden_layer = np.dot(x, self.weights_input_to_hidden) + self.bias_hidden
    #     self.hidden_output = self.relu(self.hidden_layer)
    #
    #     self.output_layer = np.dot(self.hidden_output, self.weights_hidden_to_output) + self.bias_output
    #     self.output_output_pred_relu = self.relu(self.output_layer)
    #
    #     return self.output_output_pred_relu


    # sigmoid(sigmoid(x * weight + bias)* weight + bias)

    def feedforward_sigmoid(self, x):
        self.hidden_layer = np.dot(x,self.weights_input_to_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_layer)

        self.output_layer = np.dot(self.hidden_output, self.weights_hidden_to_output) + self.bias_output
        self.output_output_pred_sigmoid = self.sigmoid(self.output_layer)

        return self.output_output_pred_sigmoid

    def backward_pass(self, X, y, learning_rate):
        """ dL/do = (output(predicted) - target(actual)) """
        out_error = self.output_output_pred_sigmoid - y
        """ dL/dz = (o-t)*deriv(sigmoid(z)) """
        output_delta = out_error * self.sigmoid_deriv(self.output_output_pred_sigmoid)

        """dL/dH = dL/dz * dz/dH
        z = H*W + b
        therefore, dz/dH = W  
        dL/dH = W * dL/dz """
        hidden_error = np.dot(output_delta, self.weights_hidden_to_output.T)
        hidden_delta = hidden_error * self.sigmoid_deriv(self.hidden_output)

        self.weights_hidden_to_output -= np.dot(self.hidden_output.T, output_delta)*learning_rate
        self.bias_output -= np.sum(output_delta, axis=0, keepdims=True)*learning_rate
        self.weights_input_to_hidden -= np.dot(X.T, hidden_delta)*learning_rate
        self.bias_hidden -= np.sum(hidden_delta, axis=0, keepdims=True)*learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward_sigmoid(X)
            self.backward_pass(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss:{loss}")

nn = NeuralNetwork(input_size=3, hidden_size=4, output_size=1)
nn.train(X,y,epochs=1000,learning_rate=0.1)

output = nn.feedforward_sigmoid(X)
print("Final predictions:")
print(output)


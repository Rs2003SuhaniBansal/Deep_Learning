"""Implement a 2-layer (input layer, hidden layer and output layer)
neural network from scratch for the XOR operation.
This includes implementing forward and backward passes from scratch."""

import numpy as np

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weights and biases: input -> hidden
        self.weights_input_to_hidden = np.random.randn(
            self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))

        # Weights and biases: hidden -> output
        self.weights_hidden_to_output = np.random.randn(
            self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        # Input -> Hidden
        self.hidden_layer = (np.dot(X, self.weights_input_to_hidden)
            + self.bias_hidden)

        self.hidden_output = self.sigmoid(self.hidden_layer)

        # Hidden -> Output
        self.output_layer = (np.dot(self.hidden_output, self.weights_hidden_to_output)
            + self.bias_output)

        self.output_pred = self.sigmoid(self.output_layer)

        return self.output_pred

    def backward_pass(self, X, y, learning_rate):
        # Output error
        # dL/do = predicted - actual
        output_error = self.output_pred - y

        # dL/dz = dL/do * sigmoid'(z)
        output_delta = (output_error* self.sigmoid_deriv(self.output_pred))

        # Error propagated back to hidden layer
        hidden_error = np.dot(output_delta,self.weights_hidden_to_output.T)

        # Hidden-layer delta
        hidden_delta = (hidden_error* self.sigmoid_deriv(self.hidden_output))

        # Gradients for hidden -> output weights
        dW_hidden_output = np.dot(self.hidden_output.T,output_delta)

        db_output = np.sum(output_delta,axis=0,keepdims=True)

        # Gradients for input -> hidden weights
        dW_input_hidden = np.dot(X.T,hidden_delta)

        db_hidden = np.sum(hidden_delta,axis=0,keepdims=True)

        # Update weights and biases
        self.weights_hidden_to_output -= (learning_rate * dW_hidden_output)

        self.bias_output -= (learning_rate * db_output)

        self.weights_input_to_hidden -= (learning_rate * dW_input_hidden)

        self.bias_hidden -= (learning_rate * db_hidden)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):

            # Forward pass
            output = self.feedforward(X)

            # Calculate loss
            loss = np.mean((y - output) ** 2)

            # Backward pass
            self.backward_pass(X,y,learning_rate)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")


# For reproducible results
np.random.seed(42)

# Create network:
# 2 input neurons -> 4 hidden neurons -> 1 output neuron
nn = NeuralNetwork(input_size=2,hidden_size=4,output_size=1)

# Train the network
nn.train(X,y,epochs=10000,learning_rate=0.1)

# Final predictions
predictions = nn.feedforward(X)

print("\nFinal Predictions:")
print(predictions)

print("\nRounded Predictions:")
print(np.round(predictions))

print("\nExpected Output:")
print(y)
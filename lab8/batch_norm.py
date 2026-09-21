import torch

class BatchNormScratch:

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum

        # Learnable scale and shift
        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)

        # Used during testing
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, x, training=True):

        if training:
            # Mean and variance across batch
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            # Update running statistics
            self.running_mean = ((1 - self.momentum) * self.running_mean
                + self.momentum * mean)

            self.running_var = ((1 - self.momentum) * self.running_var
                + self.momentum * var)

        else:
            # Use stored statistics during testing
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        y = self.gamma * x_hat + self.beta

        return y


# Example
x = torch.tensor([
    [10., 20., 30.],
    [40., 50., 60.],
    [70., 80., 90.]
])

bn = BatchNormScratch(num_features=3)

output = bn.forward(x)

print("Original:")
print(x)

print("\nAfter Batch Normalization:")
print(output)
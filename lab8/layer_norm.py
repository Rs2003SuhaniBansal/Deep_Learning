import torch


class LayerNormScratch:

    def __init__(self, num_features, eps=1e-5):
        self.eps = eps

        # Learnable scale and shift
        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)

    def forward(self, x):

        # Mean for every sample
        mean = x.mean(dim=1, keepdim=True)

        # Variance for every sample
        var = x.var(
            dim=1,
            keepdim=True,
            unbiased=False
        )

        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        y = self.gamma * x_hat + self.beta

        return y


# Example
x = torch.tensor([
    [10., 20., 30.],
    [40., 50., 60.],
    [70., 8., 9.]
])

ln = LayerNormScratch(num_features=3)

output = ln.forward(x)

print("Original:")
print(x)

print("\nAfter Layer Normalization:")
print(output)
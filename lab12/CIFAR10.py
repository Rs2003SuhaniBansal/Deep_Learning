import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])

train_data = datasets.CIFAR10(root="../data",train=True,download=True,transform=transform)

test_data = datasets.CIFAR10(root="../data",train=False,download=True,transform=transform)

train_loader = DataLoader(train_data,batch_size=128,shuffle=True)

test_loader = DataLoader(test_data,batch_size=128,shuffle=False)

class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

model = CIFAR_CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
train_errors = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    train_error = 100 - train_accuracy
    train_errors.append(train_error)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {total_loss/len(train_loader):.4f} "
        f"Train Error: {train_error:.2f}%"
    )

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_errors, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Error (%)")
plt.title("CIFAR-10 Training Error")
plt.grid()
plt.show()


class DeepCNN(nn.Module):
    def __init__(self, num_layers):
        super().__init__()

        layers = []
        in_channels = 3
        out_channels = 32

        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels

            if i % 2 == 1:
                layers.append(nn.MaxPool2d(2))
                out_channels = min(out_channels * 2, 256)

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train(mydataloader, model, loss_fn, optimizer, device, epochs):
    size = len(mydataloader.dataset)

    for epoch in range(epochs):
        model.train()

        for batch, (X, y) in enumerate(mydataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss_value = loss.item()
                current = (batch + 1) * len(X)

                print(f"loss: {loss_value:>7f} "f"[{current:>5d}/{size:>5d}]")

def test(mydataloader, model, loss_fn, device):
    size = len(mydataloader.dataset)
    num_batches = len(mydataloader)

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in mydataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += ((pred.argmax(1) == y).type(torch.float).sum().item())

    test_loss /= num_batches
    correct /= size

    test_error = 100 - (100 * correct)

    print(
        f"Test Error: "
        f"Accuracy: {100 * correct:.1f}%, "
        f"Avg loss: {test_loss:.6f}")

    return test_error


def calculate_error(dataloader, model, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

    accuracy = 100 * correct / total
    return 100 - accuracy


depths = [2, 4, 6, 8, 10]
results = []

for depth in depths:
    print(f"\nTraining network with {depth} layers")

    model = DeepCNN(depth).to(device)

    optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    train(train_loader,model,criterion,optimizer,device,epochs=10)

    train_error = calculate_error(train_loader,model,device)
    results.append(train_error)
    print(
        f"Depth: {depth}, "
        f"Training Error: {train_error:.2f}%")

plt.figure(figsize=(8, 5))
plt.plot(depths, results, marker="o")
plt.xlabel("Number of Convolutional Layers")
plt.ylabel("Training Error (%)")
plt.title("Training Error vs Network Depth")
plt.grid()
plt.show()
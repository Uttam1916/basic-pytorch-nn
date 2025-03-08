import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from timeit import default_timer as timer

# Download and Load CIFAR-10 Dataset
train_data = datasets.CIFAR10(
    root="data", train=True, download=True, transform=transforms.ToTensor()
)
test_data = datasets.CIFAR10(
    root="data", train=False, download=True, transform=transforms.ToTensor()
)

# Display a sample image

class_names = train_data.classes


# Display multiple random images
torch.manual_seed(42)

BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train Dataset Size: {len(train_data)}, Test Dataset Size: {len(test_data)}")
print(f"Train Batches: {len(train_dataloader)}, Test Batches: {len(test_dataloader)}")

# Define CNN Model
class CIFARV0(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, hidden_features: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_features * 8 * 8, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


# Initialize Model
torch.manual_seed(42)
model_0 = CIFARV0(input_shape=3, hidden_features=10, output_shape=len(class_names)).to("cuda")

# Define Accuracy Function
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = correct / len(y_pred)*100
    return acc


# Define Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# Training Function
def train_model(model, train_dataloader, loss_fn, optimizer, accuracy_fn,device="cuda"):
    train_loss, train_acc = 0, 0
    model.train()

    for batch_idx, (X, y) in enumerate(train_dataloader):
        X,y=X.to(device),y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")

# Testing Function
def test_model(model, test_dataloader, loss_fn, accuracy_fn,device="cuda"):
    test_loss, test_acc = 0, 0
    model.eval()

    with torch.no_grad():
        for X, y in test_dataloader:
            X,y=X.to(device),y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            test_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

# Train the Model
train_time = timer()
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch+1}/{epochs}")
    train_model(model=model_0, train_dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn)
    test_model(model=model_0, test_dataloader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn)

end_timer = timer()
print(f"Total Training Time: {end_timer - train_time:.2f} seconds")

train_time = timer()
epochs = 100

for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch+1}/{epochs}")
    train_model(model=model_0, train_dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn)
    test_model(model=model_0, test_dataloader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn)

end_timer = timer()
print(f"Total Training Time: {end_timer - train_time:.2f} seconds")
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

train_data=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data=datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

fig=plt.figure(figsize=(9,9))
rows=4
cols=4

for i in range(1,cols*rows+1):
  randin=torch.randint(0,len(train_data),size=[1]).item()
  img, label=train_data[randin]
  fig.add_subplot(rows,cols,i)
  plt.imshow(img.squeeze(),cmap="gray")
  plt.title(train_data.classes[label])
  plt.axis(False)

from torch.utils.data import DataLoader

BATCH_SIZE=32

class_names=train_data.classes

train_dataloader= DataLoader(dataset=train_data,
                             batch_size=BATCH_SIZE,
                             shuffle=True)
test_dataloader= DataLoader(dataset=test_data,
                            batch_size=BATCH_SIZE,
                            shuffle=False)


class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.ReLU(),
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = number of features in a data sample (784 pixels)
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)


model_0= FashionMNISTModelV0 (input_shape=784,hidden_units=10,output_shape=len(class_names)).to("cpu")

import requests
from pathlib import Path
if Path("hi.py").is_file():
  print("it alr ther")
else:
  request=requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
  with open("hi.py","wb") as f:
    f.write(request.content)

from hi import accuracy_fn

loss_fn = nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(params=model_0.parameters(),lr=0.1)

epochs=3
for epoch in range(epochs):
  print(f"Epoch:{epoch}\n------------")
  train_loss=0
  for batch,(x,y) in enumerate(train_dataloader):
    model_0.train()

    y_pred=model_0(x)

    loss=loss_fn(y_pred,y)
    train_loss+=loss

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if batch%400==0:
      print(f"looked at {batch*len(x)}/{len(train_dataloader.dataset)} samples.")

  train_loss/=len(train_dataloader)
  test_loss, test_acc = 0, 0
  model_0.eval()
  with torch.inference_mode():
      for X, y in test_dataloader:
            # 1. Forward pass
            test_pred = model_0(X)

            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
      test_loss /= len(test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
      test_acc /= len(test_dataloader)

    ## Print out what's happening
  print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")


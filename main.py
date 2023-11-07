from dataset import GoodSoundsDataset
from cnn import ConvNet
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

# Apply the dataset class
root_path = '/home/daven/data/audio/good-sounds'
width, height = 256, 256

dataset = GoodSoundsDataset(root_path, 65536, width, height)
model = ConvNet(width, width, dataset.n_categories)
optim = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

epochs = 10

for epoch in range(epochs):
    running_loss = 0.0

    for inputs, labels in dataloader:
        optim.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

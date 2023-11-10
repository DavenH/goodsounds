import torch

from dataset import GoodSoundsDataset
from cnn import ConvNetModel
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

# Apply the dataset class
root_path = '/home/daven/data/audio/good-sounds'
width, height = 256, 128
truncate_len = 32768
batch_size = 128
epochs = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = GoodSoundsDataset(root_path, truncate_len, width, height)
model = ConvNetModel(width, height, len(dataset.categories)).to(device)
optim = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss().to(device)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model.train()

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

# start_epoch = load_checkpoint('checkpoint.pth.tar')

count = 0
start_epoch = 0
for epoch in range(start_epoch, epochs):
    running_loss = 0.0
    for map in dataloader:
        optim.zero_grad()
        inputs = map['spectrogram'].to(device)
        labels = map['index'].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        count += 1
        print(f"Epoch: {epoch}, step {count}, loss: {loss.item():3.3}")
        if count % 100 == 0:
            filename = f"checkpoints/checkpoint_epoch_{epoch}_step_{count}.pth.tar"
            print(f"Saving checkpoint {filename}...")

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
            }, filename=filename)

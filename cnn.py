import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, width: int, height: int, n_categories):
        super(ConvNet, self).__init__()

        pool_w = 2
        pool_h = 2

        channels = 1
        self.conv_blocks = []

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d((pool_w, pool_h))

        self.final_size = width * height / ((pool_w * pool_h) ** 2) * 32

        self.fc1 = nn.Linear(self.final_size, 256)
        self.fc2 = nn.Linear(256, n_categories)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        if self.training:
            x = self.dropout(x)
        x = x.view(-1, self.final_size)
        x = F.relu(self.fc1(x))
        if self.training:
            x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


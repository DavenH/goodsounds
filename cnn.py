import math

import torch.nn as nn
import torch.nn.functional as F

class ConvNetModel(nn.Module):
    def __init__(self, width: int, height: int, n_categories):
        super(ConvNetModel, self).__init__()

        pool_w = 2
        pool_h = 2
        n_final_maps = 64
        dropout_rate = 0.3  # Typically between 0.3 and 0.5

        self.conv1 = nn.Conv2d(1, 16, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, n_final_maps, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(n_final_maps)

        self.pool = nn.MaxPool2d((pool_w, pool_h))
        self.dropout = nn.Dropout(dropout_rate)

        final_size = width * height # starting parameter size
        final_size //= (pool_w * pool_h) ** 2  # reduction due to 2x pooling
        final_size //= (2 * 2) ** 2 # reduction due to the strides (2 in each of width and height, in 2 convolutions)
        final_size *= n_final_maps

        # average between to the sizes in log-space
        intermediate_size = 2 ** round(0.5 * (math.log2(final_size) + math.log2(n_categories)))

        self.fc1 = nn.Linear(final_size, intermediate_size)

        # 1 more so we can predict the note
        self.fc2 = nn.Linear(intermediate_size, n_categories)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.pool(F.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


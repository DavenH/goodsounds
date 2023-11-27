import math

import torch
import torch.nn as nn
import torch.nn.functional as fn

class ConvNetModel(nn.Module):
    def __init__(self, width: int, height: int, n_categories):
        super(ConvNetModel, self).__init__()

        pool_w = 2
        pool_h = 2
        dropout_rate = 0.1

        self.conv = []
        in_channels = 1
        # starting parameter size
        w, h = width, height
        n_total_param = 0

        conv_layers = [
            dict(maps=16, kernel_size=7, stride=4, pool=False),
            dict(maps=64, kernel_size=3, stride=2, pool=True),
            dict(maps=128, kernel_size=3, stride=2, pool=False),
            dict(maps=512, kernel_size=3, stride=1, pool=True),
            dict(maps=1024, kernel_size=3, stride=1, pool=True)
        ]

        print(f"Shape:\t\t\t1 x {w} x {h},\t{w * h}")

        modules = self.__dict__.get('_modules')
        for idx, conv_layer in enumerate(conv_layers):
            n_maps = conv_layer["maps"]
            stride = conv_layer["stride"]
            kernel_size = conv_layer["kernel_size"]
            conv_config = dict(
                    conv=nn.Conv2d(in_channels, n_maps, kernel_size, stride, math.floor(kernel_size / 2)),
                    batch_norm=nn.BatchNorm2d(n_maps),
                    pool=conv_layer["pool"]
                )
            self.conv.append(conv_config)

            # necessary for the module.parameters() call to propagate to child modules
            # these aren't directly set by self.conv = nn.Conv2d (in that case the link would be
            # automatically made by the base class's overloaded __attr__() method)
            modules[f"conv{idx}"] = conv_config["conv"]
            modules[f"bn{idx}"] = conv_config["batch_norm"]
            in_channels = n_maps

            w //= stride
            h //= stride
            n_parameters = n_maps * w * h
            print(f"After conv {idx+1}:\t{n_maps} x {w} x {h}\t{n_parameters}")

            if conv_layer["pool"]:
                w //= pool_w
                h //= pool_h
                n_parameters = n_maps * w * h
                print(f"After pool {idx+1}:\t{n_maps} x {w} x {h}\t{n_parameters}")

            n_total_param += n_parameters


        final_size = conv_layers[-1]["maps"] * w * h

        self.pool = nn.MaxPool2d((pool_w, pool_h))
        self.dropout = nn.Dropout(dropout_rate)

        # average between to the sizes in log-space
        intermediate_size = 80 # 2 ** round(0.5 * (math.log2(final_size) + math.log2(n_categories)))

        self.fc1 = nn.Linear(final_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, n_categories)

        print(f"After linear 1:\t{final_size} x {intermediate_size}\t{final_size * intermediate_size}")
        n_total_param += final_size * intermediate_size

        print(f"After linear 2:\t{intermediate_size} x {n_categories}\t{intermediate_size * n_categories}")

        n_total_param += final_size * n_categories
        print(f"Total Parameters:\t{n_total_param}")

        self.config = dict(
            conv_layers=conv_layers,
            dropout=dropout_rate,
            width=width,
            height=height,
            pool_w=pool_w,
            pool_h=pool_h,
            fc1_in=final_size,
            fc1_out=n_categories
        )

        # quick test to be sure it all works before waiting for dataset load
        test_tensor = torch.zeros([1, 1, width, height])
        test_out = self.forward(test_tensor)


    def forward(self, x):
        for conv_layer in self.conv:
            x = conv_layer["conv"](x)
            x = fn.relu(x)
            if conv_layer["pool"]:
                x = self.pool(x)
            x = conv_layer["batch_norm"](x)

        # flatten everything except batch dimension
        x = x.view(x.shape[0], -1)
        x = fn.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


    def get_config(self) -> dict:
        return self.config
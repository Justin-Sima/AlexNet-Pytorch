import torch
import torch.nn as nn
import torch.functional as F


class AlexNetPytorch(nn.Module):
    def __init__(self, num_channels, channel_index):
        self.conv_1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=96,
            kernel_size=(11,11),
            stride=(4,4),
            padding='same'  
        )
        nn.init.normal_(self.conv_1.weight, 0, 0.01)
        nn.init.constant_(self.conv_1.bias, 0)
        self.conv1_maxpool = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=(2,2),
            padding='same'
        )
        # TODO: Parameters for local response regularization layer.

        self.conv_2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=256,
            kernel_size=(5,5),
            stride=(1,1),
            padding='same'
        )
        nn.init.normal_(self.conv_2.weight, 0, 0.01)
        nn.init.constant_(self.conv_2.bias, 1)
        self.conv2_maxpool = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=(2,2),
            padding='same'
        )

        self.conv_3 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=384,
            kernel_size=(3,3),
            stride=(1,1),
            padding='same'
        )
        nn.init.normal_(self.conv_3.weight, 0, 0.01)
        nn.init.constant_(self.conv_3.bias, 1)

        self.conv_4 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=384,
            kernel_size=(3,3),
            stride=(1,1),
            padding='same'
        )
        nn.init.normal_(self.conv_4.weight, 0, 0.01)
        nn.init.constant_(self.conv_4.bias, 0)

        self.conv_5 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            padding='same'
        )
        nn.init.normal_(self.conv_5.weight, 0, 0.01)
        nn.init.constant_(self.conv_5.bias, 1)
        self.conv5_maxpool = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=(2,2),
            padding='same'
        )

        self.flattening_layer = nn.Flatten()

        self.dense_1 = nn.Linear(
            in_features=,
            out_features=4096
        )
        nn.init.normal_(self.dense_1.weight, 0, 0.01)
        nn.init.constant_(self.dense_1.bias, 1)
        self.dense1_dropout = torch.nn.Dropout(p=0.5, inplace=False)

        self.dense_2 = nn.Linear(
            in_features=4096,
            out_features=4096
        )
        nn.init.normal_(self.dense_2.weight, 0, 0.01)
        nn.init.constant_(self.dense_2.bias, 1)
        self.dense2_dropout = torch.nn.Dropout(p=0.5, inplace=False)

        self.output_layer = nn.Linear(
            in_features=4096,
            out_features=1000,
            bias=False
        )
        nn.init.normal_(self.output_layer.weight, 0, 0.01)

    def forward(self, input_tensor):
        x = self.conv_1(input_tensor)
        x = self.conv1_maxpool(x)
        x = F.local_response_norm(x, size, alpha=0.0001, beta=0.75, k=1.0)
        x = self.conv_2(x)
        x = self.conv2_maxpool(x)
        x = F.local_response_norm(x, size, alpha=0.0001, beta=0.75, k=1.0)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv5_maxpool(x)
        x = self.flattening_layer(x)
        # x.view(x.size(0), 256 * 36)
        x = self.dense_1(x)
        x = self.dense1_dropout(x)
        x = self.dense_2(x)
        x = self.dense2_dropout(x)
        
        return self.output_layer(x)

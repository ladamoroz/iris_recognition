import torch
import torch.nn as nn
class EyeNet(nn.Module):
    def __init__(self, activation):
        super().__init__()

        self.shared_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            activation(),
            nn.BatchNorm2d(64),
        )

        self.shared_conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            activation(),
            nn.BatchNorm2d(128),
        )

        self.shared_conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            activation(),
            nn.BatchNorm2d(256),
        )

        self.base_convt1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=64,
                kernel_size=3,
                stride=4,
                padding=0,
                output_padding=1,
            ),
            activation(),
            nn.BatchNorm2d(64),
        )

        self.base_conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            activation(),
            nn.BatchNorm2d(64),
        )
        self.base_conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
            activation(),
            nn.Sigmoid(),
        )

        self.aux_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            activation(),
            nn.BatchNorm2d(256),
        )
        self.aux_convt1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                output_padding=1,
                padding=1,
            ),
            activation(),
            nn.BatchNorm2d(128),
        )
        self.aux_convt2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                output_padding=1,
                padding=1,
            ),
            activation(),
            nn.BatchNorm2d(64),
        )
        self.aux_conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # the shared part of the network
        x_skip = self.shared_conv1(x)
        x = self.shared_conv2(x_skip)
        x = self.shared_conv3(x)

        # the base part of the network
        base_x = self.base_convt1(x)
        base_x = torch.cat([base_x, x_skip], dim=1)
        base_x = self.base_conv2(base_x)
        base_x = self.base_conv3(base_x)

        # the auxiliary network
        aux_x = self.aux_conv1(x)
        aux_x = self.aux_convt1(aux_x)
        aux_x = self.aux_convt2(aux_x)
        aux_x = self.aux_conv2(aux_x)
        return base_x, aux_x
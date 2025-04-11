import torch.nn as nn

from lightning_template.base_model import BaseModel


class MNISTModel(BaseModel):
    def __init__(self):
        super(MNISTModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),  # 28x28x1 -> 28x28x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28x32 -> 14x14x32
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),  # 14x14x32 -> 14x14x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14x64 -> 7x7x64
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(7 * 7 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def get_loss_func(self):
        return nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

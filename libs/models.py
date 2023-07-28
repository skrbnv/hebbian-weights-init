# import torch
import torch.nn as nn


class HebbianModels(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        super().forward(x)


class HEBB1(HebbianModels):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            *[
                nn.Conv2d(3, 50, 3, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(50),
                nn.Conv2d(50, 100, 3, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(100),
                nn.Conv2d(100, 250, 3, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(250),
                nn.Conv2d(250, 500, 3, 2, 1),
                nn.ReLU(),
                nn.BatchNorm2d(500),
                nn.Flatten(),
            ]
        )
        self.final = nn.Linear(2000, 10)

    def forward(self, x):
        x = self.seq(x)
        x = self.final(x)
        return x

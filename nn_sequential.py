import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Conv3d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Yefeiji(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.conv1 = Conv2d(3, 32, 5, 1,2)
        # self.maxpoo1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, 1, 2)
        # self.maxpoo2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, 1, 2)
        # self.maxpoo3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)
        self.model1 = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpoo1(x)
        # x = self.conv2(x)
        # x = self.maxpoo2(x)
        # x = self.conv3(x)
        # x = self.maxpoo3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x= self.model1(x)
        return x

yefeiji = Yefeiji()
print(yefeiji)

input = torch.ones(64, 3, 32, 32)
output = yefeiji(input)

print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(yefeiji, input)
writer.close()
import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained =False)

torch.save(vgg16, "./model/vgg16_method1.pth")

torch.save(vgg16.state_dict(), "./model/vgg16_method2.pth")

class Yefeiji(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

yefeiji = Yefeiji()
torch.save(yefeiji, "./model/yefeiji_method1.pth")
import torch
import torchvision.models
from torch import nn
from model_save import *

# model = torch.load("./model/vgg16_method1.pth", weights_only=False)
# print(model)

vgg16 = torchvision.models.vgg16(pretrained = False)
vgg16.load_state_dict(torch.load("./model/vgg16_method2.pth"))
# model = torch.load("./model/vgg16_method2.pth")
# print(vgg16)

# class Yefeiji(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

model = torch.load("./model/yefeiji_method1.pth" , weights_only=False)
print(model)
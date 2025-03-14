from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from wheel.cli.convert import wininst_re

from test_tb import writer

"""

"""

img_path = "dataset/ants/0013035.jpg"
img = Image.open(img_path)

# 先创建ToTensor对象再调用方法
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer =  SummaryWriter("logs")
#
# writer.add_image("Tensor_img", tensor_img)

# Norm
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])

writer.add_image("Norm", img_norm)

writer.close()


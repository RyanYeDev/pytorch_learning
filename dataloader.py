import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# train data
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True,num_workers=0, drop_last=True)

# img, target = test_data[0]
# print(img.shape)
# print(target)

writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step = step + 1
writer.close()
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10("./dataset", True, torchvision.transforms.ToTensor(), download=True)

test_data = torchvision.datasets.CIFAR10("./dataset", False,torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练集长度为：{}". format(train_data_size))
print("测试集长度为：{}". format(test_data_size))

train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

class Yefeiji(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# model
yefeiji = Yefeiji()
yefeiji = yefeiji.cuda()

# loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
# optimizer
learning_rate = 1e-2
optimizer = torch.optim.SGD(yefeiji.parameters(), lr = learning_rate)

# count train
total_train_step = 0
# count test
total_test_step = 0
# count epoch
epoch = 30

writer = SummaryWriter("./logs")

for i in range(epoch):
    print("-------第{}轮训练开始-------".format(i + 1))
    # start train
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = yefeiji(imgs)
        loss = loss_fn(outputs, targets)
        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss：{}". format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # start test
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = yefeiji(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum().item()
            total_accuracy += accuracy

    print("total loss: {}".format(total_test_loss))
    print("total accuracy: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(yefeiji, "./model/yefeiji_{}.pth".format(i))
    # torch.save(yefeiji.state_dict(), "./model/yefeiji_{}.pth".format(i))
    print("model saved")

writer.close()


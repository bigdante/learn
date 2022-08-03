import torchvision
from tensorboardX import SummaryWriter
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10("./dataset", train=True, download=True, transform=data_transform)
test_set = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=data_transform)
img, label = test_set[0]
img.show()

writer = SummaryWriter("p10")
for i in range(10):
    img, label = test_set[i]
    writer.add_image("test_set", img, i)

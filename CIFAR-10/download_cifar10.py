import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",      # 数据保存位置
    train=True,         # 训练集
    download=True,      # 自动下载
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,        # 测试集
    download=True,
    transform=transform
)

print("训练集数量:", len(train_dataset))
print("测试集数量:", len(test_dataset))
print("类别:", train_dataset.classes)
print("数据形状:", train_dataset.data.shape)
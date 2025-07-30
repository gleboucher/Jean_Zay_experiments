import torchvision.datasets as datasets

# Root directory for datasets
root = "./data"

# Download torchvision datasets
mnist = datasets.MNIST(root, train=True, download=True)
fashion_mnist = datasets.FashionMNIST(root, train=True, download=True)
kmnist = datasets.KMNIST(root, train=True, download=True)
emnist = datasets.EMNIST(root, split='balanced', train=True, download=True)
cifar10 = datasets.CIFAR10(root, train=True, download=True)
cifar100 = datasets.CIFAR100(root, train=True, download=True)
svhn = datasets.SVHN(root, split='train', download=True)
qmnist = datasets.QMNIST(root, train=True, download=True)

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.datasets import make_blobs, make_classification
import requests
import os
import tarfile
import zipfile
from PIL import Image
import h5py
import pennylane as qml
from pennylane import numpy as np

class DatasetLoader:
    """
    Comprehensive dataset loader for all datasets mentioned in the memo.tex file.
    Includes classical ML datasets, graph datasets, and QML datasets.
    """
    
    def __init__(self, root_dir="./data", download=True):
        self.root_dir = root_dir
        self.download = download
        os.makedirs(root_dir, exist_ok=True)
    
    # Classical ML Datasets (Computer Vision)
    def load_mnist(self, train=True, transform=None):
        """28x28 handwritten digits (0-9), 60k train/10k test"""
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return datasets.MNIST(self.root_dir, train=train, download=self.download, transform=transform)
    
    def load_fashion_mnist(self, train=True, transform=None):
        """28x28 clothing items, 60k train/10k test"""
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        return datasets.FashionMNIST(self.root_dir, train=train, download=self.download, transform=transform)
    
    def load_kmnist(self, train=True, transform=None):
        """28x28 cursive Japanese characters, 60k train/10k test"""
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1918,), (0.3483,))])
        return datasets.KMNIST(self.root_dir, train=train, download=self.download, transform=transform)
    
    def load_emnist(self, split='balanced', train=True, transform=None):
        """28x28 extended MNIST with digits and letters, 112.8k train/18.8k test"""
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,))])
        return datasets.EMNIST(self.root_dir, split=split, train=train, download=self.download, transform=transform)
    
    def load_cifar10(self, train=True, transform=None):
        """32x32 RGB everyday objects, 50k train/10k test"""
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        return datasets.CIFAR10(self.root_dir, train=train, download=self.download, transform=transform)
    
    def load_cifar100(self, train=True, transform=None):
        """32x32 RGB fine-grained objects, 50k train/10k test"""
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        return datasets.CIFAR100(self.root_dir, train=train, download=self.download, transform=transform)
    
    def load_svhn(self, split='train', transform=None):
        """32x32 RGB street view house numbers, 73k train/26k test"""
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])
        return datasets.SVHN(self.root_dir, split=split, download=self.download, transform=transform)
    
    def load_qmnist(self, train=True, transform=None):
        """28x28 extended MNIST with metadata, 60k train/10k test"""
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return datasets.QMNIST(self.root_dir, train=train, download=self.download, transform=transform)
    
    def load_omniglot(self, background=True, transform=None):
        """105x105 handwritten characters from 50+ alphabets"""
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        return datasets.Omniglot(self.root_dir, background=background, download=self.download, transform=transform)
    
    def load_stl10(self, split='train', transform=None):
        """96x96 RGB similar to CIFAR, 5k train/8k test"""
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
            ])
        return datasets.STL10(self.root_dir, split=split, download=self.download, transform=transform)
    
    def load_celeba(self, split='train', transform=None):
        """178x218 RGB celebrity faces with attributes, 200k+ images"""
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        return datasets.CelebA(self.root_dir, split=split, download=self.download, transform=transform)
    
    def load_lfw(self, split='train', transform=None):
        """Face classification dataset, 13k images"""
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        return datasets.LFWPeople(self.root_dir, split=split, download=self.download, transform=transform)
    
    def load_caltech101(self, transform=None):
        """Object categories with moderate intra-class variation, 7k+ images"""
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        return datasets.Caltech101(self.root_dir, download=self.download, transform=transform)
    
    def load_caltech256(self, transform=None):
        """Extended Caltech-101 with more categories, 30k+ images"""
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        return datasets.Caltech256(self.root_dir, download=self.download, transform=transform)
    
    def load_flowers102(self, split='train', transform=None):
        """Oxford flowers dataset, 8k images, 102 classes"""
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        return datasets.Flowers102(self.root_dir, split=split, download=self.download, transform=transform)
    
    # QML Datasets (Synthetic)
    def load_binary_blobs(self, n_samples=1000, n_features=2, centers=2, random_state=42):
        """Synthetic binary classification dataset with blob clusters"""
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, 
                         cluster_std=1.0, random_state=random_state)
        return torch.FloatTensor(X), torch.LongTensor(y)
    
    def load_hyperplanes(self, n_samples=1000, n_features=2, n_classes=2, random_state=42):
        """Synthetic dataset with hyperplane separation"""
        X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                 n_classes=n_classes, n_redundant=0, n_informative=n_features,
                                 random_state=random_state)
        return torch.FloatTensor(X), torch.LongTensor(y)
    
    def load_linearly_separable(self, n_samples=1000, n_features=2, random_state=42):
        """Simple linearly separable dataset"""
        X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                 n_classes=2, n_redundant=0, n_informative=n_features,
                                 n_clusters_per_class=1, random_state=random_state)
        return torch.FloatTensor(X), torch.LongTensor(y)
    
    def load_ising_dataset(self, n_samples=1000, lattice_size=4, temperature=2.0, random_state=42):
        """Ising model samples for binary/multiclass classification"""
        np.random.seed(random_state)
        
        # Generate Ising configurations
        samples = []
        labels = []
        
        for _ in range(n_samples):
            # Generate random spin configuration
            spins = np.random.choice([-1, 1], size=(lattice_size, lattice_size))
            
            # Calculate energy (simplified nearest neighbor interaction)
            energy = 0
            for i in range(lattice_size):
                for j in range(lattice_size):
                    # Periodic boundary conditions
                    neighbors = [
                        spins[(i+1) % lattice_size, j],
                        spins[i, (j+1) % lattice_size],
                        spins[(i-1) % lattice_size, j],
                        spins[i, (j-1) % lattice_size]
                    ]
                    energy -= spins[i, j] * sum(neighbors)
            
            # Binary classification based on energy
            label = 1 if energy > 0 else 0
            
            samples.append(spins.flatten())
            labels.append(label)
        
        return torch.FloatTensor(samples), torch.LongTensor(labels)
    
    def load_custom_boson_sampler_dataset(self, n_samples=1000, n_modes=4, n_photons=2, random_state=42):
        """Custom boson sampler dataset for quantum ML"""
        np.random.seed(random_state)
        
        # Generate input Fock states (labels)
        labels = []
        samples = []
        
        for _ in range(n_samples):
            # Generate random input state
            input_state = np.zeros(n_modes)
            photon_positions = np.random.choice(n_modes, n_photons, replace=True)
            for pos in photon_positions:
                input_state[pos] += 1
            
            # Simulate boson sampling output (simplified)
            # In practice, this would use actual quantum simulation
            output_state = np.random.multinomial(n_photons, np.ones(n_modes)/n_modes)
            
            samples.append(output_state)
            labels.append(tuple(input_state.astype(int)))
        
        # Convert labels to class indices
        unique_labels = list(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y = [label_to_idx[label] for label in labels]
        
        return torch.FloatTensor(samples), torch.LongTensor(y)
    
    # Graph Datasets (simplified implementations)
    def load_cora(self):
        """Cora citation network for node classification"""
        # Note: In practice, you'd use PyTorch Geometric or DGL
        # This is a placeholder implementation
        print("For graph datasets like Cora, use PyTorch Geometric:")
        print("from torch_geometric.datasets import Planetoid")
        print("dataset = Planetoid(root='./data', name='Cora')")
        return None
    
    def load_citeseer(self):
        """CiteSeer citation network for node classification"""
        print("For graph datasets like CiteSeer, use PyTorch Geometric:")
        print("from torch_geometric.datasets import Planetoid")
        print("dataset = Planetoid(root='./data', name='CiteSeer')")
        return None
    
    def load_pubmed(self):
        """PubMed citation network for node classification"""
        print("For graph datasets like PubMed, use PyTorch Geometric:")
        print("from torch_geometric.datasets import Planetoid")
        print("dataset = Planetoid(root='./data', name='PubMed')")
        return None
    
    def get_dataset_info(self, dataset_name):
        """Get information about a specific dataset"""
        info = {
            'mnist': {'size': '28x28', 'classes': 10, 'train_size': 60000, 'test_size': 10000},
            'fashion_mnist': {'size': '28x28', 'classes': 10, 'train_size': 60000, 'test_size': 10000},
            'kmnist': {'size': '28x28', 'classes': 10, 'train_size': 60000, 'test_size': 10000},
            'emnist': {'size': '28x28', 'classes': 47, 'train_size': 112800, 'test_size': 18800},
            'cifar10': {'size': '32x32x3', 'classes': 10, 'train_size': 50000, 'test_size': 10000},
            'cifar100': {'size': '32x32x3', 'classes': 100, 'train_size': 50000, 'test_size': 10000},
            'svhn': {'size': '32x32x3', 'classes': 10, 'train_size': 73257, 'test_size': 26032},
            'stl10': {'size': '96x96x3', 'classes': 10, 'train_size': 5000, 'test_size': 8000},
            'celeba': {'size': '178x218x3', 'classes': 'N/A', 'train_size': 162770, 'test_size': 19962},
            'caltech101': {'size': 'varied', 'classes': 101, 'train_size': 7000, 'test_size': 'varied'},
            'caltech256': {'size': 'varied', 'classes': 256, 'train_size': 30000, 'test_size': 'varied'},
            'flowers102': {'size': 'varied', 'classes': 102, 'train_size': 8000, 'test_size': 'varied'},
        }
        return info.get(dataset_name.lower(), 'Dataset not found')
    
    def list_available_datasets(self):
        """List all available datasets"""
        datasets = {
            'Classical Vision': [
                'mnist', 'fashion_mnist', 'kmnist', 'emnist', 'cifar10', 'cifar100',
                'svhn', 'qmnist', 'omniglot', 'stl10', 'celeba', 'lfw',
                'caltech101', 'caltech256', 'flowers102'
            ],
            'QML Synthetic': [
                'binary_blobs', 'hyperplanes', 'linearly_separable', 'ising_dataset',
                'custom_boson_sampler_dataset'
            ],
            'Graph': [
                'cora', 'citeseer', 'pubmed', 'reddit', 'ogbn_arxiv', 'ogbg_molhiv'
            ]
        }
        return datasets

# Example usage
if __name__ == "__main__":
    loader = DatasetLoader()
    
    # Load a vision dataset
    mnist_train = loader.load_mnist(train=True)
    mnist_test = loader.load_mnist(train=False)
    
    # Load a QML dataset
    blobs_X, blobs_y = loader.load_binary_blobs(n_samples=1000)
    
    # Load custom boson sampler dataset
    boson_X, boson_y = loader.load_custom_boson_sampler_dataset(n_samples=500)
    
    # Get dataset info
    print("MNIST info:", loader.get_dataset_info('mnist'))
    
    # List all available datasets
    print("Available datasets:", loader.list_available_datasets())
    
    print(f"MNIST train samples: {len(mnist_train)}")
    print(f"Binary blobs shape: {blobs_X.shape}")
    print(f"Boson sampler shape: {boson_X.shape}")
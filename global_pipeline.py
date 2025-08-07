import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import numpy as np
import time
from typing import Dict, Tuple, Any
from pathlib import Path


from Architectures.hybrid_architectures import get_architecture


class EMNISTLettersAdjusted(datasets.EMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __getitem__(self, index):
        # Récupération image + label d'origine
        img, target = super().__getitem__(index)

        # Corriger le label (de 1-26 → 0-25)
        target = target - 1

        return img, target


class MNISTFromNPZ(Dataset):
    def __init__(self, npz_path, train=True, transform=None):
        data = np.load(npz_path)
        self.transform = transform
        if train:
            self.images = data['x_train']
            self.labels = data['y_train']
        else:
            self.images = data['x_test']
            self.labels = data['y_test']

        # Ensure proper shape and type
        self.images = self.images.astype(np.float32) / 255.0  # Normalize to [0,1]
        if self.images.ndim == 3:
            self.images = np.expand_dims(self.images, 1)  # Add channel dimension (1, 28, 28)

        self.labels = self.labels.astype(np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


class DatasetLoader:
    """
    Handles loading and preprocessing of different image datasets
    """
    
    @staticmethod
    def get_transform(dataset_name: str, is_training: bool = True) -> transforms.Compose:
        """Get appropriate transforms for each dataset"""
        base_transforms = []
        
        if dataset_name.lower() in ['mnist', 'fashion-mnist', 'kmnist', 'emnist', 'qmnist']:
            # Grayscale datasets
            if is_training:
                base_transforms = [
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
                ]
            else:
                base_transforms = [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
        
        elif dataset_name.lower() in ['cifar10', 'cifar100']:
            # RGB datasets
            if is_training:
                base_transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]
            else:
                base_transforms = [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]
        
        elif dataset_name.lower() == 'svhn':
            # RGB SVHN
            if is_training:
                base_transforms = [
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
                ]
            else:
                base_transforms = [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
                ]
        
        else:
            # Generic RGB transforms
            if is_training:
                base_transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ]
            else:
                base_transforms = [
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ]
        
        return transforms.Compose(base_transforms)
    
    @staticmethod
    def load_dataset(dataset_name: str, data_root: str = './data') -> Tuple[DataLoader, DataLoader, Tuple, int]:
        """Load specified dataset and return train/test loaders"""
        dataset_name = dataset_name.lower()
        
        train_transform = DatasetLoader.get_transform(dataset_name, is_training=True)
        test_transform = DatasetLoader.get_transform(dataset_name, is_training=False)

        
        if dataset_name == 'mnist':
            npz_path =  os.environ['DSDIR'] + '/MNIST/mnist.npz'
            train_dataset = MNISTFromNPZ(npz_path, train=True)
            test_dataset = MNISTFromNPZ(npz_path, train=False)
            input_shape = (1, 28, 28)
            num_classes = 10
            
        elif dataset_name == 'emnist':
            train_dataset = EMNISTLettersAdjusted(
                root="../datasets", split='letters', train=True, download=False, transform=train_transform
            )
            test_dataset = EMNISTLettersAdjusted(
                root="../datasets", split='letters', train=False, download=False, transform=test_transform
            )
            input_shape = (1, 28, 28)
            num_classes = 26

        elif dataset_name == 'kmnist':
            train_dataset = datasets.KMNIST(
                root="../datasets",  train=True, download=False, transform=train_transform
            )
            test_dataset = datasets.KMNIST(
                root="../datasets", train=False, download=False, transform=test_transform
            )
            input_shape = (1, 28, 28)
            num_classes = 10
            
        elif dataset_name == 'cifar10':
            cifar_path = os.environ['DSDIR'] + '/CIFAR-10-images'
            train_dir = cifar_path + '/train'
            test_dir = cifar_path + '/test'
            train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
            test_dataset = datasets.ImageFolder(root=test_dir, transform=train_transform)
            input_shape = (3, 32, 32)
            num_classes = 10
            
        elif dataset_name == 'cifar100':
            train_dataset = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR100(data_root, train=False, transform=test_transform)
            input_shape = (3, 32, 32)
            num_classes = 100
            
        elif dataset_name == 'svhn':
            train_dataset = datasets.SVHN(data_root, split='train', download=True, transform=train_transform)
            test_dataset = datasets.SVHN(data_root, split='test', download=True, transform=test_transform)
            input_shape = (3, 32, 32)
            num_classes = 10
            
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        return train_dataset, test_dataset, input_shape, num_classes


class HybridTrainer:
    """
    Training pipeline for hybrid architectures
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def fit_pca_components(self, model: nn.Module, train_loader: DataLoader):
        """Fit PCA components for architectures that require it"""
        if hasattr(model, 'pca') or hasattr(model, 'scaler'):
            print("Fitting PCA components...")
            model.eval()
            
            # Collect features for PCA fitting
            features = []
            torch_features = []
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(train_loader):
                    if batch_idx > 50:  # Use subset for PCA fitting
                        break
                    data = data.to(self.device)
                    batch_size = data.size(0)
                    data_flat = data.view(batch_size, -1)
                    torch_features.append(data_flat)
                    features.append(data_flat.cpu().numpy())
            
            features = np.vstack(features)
            torch_features = torch.cat(torch_features, dim=0)
            # Fit scaler and PCA
            if hasattr(model, 'scaler'):
                model.scaler.fit(features)
                features_scaled = model.scaler.transform(features)
            if hasattr(model, 'scaler_torch'):
                model.scaler.fit(torch_features)
                features_scaled = model.scaler.transform(features)
            else:
                features_scaled = features
            
            if hasattr(model, 'pca'):
                model.pca = PCA(n_components=model.pca_components)
                model.pca.fit(features_scaled)
                print(f"PCA fitted with {model.pca_components} components")
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        return avg_loss, accuracy
    
    def evaluate(self, model: nn.Module, test_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate model"""
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_loss /= len(test_loader)
        accuracy = 100. * correct / total
        return test_loss, accuracy
    
    def train_model(self, architecture_name: str, dataset_name: str, 
                   hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Train a specific architecture on a dataset"""
        print(f"\nTraining {architecture_name} on {dataset_name}")
        print(f"Hyperparameters: {hyperparams}")
        
        # Load dataset
        train_dataset, test_dataset, input_shape, num_classes = DatasetLoader.load_dataset(
            dataset_name, self.config['data_root']
        )

        
        train_loader = DataLoader(
            train_dataset, batch_size=hyperparams['batch_size'], 
            shuffle=True, num_workers=self.config.get('num_workers', 2)
        )
        test_loader = DataLoader(
            test_dataset, batch_size=hyperparams['batch_size'], 
            shuffle=False, num_workers=self.config.get('num_workers', 2)
        )

        # Create model
        model_config = {
            'dropout_rate': hyperparams['dropout_rate'],

        }

        model = get_architecture(architecture_name, input_shape, num_classes, model_config)
        model = model.to(self.device)
        if model.quantum is not None:
            model.quantum.to(self.device)


        # Fit PCA if needed

        self.fit_pca_components(model, train_loader)

        # Setup optimizer and loss
        if hyperparams['optimizer'] == 'adam':
            optimizer = optim.Adam(
                model.parameters(), 
                lr=hyperparams['learning_rate'],
                weight_decay=1e-4
            )
        elif hyperparams['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                model.parameters(), 
                lr=hyperparams['learning_rate'],
                weight_decay=1e-2,
                momentum=0.9
            )
        else:  # rmsprop
            optimizer = optim.RMSprop(
                model.parameters(), 
                lr=hyperparams['learning_rate'],
                weight_decay=5e-4
            )

        
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # Training loop
        best_accuracy = 0.0
        results = {
            'architecture': architecture_name,
            'dataset': dataset_name,
            'hyperparams': hyperparams,
            'train_losses': [],
            'train_accuracies': [],
            'test_losses': [],
            'test_accuracies': [],
            'best_accuracy': 0.0,
            'training_time': 0.0
        }
        
        start_time = time.time()
        test_loss, test_acc = self.evaluate(model, test_loader, criterion)
        print(f"Starting training, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {trainable_params}")

        for epoch in range(self.config['num_epochs']):

            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Evaluate
            test_loss, test_acc = self.evaluate(model, test_loader, criterion)
            
            # Record results
            results['train_losses'].append(train_loss)
            results['train_accuracies'].append(train_acc)
            results['test_losses'].append(test_loss)
            results['test_accuracies'].append(test_acc)
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                results['best_accuracy'] = best_accuracy
            
            scheduler.step()
            

            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        results['training_time'] = time.time() - start_time
        print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
        
        return results


def run_experiments(config: Dict[str, Any]):
    """Run experiments across architectures and datasets"""
    trainer = HybridTrainer(config)
    
    architectures = config['architectures']
    datasets = config['datasets']
    
    all_results = []
    
    for arch_name in architectures:
        for dataset_name in datasets:
            # Use default hyperparameters or sample from ranges
            hyperparams = {
                'learning_rate': config.get('learning_rate', 1e-3),
                'optimizer': config.get('optimizer', 'adam'),
                'batch_size': config.get('batch_size', 64),
                'dropout_rate': config.get('dropout_rate', 0.2),
                'weight_decay': config.get('weight_decay', 1e-4)
            }

            #try:
            results = trainer.train_model(arch_name, dataset_name, hyperparams)
            all_results.append(results)

            # Save intermediate results
            output_dir = Path(config['output_dir'])
            output_dir.mkdir(exist_ok=True)

            result_file = output_dir / f"{arch_name}_{dataset_name}_results.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
                    
            #except Exception as e:
             #   print(f"Error training {arch_name} on {dataset_name}: {str(e)}")
              #  continue
    
    # Save all results
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    
    for result in all_results:
        print(f"{result['architecture']} on {result['dataset']}: "
              f"{result['best_accuracy']:.2f}% (Training time: {result['training_time']:.1f}s)")


import argparse
import json
import os
from itertools import product
from copy import deepcopy



def generate_grid_configs(base_config, grid_params):
    keys = list(grid_params.keys())
    values = list(product(*grid_params.values()))

    for combination in values:
        config = deepcopy(base_config)
        for key, val in zip(keys, combination):
            config[key] = val
        yield config


def main():
    parser = argparse.ArgumentParser(description='Hybrid Architecture Pipeline')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Base configuration file path')
    parser.add_argument('--grid_config', type=str, default='grid_config.json',
                        help='Grid search configuration file path (for hyperparameter search)')
    parser.add_argument('--gridsearch', action='store_true',
                        help='Enable grid search over hyperparameters')
    parser.add_argument('--architectures', nargs='+',
                        default=['boson_preprocessor_mlp', 'cnn_boson_mlp'],
                        help='Architectures to test')
    parser.add_argument('--datasets', nargs='+',
                        default=['mnist', 'cifar10'],
                        help='Datasets to test')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')

    args = parser.parse_args()

    # Default base config
    base_config = {
        'architectures': args.architectures,
        'datasets': args.datasets,
        'num_epochs': args.epochs,
        'output_dir': args.output_dir,
        'data_root': './data',
        'num_workers': 2,
        'learning_rate': 1e-3,
        'optimizer': 'adam',
        'batch_size': 64,
        'network_depth': 3,
        'dropout_rate': 0.2,
        'weight_decay': 1e-4
    }

    # Load base config if it exists
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        base_config.update(file_config)

    if args.gridsearch:
        if not os.path.exists(args.grid_config):
            raise FileNotFoundError(f"Grid config file not found: {args.grid_config}")

        with open(args.grid_config, 'r') as f:
            grid_params = json.load(f)

        print("Running Grid Search on:")
        print(json.dumps(grid_params, indent=2))

        for i, config in enumerate(generate_grid_configs(base_config, grid_params)):
            print(f"\nRunning configuration {i + 1}")
            run_experiments(config)
    else:
        print("Running single experiment with configuration:")
        print(json.dumps(base_config, indent=2))
        run_experiments(base_config)


if __name__ == "__main__":
    main()

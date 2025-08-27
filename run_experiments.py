#!/usr/bin/env python3
"""
Example script to run hybrid architecture experiments
"""
import numpy as np
from distutils.command.config import config

import torch
import json
import argparse
from pathlib import Path

from global_pipeline import run_experiments, HybridTrainer, DatasetLoader
from Architectures.hybrid_architectures import get_architecture


def quick_test():
    """Quick test to verify everything works"""
    print("Running quick test...")
    
    # Test dataset loading
    try:
        train_dataset, test_dataset, input_shape, num_classes = DatasetLoader.load_dataset('mnist')
        print(f"✓ Dataset loading works: {input_shape}, {num_classes} classes")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return
    
    # Test architecture creation
    try:
        model = get_architecture('boson_preprocessor_mlp', input_shape, num_classes)
        print(f"✓ Architecture creation works: {type(model).__name__}")
    except Exception as e:
        print(f"✗ Architecture creation failed: {e}")
        return
    
    # Test forward pass
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        dummy_input = torch.randn(2, *input_shape).to(device)
        output = model(dummy_input)
        print(f"✓ Forward pass works: output shape {output.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return
    
    print("All tests passed! ✓")


def run_small_experiment(gpu, arch, dataset):
    """Run a small experiment on MNIST with one architecture"""
    batch_size = 32
    if gpu == "v100":
        batch_size = 32
    elif gpu == "a100":
        batch_size = 256
    elif gpu == "h100":
        batch_size = 256

    arch = [arch]
    dataset = [dataset]
    config = {
        'architectures': arch,
        'datasets': dataset,
        'num_epochs': 20,  # Quick test
        'output_dir': './test_results',
        'data_root': './data',
        'num_workers': 0,  # Avoid multiprocessing issues
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'optimizer': 'adam',
        'batch_size': batch_size,
        'network_depth': 2,
        'dropout_rate': 0.2,
        'max_modes': 20,
        'n_photons': 3,
        'output_strategy': None,
        'output_size': None,
    }
    
    print("Running small experiment...")
    run_experiments(config)


def run_full_experiment():
    """Run full experiments from config file"""
    config_file = 'config.json'
    
    if not Path(config_file).exists():
        print(f"Configuration file {config_file} not found!")
        return
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Use default hyperparameters
    config.update(config['default_hyperparams'])
    
    print("Running full experiment with configuration:")
    print(json.dumps(config, indent=2))
    
    run_experiments(config)


def hyperparameter_search(gpu, arch, dataset, batch_size):
    """Run hyperparameter search on a subset"""
    import itertools
    import random
    
    # Define search space (reduced for efficiency)
    if batch_size is None:
        if gpu == "v100":
            batch_size = 32
        elif gpu == "a100":
            batch_size = 128
        elif gpu == "h100":
            batch_size = 256
    batch_size = int(batch_size)

    search_space = {
        'learning_rate': [1e-3],
        'optimizer': ['adam'],
        'batch_size': [batch_size],
        'dropout_rate': [0.15],
        'max_modes': [32],
        'n_photons': [3],
        'output_strategy':[None],
        'output_size': [32]
    }
    
    # Generate all combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))
    
    # Sample a subset for efficiency
    max_trials = 18
    if len(combinations) > max_trials:
        combinations = random.sample(combinations, max_trials)
    
    print(f"Running hyperparameter search with {len(combinations)} combinations...")
    
    trainer = HybridTrainer({
        'num_epochs': 15,
        'data_root': './data',
        'num_workers': 0
    })
    
    all_results = []
    rand_int = np.random.randint(low=0, high=10000)
    filename = f'./hyperparameter_search/results_{arch}_{dataset}_{rand_int}.json'

    for i, combo in enumerate(combinations):
        hyperparams = dict(zip(keys, combo))
        hyperparams['network_depth'] = 3  # Fixed
        
        print(f"\nTrial {i+1}/{len(combinations)}: {hyperparams}")
        
        try:
            results = trainer.train_model(arch, dataset, hyperparams)
            all_results.append(results)
            Path('./hyperparameter_search').mkdir(exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(all_results, f, indent=2)
        except Exception as e:
            print(f"Trial failed: {e}")
            continue
    
    # Find best configuration
    if all_results:
        best_result = max(all_results, key=lambda x: x['best_accuracy'])
        print(f"\nBest configuration:")
        print(f"Hyperparams: {best_result['hyperparams']}")
        print(f"Best accuracy: {best_result['best_accuracy']:.2f}%")
        

def print_architecture_info():
    """Print information about available architectures"""
    architectures = [
        'boson_preprocessor_mlp',
        'cnn_boson_mlp', 
        'boson_decoder',
        'boson_layer_nn',
        'dual_path_cnn_boson',
        'boson_pca_classifier',
        'variational_boson_ae'
    ]
    
    print("Available architectures:")
    print("=" * 50)
    
    for arch in architectures:
        try:
            # Create a dummy model to get info
            model = get_architecture(arch, (1, 28, 28), 10)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"{arch:25} | {param_count:>8} parameters")
        except Exception as e:
            print(f"{arch:25} | Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run hybrid architecture experiments')
    parser.add_argument('--mode', choices=['test', 'small', 'full', 'hypersearch', 'info'],
                       default='test', help='Experiment mode')
    parser.add_argument('--gpu', choices=['v100', 'a100', 'h100'],
                        default='test', help='Gpu choice mode')
    parser.add_argument('--arch', choices=['boson_preprocessor_mlp', 'cnn_boson_mlp',
                                           'boson_layer_nn', 'dual_path_cnn_boson', 'quantum_vit',
                                           'variational_boson_ae', 'boson_decoder', 'classical_vit',
                                           'classical_cnn', 'dual_path_vit_vba'],
                        help='Architecture mode')
    parser.add_argument('--dataset', choices=['mnist', 'emnist', 'kmnist', 'cifar10'],
                        default='cifar10',)
    parser.add_argument('--batch_size',
                        default=None)

    args = parser.parse_args()
    
    if args.mode == 'test':
        quick_test()
    elif args.mode == 'small':
        run_small_experiment(args.gpu, args.arch, args.dataset)
    elif args.mode == 'full':
        run_full_experiment()
    elif args.mode == 'hypersearch':
        hyperparameter_search(args.gpu, args.arch, args.dataset, args.batch_size)
    elif args.mode == 'info':
        print_architecture_info()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Example script to run hybrid architecture experiments
"""

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


def run_small_experiment():
    """Run a small experiment on MNIST with one architecture"""
    config = {
        'architectures': ['cnn_boson_mlp', 'boson_layer_nn'],
        'datasets': ['mnist'],
        'num_epochs': 10,  # Quick test
        'output_dir': './test_results',
        'data_root': './data',
        'num_workers': 0,  # Avoid multiprocessing issues
        'learning_rate': 1e-3,
        'optimizer': 'adam',
        'batch_size': 64,
        'network_depth': 2,
        'dropout_rate': 0.1,
        'weight_decay': 1e-4
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


def hyperparameter_search():
    """Run hyperparameter search on a subset"""
    import itertools
    import random
    
    # Define search space (reduced for efficiency)
    search_space = {
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'optimizer': ['adam', 'sgd'],
        'batch_size': [32, 64],
        'dropout_rate': [0.1, 0.2],
        'weight_decay': [1e-5, 1e-4]
    }
    
    # Generate all combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))
    
    # Sample a subset for efficiency
    max_trials = 10
    if len(combinations) > max_trials:
        combinations = random.sample(combinations, max_trials)
    
    print(f"Running hyperparameter search with {len(combinations)} combinations...")
    
    trainer = HybridTrainer({
        'num_epochs': 20,
        'data_root': './data',
        'num_workers': 0
    })
    
    all_results = []
    
    for i, combo in enumerate(combinations):
        hyperparams = dict(zip(keys, combo))
        hyperparams['network_depth'] = 3  # Fixed
        
        print(f"\nTrial {i+1}/{len(combinations)}: {hyperparams}")
        
        try:
            results = trainer.train_model('boson_preprocessor_mlp', 'mnist', hyperparams)
            all_results.append(results)
        except Exception as e:
            print(f"Trial failed: {e}")
            continue
    
    # Find best configuration
    if all_results:
        best_result = max(all_results, key=lambda x: x['best_accuracy'])
        print(f"\nBest configuration:")
        print(f"Hyperparams: {best_result['hyperparams']}")
        print(f"Best accuracy: {best_result['best_accuracy']:.2f}%")
        
        # Save results
        Path('./hyperparameter_search').mkdir(exist_ok=True)
        with open('./hyperparameter_search/results.json', 'w') as f:
            json.dump(all_results, f, indent=2)


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
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        quick_test()
    elif args.mode == 'small':
        run_small_experiment()
    elif args.mode == 'full':
        run_full_experiment()
    elif args.mode == 'hypersearch':
        hyperparameter_search()
    elif args.mode == 'info':
        print_architecture_info()


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple, Optional
import warnings
from merlin import QuantumLayer, OutputMappingStrategy
from Boson_samplers





class Architecture1_BosonPreprocessor_MLP(nn.Module):
    """
    Data → Boson Sampler → Histogram → PCA → MLP
    Modified: Data → Normalization → Linear → PCA → MLP
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = [256, 128], 
                 pca_components: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.boson_replacement = QuantumLayer(
                    input_size=input_dim,
                    output_size=None,
                    circuit=circuit,
                    n_photons=N,
                    input_state=input_state,# Random Initial quantum state used only for initialization
                    output_mapping_strategy=OutputMappingStrategy.NONE,
                    input_parameters=["phi"],# Optional: Specify device
                    trainable_parameters=[],
                    shots=1000,  # Optional: Enable quantum measurement sampling
                    no_bunching=True,
                    sampling_method='multinomial', # Optional: Specify sampling method
                )(input_dim, hidden_dims[0])
        self.pca_components = pca_components
        self.pca = None  # Will be fitted during training
        
        # MLP after PCA
        mlp_layers = []
        prev_dim = pca_components
        for hidden_dim in hidden_dims[1:]:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*mlp_layers)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.input_norm(x)
        x = self.boson_replacement(x)
        
        # Apply PCA (in eval mode or after fitting)
        if self.pca is not None:
            x_np = x.detach().cpu().numpy()
            x_pca = self.pca.transform(x_np)
            x = torch.tensor(x_pca, dtype=torch.float32, device=x.device)
        
        return self.mlp(x)


class Architecture2_CNN_Boson_MLP(nn.Module):
    """
    Image → CNN → Boson Sampler → Flatten → MLP
    Modified: Image → Normalization → CNN → Linear → Flatten → MLP
    """
    def __init__(self, input_channels: int, num_classes: int, cnn_channels: List[int] = [32, 64],
                 mlp_hidden: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(input_channels)
        
        # CNN layers
        cnn_layers = []
        prev_channels = input_channels
        for channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(prev_channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            prev_channels = channels
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate CNN output size (assuming input is square)
        self.cnn_output_size = None
        
        # Boson sampler replacement and MLP will be defined after first forward pass
        self.boson_replacement = None
        self.mlp = None
        self.mlp_hidden = mlp_hidden
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
    def forward(self, x):
        x = self.input_norm(x)
        x = self.cnn(x)
        
        # Initialize boson replacement and MLP on first forward pass
        if self.boson_replacement is None:
            batch_size = x.size(0)
            self.cnn_output_size = x.numel() // batch_size
            self.boson_replacement = BosonSamplerReplacement(
                self.cnn_output_size, self.mlp_hidden, dropout_rate=self.dropout_rate
            ).to(x.device)
            self.mlp = nn.Sequential(
                nn.BatchNorm1d(self.mlp_hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.mlp_hidden, self.num_classes)
            ).to(x.device)
        
        x = x.view(x.size(0), -1)
        x = self.boson_replacement(x)
        return self.mlp(x)


class Architecture3_Boson_Decoder(nn.Module):
    """
    Data → Boson Sampler → Latent Vector → Decoder (CNN/MLP)
    Modified: Data → Normalization → Linear → Latent Vector → Decoder
    """
    def __init__(self, input_dim: int, num_classes: int, latent_dim: int = 64,
                 decoder_hidden: List[int] = [128, 256], dropout_rate: float = 0.2):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.boson_replacement = BosonSamplerReplacement(input_dim, latent_dim, dropout_rate=dropout_rate)
        
        # Decoder MLP
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in decoder_hidden:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, num_classes))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.input_norm(x)
        x = self.boson_replacement(x)
        return self.decoder(x)


class Architecture4_Boson_Layer_NN(nn.Module):
    """
    Input → Dense → Boson Sampler → Dense → Output
    Modified: Input → Normalization → Dense → Linear → Dense → Output
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128,
                 boson_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.dense1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.boson_replacement = BosonSamplerReplacement(hidden_dim, boson_dim, dropout_rate=dropout_rate)
        self.dense2 = nn.Sequential(
            nn.BatchNorm1d(boson_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(boson_dim, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.input_norm(x)
        x = self.dense1(x)
        x = self.boson_replacement(x)
        return self.dense2(x)


class Architecture5_DualPath_CNN_Boson(nn.Module):
    """
    Image → [CNN // Boson Sampler] → Concatenation → MLP
    Modified: Image → Normalization → [CNN // Linear] → Concatenation → MLP
    """
    def __init__(self, input_channels: int, num_classes: int, cnn_channels: List[int] = [32, 64],
                 boson_hidden: int = 128, mlp_hidden: int = 256, dropout_rate: float = 0.2):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(input_channels)
        
        # CNN path
        cnn_layers = []
        prev_channels = input_channels
        for channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(prev_channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            prev_channels = channels
        self.cnn_path = nn.Sequential(*cnn_layers)
        
        # Boson path (operates on flattened input)
        self.boson_path_init = None
        self.boson_replacement = None
        
        # MLP for concatenated features
        self.mlp = None
        self.boson_hidden = boson_hidden
        self.mlp_hidden = mlp_hidden
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
    def forward(self, x):
        batch_size = x.size(0)
        normalized_x = self.input_norm(x)
        
        # CNN path
        cnn_features = self.cnn_path(normalized_x)
        cnn_flat = cnn_features.view(batch_size, -1)
        
        # Boson path (linear transformation of flattened input)
        input_flat = normalized_x.view(batch_size, -1)
        if self.boson_replacement is None:
            input_dim = input_flat.size(1)
            self.boson_replacement = BosonSamplerReplacement(
                input_dim, self.boson_hidden, dropout_rate=self.dropout_rate
            ).to(x.device)
            
            # Initialize MLP after knowing feature dimensions
            concat_dim = cnn_flat.size(1) + self.boson_hidden
            self.mlp = nn.Sequential(
                nn.Linear(concat_dim, self.mlp_hidden),
                nn.BatchNorm1d(self.mlp_hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.mlp_hidden, self.num_classes)
            ).to(x.device)
        
        boson_features = self.boson_replacement(input_flat)
        
        # Concatenate and classify
        combined_features = torch.cat([cnn_flat, boson_features], dim=1)
        return self.mlp(combined_features)


class Architecture7_Boson_PCA_Classifier(nn.Module):
    """
    Image → PCA → Boson Sampler → Histogram → Classifier
    Modified: Image → Normalization → PCA → Linear → Classifier
    """
    def __init__(self, input_dim: int, num_classes: int, pca_components: int = 64,
                 hidden_dims: List[int] = [128, 64], dropout_rate: float = 0.2):
        super().__init__()
        self.pca_components = pca_components
        self.pca = None
        self.scaler = StandardScaler()
        
        self.boson_replacement = BosonSamplerReplacement(
            pca_components, hidden_dims[0], dropout_rate=dropout_rate
        )
        
        # Classifier
        classifier_layers = []
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            classifier_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        classifier_layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Apply PCA if fitted
        if self.pca is not None:
            x_np = x.detach().cpu().numpy()
            x_scaled = self.scaler.transform(x_np)
            x_pca = self.pca.transform(x_scaled)
            x = torch.tensor(x_pca, dtype=torch.float32, device=x.device)
        else:
            # If PCA not fitted, use input as-is (for initialization)
            warnings.warn("PCA not fitted yet, using raw input")
            
        x = self.boson_replacement(x)
        return self.classifier(x)


class Architecture8_Variational_Boson_Autoencoder(nn.Module):
    """
    Input → Encoder (NN) → Boson Sampler (latent space) → Decoder (NN)
    Modified: Input → Normalization → Encoder → Linear (latent) → Decoder
    """
    def __init__(self, input_dim: int, num_classes: int, latent_dim: int = 64,
                 encoder_hidden: List[int] = [256, 128], decoder_hidden: List[int] = [128, 256],
                 dropout_rate: float = 0.2):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in encoder_hidden:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space (Boson sampler replacement)
        self.boson_replacement = BosonSamplerReplacement(
            prev_dim, latent_dim, dropout_rate=dropout_rate
        )
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in decoder_hidden:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, num_classes))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.input_norm(x)
        encoded = self.encoder(x)
        latent = self.boson_replacement(encoded)
        return self.decoder(latent)


# Hyperparameter configurations from the memo
HYPERPARAMETERS = {
    'learning_rates': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    'optimizers': ['adam', 'sgd', 'rmsprop'],
    'batch_sizes': [16, 32, 64, 128],
    'network_depths': [2, 3, 4, 5, 6],
    'dropout_rates': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    'output_mapping': [OutputMappingStrategy.NONE, OutputMappingStrategy.LINEAR, OutputMappingStrategy.LEXGROUPING]
}


def get_architecture(arch_name: str, input_shape: Tuple[int, ...], num_classes: int, 
                    config: Dict[str, Any] = None) -> nn.Module:
    """
    Factory function to create architecture instances
    """
    if config is None:
        config = {}
        
    if len(input_shape) == 3:  # Image input (C, H, W)
        input_channels, height, width = input_shape
        input_dim = input_channels * height * width
    else:  # Flattened input
        input_dim = input_shape[0]
        input_channels = input_shape[0] if len(input_shape) == 1 else input_shape[0]
    
    architectures = {
        'boson_preprocessor_mlp': lambda: Architecture1_BosonPreprocessor_MLP(
            input_dim, num_classes, **config
        ),
        'cnn_boson_mlp': lambda: Architecture2_CNN_Boson_MLP(
            input_channels, num_classes, **config
        ),
        'boson_decoder': lambda: Architecture3_Boson_Decoder(
            input_dim, num_classes, **config
        ),
        'boson_layer_nn': lambda: Architecture4_Boson_Layer_NN(
            input_dim, num_classes, **config
        ),
        'dual_path_cnn_boson': lambda: Architecture5_DualPath_CNN_Boson(
            input_channels, num_classes, **config
        ),
        'boson_pca_classifier': lambda: Architecture7_Boson_PCA_Classifier(
            input_dim, num_classes, **config
        ),
        'variational_boson_ae': lambda: Architecture8_Variational_Boson_Autoencoder(
            input_dim, num_classes, **config
        )
    }
    
    if arch_name not in architectures:
        raise ValueError(f"Unknown architecture: {arch_name}")
        
    return architectures[arch_name]()
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
from Architectures.Boson_samplers import BosonSampler
import perceval as pcvl

class MinMaxNorm1d(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-8):
        """

        :rtype: torch.tensor
        """
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Running stats (like in BatchNorm)
        self.register_buffer('running_min', torch.zeros(1, num_features))
        self.register_buffer('running_max', torch.ones(1, num_features))

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input (batch, features), got {x.shape}")

        if self.training:
            batch_min = x.min(dim=0, keepdim=True).values
            batch_max = x.max(dim=0, keepdim=True).values

            # Update running stats
            self.running_min = (1 - self.momentum) * self.running_min + self.momentum * batch_min
            self.running_max = (1 - self.momentum) * self.running_max + self.momentum * batch_max

            min_val = batch_min
            max_val = batch_max
        else:
            # Use running stats at inference
            min_val = self.running_min
            max_val = self.running_max

        # Min-max normalize
        out = (x - min_val) / (max_val - min_val + self.eps)
        out = out.clamp(0.0, 1.0)
        return out

def create_quantum_circuit(input_size, n_photons, max_modes=20):
    # 1. Left interferometer - trainable transformation

    k = input_size // max_modes
    if k == 0:
        num_modes = input_size
        last_layer = input_size
    else:
        num_modes = max_modes
        last_layer = input_size % num_modes
    input_state = [1] * n_photons + [0] * (num_modes - n_photons)

    print("number of modes", num_modes, "number of reps", k, "input_size", input_size)
    wl = pcvl.GenericInterferometer(
        num_modes,
        lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_li{i}")) //
                 pcvl.BS() // pcvl.PS(pcvl.P(f"theta_lo{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE
    )
    circuit = wl
    for j in range(k):
        # 2. Input encoding - maps classical data to quantum parameters
        c_var = pcvl.Circuit(num_modes)
        for i in range(num_modes):  # 4 input features
            px = pcvl.P(f"px{i}_{j}")
            c_var.add(i, pcvl.PS(px))
        w_enc = pcvl.Circuit(num_modes)
        for i in range(0, num_modes, 2):
            w_enc = w_enc // (i, pcvl.BS()) // (i, pcvl.PS(pcvl.P(f"theta_ri{i}_{j}")))

        # 3. Right interferometer - trainable transformation


        circuit = circuit // c_var // w_enc
    if last_layer > 0:
        c_var = pcvl.Circuit(num_modes)
        for i in range(last_layer):  # 4 input features
            px = pcvl.P(f"px{i}_{k}")
            c_var.add(i, pcvl.PS(px))
        circuit = circuit // c_var
        # 3. Right interferometer - trainable transformation

    wr = pcvl.GenericInterferometer(
        num_modes,
        lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ri{i}_{k}")) //
                  pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ro{i}_{k}")),
        shape=pcvl.InterferometerShape.RECTANGLE
    )
    circuit = circuit // wr

    # Combine all components
    return circuit, input_state

def map_output_strategy(output_strategy):
    if output_strategy is None:
        return OutputMappingStrategy.NONE
    if output_strategy == "lexgrouping":
        return OutputMappingStrategy.LEXGROUPING
    if output_strategy == "modgrouping":
        return OutputMappingStrategy.MODGROUPING
    if output_strategy == "linear":
        return OutputMappingStrategy.LINEAR
    else:
        raise ValueError(f"Unknown output strategy {output_strategy}")


class Architecture1_BosonPreprocessor_MLP(nn.Module):
    """
    Data → Boson Sampler → Histogram → PCA → MLP
    Modified: Data → Normalization → Linear → PCA → MLP
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dims=None,
                 pca_components: int = 16, dropout_rate: float = 0.2,
                 network_depth=None, n_photons=3, max_modes=20,
                 output_strategy=None, output_size=None):
        super().__init__()
        if output_strategy is None:
            output_size = None
        output_strategy = map_output_strategy(output_strategy)
        if hidden_dims is None:
            hidden_dims = [128]
            if network_depth is not None:
                hidden_dims *=network_depth

        self.input_norm = nn.BatchNorm1d(input_dim)
        circuit, input_state = create_quantum_circuit(pca_components, n_photons, max_modes)
        self.quantum_norm = MinMaxNorm1d(pca_components)

        self.quantum = QuantumLayer(
                    input_size=pca_components,
                    output_size=output_size,
                    circuit=circuit,
                    input_state=input_state,# Random Initial quantum state used only for initialization
                    output_mapping_strategy=output_strategy,
                    input_parameters=["px"],
                    trainable_parameters=["theta"],
                    no_bunching=True,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                )

        self.pca_components = pca_components
        self.pca = None  # Will be fitted during training
        
        # MLP after PCA
        mlp_layers = []
        prev_dim = self.quantum.output_size

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
        # Apply PCA (in eval mode or after fitting)
        if self.pca is not None:
            x_np = x.detach().cpu().numpy()
            x_pca = self.pca.transform(x_np)
            x = torch.tensor(x_pca, dtype=torch.float32, device=x.device)
        x = self.quantum_norm(x)
        x = self.quantum(x)
        return self.mlp(x)


class Architecture2_CNN_Boson_MLP(nn.Module):
    """
    Image → CNN → Boson Sampler → Flatten → MLP
    Modified: Image → Normalization → CNN → Linear → Flatten → MLP
    """
    def __init__(self, input_channels: int, num_classes: int, cnn_channels=None,
                 hidden_dims=None, dropout_rate: float = 0.1, n_photons=3,
                 network_depth=None, boson_modes=20,
                 output_strategy=None, output_size=None):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 32]
        if hidden_dims is None:
            hidden_dims = [128]
            if network_depth is not None:
                hidden_dims *=network_depth
        self.input_norm = nn.BatchNorm2d(input_channels)
        self.n_photons = n_photons
        if output_strategy is None:
            output_size = None
        self.output_size = output_size
        self.output_strategy = map_output_strategy(output_strategy)
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
        self.quantum = None
        self.mlp = None
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.boson_modes = boson_modes


    def forward(self, x):
        x = self.input_norm(x)
        x = self.cnn(x)
        # Initialize boson replacement and MLP on first forward pass
        if self.quantum is None:
            batch_size = x.size(0)
            self.cnn_output_size = x.numel() // batch_size

            circuit, input_state= create_quantum_circuit(self.cnn_output_size, self.n_photons, self.boson_modes)



            self.quantum_norm = MinMaxNorm1d(self.cnn_output_size).to(x.device)

            self.quantum = QuantumLayer(
                input_size=self.cnn_output_size,
                output_size=self.output_size,
                circuit=circuit,
                input_state=input_state,  # Random Initial quantum state used only for initialization
                output_mapping_strategy=self.output_strategy,
                input_parameters=["px"],
                trainable_parameters=["theta"],
                no_bunching=True,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ).to(x.device)
            print("Quantum Layer defined with input size:", self.cnn_output_size)
            mlp_layers = []
            prev_dim = self.quantum.output_size
            for hidden_dim in self.hidden_dims[1:]:
                mlp_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate)
                ])
                prev_dim = hidden_dim
            mlp_layers.append(nn.Linear(prev_dim, self.num_classes))
            self.mlp = nn.Sequential(*mlp_layers).to(x.device)

        x = x.view(x.size(0), -1)
        x = self.quantum_norm(x)
        x = self.quantum(x)
        return self.mlp(x)


class Architecture3_Boson_Decoder(nn.Module):
    """
    Data → Boson Sampler → Latent Vector → Decoder (CNN/MLP)
    Modified: Data → Normalization → Linear → Latent Vector → Decoder
    """
    def __init__(self, input_dim: int, num_classes: int, latent_dim: int = 64,
                 decoder_hidden: List[int] = [128, 256], dropout_rate: float = 0.2,
                 n_photons=3, max_modes=20, output_strategy=None, output_size=None):
        super().__init__()
        self.quantum_norm = MinMaxNorm1d(input_dim)
        circuit, input_state = create_quantum_circuit(input_dim, n_photons, max_modes)
        if output_strategy is None:
            output_size = None
        output_strategy = map_output_strategy(output_strategy)
        self.quantum = QuantumLayer(
            input_size=input_dim,
            output_size=output_size,
            circuit=circuit,
            input_state=input_state,  # Random Initial quantum state used only for initialization
            output_mapping_strategy=output_strategy,
            input_parameters=["px"],
            trainable_parameters=["theta"],
            no_bunching=True,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Decoder MLP
        decoder_layers = []
        prev_dim = self.quantum.output_size
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
        x = self.quantum_norm(x)
        x = self.quantum(x)
        return self.decoder(x)


class Architecture4_Boson_Layer_NN(nn.Module):
    """
    Input → Dense → Boson Sampler → Dense → Output
    Modified: Input → Normalization → Dense → Linear → Dense → Output
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dims=None,
                 dropout_rate: float = 0.2,
                 n_photons=3, network_depth=None, max_modes=20,
                 output_strategy="lexgrouping", output_size=64):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16]
            if network_depth is not None:
                hidden_dims *= network_depth
        if output_strategy is None:
            output_size = None
        output_strategy = map_output_strategy(output_strategy)
        self.input_norm = nn.BatchNorm1d(input_dim)
        mlp_layers = []
        prev_dim = input_dim
        n_dims = len(hidden_dims)
        for hidden_dim in hidden_dims[:n_dims-1]:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        self.dense1 = nn.Sequential(*mlp_layers)

        circuit, input_state = create_quantum_circuit(hidden_dims[-1], n_photons, max_modes)

        self.quantum = QuantumLayer(
            input_size=hidden_dims[-1],
            output_size=output_size,
            circuit=circuit,
            input_state=input_state,  # Random Initial quantum state used only for initialization
            output_mapping_strategy=output_strategy,
            input_parameters=["px"],
            trainable_parameters=["theta"],
            no_bunching=True,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        print(f"Quantum Layer definer with input size {hidden_dims[-1]} and output size {output_size} ")
        if output_size is None:
            output_size = self.quantum.output_size
        self.quantum_norm = MinMaxNorm1d(hidden_dims[-1])
        self.dense2 = nn.Sequential(
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_size, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.input_norm(x)
        x = self.dense1(x)
        x = self.quantum_norm(x)
        x = self.quantum(x)
        return self.dense2(x)


class Architecture5_DualPath_CNN_Boson(nn.Module):
    """
    Image → [CNN // Boson Sampler] → Concatenation → MLP
    Modified: Image → Normalization → [CNN // Linear] → Concatenation → MLP
    """
    def __init__(self, input_channels: int, num_classes: int, cnn_channels: List[int] = [32, 64],
                 output_size: int = 64, mlp_hidden: int = 256, dropout_rate: float = 0.2,
                 n_photons=3, max_modes=20, output_strategy="lexgrouping"):
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
        self.quantum = None
        self.n_photons = n_photons
        self.max_modes = max_modes
        # MLP for concatenated features
        self.mlp = None

        if output_strategy is None:
            output_size = None
        self.output_strategy = map_output_strategy(output_strategy)
        self.boson_hidden = output_size
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
        if self.quantum is None:
            input_dim = input_flat.size(1)
            self.quantum_norm = MinMaxNorm1d(input_dim)
            circuit, input_state = create_quantum_circuit(input_dim, self.n_photons, self.max_modes)
            self.quantum = QuantumLayer(
                input_size=input_dim,
                output_size=self.boson_hidden,
                circuit=circuit,
                input_state=input_state,  # Random Initial quantum state used only for initialization
                output_mapping_strategy=self.output_strategy,
                input_parameters=["px"],
                trainable_parameters=["theta"],
                no_bunching=True,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            if self.boson_hidden is None:
                self.boson_hidden = self.quantum.output_size
            # Initialize MLP after knowing feature dimensions
            concat_dim = cnn_flat.size(1) + self.boson_hidden
            self.mlp = nn.Sequential(
                nn.Linear(concat_dim, self.mlp_hidden),
                nn.BatchNorm1d(self.mlp_hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.mlp_hidden, self.num_classes)
            ).to(x.device)
        
        boson_features = self.quantum(input_flat)
        
        # Concatenate and classify
        combined_features = torch.cat([cnn_flat, boson_features], dim=1)
        return self.mlp(combined_features)



class Architecture8_Variational_Boson_Autoencoder(nn.Module):
    """
    Input → Encoder (NN) → Boson Sampler (latent space) → Decoder (NN)
    Modified: Input → Normalization → Encoder → Linear (latent) → Decoder
    """
    def __init__(self, input_dim: int, num_classes: int, latent_dim: int = 64,
                 encoder_hidden: List[int] = [256, 128, 64, 32], decoder_hidden: List[int] = [32, 64, 128, 256],
                 dropout_rate: float = 0.2, n_photons=3, max_modes=20, output_strategy=None, output_size=None):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        if output_strategy is None:
            output_size = None
        output_strategy = map_output_strategy(output_strategy)
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
        print("max modes", max_modes)
        circuit, input_state = create_quantum_circuit(prev_dim, n_photons, max_modes)


        self.quantum_norm = MinMaxNorm1d(prev_dim)
        self.quantum = QuantumLayer(
            input_size=prev_dim,
            output_size=output_size,
            circuit=circuit,
            input_state=input_state,  # Random Initial quantum state used only for initialization
            output_mapping_strategy=output_strategy,
            input_parameters=["px"],
            trainable_parameters=["theta"],
            no_bunching=True,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Decoder
        decoder_layers = []
        prev_dim = self.quantum.output_size
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
        encoded = self.quantum_norm(encoded)
        latent = self.quantum(encoded)
        return self.decoder(latent)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)  # [B, embed_dim, N]
        x = x.transpose(1, 2)  # [B, N, embed_dim]
        return x


# ======================================
# Transformer Encoder Block
# ======================================
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self Attention
        x_res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x_res + x  # Residual

        # MLP
        x_res = x
        x = self.norm2(x)
        x = x_res + self.mlp(x)
        return x


# ======================================
# Vision Transformer
# ======================================
class QuantumVisionTransformer(nn.Module):
    def __init__(self, input_size=32, num_classes=10, patch_size=4, in_chans=3, embed_dim=64,
                 depth=6, num_heads=8, mlp_ratio=4.0, dropout_rate=0.2, n_photons=3, max_modes=20,
                 output_strategy=None, output_size=None):
        super().__init__()
        print(input_size, in_chans)
        if output_strategy is None:
            output_size = None
        output_strategy = map_output_strategy(output_strategy)

        self.patch_embed = PatchEmbedding(input_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_rate)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout_rate)
            for _ in range(depth)
        ])

        circuit, input_state = create_quantum_circuit(embed_dim, n_photons, max_modes)
        self.quantum_norm = MinMaxNorm1d(embed_dim)
        self.quantum = QuantumLayer(
            input_size=embed_dim,
            output_size=output_size,
            circuit=circuit,
            input_state=input_state,  # Random Initial quantum state used only for initialization
            output_mapping_strategy=output_strategy,
            input_parameters=["px"],
            trainable_parameters=["theta"],
            no_bunching=True,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.norm = nn.LayerNorm(self.quantum.output_size)
        self.head = nn.Linear(self.quantum.output_size, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.quantum_norm(x[:, 0])
        x = self.quantum(x)
        x = self.norm(x)
        return self.head(x)


class ClassicalVisionTransformer(nn.Module):
    def __init__(self, input_size=32, num_classes=10, patch_size=4, in_chans=3, embed_dim=64,
                 depth=6, num_heads=8, mlp_ratio=4.0, dropout_rate=0.2, n_photons=3, max_modes=20,
                 output_strategy=None, output_size=None):
        super().__init__()
        print(input_size, in_chans)
        self.quantum = None
        self.patch_embed = PatchEmbedding(input_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_rate)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])

class CompactCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        # Conv layers
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),  # 32x28x28 (or 32x32x32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x7x7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128x7x7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 128x1x1
        )
        self.quantum = None
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Hyperparameter configurations from the memo
HYPERPARAMETERS = {
    'learning_rates': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    'optimizers': ['adam', 'sgd', 'rmsprop'],
    'batch_sizes': [16, 32, 64, 128],
    'network_depths': [2, 3, 4, 5, 6],
    'dropout_rates': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
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
        ),
        'quantum_vit': lambda: QuantumVisionTransformer(input_shape[1], num_classes, in_chans=input_channels, **config),

        'classical_cnn': lambda: CompactCNN(input_channels, num_classes),

        'classical_vit': lambda: ClassicalVisionTransformer(input_shape[1], num_classes, in_chans=input_channels, **config),

    }
    
    if arch_name not in architectures:
        raise ValueError(f"Unknown architecture: {arch_name}")

    return architectures[arch_name]()
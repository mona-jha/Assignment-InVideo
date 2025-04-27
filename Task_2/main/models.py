import torch
import torch.nn as nn
from transformers import Dinov2Model

class DinoEmbedder(nn.Module):
    """
    DINOV2-based embedder with optional fine-tuning.
    Takes an input image and outputs a compact embedding vector.
    """
    def __init__(self, projection_dim=512, finetune_backbone=True):
        super().__init__()
        # Load pre-trained DINOV2 small model
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-small")
        
        # Optionally freeze backbone if fine-tuning not required
        for param in self.backbone.parameters():
            param.requires_grad = finetune_backbone
            
        # Project final DINOV2 features (384 dim) to desired output dim (e.g., 512)
        self.projection = nn.Linear(384, projection_dim)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = {"pixel_values": x}  # Dinov2Model expects dict input
        feats = self.backbone(**x).last_hidden_state.mean(dim=1)  # Global average pooling
        return self.projection(feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device=None):
        state_dict = torch.load(path, map_location=device if device else 'cpu')
        self.load_state_dict(state_dict)
        return self

class ResBlock(nn.Module):
    """
    Simple residual block with 2 convolutional layers.
    Helps the generator model learn better features.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class EmbeddingToImageGenerator(nn.Module):
    """
    Generator model that takes embedding vectors and reconstructs 128x128 images.
    Progressive upsampling using residual blocks.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 512 * 8 * 8)  # Fully connected layer to project to 8x8x512

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 8x8 -> 16x16
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(256),

            nn.Upsample(scale_factor=2),  # 16x16 -> 32x32
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(128),

            nn.Upsample(scale_factor=2),  # 32x32 -> 64x64
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),  # 64x64 -> 128x128
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output range [-1, 1] for image
        )

    def forward(self, x):
        x = self.fc(x).view(-1, 512, 8, 8)  # Reshape to 8x8 feature map
        return self.decoder(x)

# Instantiate generator easily from anywhere
Generator = EmbeddingToImageGenerator()

import torch
import torch.nn as nn
from transformers import Dinov2Model

class DinoEmbedder(nn.Module):
    """
    DINOV2-based face embedder model with fine-tuning support
    """
    def __init__(self, projection_dim=512, finetune_backbone=True):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-small")
        
        # Enable or disable fine-tuning of the backbone
        for param in self.backbone.parameters():
            param.requires_grad = finetune_backbone
            
        self.projection = nn.Linear(384, projection_dim)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = {"pixel_values": x}
        feats = self.backbone(**x).last_hidden_state[:, 0]
        return self.projection(feats)
    
    def save(self, path):
        """Save model weights to path"""
        torch.save(self.state_dict(), path)
        
    def load(self, path, device=None):
        """Load model weights from path"""
        state_dict = torch.load(path, map_location=device if device else 'cpu')
        self.load_state_dict(state_dict)
        return self

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),           
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),           
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class EmbeddingToImageGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 512 * 8 * 8)
       self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2),        # 8x8 → 16x16
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.Dropout(0.2),
                nn.ReLU(inplace=True),
                ResBlock(256),
            
                nn.Upsample(scale_factor=2),        # 16x16 → 32x32
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.Dropout(0.2),
                nn.ReLU(inplace=True),
                ResBlock(128),
            
                nn.Upsample(scale_factor=2),        # 32x32 → 64x64
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.Dropout(0.2),
                nn.ReLU(inplace=True),
            
                nn.Upsample(scale_factor=2),        # 64x64 → 128x128   (Added Step)
                nn.Conv2d(64, 32, 3, 1, 1),         # 64 channels → 32 channels
                nn.BatchNorm2d(32),                 
                nn.Dropout(0.2),
                nn.ReLU(inplace=True),
            
                nn.Conv2d(32, 3, 3, 1, 1),          # Output 3 channels RGB
                nn.Tanh()
            )


    def forward(self, x):
        x = self.fc(x).view(-1, 512, 8, 8)
        return self.decoder(x)

# For backward compatibility
Generator = EmbeddingToImageGenerator

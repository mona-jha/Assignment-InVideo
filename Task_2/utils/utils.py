import torch
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VGG16 for perceptual loss
vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval().to(device)
for param in vgg.parameters():
    param.requires_grad = False

def perceptual_loss(generated, target):
    return F.l1_loss(vgg(generated), vgg(target))

def save_model(model, filename, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, filename))
    print(f"💾 Model saved at {os.path.join(save_dir, filename)}")

def setup_device():
    """Set up and return the device (CPU/GPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def save_embeddings(embeddings, output_dir, prefix="embedding"):
    """Save embeddings to disk"""
    os.makedirs(output_dir, exist_ok=True)
    for i, embedding in enumerate(embeddings):
        torch.save(embedding.cpu(), os.path.join(output_dir, f"{prefix}_{i}.pt"))
    print(f"✅ Saved {len(embeddings)} embeddings to {output_dir}")

def visualize_embeddings(embeddings, labels=None, output_path=None):
    """
    Visualize embeddings using PCA
    
    Args:
        embeddings: List of embedding tensors
        labels: Optional list of labels for each embedding
        output_path: Optional path to save the visualization
    """
    # Convert embeddings to numpy array
    embeddings_np = torch.stack(embeddings).cpu().numpy()
    
    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_np)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                          c=np.arange(len(embeddings)) if labels is None else labels, 
                          cmap='viridis', 
                          alpha=0.8)
    
    # Add labels if provided
    if labels is not None:
        plt.colorbar(scatter, label='Label')
    
    plt.title('PCA Visualization of Face Embeddings')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    plt.show()
    
    return embeddings_2d, pca
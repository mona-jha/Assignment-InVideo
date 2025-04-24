import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from main.models import DinoEmbedder, EmbeddingToImageGenerator
from utils.datasets import EmbeddingImageDataset

def embedding_image_collate(batch):
    """
    Custom collate function for EmbeddingImageDataset.
    Properly handles the (embedding, image) pairs.
    """
    embeddings = torch.stack([item[0] for item in batch])
    images = torch.stack([item[1] for item in batch])
    return embeddings, images

def parse_args():
    parser = argparse.ArgumentParser(description="Face Generation Inference")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input image folder')
    parser.add_argument('--output_dir', type=str, default='face_generation_results', help='Output directory')
    parser.add_argument('--embedder_path', type=str, required=True, help='Path to the embedder model')
    parser.add_argument('--generator_path', type=str, required=True, help='Path to the generator model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--cache_dir', type=str, default='embedding_cache', help='Directory to cache embeddings')
    parser.add_argument('--comparison', action='store_true', help='Generate side-by-side comparisons')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run inference on')
    
    return parser.parse_args()

def load_models(embedder_path, generator_path, device):
    """Load embedder and generator models"""
    print("\nLoading models...")
    
    # Load embedder
    embedder = DinoEmbedder().to(device)
    embedder.load_state_dict(torch.load(embedder_path, map_location=device))
    embedder.eval()
    print("✅ DINOV2 embedder loaded")
    
    # Load generator
    generator = EmbeddingToImageGenerator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    print("✅ Generator loaded")
    
    return embedder, generator

def create_comparison_grid(real_images, generated_images, output_path):
    """Create a grid of real vs generated images"""
    batch_size = min(len(real_images), len(generated_images))
    rows = min(batch_size, 5)  # Display at most 5 rows
    
    plt.figure(figsize=(10, 2*rows))
    for i in range(rows):
        # Convert tensors to numpy arrays for display
        real_img = real_images[i].cpu().permute(1, 2, 0).numpy()
        gen_img = generated_images[i].cpu().permute(1, 2, 0).numpy()
        
        # Denormalize from [-1, 1] to [0, 1]
        real_img = np.clip((real_img + 1) / 2, 0, 1)
        gen_img = np.clip((gen_img + 1) / 2, 0, 1)
        
        # Plot real image
        plt.subplot(rows, 2, 2*i+1)
        plt.imshow(real_img)
        plt.title(f"Original {i+1}")
        plt.axis('off')
        
        # Plot generated image
        plt.subplot(rows, 2, 2*i+2)
        plt.imshow(gen_img)
        plt.title(f"Generated {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Comparison grid saved to {output_path}")

def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Load models
    device = torch.device(args.device)
    embedder, generator = load_models(args.embedder_path, args.generator_path, device)
    
    # Create dataset with the embedded images
    print(f"\nCreating dataset from images in {args.input_dir}...")
    dataset = EmbeddingImageDataset(
        root=args.input_dir,
        embedder=embedder,
        transform_size=128,
        device=device,
        cache_dir=args.cache_dir
    )
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=embedding_image_collate
    )
    
    # Generate images
    print("\nGenerating face images...")
    all_real_images = []
    all_generated_images = []
    
    with torch.no_grad():
        for batch_idx, (embeddings, real_images) in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Move data to device
            embeddings = embeddings.to(device)
            real_images = real_images.to(device)
            
            # Generate images from embeddings
            generated_images = generator(embeddings)
            
            # Save each generated image
            for i, generated_img in enumerate(generated_images):
                idx = batch_idx * args.batch_size + i
                # Normalize to [0,1] for saving
                normalized_img = (generated_img.cpu().clamp(-1, 1) + 1) / 2
                save_image(normalized_img, os.path.join(args.output_dir, f"generated_{idx:04d}.png"))
                
                # Store images for comparison if needed
                if args.comparison:
                    all_real_images.append(real_images[i])
                    all_generated_images.append(generated_images[i])
    
    # Create comparison grid if requested
    if args.comparison and all_real_images:
        print("\nCreating comparison visualization...")
        create_comparison_grid(
            all_real_images, 
            all_generated_images, 
            os.path.join(args.output_dir, "comparison_grid.png")
        )
    
    print(f"\n✨ Done! Generated {len(dataset)} face images in {args.output_dir}")

if __name__ == "__main__":
    main()

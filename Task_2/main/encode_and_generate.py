"""
Demonstration script for the encoding and generation pipeline:
1. Load a pre-fine-tuned DINOV2 model
2. Encode face images into embeddings
3. Feed embeddings to the generator to reconstruct faces
"""

import os
import torch
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

from main.models import DinoEmbedder, EmbeddingToImageGenerator

def parse_args():
    parser = argparse.ArgumentParser(description="Encode Images and Generate Faces")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing input face images")
    parser.add_argument("--output_dir", type=str, default="encoded_generated_results",
                        help="Directory to save results")
    parser.add_argument("--embedder_path", type=str, required=True,
                        help="Path to fine-tuned DINOV2 embedder model")
    parser.add_argument("--generator_path", type=str, required=True,
                        help="Path to GAN generator model")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to process")
    parser.add_argument("--save_embeddings", action="store_true",
                        help="Save the embeddings to disk")
    return parser.parse_args()

def encode_image(img_path, encoder, device):
    """Encode a single image using the DINOV2 encoder"""
    # Transform image for encoder
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    # Load and transform image
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Extract embedding
    with torch.no_grad():
        embedding = encoder(img_tensor)
    
    return embedding, img

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_embeddings:
        os.makedirs(os.path.join(args.output_dir, "embeddings"), exist_ok=True)
    
    # Step 1: Load the DINOV2 encoder (fine-tuned)
    print("\n1️ Loading DINOV2 encoder model...")
    encoder = DinoEmbedder().to(device)
    encoder.load_state_dict(torch.load(args.embedder_path, map_location=device))
    encoder.eval()
    print(" DINOV2 encoder loaded successfully")
    
    # Step 2: Load the generator
    print("\n2️ Loading image generator model...")
    generator = EmbeddingToImageGenerator().to(device)
    generator.load_state_dict(torch.load(args.generator_path, map_location=device))
    generator.eval()
    print(" Generator loaded successfully")
    
    # Step 3: Find input images
    all_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    selected_files = all_files[:args.num_samples] if args.num_samples > 0 else all_files
    selected_paths = [os.path.join(args.input_dir, f) for f in selected_files]
    
    print(f"\n3️ Found {len(selected_paths)} images to process")
    
    # Step 4: Process each image - encode and generate
    print("\n4️ Encoding images and generating reconstructions...")
    results = []
    
    for i, img_path in enumerate(tqdm(selected_paths)):
        # Step 4a: Encode the image using DINOV2
        embedding, original_img = encode_image(img_path, encoder, device)
        
        # Save embedding if requested
        if args.save_embeddings:
            torch.save(embedding.cpu(), os.path.join(args.output_dir, "embeddings", f"embedding_{i}.pt"))
        
        # Step 4b: Generate face image from the embedding using GAN
        with torch.no_grad():
            generated_img = generator(embedding)
            # Convert from [-1, 1] to [0, 1] range for visualization
            generated_img = (generated_img.clamp(-1, 1) + 1) / 2
        
        # Step 4c: Save the results
        # Save original image
        original_img.save(os.path.join(args.output_dir, f"original_{i}.png"))
        
        # Save generated image
        save_image(generated_img, os.path.join(args.output_dir, f"generated_{i}.png"))
        
        # Create side-by-side comparison
        original_tensor = transforms.ToTensor()(original_img.resize((128, 128)))
        comparison = torch.cat([original_tensor, generated_img.cpu().squeeze(0)], dim=2)
        save_image(comparison, os.path.join(args.output_dir, f"comparison_{i}.png"))
        
        # Store for visualization
        results.append({
            'original': original_img,
            'generated': generated_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
        })
    
    # Step 5: Visualize the results
    if len(results) > 0:
        n_samples = min(len(results), 5)  # Show up to 5 samples
        plt.figure(figsize=(12, 2*n_samples))
        
        for i in range(n_samples):
            # Show original
            plt.subplot(n_samples, 2, i*2+1)
            plt.imshow(results[i]['original'])
            plt.title(f"Original Image {i+1}")
            plt.axis('off')
            
            # Show generated
            plt.subplot(n_samples, 2, i*2+2)
            plt.imshow(results[i]['generated'])
            plt.title(f"Generated from Embedding {i+1}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "visualization.png"))
        plt.show()
    
    print(f"\n Processing complete! Results saved to: {args.output_dir}")
    print("Summary of steps performed:")
    print("1. Loaded fine-tuned DINOV2 encoder model")
    print("2. Loaded GAN generator model")
    print("3. For each input image:")
    print("   a. Encoded the image into an embedding vector using DINOV2")
    print("   b. Fed the embedding to the GAN generator to produce a reconstructed face")
    print("   c. Saved original, generated, and comparison images")

if __name__ == "__main__":
    main()

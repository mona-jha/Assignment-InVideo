import os
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from main.models import DinoEmbedder, EmbeddingToImageGenerator
from utils.datasets import EmbeddingImageDataset
from utils.evaluation import generate_and_compare_samples

def embedding_image_collate(batch):
    """Custom collate function for EmbeddingImageDataset"""
    embeddings = torch.stack([item[0] for item in batch])
    images = torch.stack([item[1] for item in batch])
    return embeddings, images

def run_inference(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "embedding_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    
    # Load embedding model
    embedder = DinoEmbedder().to(device)
    embedder.load_state_dict(torch.load(args.embedder_path, map_location=device))
    embedder.eval()
    print(f"✓ Loaded embedder from {args.embedder_path}")
    
    # Load generator model
    generator = EmbeddingToImageGenerator().to(device)
    generator.load_state_dict(torch.load(args.generator_path, map_location=device))
    generator.eval()
    print(f"✓ Loaded generator from {args.generator_path}")
    
    # Create dataset
    print(f"Creating dataset from {args.data_dir}...")
    dataset = EmbeddingImageDataset(
        root=args.data_dir,
        embedder=embedder,
        transform_size=128,
        device=device,
        cache_dir=cache_dir
    )
    
    # Limit samples if needed
    if args.num_samples > 0 and args.num_samples < len(dataset):
        indices = list(range(args.num_samples))
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Limited to {args.num_samples} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=embedding_image_collate
    )
    
    # Generate images
    print("Generating face images...")
    # Remove the break condition to process all samples
    generate_and_compare_samples(generator, dataloader, args.output_dir, remove_limit=True)
    
    print(f"✓ Generation complete! Results saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Face Generation Inference")
    parser.add_argument("--mode", type=str, default="inference", 
                        help="Mode to run (always 'inference' for this script)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing test images")
    parser.add_argument("--embedder_path", type=str, required=True,
                        help="Path to the saved embedder model")
    parser.add_argument("--generator_path", type=str, required=True,
                        help="Path to the saved generator model")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save generated images")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of samples to process (default: all)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    run_inference(args)

if __name__ == "__main__":
    main()

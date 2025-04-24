"""
Main script for face generation pipeline
"""

import os
import argparse
import torch
from tqdm import tqdm

from main.models import DinoEmbedder, EmbeddingToImageGenerator
from utils.datasets import FaceDataset, EmbeddingImageDataset, create_dataloaders
from utils.training import train_embedder
from utils.evaluation import generate_images, test_zero_shot_generalization
from .utils import setup_device

def parse_args():
    parser = argparse.ArgumentParser(description="Face Generation Pipeline")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["train_embedder", "train_gan", "generate", "evaluate", "zero_shot"],
                        help="Mode to run")
    parser.add_argument("--data_dir", type=str, default="faces_dataset",
                        help="Directory containing face images")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save outputs")
    parser.add_argument("--embedder_path", type=str, default="finetuned_dinov2_faceembedder.pth",
                        help="Path to fine-tuned embedder model")
    parser.add_argument("--generator_path", type=str, default="gan_generator.pth",
                        help="Path to generator model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate")
    parser.add_argument("--finetune_backbone", action="store_true",
                        help="Fine-tune DINOV2 backbone (not just projection head)")
    parser.add_argument("--max_train_time", type=float, default=6.0,
                        help="Maximum training time in hours")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples for generation/evaluation")
    parser.add_argument("--use_cache", action="store_true",
                        help="Cache embeddings to disk for faster training")
    return parser.parse_args()

def main():
    args = parse_args()
    device = setup_device()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "train_embedder":
        print("===== Fine-tuning DINOV2 Embedder =====")
        train_embedder(
            data_dir=args.data_dir,
            output_path=args.embedder_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            finetune_backbone=args.finetune_backbone
        )
        
    elif args.mode == "train_gan":
        print("===== Training Image Generator =====")
        # First, load the fine-tuned embedder
        print(f"Loading fine-tuned embedder from {args.embedder_path}")
        embedder = DinoEmbedder().to(device)
        embedder.load_state_dict(torch.load(args.embedder_path, map_location=device))
        embedder.eval()
        print("✅ Embedder loaded successfully")
        
        # Then train the generator
        train_gan(
            data_dir=args.data_dir,
            embedder_path=args.embedder_path,
            output_dir=os.path.join(args.output_dir, "checkpoints"),
            generator_path=args.generator_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            max_train_time_hours=args.max_train_time,
            device=device,
            use_embedding_cache=args.use_cache
        )
        
    elif args.mode == "generate":
        print("===== Generating Face Images =====")
        # Load the fine-tuned embedder
        print(f"Loading fine-tuned embedder from {args.embedder_path}")
        embedder = DinoEmbedder().to(device)
        embedder.load_state_dict(torch.load(args.embedder_path, map_location=device))
        embedder.eval()
        
        # Get test images
        test_dir = args.data_dir
        all_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                     if f.lower().endswith(('.jpg','.png'))]
        test_paths = all_files[:args.num_samples]
        
        # Generate images
        generate_images(
            model_path=args.generator_path,
            embedder=embedder,
            image_paths=test_paths,
            output_dir=os.path.join(args.output_dir, "generated_images"),
            device=device
        )
        
    elif args.mode == "zero_shot":
        print("===== Testing Zero-Shot Generalization =====")
        # Load the fine-tuned embedder
        print(f"Loading fine-tuned embedder from {args.embedder_path}")
        embedder = DinoEmbedder().to(device)
        embedder.load_state_dict(torch.load(args.embedder_path, map_location=device))
        embedder.eval()
        
        # Test zero-shot generalization
        test_zero_shot_generalization(
            model_path=args.generator_path,
            embedder=embedder,
            test_dir=args.data_dir,
            output_dir=os.path.join(args.output_dir, "zero_shot_results"),
            num_samples=args.num_samples,
            device=device
        )
    
    print(f"✅ Task '{args.mode}' completed successfully!")

if __name__ == "__main__":
    main()



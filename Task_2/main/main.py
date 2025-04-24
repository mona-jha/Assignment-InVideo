 

import os
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from main.models import DinoEmbedder, EmbeddingToImageGenerator
from utils.datasets import FaceDataset, EmbeddingImageDataset, create_dataloaders
from utils.training import train_embedder, train_gan
from utils.evaluation import generate_images, test_zero_shot_generalization
from utils.utils import setup_device

class ImagePathDataset(Dataset):
    """Dataset that loads raw images from a list of file paths."""
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path

def parse_args():
    parser = argparse.ArgumentParser(description="Face Generation Pipeline")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["train_embedder", "train_gan", "generate", "evaluate", "zero_shot"],
                        help="Mode to run")
    parser.add_argument("--data_dir", type=str, default="faces_dataset",
                        help="Directory containing face images (train or test)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save outputs")
    parser.add_argument("--embedder_path", type=str, default="finetuned_dinov2_faceembedder.pth",
                        help="Path to fine-tuned embedder checkpoint")
    parser.add_argument("--generator_path", type=str, default="gan_generator.pth",
                        help="Path to trained generator checkpoint")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training or inference")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate")
    parser.add_argument("--finetune_backbone", action="store_true",
                        help="Fine-tune the DINOv2 backbone")
    parser.add_argument("--max_train_time", type=float, default=6.0,
                        help="Max training time (hours)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples for generation/evaluation")
    parser.add_argument("--use_cache", action="store_true",
                        help="Cache embeddings to disk during training")
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
        print(f"Loading embedder from {args.embedder_path}")
        embedder = DinoEmbedder().to(device)
        embedder.load_state_dict(torch.load(args.embedder_path, map_location=device))
        embedder.eval()
        print(" Embedder loaded")

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
        # Load embedder
        print(f"Loading embedder from {args.embedder_path}")
        embedder = DinoEmbedder().to(device)
        embedder.load_state_dict(torch.load(args.embedder_path, map_location=device))
        embedder.eval()
        # Load generator
        print(f"Loading generator from {args.generator_path}")
        generator = EmbeddingToImageGenerator().to(device)
        generator.load_state_dict(torch.load(args.generator_path, map_location=device))
        generator.eval()

        # Build test DataLoader
        all_files = [os.path.join(args.data_dir, f)
                     for f in os.listdir(args.data_dir)
                     if f.lower().endswith(('.jpg', '.png'))]
        sample_files = all_files[: args.num_samples]
        ds = ImagePathDataset(sample_files,
                              transform=None)  # or your preprocessing
        loader = DataLoader(ds,
                            batch_size=args.batch_size,
                            shuffle=False)

        # Generate & save
        generate_images(
            embedder,
            generator,
            loader,
            os.path.join(args.output_dir, "generated_images"),
            device
        )

    elif args.mode == "evaluate":
        print("===== Evaluating Reconstructions =====")
        # You can implement evaluate analogous to generate, calling appropriate util

    elif args.mode == "zero_shot":
        print("===== Zero-Shot Generalization =====")
        # Load embedder + generator as above...
        # Then:
        test_zero_shot_generalization(
            embedder,
            generator,
            loader,  # reuse loader above or rebuild
            os.path.join(args.output_dir, "zero_shot_results"),
            device
        )

    print(f" Task '{args.mode}' completed!")

if __name__ == "__main__":
    main()

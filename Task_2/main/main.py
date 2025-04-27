import os
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from main.models import *  
from utils.datasets import *
from utils.training import *
from utils.evaluation import *
from utils.utils import *



class ImagePathDataset(Dataset):
    """Dataset that loads raw images from a list of file paths."""
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        # Add default transform if not provided to convert PIL images to tensors
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])

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
                        choices=["train_embedder", "train_gan", "generate", "evaluate", "zero_shot", "inference_test"],
                        help="Mode to run")
    parser.add_argument("--data_dir", type=str, default="faces_dataset",
                        help="Directory containing face images (train or test)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name for inference_test mode")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save outputs")
    parser.add_argument("--embedder_path", type=str, default="finetuned_dinov2_faceembedder.pth",
                        help="Path to fine-tuned embedder checkpoint")
    parser.add_argument("--generator_path", type=str, default="gan_generator.pth",
                        help="Path to trained generator checkpoint")
    parser.add_argument("--inference_ckp", type=str, default=None,
                        help="Path to checkpoint for inference mode")
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
        embedder.load_state_dict(torch.load(args.embedder_path, map_location=device, weights_only=True))
        embedder.eval()
        # Load generator
        print(f"Loading generator from {args.generator_path}")
        generator = EmbeddingToImageGenerator().to(device)
        generator.load_state_dict(torch.load(args.generator_path, map_location=device, weights_only=True))
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

    elif args.mode == "inference_test":
        print(f"===== Running Inference Test on {args.dataset} =====")
        
        if args.inference_ckp is None:
            print("Error: --inference_ckp parameter is required for inference_test mode")
            return
        
        # Set data directory based on dataset name if provided
        data_dir = args.data_dir
        if args.dataset:
            # You can customize this mapping based on your dataset structure
            data_dir = os.path.join("datasets", args.dataset) if not os.path.exists(args.dataset) else args.dataset
            print(f"Using dataset directory: {data_dir}")
        
        # Create output directory with dataset name
        output_dir = os.path.join(args.output_dir, f"inference_{args.dataset}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load checkpoint
        print(f"Loading checkpoint from {args.inference_ckp}")
        checkpoint = torch.load(args.inference_ckp, map_location=device, weights_only=True)
        
        # Initialize models
        embedder = DinoEmbedder().to(device)
        generator = EmbeddingToImageGenerator().to(device)
        
        # Load model weights from checkpoint
        # Adapt this based on your actual checkpoint structure
        if isinstance(checkpoint, dict):
            if 'encoder' in checkpoint and 'generator' in checkpoint:
                embedder.load_state_dict(checkpoint['encoder'])
                generator.load_state_dict(checkpoint['generator'])
            elif 'embedder' in checkpoint and 'gen' in checkpoint:
                embedder.load_state_dict(checkpoint['embedder'])
                generator.load_state_dict(checkpoint['gen'])
            else:
                print("Warning: Checkpoint format not recognized. Trying direct load.")
                embedder.load_state_dict(torch.load(args.embedder_path, map_location=device, weights_only=True))
                generator.load_state_dict(checkpoint)
        else:
            print("Warning: Checkpoint is not a dictionary. Using default paths.")
            embedder.load_state_dict(torch.load(args.embedder_path, map_location=device, weights_only=True))
            generator.load_state_dict(torch.load(args.generator_path, map_location=device, weights_only=True))
        
        embedder.eval()
        generator.eval()
        print("✓ Models loaded successfully")
        
        # Create dataset and dataloader
        dataset = EmbeddingImageDataset(
            root=data_dir,
            embedder=embedder,
            transform_size=128,
            device=device,
            cache_dir=os.path.join(output_dir, 'embedding_cache')
        )
        
        # Custom collate function to fix PIL Image handling
        def embedding_image_collate(batch):
            embeddings = torch.stack([item[0] for item in batch])
            images = torch.stack([item[1] for item in batch])
            return embeddings, images
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=embedding_image_collate
        )
        
        # Generate images
        print("Generating images...")
        generate_and_compare_samples(generator, dataloader, output_dir=output_dir)
        
        print(f"✓ Inference completed. Results saved to {output_dir}")
        
    print(f" Task '{args.mode}' completed!")

if __name__ == "__main__":
    main()

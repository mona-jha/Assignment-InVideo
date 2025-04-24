

import argparse
from main.main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Face Generation Pipeline")

    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True, help='train or inference mode')
    parser.add_argument('--inference_ckp', type=str, default=None, help='Checkpoint path for inference')

    args = parser.parse_args()
    print("🚀 Launching Face Generation Pipeline...")
    main(dataset_path=args.dataset, mode=args.mode, inference_ckp=args.inference_ckp)
import argparse
from face_generation.main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face generator from embeddings using DINOv2")
    parser.add_argument('--dataset', type=str, required=True, help="Path to image dataset folder")
    parser.add_argument('--dino_ckpt', type=str, required=True, help="Path to pretrained DinoEmbedder checkpoint")

    args = parser.parse_args()
    print("🚀 Starting training pipeline...")
    main(dataset_path=args.dataset, dino_ckpt_path=args.dino_ckpt)

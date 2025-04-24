

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


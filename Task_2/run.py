 
import argparse
import sys
from main.main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Face Generation Pipeline")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to training folder or test image folder')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'inference'],
                        help='Choose "train" (GAN training) or "inference" (image generation)')
    parser.add_argument('--inference_ckp', type=str, default=None,
                        help='Path to generator checkpoint for inference (optional)')
    args = parser.parse_args()

    # Forward to main.main() with the correct internal mode flags
    sys.argv = [sys.argv[0]]
    if args.mode == 'train':
        sys.argv += ['--mode', 'train_gan', '--data_dir', args.dataset]
    else:
        sys.argv += ['--mode', 'generate', '--data_dir', args.dataset]
        if args.inference_ckp:
            sys.argv += ['--generator_path', args.inference_ckp]

    print(f"🚀 Launching pipeline in mode={sys.argv[2]} on {args.dataset}")
    main()

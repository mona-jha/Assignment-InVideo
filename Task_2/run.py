
 
import argparse
import sys
from main.main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Face Generation Pipeline")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset directory for training or inference')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'inference'],
                        help='train (GAN) or inference mode')
    parser.add_argument('--embedder_path', type=str, default='dinov2_encoder.pth',
                        help='Path to DINOv2 encoder checkpoint')
    parser.add_argument('--generator_path', type=str, default='generator.pth',
                        help='Path to generator model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save generated images or model outputs')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to process in inference or evaluation')
    parser.add_argument('--inference_ckp', type=str, default=None,
                        help='[Deprecated] alias for --generator_path in inference mode')

    args = parser.parse_args()

    # Build args for the core pipeline
    sys.argv = [sys.argv[0]]  # keep script name

    # Map wrapper modes to pipeline modes
    if args.mode == 'train':
        sys.argv += ['--mode', 'train_gan']
    else:
        sys.argv += ['--mode', 'generate']

    # Common flags
    sys.argv += ['--data_dir', args.dataset]
    sys.argv += ['--embedder_path', args.embedder_path]

    # Generator checkpoint: use explicit generator_path or fallback to inference_ckp
    gen_ckp = args.inference_ckp or args.generator_path
    sys.argv += ['--generator_path', gen_ckp]

    # Output and sampling
    sys.argv += ['--output_dir', args.output_dir]
    sys.argv += ['--num_samples', str(args.num_samples)]

    print(f" Launching Face Generation Pipeline in mode={sys.argv[2]}...")
    main()



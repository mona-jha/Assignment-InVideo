import argparse
import sys
from main.main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Face Generation Pipeline")
    parser.add_argument(
        '--dataset', type=str, required=True,
        help='Path to training folder (for train) or test images folder (for inference)'
    )
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['train', 'inference', 'inference_test'],
        help='train → GAN training; inference → generate images; inference_test → test with specific checkpoint'
    )
    parser.add_argument(
        '--embedder_path', type=str, default='dinov2_encoder.pth',
        help='Path to fine-tuned DINOv2 encoder checkpoint'
    )
    parser.add_argument(
        '--generator_path', type=str, default='gan_generator.pth',
        help='Path to trained generator checkpoint'
    )
    parser.add_argument(
        '--inference_ckp', type=str, default=None,
        help='Path to checkpoint for inference_test mode'
    )
    parser.add_argument(
        '--output_dir', type=str, default='outputs',
        help='Directory to save outputs (checkpoints or generated images)'
    )
    parser.add_argument(
        '--num_samples', type=int, default=10,
        help='Number of images to generate in inference mode'
    )

    args = parser.parse_args()

    # Build argument list for internal main
    sys.argv = [sys.argv[0]]

    if args.mode == 'train':
        # GAN training uses internal mode "train_gan"
        sys.argv += ['--mode', 'train_gan']
        sys.argv += ['--data_dir', args.dataset]
        sys.argv += ['--embedder_path', args.embedder_path]
        sys.argv += ['--generator_path', args.generator_path]
        sys.argv += ['--output_dir', args.output_dir]
    elif args.mode == 'inference_test':
        # Inference test mode with specific checkpoint
        sys.argv += ['--mode', 'inference_test']
        sys.argv += ['--dataset', args.dataset]  # Pass as dataset name
        sys.argv += ['--inference_ckp', args.inference_ckp]
        sys.argv += ['--output_dir', args.output_dir]
        sys.argv += ['--batch_size', '8']  # Smaller batch size for inference
    else:
        # Regular inference uses internal mode "generate"
        sys.argv += ['--mode', 'generate']
        sys.argv += ['--data_dir', args.dataset]
        sys.argv += ['--embedder_path', args.embedder_path]
        sys.argv += ['--generator_path', args.generator_path]
        sys.argv += ['--output_dir', args.output_dir]
        sys.argv += ['--num_samples', str(args.num_samples)]

    print(f" Launching pipeline with arguments: {sys.argv[1:]}\n")
    main()

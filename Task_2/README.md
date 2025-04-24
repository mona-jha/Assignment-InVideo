# Face Generation from DINOV2 Embeddings

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project fine-tunes DINOV2 on face datasets and trains a generator model to reconstruct high-quality face images from embeddings. The system follows a two-step process:

1. **Encoding Stage**: A fine-tuned DINOV2 model extracts meaningful face embeddings from input images
2. **Generation Stage**: A trained generator network reconstructs face images from these embeddings



## How It Works

### Part 1: Encoding with DINOV2

The DINOV2 model is fine-tuned on a face dataset to create a specialized face embedding model:

1. We start with a pre-trained DINOV2-small model from Facebook AI Research
2. Fine-tune it on our face dataset using contrastive learning techniques
3. The resulting encoder maps each face image to a 512-dimensional embedding vector
4. These embeddings capture essential facial features in a compact representation

### Part 2: Image Generation from Embeddings

A generator network is trained to convert face embeddings back into realistic face images:

1. The generator takes the 512-dimensional face embedding as input
2. Through a series of upsampling blocks and residual connections, it produces a 128×128 RGB image
3. The model is trained using a combination of L1 loss and perceptual loss
4. This ensures both pixel accuracy and perceptual quality in the reconstructed faces

## Features

- Fine-tune DINOV2 models on face datasets for better face embeddings
- Train a generator to generate faces from these embeddings
- Evaluate model quality with various metrics (SSIM, PSNR, embedding similarity)
- Zero-shot generalization to unseen faces
- Comprehensive evaluation and visualization tools

## Quick Start Guide

### Encoding and Generating Faces

```bash
python encode_and_generate.py \
  --input_dir /path/to/your/face/images \
  --embedder_path finetuned_dinov2_faceembedder.pth \
  --generator_path generator.pth \
  --output_dir results \
  --save_embeddings
```

This will:
1. Load the fine-tuned DINOV2 encoder
2. Encode each input image into an embedding vector
3. Feed these embeddings to the generator to produce reconstructed faces
4. Save the original images, generated images, and side-by-side comparisons



## Command-Line Interface

### Fine-tune the DINOV2 Embedder

```bash
python finetune_dinov2.py \
  --data_dir /path/to/face/data \
  --output_path finetuned_dinov2_faceembedder.pth \
  --batch_size 32 \
  --epochs 5 \
  --lr 5e-5 \
  --finetune_backbone  # Optional: fine-tune backbone network
```

### Train the Generator

```bash
python -m main.main \
  --mode train \
  --data_dir /path/to/face/data \
  --embedder_path finetuned_dinov2_faceembedder.pth \
  --generator_path generator.pth \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4
```




### Test  Generalization

```bash
python -m main.main \
  --mode inference \
  --data_dir /path/to/test/data \
  --embedder_path finetuned_dinov2_faceembedder.pth \
  --generator_path generator.pth \
  --output_dir outputs \
  --num_samples 5
```









## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The DINOV2 team at Facebook AI Research
- The original datasets creators
- All contributors to this project

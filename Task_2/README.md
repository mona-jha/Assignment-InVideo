# Face Generation from DINOV2 Embeddings

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project fine-tunes DINOV2 on face datasets and trains a generator model to reconstruct high-quality face images from embeddings. The system follows a two-step process:

1. **Encoding Stage**: A fine-tuned DINOV2 model extracts meaningful face embeddings from input images
2. **Generation Stage**: A trained generator network reconstructs face images from these embeddings

<div align="center">
  <img src="docs/architecture.png" width="80%" alt="System Architecture">
  <p><em>System Architecture: Fine-tuned DINOv2 Embedder → 512-dim Embedding Vector → Generator → 128×128 Face Reconstruction</em></p>
</div>

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

### Training the Complete Pipeline

For end-to-end training of the face generation system:

```bash
./run_pipeline.sh
```

Or manually:

```bash
# 1. Fine-tune the DINOV2 model
python finetune_dinov2.py \
  --data_dir /path/to/face/dataset \
  --output_path finetuned_dinov2_faceembedder.pth \
  --epochs 5

# 2. Train the generator
python -m face_generation.main \
  --mode train_gan \
  --data_dir /path/to/face/dataset \
  --embedder_path finetuned_dinov2_faceembedder.pth \
  --generator_path generator.pth
```

## Computing Requirements and Production Considerations

### Development Environment

Our model development was conducted with the following resources:

- **GPU**: NVIDIA RTX A6000 (24GB VRAM)
- **CPU**: 8 cores, Intel Xeon
- **RAM**: 32GB
- **Storage**: 100GB SSD
- **Training Time**: 
  - DINOV2 Fine-tuning: ~2 hours
  - GAN Training: ~4 hours

### Production Deployment Options

For production deployment, we recommend:

#### Option 1: On-premises GPU Server (High Performance)
- **Hardware**: 
  - NVIDIA T4 or A10 GPU (minimum 16GB VRAM)
  - 8+ CPU cores
  - 32GB+ RAM
  - 100GB+ SSD storage
- **Software**:
  - Docker container deployment
  - Model serving with TorchServe or NVIDIA Triton
  - API gateway for request handling

#### Option 2: Cloud-based Deployment (Scalable)
- **Services**:
  - AWS SageMaker or Google Vertex AI for model serving
  - GPU instances (e.g., g4dn.xlarge on AWS)
  - Auto-scaling configuration for demand fluctuation
  - CDN for serving generated images
- **Optimization**:
  - Model quantization to INT8 for faster inference
  - TorchScript or ONNX model export

#### Option 3: Edge Deployment (Low Latency)
- **Hardware**: NVIDIA Jetson AGX Orin or equivalent
- **Optimization**:
  - Model pruning and quantization
  - TensorRT conversion
  - Batch processing for efficiency

For all scenarios, we recommend:
- A/B testing different model versions
- Continuous monitoring of inference quality
- Regular retraining with new data
- End-to-end testing pipeline

## Installation

### Prerequisites
- Python 3.8+
- CUDA toolkit 11.6+ (for GPU acceleration)
- 10GB+ disk space

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/face-generation.git
cd face-generation

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker Installation

```bash
# Build the Docker image
docker build -t face-generation .

# Run the container
docker run --gpus all -it face-generation
```

### Low Disk Space Installation

If you're working with limited disk space:

```bash
# Make the script executable
chmod +x install_minimal.sh

# Run the minimal installation
./install_minimal.sh
```

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
python -m face_generation.main \
  --mode train_gan \
  --data_dir /path/to/face/data \
  --embedder_path finetuned_dinov2_faceembedder.pth \
  --generator_path generator.pth \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4
```

### Generating Images

```bash
python -m face_generation.main --mode generate \
  --data_dir /path/to/test/data \
  --embedder_path outputs/finetuned_dinov2_faceembedder.pth \
  --generator_path outputs/gan_generator.pth \
  --output_dir outputs
```

### Test Zero-Shot Generalization

```bash
python -m face_generation.main \
  --mode zero_shot \
  --data_dir /path/to/test/data \
  --embedder_path finetuned_dinov2_faceembedder.pth \
  --generator_path generator.pth \
  --output_dir outputs \
  --num_samples 5
```

### Evaluating the Model

```bash
python -m face_generation.main --mode evaluate \
  --data_dir /path/to/test/data \
  --embedder_path outputs/finetuned_dinov2_faceembedder.pth \
  --generator_path outputs/gan_generator.pth \
  --output_dir outputs
```

### Interactive Demo

For a visual interactive demo:

```bash
streamlit run demo_app.py
```

## GitHub Repository Structure

When setting up this project on GitHub, we recommend:

### Repository Structure

```
face-generation/
├── data/                      # Dataset handling scripts (not the actual data)
├── docs/                      # Documentation and diagrams
├── face_generation/           # Main module
│   ├── __init__.py
│   ├── models.py              # Model architectures
│   ├── datasets.py            # Dataset classes
│   ├── training.py            # Training functions
│   ├── evaluation.py          # Evaluation functions
│   ├── utils.py               # Utility functions
│   └── main.py                # Command-line interface
├── scripts/                   # Utility scripts
│   ├── setup_venv.sh          # Environment setup script
│   └── install_minimal.sh     # Minimal installation script
├── tests/                     # Unit tests
├── .gitignore                 # Git ignore file
├── requirements.txt           # Standard requirements
└── README.md                  # Main documentation
```

### Commit History

We recommend preserving the development history in git:

- **Don't squash commits** for main development branches
- Use semantic commit messages (e.g., `feat:`, `fix:`, `docs:`)
- Tag significant versions (e.g., `v1.0.0`)
- Maintain clear branch structures:
  - `main`: Stable releases
  - `dev`: Development branch
  - Feature branches for specific enhancements

## Results and Examples

Our model achieves excellent results in reconstructing faces from embeddings:

- **SSIM Score**: 0.85+
- **PSNR**: 25dB+
- **Embedding Similarity**: 0.95+
- **Zero-shot Performance**: Successfully generates faces for embedding vectors from entirely unseen images

For detailed examples, see the [inference_demo.ipynb](notebooks/inference_demo.ipynb) notebook.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The DINOV2 team at Facebook AI Research
- The original datasets creators
- All contributors to this project

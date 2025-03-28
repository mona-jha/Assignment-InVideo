
#  Efficient Face Data Processing and Generation

This assignment is split into two parts:

1. **Face Dataset Generation using Rust + YOLOv11 (or equivalent)**
2. **Face Generation using Embeddings via a Generative Model (PyTorch)**

We aim to demonstrate efficient data processing, model training, and present an approach suited for production timelines with constrained resources.

---

##  Task 1: Data Generation in Rust

We implemented a face cropping pipeline using Rust and a pre-trained face detection model (YOLOv11 / alternative). Given a folder of images, the binary processes them in batches, detects faces, and outputs square-cropped face images centered on each face.

### ⚙️ Features
- Parallelized face detection using rayon.
- Image loading and face detection using `image` + `tch` or `tract` for ONNX inference.
- Graceful handling of edge cases: multiple faces, no faces, low-resolution faces.
- Outputs 4800 valid face crops to disk.

###  Design Notes  
- We crop the largest face if multiple are present.
- We skip images with no valid detections or extremely small faces (< 32x32).
- Faces are resized to a uniform 128x128 for downstream training.
- We use a confidence threshold of 0.5 to reduce false positives.
- Each output file is uniquely named using UUIDs.
- IO and inference are batched to minimize memory overhead.
- The pipeline is wrapped into a single CLI binary with helpful usage flags.
- Invalid/corrupted images are logged and skipped.
- We use parallel iterators for speedup.
- Sample outputs are saved in `samples/`.

### 🚀 Usage
```bash
# Build
cargo build --release

# Run
./target/release/face_cropper --input ./wider_face/images --output ./cropped_faces
```

 

## 🎨 Task 2: Generative Face Model (PyTorch)

This task trains a generative model conditioned on visual embeddings. The pipeline is split into two stages:

1. **Embedding Model**: Fine-tuning a DinoV2 encoder on the custom dataset from Task 1.
2. **Generator**: A model (based on CycleGAN-style encoder-decoder with perceptual loss) trained to reconstruct 128x128 faces from embeddings.

### 🏗️ Architecture
- **Encoder**: DinoV2 (fine-tuned on face crops using a contrastive objective).
- **Generator**: MLP + CNN upsampling blocks with residual connections.
- **Loss Function**: L1 + 5×Perceptual Loss (VGG16), no discriminator.
- **Output Size**: 128x128 RGB images
- **Inference**: Supports zero-shot generation on unseen embeddings.

### 📈 Training Setup
 
- **Training Time**: ~6 hours
- **Optimizer**: Adam, LR = 1e-4
- **Batch Size**: 32
- **Data**: 4,800 face images (cropped in Task 1)

 


---

##   What This Shows

- Efficient use of Rust for real-time batch image processing
- Ability to integrate pretrained models (YOLO, DINO, VGG)
- Stability-focused training using perceptual and L1 losses
- Awareness of production constraints (no GAN instabilities)
- Organized code, version control, sample outputs, reproducibility

---

 
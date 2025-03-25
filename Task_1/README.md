# Image Face Detection and Cropping Tool

This Rust application detects faces in images and creates center-cropped versions focused on the detected faces. It uses a TorchScript model for face detection and OpenCV for image processing.

## Features

- Face detection using a PyTorch model
- Automatic center-cropping around detected faces
- Batch processing of images in a directory
- Configurable confidence threshold for detection

## Prerequisites

### Installing Rust

1. Install Rust using rustup (the Rust installer and version management tool):

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. Follow the on-screen instructions to complete the installation.

3. Add Rust to your current shell's path:

   ```bash
   source $HOME/.cargo/env
   ```

4. Verify the installation:

   ```bash
   rustc --version
   cargo --version
   ```

### Additional Dependencies

This project requires:

- OpenCV development libraries
- LibTorch (PyTorch C++ API)

Install these dependencies:

#### Ubuntu/Debian

```bash
# Install OpenCV dependencies
sudo apt update
sudo apt install libopencv-dev clang libclang-dev

# Install LibTorch dependencies
sudo apt install libgomp1
```



## Usage

Run the application with the following command:

```bash
./target/release/image-generation --input-dir /path/to/images --output-dir /path/to/output [OPTIONS]
```

### Command Line Arguments

- `-i, --input-dir`: Directory containing images to process (required)
- `-o, --output-dir`: Directory where cropped images will be saved (required)
- `-m, --model-path`: Path to the TorchScript model file (default: "model.pt")
- `-c, --conf-thresh`: Confidence threshold for face detection (default: 0.5)

### Example

```bash
./target/release/image-generation --input-dir ./input_images --output-dir ./cropped_faces --conf-thresh 0.6
```

## Notes

- The application only processes images with a single detected face meeting the confidence threshold
- Supported image formats: JPG, JPEG, PNG
- The face cropping uses a scaling factor of 2.0 around the detected face bounding box



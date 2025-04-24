import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision.utils import save_image

def show_generated_images(output_dir="generated_images", max_images=10):
    """Display generated face images compared to real ones
    
    Args:
        output_dir (str): Directory containing the generated and real images
        max_images (int): Maximum number of images to display
    """
    
    # Check if directory exists
    if not os.path.exists(output_dir):
        print(f"Directory not found: {output_dir}")
        return
    
    # Find all generated and real images
    generated_files = sorted(glob.glob(os.path.join(output_dir, "gen_*.png")))
    real_files = sorted(glob.glob(os.path.join(output_dir, "real_*.png")))
    
    # Limit the number of images to display
    num_images = min(len(generated_files), len(real_files), max_images)
    
    if num_images == 0:
        print("No images found!")
        # Look for any images in the directory
        all_images = glob.glob(os.path.join(output_dir, "*.png"))
        if all_images:
            print(f"Found {len(all_images)} images with different naming pattern:")
            for img in all_images[:5]:
                print(f" - {os.path.basename(img)}")
            if len(all_images) > 5:
                print(f" - ... and {len(all_images) - 5} more")
        return
        
    print(f"Displaying {num_images} images from {output_dir}")
    
    # Create a figure with subplots
    fig, axs = plt.subplots(num_images, 2, figsize=(10, 2*num_images))
    
    # Display single image pair if only one image
    if num_images == 1:
        real_img = Image.open(real_files[0])
        axs[0].imshow(np.array(real_img))
        axs[0].set_title("Real")
        axs[0].axis("off")
        
        gen_img = Image.open(generated_files[0])
        axs[1].imshow(np.array(gen_img))
        axs[1].set_title("Generated")
        axs[1].axis("off")
    else:
        # Display multiple image pairs
        for i in range(num_images):
            real_img = Image.open(real_files[i])
            axs[i, 0].imshow(np.array(real_img))
            axs[i, 0].set_title(f"Real {i+1}")
            axs[i, 0].axis("off")
            
            gen_img = Image.open(generated_files[i])
            axs[i, 1].imshow(np.array(gen_img))
            axs[i, 1].set_title(f"Generated {i+1}")
            axs[i, 1].axis("off")
    
    plt.tight_layout()
    plt.show()

def save_real_images(dataloader, output_dir="generated_images"):
    """Save the real images from the dataloader for comparison
    
    Args:
        dataloader: PyTorch DataLoader containing the images
        output_dir (str): Directory to save the real images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (images, _) in enumerate(dataloader):
        if isinstance(images, torch.Tensor):
            # Save each image in the batch
            for j in range(images.size(0)):
                img = images[j].cpu()
                # Denormalize if normalized
                if img.min() < 0:
                    img = (img + 1) / 2
                save_image(img, os.path.join(output_dir, f"real_{i*images.size(0)+j}.png"))
        elif isinstance(images, list) and isinstance(images[0], Image.Image):
            # Handle PIL images
            for j, img in enumerate(images):
                img.save(os.path.join(output_dir, f"real_{i*len(images)+j}.png"))
    
    print(f"Saved real images to {output_dir}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize generated images")
    parser.add_argument("--dir", type=str, default="generated_images", 
                        help="Directory containing generated and real images")
    parser.add_argument("--max", type=int, default=10, 
                        help="Maximum number of images to display")
    
    args = parser.parse_args()
    show_generated_images(args.dir, args.max)

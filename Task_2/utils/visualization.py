import os
import glob
from PIL import Image
import numpy as np

def create_comparison_grid(output_dir="generated_images", max_images=10, save_name="comparison_grid.png"):
    """Create and save a grid of real vs generated images as a single image file
    
    Args:
        output_dir (str): Directory containing the generated and real images
        max_images (int): Maximum number of images to include
        save_name (str): Filename to save the comparison grid
    """
    print(f"Creating comparison grid from images in {output_dir}")
    
    # Check if directory exists
    if not os.path.exists(output_dir):
        print(f"Directory not found: {output_dir}")
        return
    
    # Find all generated and real images with various possible naming patterns
    gen_patterns = ["gen_*.png", "generated_*.png", "*_generated.png", "*_gen.png", "compare_*.jpg"]
    real_patterns = ["real_*.png", "original_*.png", "*_real.png", "*_orig.png"]
    
    # Try to find files matching the patterns
    generated_files = []
    for pattern in gen_patterns:
        generated_files.extend(glob.glob(os.path.join(output_dir, pattern)))
    generated_files = sorted(generated_files)
    
    real_files = []
    for pattern in real_patterns:
        real_files.extend(glob.glob(os.path.join(output_dir, pattern)))
    real_files = sorted(real_files)
    
    # If no files found with specific patterns, get all PNG files
    if not generated_files or not real_files:
        all_files = sorted(glob.glob(os.path.join(output_dir, "*.png")))
        if len(all_files) >= 2:
            # Assume first half are real, second half are generated
            middle = len(all_files) // 2
            real_files = all_files[:middle]
            generated_files = all_files[middle:]
    
    # Look for comparison files if we still don't have pairs
    if (not real_files or not generated_files) and os.path.exists(output_dir):
        comparison_files = glob.glob(os.path.join(output_dir, "compare_*.jpg"))
        if comparison_files:
            print(f"Found {len(comparison_files)} pre-made comparison images")
            # Create a grid from these comparison images
            comparisons = [Image.open(f) for f in comparison_files[:max_images]]
            if not comparisons:
                print("No comparison images found")
                return
            
            # Get sizes
            width = comparisons[0].width
            height = comparisons[0].height
            
            # Create grid image
            grid_width = width
            grid_height = sum(img.height for img in comparisons)
            
            # Create a new blank image for the grid
            grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
            
            # Paste images
            y_offset = 0
            for img in comparisons:
                grid_img.paste(img, (0, y_offset))
                y_offset += img.height
            
            # Save the grid
            grid_path = os.path.join(output_dir, save_name)
            grid_img.save(grid_path)
            print(f"✓ Saved comparison grid to {grid_path}")
            return grid_path
    
    # Limit the number of images to display
    num_images = min(len(generated_files), len(real_files), max_images)
    
    if num_images == 0:
        print("No image pairs found!")
        all_images = glob.glob(os.path.join(output_dir, "*.png"))
        if all_images:
            print(f"Found {len(all_images)} images with different naming pattern:")
            for img in all_images[:5]:
                print(f" - {os.path.basename(img)}")
            if len(all_images) > 5:
                print(f" - ... and {len(all_images) - 5} more")
        return
    
    print(f"Creating grid with {num_images} image pairs")
    
    # Load images
    real_images = []
    gen_images = []
    
    for i in range(num_images):
        try:
            real_img = Image.open(real_files[i])
            gen_img = Image.open(generated_files[i])
            
            # Resize if needed
            target_size = (128, 128)  # Standard size for all images
            if real_img.size != target_size:
                real_img = real_img.resize(target_size)
            if gen_img.size != target_size:
                gen_img = gen_img.resize(target_size)
                
            real_images.append(real_img)
            gen_images.append(gen_img)
        except Exception as e:
            print(f"Error processing image pair {i}: {e}")
    
    if not real_images or not gen_images:
        print("Failed to load any valid image pairs")
        return
        
    # Create the grid
    # Each row has a real image and its generated counterpart
    pair_height = real_images[0].height
    pair_width = real_images[0].width + gen_images[0].width
    
    grid_width = pair_width
    grid_height = pair_height * len(real_images)
    
    # Create a new blank image for the grid
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images into grid
    for i in range(len(real_images)):
        # Paste real image on the left
        grid_img.paste(real_images[i], (0, i * pair_height))
        # Paste generated image on the right
        grid_img.paste(gen_images[i], (real_images[i].width, i * pair_height))
    
    # Save the grid
    grid_path = os.path.join(output_dir, save_name)
    grid_img.save(grid_path)
    print(f"✓ Saved comparison grid to {grid_path}")
    return grid_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create comparison grid of real and generated images")
    parser.add_argument("--dir", type=str, required=True, 
                        help="Directory containing generated and real images")
    parser.add_argument("--max", type=int, default=10, 
                        help="Maximum number of images to include")
    parser.add_argument("--output", type=str, default="comparison_grid.png", 
                        help="Output filename for the comparison grid")
    
    args = parser.parse_args()
    create_comparison_grid(args.dir, args.max, args.output)

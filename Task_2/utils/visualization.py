import os
import glob
import re
from PIL import Image

def numerical_sort_key(s):
    """Helper for natural number sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def create_side_by_side_comparison(output_dir="generated_images", max_pairs=10, save_name="side_by_side_comparison.png"):
    """Create and save a side-by-side comparison (Real || Generated) vertically stacked."""

    print(f"Creating side-by-side comparison from {output_dir}")
    
    if not os.path.exists(output_dir):
        print(f"Directory not found: {output_dir}")
        return
    
    # Search patterns
    gen_patterns = ["gen_*.png", "generated_*.png", "*_generated.png", "*_gen.png"]
    real_patterns = ["real_*.png", "original_*.png", "*_real.png", "*_orig.png"]
    
    generated_files, real_files = [], []
    
    for pattern in gen_patterns:
        generated_files.extend(glob.glob(os.path.join(output_dir, pattern)))
    for pattern in real_patterns:
        real_files.extend(glob.glob(os.path.join(output_dir, pattern)))
    
    # Sort files naturally
    generated_files = sorted(generated_files, key=numerical_sort_key)
    real_files = sorted(real_files, key=numerical_sort_key)
    
    # Fallback: assume first half real, second half generated
    if not real_files or not generated_files:
        all_files = sorted(glob.glob(os.path.join(output_dir, "*.png")), key=numerical_sort_key)
        if len(all_files) >= 2:
            mid = len(all_files) // 2
            real_files = all_files[:mid]
            generated_files = all_files[mid:]
    
    num_pairs = min(len(real_files), len(generated_files), max_pairs)
    
    if num_pairs == 0:
        print("No real-generated pairs found.")
        return

    real_images = []
    gen_images = []
    
    for i in range(num_pairs):
        try:
            real_img = Image.open(real_files[i]).resize((128, 128))
            gen_img = Image.open(generated_files[i]).resize((128, 128))
            real_images.append(real_img)
            gen_images.append(gen_img)
        except Exception as e:
            print(f"Error loading image pair {i}: {e}")
    
    if not real_images or not gen_images:
        print("Failed to load images.")
        return

    # Create one large side-by-side image (stack vertically)
    pair_width = real_images[0].width + gen_images[0].width
    pair_height = real_images[0].height

    total_height = num_pairs * pair_height

    comparison_img = Image.new('RGB', (pair_width, total_height), color='white')

    for idx, (real_img, gen_img) in enumerate(zip(real_images, gen_images)):
        y = idx * pair_height
        # Paste real on left
        comparison_img.paste(real_img, (0, y))
        # Paste generated on right
        comparison_img.paste(gen_img, (real_img.width, y))
    
    save_path = os.path.join(output_dir, save_name)
    comparison_img.save(save_path)
    print(f"✓ Saved side-by-side comparison to {save_path}")

    return save_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create side-by-side comparison of real and generated images")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing generated and real images")
    parser.add_argument("--max", type=int, default=10, help="Maximum number of pairs to include")
    parser.add_argument("--output", type=str, default="side_by_side_comparison.png", help="Output filename for the comparison")
    
    args = parser.parse_args()
    create_side_by_side_comparison(output_dir=args.dir, max_pairs=args.max, save_name=args.output)

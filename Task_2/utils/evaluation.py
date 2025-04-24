import os
import numpy as np
from PIL import Image
import glob
import torch




def generate_images(encoder, generator, dataloader, output_dir="generated_images", device="cpu"):
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    encoder.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(dataloader, desc="Generating images")):
            images = images.to(device)
            embeddings = encoder(images)
            generated = generator(embeddings)
            for j in range(generated.size(0)):
                save_image(generated[j], os.path.join(output_dir, f"gen_{i*generated.size(0)+j}.png"))

def test_zero_shot_generalization(encoder, generator, dataloader, output_dir="zero_shot_results", device="cpu"):
    print(" Running zero-shot generalization test...")
    generate_images(encoder, generator, dataloader, output_dir, device)



def generate_and_compare_samples(generator, dataloader, output_dir="generated_faces_compare", epoch=None):
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        for i, (embedding, real_img) in enumerate(dataloader):
            embedding = embedding.to(generator.fc.weight.device)
            real_img = real_img.to(generator.fc.weight.device)
            fake_img = generator(embedding)

            for j in range(min(real_img.size(0), 4)):
                real = real_img[j].cpu().numpy().transpose(1, 2, 0)
                fake = fake_img[j].cpu().numpy().transpose(1, 2, 0)

                real = np.clip(((real + 1) / 2), 0, 1)
                fake = np.clip(((fake + 1) / 2), 0, 1)

                real = (real * 255).astype(np.uint8)
                fake = (fake * 255).astype(np.uint8)

                combined = np.concatenate([real, fake], axis=1)
                img_name = f"compare_{i*4 + j}.jpg" if epoch is None else f"epoch{epoch:02d}_img{i*4 + j}.jpg"
                Image.fromarray(combined).save(os.path.join(output_dir, img_name))

            if i >= 2:
                break

def create_training_progress_gif(image_folder, output_gif="training_progress.gif", max_images=100, duration=400):
    images = sorted(glob.glob(os.path.join(image_folder, "epoch*_img0.jpg")))[:max_images]
    frames = [Image.open(img) for img in images]
    if frames:
        frames[0].save(
            os.path.join(image_folder, output_gif),
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=duration,
            loop=0
        )
        print(f"🎞️ GIF saved at {os.path.join(image_folder, output_gif)}")
    else:
        print("⚠️ No images found to create GIF.")

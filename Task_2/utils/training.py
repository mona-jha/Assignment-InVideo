"""
Training functions for embedder and generative models
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from main.models import DinoEmbedder, EmbeddingToImageGenerator
from utils.datasets import FaceDataset, EmbeddingImageDataset, create_dataloaders
from utils.utils import save_images, perceptual_loss
from utils.evaluation import generate_and_compare_samples

def train_embedder(
    data_dir, 
    output_path="finetuned_dinov2_faceembedder.pth",
    batch_size=32,
    epochs=5,
    lr=5e-5,
    device=None,
    finetune_backbone=True
):
    """
    Fine-tunes a DINOV2 model on face data
    
    Args:
        data_dir: Directory containing face images
        output_path: Path to save the fine-tuned model
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on (cpu or cuda)
        finetune_backbone: Whether to fine-tune the backbone or just the projection head
    
    Returns:
        Fine-tuned embedder model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset and dataloader with augmentation
    dataset = FaceDataset(data_dir, augment=True)
    train_dataloader, val_dataloader = create_dataloaders(dataset, batch_size=batch_size)
    
    # Initialize model with fine-tuning options
    model = DinoEmbedder(finetune_backbone=finetune_backbone).to(device)
    
    # Learning rate scheduler and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # We'll use contrastive loss for face embedding fine-tuning
    # This is more suitable for face recognition than MSE
    criterion = nn.CosineEmbeddingLoss()
    
    # Track best validation loss
    best_val_loss = float('inf')
    start_time = time.time()
    
    # Fine-tune
    model.train()
    for epoch in range(epochs):
        # Training phase
        train_loss = 0
        model.train()
        for i, images in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]")):
            images = images.to(device)
            
            # Create positive pairs by augmenting existing images
            aug_images = images + 0.1 * torch.randn_like(images)  # Simple augmentation
            
            optimizer.zero_grad()
            
            # Get embeddings for both original and augmented images
            embeddings1 = model(images)
            embeddings2 = model(aug_images)
            
            # Contrastive loss with positive pairs (target=1)
            target = torch.ones(images.size(0)).to(device)
            loss = criterion(embeddings1, embeddings2, target)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Print progress every 50 batches
            if i % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Batch {i}/{len(train_dataloader)}, Loss: {loss.item():.4f}, Time: {elapsed:.1f}s")
            
        # Validation phase
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images in tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Valid]"):
                images = images.to(device)
                
                # Create positive pairs
                aug_images = images + 0.1 * torch.randn_like(images)
                
                # Get embeddings
                embeddings1 = model(images)
                embeddings2 = model(aug_images)
                
                # Contrastive loss
                target = torch.ones(images.size(0)).to(device)
                loss = criterion(embeddings1, embeddings2, target)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Update learning rate
        scheduler.step()
        
        print(f"✅ Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_path)
            print(f"✅ Saved best model with val loss: {best_val_loss:.4f}")
    
    # Ensure we've saved the final model state
    if not os.path.exists(output_path):
        torch.save(model.state_dict(), output_path)
        print(f"✅ Saved final model")
    
    print(f"✅ Training completed in {(time.time() - start_time)/60:.1f} minutes")
    
    return model

def train_generator(
    data_dir,
    embedder_path,
    output_dir="generated_faces_compare",
    generator_path="generator.pth",
    batch_size=16,
    epochs=50,
    lr=1e-4,
    device=None
):
    """
    Train a generator to reconstruct images from embeddings using L1 and perceptual loss
    
    Args:
        data_dir: Directory containing face images
        embedder_path: Path to fine-tuned embedder model
        output_dir: Directory to save generated samples
        generator_path: Path to save the generator model
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on (cpu or cuda)
        
    Returns:
        Trained generator model and dataloader
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load embedder model
    embedder = DinoEmbedder().to(device)
    embedder.load_state_dict(torch.load(embedder_path, map_location=device))
    embedder.eval()
    
    # Create dataset and dataloader
    dataset = EmbeddingImageDataset(data_dir, embedder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize generator and optimizer
    generator = EmbeddingToImageGenerator().to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        generator.train()
        total_loss = 0
        for embeddings, real_imgs in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            embeddings = embeddings.to(device)
            real_imgs = real_imgs.to(device)
            
            # Generate images
            fake_imgs = generator(embeddings)
            
            # Ensure real images have the same size as fake ones
            if fake_imgs.shape != real_imgs.shape:
                real_imgs = F.interpolate(real_imgs, size=fake_imgs.shape[2:])
            
            # Calculate losses
            l1 = F.l1_loss(fake_imgs, real_imgs)
            percep = perceptual_loss(fake_imgs, real_imgs)
            loss = l1 + 5 * percep
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Print epoch statistics
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f}")
        
        # Save comparison samples
        generate_and_compare_samples(generator, dataloader, output_dir=output_dir, epoch=epoch+1)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), os.path.join(output_dir, f"generator_epoch_{epoch+1}.pth"))
    
    # Save final model
    torch.save(generator.state_dict(), generator_path)
    print("✅ Training complete. Model saved.")
    
    return generator, dataloader

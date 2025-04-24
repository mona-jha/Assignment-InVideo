"""
Fine-tune DINOV2 model on face dataset
"""

import os
import torch
import argparse
from tqdm import tqdm
from main.models import DinoEmbedder
from utils.datasets import FaceDataset, create_dataloaders
from utils import setup_device

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DINOV2 for face embeddings")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing face images")
    parser.add_argument("--output_path", type=str, default="finetuned_dinov2_faceembedder.pth", help="Path to save fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--finetune_backbone", action="store_true", help="Fine-tune backbone network (not just projection head)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = setup_device()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    print(f"Fine-tuning DINOV2 model on face images from {args.data_dir}")
    print(f"Will save model to {args.output_path}")
    
    # Load dataset and create dataloaders
    dataset = FaceDataset(args.data_dir, augment=True)
    train_dataloader, val_dataloader = create_dataloaders(dataset, batch_size=args.batch_size)
    
    # Initialize model
    model = DinoEmbedder(finetune_backbone=args.finetune_backbone).to(device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CosineEmbeddingLoss()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0
        for images in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            images = images.to(device)
            
            # Create positive pairs with augmentation
            aug_images = images + 0.1 * torch.randn_like(images)
            
            optimizer.zero_grad()
            embeddings1 = model(images)
            embeddings2 = model(aug_images)
            
            # Contrastive loss with positive pairs
            target = torch.ones(images.size(0)).to(device)
            loss = criterion(embeddings1, embeddings2, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]"):
                images = images.to(device)
                aug_images = images + 0.1 * torch.randn_like(images)
                
                embeddings1 = model(images)
                embeddings2 = model(aug_images)
                
                target = torch.ones(images.size(0)).to(device)
                loss = criterion(embeddings1, embeddings2, target)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.output_path)
            print(f"✅ Saved best model with val loss: {best_val_loss:.4f}")
    
    print(f"Fine-tuning complete! Model saved to {args.output_path}")

if __name__ == "__main__":
    main()



import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from main.models import DinoEmbedder, EmbeddingToImageGenerator
from utils.datasets import FaceDataset, EmbeddingImageDataset
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
    Fine-tunes a DINOV2 model on face data using contrastive loss.

    Args:
        data_dir: Directory containing face images.
        output_path: Where to save the best embedder checkpoint.
        batch_size: Training batch size.
        epochs: Number of epochs.
        lr: Learning rate.
        device: 'cuda' or 'cpu'.
        finetune_backbone: If False, only projection head is trained.

    Returns:
        The fine-tuned DinoEmbedder instance.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Prepare data
    dataset = FaceDataset(data_dir, augment=True)
    n = len(dataset)
    idxs = torch.randperm(n).tolist()
    split = int(0.1 * n)                           # hold out 10% for validation
    train_idxs, val_idxs = idxs[split:], idxs[:split]
    train_loader = DataLoader(Subset(dataset, train_idxs), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idxs),   batch_size=batch_size, shuffle=False)

    # 2) Model, optimizer, scheduler, loss
    model = DinoEmbedder(finetune_backbone=finetune_backbone).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CosineEmbeddingLoss()

    best_val = float('inf')
    start = time.time()

    for epoch in range(1, epochs+1):
        # ——— Training ———
        model.train()
        total_train_loss = 0
        for imgs in tqdm(train_loader, desc=f"[Embedder] Epoch {epoch}/{epochs} (train)"):
            imgs = imgs.to(device)
            aug = imgs + 0.1 * torch.randn_like(imgs)

            optimizer.zero_grad()
            e1 = model(imgs)
            e2 = model(aug)
            target = torch.ones(imgs.size(0), device=device)
            loss = criterion(e1, e2, target)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # ——— Validation ———
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs in tqdm(val_loader, desc=f"[Embedder] Epoch {epoch}/{epochs} (val)"):
                imgs = imgs.to(device)
                aug = imgs + 0.1 * torch.randn_like(imgs)
                e1 = model(imgs)
                e2 = model(aug)
                target = torch.ones(imgs.size(0), device=device)
                total_val_loss += criterion(e1, e2, target).item()

        avg_train = total_train_loss / len(train_loader)
        avg_val   = total_val_loss   / len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch} ▶ Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        # save best
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), output_path)
            print(f"  ↳ Saved best embedder (val {best_val:.4f})")

    print(f"Embedder training completed in {(time.time()-start)/60:.1f}m")
    return model


def train_generator(
    data_dir,
    embedder_path,
    output_dir="generated_samples",
    generator_path="generator_best.pth",
    batch_size=16,
    epochs=50,
    lr=1e-4,
    device=None
):
    """
    Trains the EmbeddingToImageGenerator to reconstruct 128×128 faces
    from Dino embeddings, using L1 + perceptual loss and zero-shot validation.

    Args:
        data_dir: Folder with 'embeddings.pt' and 'real_faces.pt'
        embedder_path: Path to fine-tuned DinoEmbedder checkpoint
        output_dir: Where to dump sample grids each epoch
        generator_path: Where to save best generator
        batch_size: Training batch size
        epochs: Number of epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'

    Returns:
        Trained generator model.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 1) Load fixed embedder
    embedder = DinoEmbedder(finetune_backbone=False).to(device)
    embedder.load_state_dict(torch.load(embedder_path, map_location=device))
    embedder.eval()

    # 2) Prepare dataset
    ds = EmbeddingImageDataset(data_dir, embedder)   # returns (512-d emb, 3×128×128 image)
    n = len(ds)
    idxs = torch.randperm(n).tolist()
    split = int(0.1 * n)                             # hold out 10% unseen for zero-shot
    train_idxs, val_idxs = idxs[split:], idxs[:split]
    train_loader = DataLoader(Subset(ds, train_idxs), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(ds, val_idxs),   batch_size=batch_size, shuffle=False)

    # 3) Model, optimizer, losses
    gen = EmbeddingToImageGenerator().to(device)
    opt = optim.Adam(gen.parameters(), lr=lr)
    l1   = nn.L1Loss()
    # perceptual_loss imported from utils.utils
    best_val = float('inf')
    start = time.time()

    for epoch in range(1, epochs+1):
        gen.train()
        train_loss = 0
        for emb, real in tqdm(train_loader, desc=f"[Gen] Epoch {epoch}/{epochs} (train)"):
            emb  = emb.to(device) + 0.05 * torch.randn_like(emb)  # small embedding noise
            real = real.to(device)

            fake = gen(emb)
            if fake.shape != real.shape:
                real = F.interpolate(real, size=fake.shape[2:])

            loss = l1(fake, real) + 5 * perceptual_loss(fake, real)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()

        # ——— Validation ———
        gen.eval()
        val_loss = 0
        with torch.no_grad():
            for emb, real in tqdm(val_loader, desc=f"[Gen] Epoch {epoch}/{epochs} (val)"):
                emb  = emb.to(device)
                real = real.to(device)
                fake = gen(emb)
                if fake.shape != real.shape:
                    real = F.interpolate(real, size=fake.shape[2:])
                val_loss += (l1(fake, real) + 5 * perceptual_loss(fake, real)).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"Epoch {epoch} ▶ Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        # Save sample grid & comparison
        generate_and_compare_samples(gen, val_loader, output_dir, epoch)

        # Save best
        if avg_val < best_val:
            best_val = avg_val
            torch.save(gen.state_dict(), generator_path)
            print(f"  ↳ Saved best generator (val {best_val:.4f})")

    print(f"Generator training completed in {(time.time()-start)/60:.1f}m")
    return gen

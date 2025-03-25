import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FaceDataset(Dataset):
    """
    Dataset for loading face images for embedder fine-tuning
    """
    def __init__(self, root_dir, transform_size=224, augment=True):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]
        
        # Augmentations for better fine-tuning
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((transform_size, transform_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((transform_size, transform_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)

class EmbeddingImageDataset(Dataset):
    """
    Dataset that returns both embeddings and images
    Uses caching to speed up training by avoiding repeated embeddings computation
    """
    def __init__(self, root, embedder, transform_size=128, device='cuda', cache_dir=None):
        self.paths = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.jpg','.png'))]
        print(f"Found {len(self.paths)} images in {root}")
        
        self.transform = transforms.Compose([
            transforms.Resize((transform_size, transform_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        
        self.embedder = embedder.eval().to(device)
        self.embedder_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        
        self.device = device
        self.cache_dir = cache_dir
        self.embedding_cache = {}
        
        # Create cache directory if needed
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        
        # Get filename for potential caching
        filename = os.path.basename(path)
        cache_path = os.path.join(self.cache_dir, f"{filename}.pt") if self.cache_dir else None
        
        # Try to load embedding from cache
        embedding = None
        if cache_path and os.path.exists(cache_path):
            try:
                embedding = torch.load(cache_path)
            except:
                embedding = None
        
        # Compute embedding if not cached
        if embedding is None:
            input_tensor = self.embedder_transforms(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.embedder(input_tensor).squeeze(0).cpu()
            
            # Save to cache if enabled
            if cache_path:
                torch.save(embedding, cache_path)
        
        return embedding, self.transform(image)

def create_dataloaders(dataset, batch_size=32, train_split=0.9, num_workers=4):
    """
    Creates train and validation dataloaders from a dataset
    """
    # Use more workers for faster data loading
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_dataloader, val_dataloader

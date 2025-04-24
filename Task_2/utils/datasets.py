import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FaceDataset(Dataset):
    """
    Dataset for loading face images for embedder fine-tuning.
    Returns: (image_tensor, dummy_label)
    """
    def __init__(self, root_dir, transform_size=224, augment=True):
        self.root_dir = root_dir
        self.image_files = [
            f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        self.transform = transforms.Compose([
            transforms.Resize((transform_size, transform_size)),
            *([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
            ] if augment else []),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, 0  # Dummy label for compatibility


class EmbeddingImageDataset(Dataset):
    """
    Dataset that returns (embedding, image_tensor) using a pretrained embedder.
    Optionally caches embeddings to disk for efficiency.
    """
    def __init__(self, root, embedder, transform_size=128, device='cuda', cache_dir=None):
        self.paths = [
            os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        print(f"Found {len(self.paths)} images in {root}")

        # Transforms for generator input
        self.image_transform = transforms.Compose([
            transforms.Resize((transform_size, transform_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        # Transforms for embedding model input
        self.embedder_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        self.embedder = embedder.eval().to(device)
        self.device = device
        self.cache_dir = cache_dir

        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert("RGB")
        file_id = os.path.basename(img_path).split('.')[0]
        cache_path = os.path.join(self.cache_dir, f"{file_id}.pt") if self.cache_dir else None

        # Try loading cached embedding
        if cache_path and os.path.exists(cache_path):
            try:
                embedding = torch.load(cache_path)
            except:
                embedding = None
        else:
            embedding = None

        # Generate embedding if not cached
        if embedding is None:
            input_tensor = self.embedder_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.embedder(input_tensor).squeeze(0).cpu()
            if cache_path:
                torch.save(embedding, cache_path)

        return embedding, self.image_transform(image)


def create_dataloaders(dataset, batch_size=32, train_split=0.9, num_workers=4):
    """
    Splits dataset into training and validation DataLoaders.
    """
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


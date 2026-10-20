from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
BATCH = 128

def carregar_dados():
    transform = transforms.ToTensor()
    full = datasets.MNIST(str(ROOT / 'data'), train=True, download=True, transform=transform)
    test = datasets.MNIST(str(ROOT / 'data'), train=False, download=True, transform=transform)
    train, val = random_split(full, [50_000, 10_000],
                              generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(test, batch_size=BATCH, shuffle=False)
    return train_loader, val_loader, test_loader, test

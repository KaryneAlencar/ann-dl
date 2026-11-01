from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent

# Define o tamanho dos lotes de treinamento
BATCH = 128

def carregar_dados():
    # Converte as imagens em tensores com pixels entre 0 e 1
    transform = transforms.ToTensor()

    # Carrega os conjuntos oficiais de treino e teste do MNIST
    full = datasets.MNIST(str(ROOT / 'data'), train=True, download=True, transform=transform)
    test = datasets.MNIST(str(ROOT / 'data'), train=False, download=True, transform=transform)

    # Separa 50.000 imagens para treino e 10.000 para validação,
    # usando uma semente fixa para garantir a reprodução da divisão
    train, val = random_split(full, [50_000, 10_000],
                              generator=torch.Generator().manual_seed(42))

    # Embaralha apenas o treino; validação e teste mantêm a ordem
    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(test, batch_size=BATCH, shuffle=False)
    return train_loader, val_loader, test_loader, test

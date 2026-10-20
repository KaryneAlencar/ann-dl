from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ex1b import VAE, loss_terms

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train(train)
    totals = np.zeros(2, dtype=np.float64)
    count = 0
    with torch.set_grad_enabled(train):
        for x, _ in loader:
            x = x.to(DEVICE).flatten(1)
            xhat, mu, logvar = model(x)
            reconstruction, kl = loss_terms(x, xhat, mu, logvar)
            if train:
                optimizer.zero_grad()
                (reconstruction + kl).backward()
                optimizer.step()
            n = x.size(0)
            totals += n * np.array([reconstruction.item(), kl.item()])
            count += n
    return (totals / count).tolist()

def train_model(d, train_loader, val_loader):
    torch.manual_seed(42)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    model = VAE(d).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    hist = {'train_rec': [], 'train_kl': [], 'val_rec': [], 'val_kl': []}
    for epoch in range(EPOCHS):
        a, b = run_epoch(model, train_loader, optimizer)
        c, e = run_epoch(model, val_loader)
        for key, value in zip(hist.keys(), (a, b, c, e)):
            hist[key].append(value)
        print(f'd={d:2d} epoca {epoch+1:02d}/{EPOCHS}: treino rec={a:.3f} KL={b:.3f}; val rec={c:.3f} KL={e:.3f}', flush=True)
    return model, hist

def fig1(hist):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in [(axs[0], 'rec', 'Reconstrucao (BCE)'), (axs[1], 'kl', 'Divergencia KL')]:
        ax.plot(range(1, EPOCHS+1), hist['train_'+metric], label='Treino')
        ax.plot(range(1, EPOCHS+1), hist['val_'+metric], label='Validacao')
        ax.set(xlabel='Epoca', ylabel='nats por imagem', title=title)
        ax.legend()
    fig.suptitle('Figura 1 — Perdas do VAE d=16')
    fig.tight_layout()
    fig.savefig(FIG/'fig1.png', dpi=160)
    plt.close(fig)

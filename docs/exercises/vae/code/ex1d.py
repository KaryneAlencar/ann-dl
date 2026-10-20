from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ex1c import DEVICE, FIG

def display_img(ax, flat):
    ax.imshow(flat.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')

def fig2(model, test):
    indices = {}
    for i, (_, y) in enumerate(test):
        digit = int(y)
        if digit not in indices:
            indices[digit] = i
        if len(indices) == 10:
            break
    images = torch.stack([test[indices[d]][0] for d in range(10)]).to(DEVICE).flatten(1)
    with torch.no_grad():
        mu, _ = model.encode(images)
        outputs = model.decoder(mu)
    fig, ax = plt.subplots(2, 10, figsize=(13, 3.3))
    for d in range(10):
        display_img(ax[0, d], images[d].cpu().numpy())
        display_img(ax[1, d], outputs[d].cpu().numpy())
        ax[0, d].set_title(str(d))
    ax[0, 0].set_ylabel('Original', fontsize=10)
    ax[1, 0].set_ylabel('Reconstrucao', fontsize=10)
    fig.suptitle('Figura 2 — Originais e reconstrucoes a partir de mu (d=16)')
    fig.tight_layout()
    fig.savefig(FIG/'fig2.png', dpi=160)
    plt.close(fig)
    return indices

def fig3(model):
    torch.manual_seed(0)
    with torch.no_grad():
        z = torch.randn(64, 16, device=DEVICE)
        samples = model.decoder(z).cpu().numpy()
    fig, axs = plt.subplots(8, 8, figsize=(8, 8))
    for ax, sample in zip(axs.ravel(), samples):
        display_img(ax, sample)
    fig.suptitle('Figura 3 — 64 amostras do prior N(0,I), d=16')
    fig.tight_layout()
    fig.savefig(FIG/'fig3.png', dpi=160)
    plt.close(fig)

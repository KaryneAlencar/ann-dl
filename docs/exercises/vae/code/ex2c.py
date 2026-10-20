import numpy as np
import torch
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ex1c import DEVICE, FIG

def fig5(model):
    coordinates = norm.ppf(np.linspace(0.05, 0.95, 15))
    xx, yy = np.meshgrid(coordinates, coordinates[::-1])
    zs = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        decoded = model.decoder(zs).cpu().numpy().reshape(15, 15, 28, 28)
    canvas = np.zeros((15*28, 15*28))
    for i in range(15):
        for j in range(15):
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = decoded[i, j]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas, cmap='gray', extent=[coordinates[0], coordinates[-1], coordinates[0], coordinates[-1]], origin='upper', vmin=0, vmax=1)
    ax.set(xlabel='z_1 (quantis N(0,1))', ylabel='z_2 (quantis N(0,1))', title='Figura 5 — Grade de decodificacao 15 x 15 (d=2)')
    fig.tight_layout()
    fig.savefig(FIG/'fig5.png', dpi=160)
    plt.close(fig)

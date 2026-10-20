import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ex1c import DEVICE, FIG

def fig4(model, test_loader):
    points, digits = [], []
    with torch.no_grad():
        for x, y in test_loader:
            mu, _ = model.encode(x.to(DEVICE).flatten(1))
            points.append(mu.cpu().numpy())
            digits.append(y.numpy())
    points = np.concatenate(points)
    digits = np.concatenate(digits)
    fig, ax = plt.subplots(figsize=(9, 7))
    for d in range(10):
        select = digits == d
        ax.scatter(points[select, 0], points[select, 1], s=3, alpha=0.45, label=str(d))
    ax.set(xlabel='mu_1', ylabel='mu_2', title='Figura 4 — Medias latentes das 10.000 imagens de teste (d=2)')
    ax.legend(title='Digito', markerscale=4, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG/'fig4.png', dpi=160)
    plt.close(fig)

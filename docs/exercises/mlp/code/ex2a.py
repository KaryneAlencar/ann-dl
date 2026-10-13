import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

centers = np.array([
    [0.0, 0.0],
    [3.0, 3.0],
    [-3.0, -3.0]
])

Xb, c = make_blobs(
    n_samples=[500, 250, 250],
    centers=centers,
    cluster_std=1.2,
    random_state=42
)

yb = (c > 0).astype(int)

Xbt, Xbe, ybt, ybe = train_test_split(
    Xb,
    yb,
    test_size=0.2,
    random_state=42,
    stratify=yb
)

plt.figure(figsize=(7, 5))

for k in [0, 1]:
    plt.scatter(
        Xb[yb == k, 0],
        Xb[yb == k, 1],
        s=10,
        alpha=0.55,
        label=f"Classe {k}"
    )

plt.title("Figura 1: Dados binários não linearmente separáveis")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()

plt.tight_layout()
plt.show()
from ex2a import Xb, Xbt, Xbe, ybt, ybe
from mlp_core import MLP
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

binary = MLP(
    layers=[2, 16, 1],
    task="binary",
    rng=rng
)

hb = binary.fit(
    Xbt,
    ybt,
    lr=0.3,
    epochs=1800
)

acc_b = binary.score(Xbe, ybe)

print(f"Arquitetura: [2, 16, 1]")
print(f"Taxa de aprendizado: 0.3")
print(f"Épocas: 1800")
print(f"Acurácia no teste: {acc_b:.2%}")

plt.figure(figsize=(7, 5))

plt.plot(np.arange(1, len(hb) + 1),hb,label="Treino")

plt.title("Figura 2: Perda de treino — binário")
plt.xlabel("Época")
plt.ylabel("Entropia cruzada binária")
plt.legend()
plt.show()

xmin = Xb[:, 0].min() - 1
xmax = Xb[:, 0].max() + 1

ymin = Xb[:, 1].min() - 1
ymax = Xb[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(xmin, xmax, 240),
    np.linspace(ymin, ymax, 240)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]

zz = binary.predict(grid_points)
zz = zz.reshape(xx.shape)

plt.figure(figsize=(7, 5))

plt.contourf(
    xx,
    yy,
    zz,
    alpha=0.22,
    levels=[-0.5, 0.5, 1.5],
    cmap="coolwarm"
)

for k in [0, 1]:
    plt.scatter(
        Xbe[ybe == k, 0],
        Xbe[ybe == k, 1],
        s=17,
        label=f"Classe {k}",
        alpha=0.75
    )

plt.title("Figura 3: Fronteira de decisão do MLP")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()
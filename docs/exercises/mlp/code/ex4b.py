import numpy as np
import matplotlib.pyplot as plt

seeds = [42, 43, 44]
runs = []

for seed in seeds:

    shallow_model = MLP(
        layers=[4, 24, 3],
        task="multi",
        seed=seed
    )

    shallow_history = shallow_model.fit(
        Xmt,
        ymt,
        lr=LR,
        epochs=EPOCHS
    )

    deep_model = MLP(
        layers=[4, 24, 24, 3],
        task="multi",
        seed=seed
    )

    deep_history = deep_model.fit(
        Xmt,
        ymt,
        lr=LR,
        epochs=EPOCHS
    )

    shallow_acc = shallow_model.score(Xme, yme)
    deep_acc = deep_model.score(Xme, yme)

    runs.append((
        seed,
        shallow_acc,
        deep_acc,
        shallow_history[-1],
        deep_history[-1]
    ))

    print(f"Semente: {seed}")
    print(f"Acurácia da rede rasa: {shallow_acc:.2%}")
    print(f"Acurácia da rede profunda: {deep_acc:.2%}")
    print(f"Perda final da rede rasa: {shallow_history[-1]:.6f}")
    print(f"Perda final da rede profunda: {deep_history[-1]:.6f}")
    print()

    if seed == 42:
        shallow = shallow_model
        deep = deep_model
        hs = shallow_history
        hdeep = deep_history

R = np.array(runs, dtype=float)

print("Quantidade de parâmetros:")
print(f"Rede rasa: {shallow.count_params()}")
print(f"Rede profunda: {deep.count_params()}")

mean_shallow = R[:, 1].mean()
mean_deep = R[:, 2].mean()

difference = mean_deep - mean_shallow

print("\nAcurácia média:")
print(f"Rede rasa: {mean_shallow:.2%}")
print(f"Rede profunda: {mean_deep:.2%}")
print(f"Diferença (profunda - rasa): {difference * 100:.2f} pontos percentuais")

plt.figure(figsize=(7, 5))

plt.plot(
    hs,
    label="1 camada oculta"
)

plt.plot(
    hdeep,
    label="2 camadas ocultas"
)

plt.title("Figura 6: Perda de treino — comparação de profundidade")
plt.xlabel("Época")
plt.ylabel("Entropia cruzada categórica")
plt.legend()
plt.show()
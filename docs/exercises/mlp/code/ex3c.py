baseline = LogisticRegression(max_iter=2000)
baseline.fit(Xmt, ymt)

base_m = baseline.score(Xme, yme)

LR = 0.2
EPOCHS = 2200

shallow = MLP(
    layers=[4, 24, 3],
    task="multi",
    seed=42
)

hs = shallow.fit(
    Xmt,
    ymt,
    lr=LR,
    epochs=EPOCHS
)

acc_s = shallow.score(Xme, yme)

print(f"Acurácia da linha de base: {base_m:.2%}")
print(f"Acurácia do MLP: {acc_s:.2%}")

plt.figure(figsize=(7, 5))

plt.plot(
    np.arange(1, len(hs) + 1),
    hs,
    label="Treino"
)

plt.title("Figura 4: Perda de treino — multiclasse")
plt.xlabel("Época")
plt.ylabel("Entropia cruzada categórica")
plt.legend()

predictions = shallow.predict(Xme)
cm = confusion_matrix(yme, predictions)

print("Matriz de confusão:")
print(cm)

plt.figure(figsize=(7, 5))

plt.imshow(cm, cmap="Blues")
plt.colorbar(label="Quantidade")

for i in range(3):
    for j in range(3):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black"
        )

plt.xticks(range(3))
plt.yticks(range(3))

plt.title("Figura 5: Matriz de confusão — MLP")
plt.xlabel("Classe prevista")
plt.ylabel("Classe real")

pairs = {}

for i in range(3):
    for j in range(i + 1, 3):
        pairs[(i, j)] = int(cm[i, j] + cm[j, i])

most_confused = max(pairs, key=pairs.get)
total_errors = pairs[most_confused]

print(f"Par mais confundido: {most_confused}")
print(f"Erros bidirecionais: {total_errors}")
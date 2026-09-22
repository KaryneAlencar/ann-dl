w_initial_sep = rng.normal(0, 0.01, size=2)
b_initial_sep = 0.0

result_sep = train_perceptron(
    X_sep, y_sep,
    w_initial=w_initial_sep,
    b_initial=b_initial_sep,
    eta=0.01,
    max_epochs=100
)

print("Pesos iniciais:", w_initial_sep)
print("w final:", result_sep["w"])
print("b final:", result_sep["b"])
print("Épocas:", result_sep["epochs"])
print(f"Acurácia final: {result_sep['accuracy']:.4f} ({100*result_sep['accuracy']:.2f}%)")
print("Atualizações por época:", result_sep["updates_history"])

def plot_boundary(ax, w, b, x_limits, label, linestyle="-"):
    x1 = np.linspace(x_limits[0], x_limits[1], 300)

    if abs(w[1]) > 1e-12:
        x2 = -(w[0] * x1 + b) / w[1]
        ax.plot(x1, x2, linestyle=linestyle, linewidth=2, label=label)
    else:
        ax.axvline(-b / w[0], linestyle=linestyle, linewidth=2, label=label)


pred_sep = predict_perceptron(X_sep, result_sep["w"], result_sep["b"])
wrong_sep = pred_sep != y_sep

fig, ax = plt.subplots()
ax.scatter(X_sep[y_sep == 0, 0], X_sep[y_sep == 0, 1],
           alpha=0.5, s=18, label="Classe 0")
ax.scatter(X_sep[y_sep == 1, 0], X_sep[y_sep == 1, 1],
           alpha=0.5, s=18, label="Classe 1")

plot_boundary(
    ax, result_sep["w"], result_sep["b"],
    (X_sep[:, 0].min(), X_sep[:, 0].max()),
    "Fronteira de decisão"
)

if np.any(wrong_sep):
    ax.scatter(X_sep[wrong_sep, 0], X_sep[wrong_sep, 1],
               facecolors="none", edgecolors="black", s=80,
               linewidths=1.5, label="Mal classificados")

ax.set_title("Figura 2: Fronteira de decisão — dados separáveis")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend()
plt.tight_layout()
plt.show()

print("Pontos mal classificados:", wrong_sep.sum())
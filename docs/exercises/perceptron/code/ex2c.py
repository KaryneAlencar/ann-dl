pred_final = predict_perceptron(
    X_overlap, result_overlap["w"], result_overlap["b"]
)
pred_pocket = predict_perceptron(
    X_overlap, result_overlap["pocket_w"], result_overlap["pocket_b"]
)

wrong_final = pred_final != y_overlap
wrong_pocket = pred_pocket != y_overlap

fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(X_overlap[y_overlap == 0, 0], X_overlap[y_overlap == 0, 1],
           alpha=0.35, s=18, label="Classe 0")
ax.scatter(X_overlap[y_overlap == 1, 0], X_overlap[y_overlap == 1, 1],
           alpha=0.35, s=18, label="Classe 1")

x_limits = (X_overlap[:, 0].min(), X_overlap[:, 0].max())

plot_boundary(
    ax, result_overlap["w"], result_overlap["b"],
    x_limits, "Fronteira final", "-"
)
plot_boundary(
    ax, result_overlap["pocket_w"], result_overlap["pocket_b"],
    x_limits, "Fronteira pocket", "--"
)

ax.scatter(X_overlap[wrong_final, 0], X_overlap[wrong_final, 1],
           facecolors="none", edgecolors="black", s=50,
           linewidths=0.8, label="Erro — final")
ax.scatter(X_overlap[wrong_pocket, 0], X_overlap[wrong_pocket, 1],
           marker="x", s=30, label="Erro — pocket")

ax.set_title("Figura 5: Fronteiras final e pocket")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

epochs_overlap = np.arange(1, result_overlap["epochs"] + 1)

plt.figure()
plt.plot(epochs_overlap, result_overlap["accuracy_history"],
         label="Pesos atuais")
plt.plot(epochs_overlap, result_overlap["pocket_history"],
         label="Melhor até agora (pocket)", linewidth=2)
plt.title("Figura 6: Acurácia atual e pocket por época")
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.show()
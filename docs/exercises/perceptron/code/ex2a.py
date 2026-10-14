from perceptron_shared import np, plt, FIGURES, prepare_overlap
rng, X_overlap, y_overlap = prepare_overlap()

# Figura 4
plt.figure()
plt.scatter(X_overlap[y_overlap == 0, 0], X_overlap[y_overlap == 0, 1],
            alpha=0.5, s=18, label="Classe 0")
plt.scatter(X_overlap[y_overlap == 1, 0], X_overlap[y_overlap == 1, 1],
            alpha=0.5, s=18, label="Classe 1")
plt.title("Figura 4: Dados sobrepostos")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "fig4.png", dpi=150, bbox_inches="tight")
plt.show()

print("Shape de X:",X_overlap.shape)

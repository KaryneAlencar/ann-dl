from perceptron_shared import np, plt, FIGURES, prepare_sep
rng, X_sep, y_sep = prepare_sep()

# Figura 1
plt.figure()
plt.scatter(X_sep[y_sep == 0, 0], X_sep[y_sep == 0, 1],
            alpha=0.55, s=18, label="Classe 0")
plt.scatter(X_sep[y_sep == 1, 0], X_sep[y_sep == 1, 1],
            alpha=0.55, s=18, label="Classe 1")
plt.title("Figura 1: Dados linearmente separáveis")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "fig1.png", dpi=150, bbox_inches="tight")
plt.show()

print("Shape de X:",X_sep.shape)

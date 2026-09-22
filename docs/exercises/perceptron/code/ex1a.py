mu0 = np.array([1.5, 1.5])
mu1 = np.array([5.0, 5.0])
cov_sep = np.array([[0.5, 0.0],
                    [0.0, 0.5]])

X0 = rng.multivariate_normal(mu0, cov_sep, size=1000)
X1 = rng.multivariate_normal(mu1, cov_sep, size=1000)

X_sep = np.vstack([X0, X1])
y_sep = np.concatenate([
    np.zeros(1000, dtype=int),
    np.ones(1000, dtype=int)
])

print("Shape de X:", X_sep.shape)
print("Classe 0:", np.sum(y_sep == 0), "amostras")
print("Classe 1:", np.sum(y_sep == 1), "amostras")
mu_overlap_0 = np.array([3.0, 3.0])
mu_overlap_1 = np.array([4.0, 4.0])
cov_overlap = np.array([[1.5, 0.0],
                        [0.0, 1.5]])

X2_0 = rng.multivariate_normal(mu_overlap_0, cov_overlap, size=1000)
X2_1 = rng.multivariate_normal(mu_overlap_1, cov_overlap, size=1000)

X_overlap = np.vstack([X2_0, X2_1])
y_overlap = np.concatenate([
    np.zeros(1000, dtype=int),
    np.ones(1000, dtype=int)
])

print("Shape de X:", X_overlap.shape)
print("Classe 0:", np.sum(y_overlap == 0), "amostras")
print("Classe 1:", np.sum(y_overlap == 1), "amostras")
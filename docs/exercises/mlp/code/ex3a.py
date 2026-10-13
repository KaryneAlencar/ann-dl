Xm, ym = make_classification(
    n_samples=1500,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    n_repeated=0,
    n_classes=3,
    n_clusters_per_class=2,
    class_sep=1.0,
    random_state=42
)

Xmt, Xme, ymt, yme = train_test_split(
    Xm,
    ym,
    test_size=0.2,
    random_state=42,
    stratify=ym
)

print(f"Total de amostras: {len(Xm)}")
print(f"Amostras de treinamento: {len(Xmt)}")
print(f"Amostras de teste: {len(Xme)}")
print(f"Quantidade de features: {Xm.shape[1]}")
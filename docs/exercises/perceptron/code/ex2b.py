w_initial_overlap = rng.normal(0, 0.01, size=2)

result_overlap = train_perceptron(
    X_overlap, y_overlap,
    w_initial=w_initial_overlap,
    b_initial=0.0,
    eta=0.01,
    max_epochs=100,
    use_pocket=True
)

print("--- Pesos finais ---")
print("w final:", result_overlap["w"])
print("b final:", result_overlap["b"])
print(f"Acurácia final: {result_overlap['accuracy']:.4f} ({100*result_overlap['accuracy']:.2f}%)")

print("\n--- Pocket ---")
print("w pocket:", result_overlap["pocket_w"])
print("b pocket:", result_overlap["pocket_b"])
print(f"Acurácia pocket: {result_overlap['pocket_accuracy']:.4f} ({100*result_overlap['pocket_accuracy']:.2f}%)")
print("Época do melhor pocket:", result_overlap["pocket_epoch"])
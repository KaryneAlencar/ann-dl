def predict_perceptron(X, w, b):
    return (X @ w + b >= 0).astype(int)


def accuracy_perceptron(X, y, w, b):
    y_hat = predict_perceptron(X, w, b)
    return np.mean(y_hat == y)


def train_perceptron(X, y, w_initial, b_initial=0.0,
                     eta=0.01, max_epochs=100, use_pocket=False):

    w = w_initial.copy().astype(float)
    b = float(b_initial)

    accuracy_history = []
    updates_history = []

    best_w = w.copy()
    best_b = b
    best_accuracy = accuracy_perceptron(X, y, w, b)
    best_epoch = 0
    pocket_history = []

    for epoch in range(1, max_epochs + 1):
        updates = 0

        for xi, yi in zip(X, y):
            y_hat = 1 if np.dot(w, xi) + b >= 0 else 0
            error = yi - y_hat

            if error != 0:
                w += eta * error * xi
                b += eta * error
                updates += 1

                if use_pocket:
                    current_accuracy = accuracy_perceptron(X, y, w, b)

                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_w = w.copy()
                        best_b = b
                        best_epoch = epoch

        epoch_accuracy = accuracy_perceptron(X, y, w, b)
        accuracy_history.append(epoch_accuracy)
        updates_history.append(updates)

        if use_pocket:
            pocket_history.append(best_accuracy)

        if updates == 0:
            break

    return {
        "w": w,
        "b": b,
        "epochs": epoch,
        "accuracy": accuracy_perceptron(X, y, w, b),
        "accuracy_history": np.array(accuracy_history),
        "updates_history": np.array(updates_history),
        "pocket_w": best_w,
        "pocket_b": best_b,
        "pocket_accuracy": best_accuracy,
        "pocket_epoch": best_epoch,
        "pocket_history": np.array(pocket_history)
    }
from ex1c import train_model, run_epoch

def treinar_vae_2d(train_loader, val_loader, test_loader):
    # Reutiliza o mesmo treinamento, alterando apenas d para 2
    model, history = train_model(2, train_loader, val_loader)

    # Calcula os valores finais de reconstrução e KL no teste
    reconstruction, kl = run_epoch(model, test_loader)
    return model, history, reconstruction, kl

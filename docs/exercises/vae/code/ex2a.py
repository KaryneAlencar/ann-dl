from ex1c import train_model, run_epoch

def treinar_vae_2d(train_loader, val_loader, test_loader):
    model, history = train_model(2, train_loader, val_loader)
    reconstruction, kl = run_epoch(model, test_loader)
    return model, history, reconstruction, kl

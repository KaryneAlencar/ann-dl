import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.mu = nn.Linear(256, d)
        self.logvar = nn.Linear(256, d)
        self.decoder = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def loss_terms(x, xhat, mu, logvar):
    reconstruction = F.binary_cross_entropy(xhat, x, reduction='sum') / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return reconstruction, kl

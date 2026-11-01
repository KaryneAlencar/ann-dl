import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, d):
        super().__init__()

        # Encoder: reduz os 784 pixels para uma representação de 256 dimensões
        self.encoder = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        
        # Calcula a média e o logaritmo da variância da distribuição latente
        self.mu = nn.Linear(256, d)
        self.logvar = nn.Linear(256, d)

        # Decoder: reconstrói os 784 pixels a partir do vetor latente
        self.decoder = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):

        # Amostra ruído de uma distribuição normal padrão
        eps = torch.randn_like(mu)

        # Aplica a reparametrização para permitir a retropropagação
        return mu + torch.exp(0.5 * logvar) * eps

    def forward(self, x):
        # Obtém os parâmetros da distribuição latente
        mu, logvar = self.encode(x)

        # Amostra o vetor latente usando a reparametrização
        z = self.reparameterize(mu, logvar)

        # Reconstrói a entrada e retorna os parâmetros da distribuição
        return self.decoder(z), mu, logvar

def loss_terms(x, xhat, mu, logvar):

    # BCE: soma os erros dos 784 pixels e calcula a média por imagem
    reconstruction = F.binary_cross_entropy(xhat, x, reduction='sum') / x.size(0)

    # KL em forma fechada: regulariza a distribuição latente
    # em relação ao prior normal padrão N(0,I)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return reconstruction, kl

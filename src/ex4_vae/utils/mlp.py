from collections import OrderedDict
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, device, input_dim=784, hidden_dim=400, latent_dim=10):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(input_dim, hidden_dim)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(hidden_dim, latent_dim))
            ])
        )

        self.mu_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        self.decoder = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(2, latent_dim)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(latent_dim, hidden_dim)),
                ('relu3', nn.ReLU()),
                ('fc3', nn.Linear(hidden_dim, input_dim)),
                ('output', nn.Sigmoid())
            ])
        )

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = self.mu_layer(x), self.logvar_layer(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decode(z)
        return x_pred, mu, logvar

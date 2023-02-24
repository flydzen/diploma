# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch import optim
import matplotlib as plt
from tqdm import tqdm
import os

import utils.dataloader as dl

# %%
# dl.download()
data = dl.get_data(2)
# %%
just_glifs = data['data'].to_list()
just_glifs = np.array(just_glifs)


# %%


class SubModule(nn.Module):
    def __init__(self, in_features, g):
        super().__init__()
        self.in_features = in_features

        self.main_layers = [
            nn.Linear(in_features, in_features * 2),
            # nn.Linear(in_features * 2, in_features * 2),
            nn.Linear(in_features * 2, g * 2),
            nn.Linear(g * 2, g),
            # nn.Linear(g * 2, g * 2),
            nn.Linear(g, g * 2),
            nn.Linear(g * 2, in_features * 2),
            # nn.Linear(in_features * 2, in_features * 2),
            nn.Linear(in_features * 2, in_features)
        ]
        self.layers = []
        self.activate = nn.Tanh()
        for layer in self.main_layers:
            self.layers.append(layer)
            self.layers.append(self.activate)
        self.layer = nn.Sequential(*self.layers)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x: torch.Tensor, t):
        x = x.view(-1, self.in_features)
        return self.layer(x)


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, rows_num=64, cols_num=11, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.rows_num = rows_num
        self.cols_num = cols_num
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self) -> torch.Tensor:
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t) -> tuple[torch.Tensor, torch.Tensor]:
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e

    def sample_timesteps(self, n) -> torch.Tensor:
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.rows_num, self.cols_num)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        return x


# %%
def save_sampled(x: np.ndarray, name):
    assert x.shape == (64, 11)
    start = x[:, 3:5]
    end = x[:, -2:]
    plt.clf()
    for st, en in zip(start, end):
        plt.plot([st[0], en[0]], [st[1], en[1]])
    plt.savefig(f'imgs/{name}')


# %%
def train(lr, epochs, batch_size=12, run_name='run', device='cpu'):
    device = device
    model = SubModule(64 * 11, 200).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(rows_num=64, cols_num=11, device=device)

    for epoch in tqdm(range(epochs), desc="Starting new epoch"):
        np.random.shuffle(just_glifs)
        for i in range(len(just_glifs) // batch_size):
            images = torch.tensor(just_glifs[i * batch_size: (i + 1) * batch_size])
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'mse: {loss.item()}')

        sampled_images = diffusion.sample(model, n=1)
        save_sampled(sampled_images[0], f'{run_name}_{epoch}.png')
        # save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))


# %%
train(lr=3e-4, epochs=15, run_name='run1')
# %%

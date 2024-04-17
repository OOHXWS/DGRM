
import torch
import torch.nn as nn
import math


class Dis(nn.Module):
    def __init__(self, nb_item):
        super(Dis, self).__init__()
        self.emb_layer = nn.Linear(10, 10)
        self.dis = nn.Sequential(
            nn.Linear(nb_item*2+10, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, condition, predict, timesteps):
        time_emb = timestep_embedding(timesteps, 10)
        emb = self.emb_layer(time_emb)
        data = torch.cat([condition, predict, emb], dim=-1)
        out = self.dis(data)
        return out


class Gen(nn.Module):
    def __init__(self, nb_item):
        super(Gen, self).__init__()
        self.emb_layer = nn.Linear(10, 10)
        self.gen = nn.Sequential(
            nn.Linear(nb_item*2+10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, nb_item),
        )

    def forward(self, condition, noise, timesteps):
        time_emb = timestep_embedding(timesteps, 10)
        emb = self.emb_layer(time_emb)
        data = torch.cat([condition ,noise, emb],dim=-1)
        out = self.gen(data)
        return out

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


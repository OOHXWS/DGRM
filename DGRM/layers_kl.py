# import torch
# import torch.nn as nn
# import numpy as np
# import torch.nn.functional as F
#
# class Dis(nn.Module):
#     def __init__(self, nb_item):
#         '''
#         :param nb_item: 项目的数量
#         '''
#         super(Dis, self).__init__()
#         self.dis = nn.Sequential(
#             nn.Linear(nb_item*2, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, condition, predict):
#         data = torch.cat([condition, predict], dim=-1)
#         out = self.dis(data)
#         return out
#
#
# class Gen(nn.Module):
#     def __init__(self, nb_item):
#         super(Gen, self).__init__()
#         self.gen = nn.Sequential(
#             nn.Linear(nb_item, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, nb_item),
#             nn.Sigmoid()
#         )
#
#     def forward(self, purchase_vec):
#         out = self.gen(purchase_vec)
#         return out
#
#
# class SelfAttention(nn.Module):
#     def __init__(self, input_dim):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(input_dim, input_dim)
#         self.key = nn.Linear(input_dim, input_dim)
#         self.value = nn.Linear(input_dim, input_dim)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x, reliability_matrix):
#         q = self.query(x)
#         q = q.unsqueeze(0)
#         k = self.key(x)
#         k = k.unsqueeze(0)
#         v = self.value(x)
#
#         alpha = self.softmax(torch.bmm(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(x.size(-1)).float()))
#         r_alpha = reliability_matrix.unsqueeze(0).expand_as(alpha)
#         alpha = alpha * r_alpha
#         y = torch.bmm(alpha, v)
#         return y

import torch
import torch.nn as nn
import math


class Dis(nn.Module):
    def __init__(self, nb_item):
        '''
        :param nb_item: 项目的数量
        '''
        super(Dis, self).__init__()
        self.emb_layer = nn.Linear(10, 10)
        self.dis = nn.Sequential(
            nn.Linear(nb_item*2+10, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
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
            nn.Sigmoid()
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


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, input_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # 计算query、key和value
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)

        # 计算注意力权重
        attention_weights = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = self.softmax(attention_weights)

        # 根据注意力权重加权求和得到上下文向量
        out = torch.matmul(attention_weights, value)

        return out

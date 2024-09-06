import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
import random
from models import Attention

class CustomDNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, n_items, emb_size, time_type="cat", norm=False, dropout=0.5):
        super().__init__()
        self.n_items = n_items
        self.time_type = time_type
        self.emb_dim = emb_size
        self.norm = norm
        self.training = False
        self.cross_attention_layer = Attention.CrossAttention(self.n_items, self.n_items, 100, self.n_items) # (input dim for x_1 and x_2, internal dimension, output_dim)
        self.self_attention_layer= Attention.SelfAttention(self.n_items, 100, self.n_items) # input dim for x, internal dimension, output_dim
        self.guidance_embedding_layer = GuidanceEmbedding(self.n_items, self.emb_dim)

        # self.linear1 = nn.Linear(n_items*2, self.emb_dim)
        self.linear1 = nn.Linear(self.n_items + self.emb_dim, self.emb_dim)
        self.linear2 = nn.Linear(self.emb_dim, self.n_items)
        self.drop = nn.Dropout(dropout)
        # self.init_weights()
    
    def init_weights(self):
        
        
        # Xavier Initialization for weights
        size = self.linear1.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.linear1.weight.data.normal_(0.0, std)

        # Normal Initialization for weights
        self.linear1.bias.data.normal_(0.0, 0.001)
        
        # Xavier Initialization for weights
        size = self.linear2.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.linear2.weight.data.normal_(0.0, std)

        # Normal Initialization for weights
        self.linear2.bias.data.normal_(0.0, 0.001)
        
    
    def forward(self, x, timesteps, guidance, p_uncond): # no torch.tanh means much worse results, larger model means worse results
        if self.training: 
            guidance = guidance.cpu() 
            p_uncond_indices = random.sample( list( range( 0, len(guidance) ) ), int(len(guidance)*p_uncond))
            for i in range(0,len(guidance)):
                if i in p_uncond_indices:
                    guidance[i] = torch.from_numpy(np.zeros_like(guidance[i]))
            guidance = guidance.to("cuda")
        guidance = self.guidance_embedding_layer(guidance)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)

        # h = self.self_attention_layer(x) # results in 400 x n_items, b x n_items

        # h = self.cross_attention_layer(x, guidance) # results in 400 x n_items, b x emb_dim
        h = torch.cat([guidance, x], dim=-1) # results in 400 x (n_items + emb_dim)

        h = self.linear1(h) # 400 x 1000
        h = torch.tanh(h)

        h = self.linear2(h) # 400 x 2810, b x n_items
        
        return h
    
class GuidanceEmbedding(nn.Module):

    def __init__(self, n_items, emb_dim):
        super().__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim

        self.emb = nn.Linear(self.n_items, self.emb_dim)

    def forward(self, guidance):
        emb_guidance = self.emb(guidance)
        return emb_guidance
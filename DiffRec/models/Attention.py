import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
import random

class CrossAttention(nn.Module):

    def __init__(self, d_in_q, d_in_kv, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in_q, d_out_kq))
        self.W_key = nn.Parameter(torch.rand(d_in_kv, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in_kv, d_out_v))

    def forward(self, x_1, x_2):
        queries_1 = torch.matmul(x_1, self.W_query)

        keys_2 = torch.matmul(x_2, self.W_key)
        values_2 = torch.matmul(x_2, self.W_value)

        attn_scores = torch.matmul(queries_1, keys_2.T)
        attn_weights = torch.softmax(attn_scores / self.d_out_kq**0.5, dim=-1)
        context_vec = torch.matmul(attn_weights, values_2)

        return context_vec
    
class SelfAttention(nn.Module):

    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))

    def forward(self, x):
        keys = torch.matmul(x, self.W_key)
        queries = torch.matmul(x, self.W_query)
        values = torch.matmul(x, self.W_value)

        attn_scores = torch.matmul(queries, keys.T)
        attn_weights = torch.softmax(attn_scores / self.d_out_kq**0.5, dim=-1)
        context_vec = torch.matmul(attn_weights, values)
        
        return context_vec
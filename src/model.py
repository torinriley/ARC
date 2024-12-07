import torch
import torch.nn as nn
import math

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len).expand(x.size(0), seq_len).to(x.device)
        return self.dropout(self.token_embedding(x) + self.position_embedding(positions))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class Residual(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, d_k, mask=None, dropout=None):
        d_k = q.shape[-1]
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ v), attention_scores

    def forward(self, q, k, v, mask=None):
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)
        query = q.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2)
        key = k.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
        value = v.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, self.d_k, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.linear_o(x)

class TransformerBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual = nn.ModuleList([Residual(features, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual[1](x, self.feed_forward_block)
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> None:
        super().__init__()
        self.embedding = InputEmbedding(d_model, vocab_size, seq_len, dropout)
        self.positional_encoding = PositionalEncoding(d_model, seq_len, dropout)
        self.layers = nn.ModuleList([TransformerBlock(d_model, MultiHeadAttention(d_model, h, dropout), FeedForward(d_model, d_ff, dropout), dropout) for _ in range(N)])
        self.norm = LayerNorm(d_model)
        self.projection_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.projection_layer(x)

    @staticmethod
    def buildTransformer(vocab_size: int, seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> 'Transformer':
        return Transformer(vocab_size, seq_len, d_model, N, h, dropout, d_ff)
    
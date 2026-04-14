'''
Transformer from scratch.
'''

import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    # embed_size - the size of the input and output vectors
    # head_dim - the size of the q, k, v vectors
    def __init__(self, embed_size, head_dim):
        super().__init__()
        self.embed_size = embed_size
        self.head_dim = head_dim

        # get weight matrices for q, k, v
        self.q_w = nn.Linear(embed_size, head_dim) # creates a weight matrix internally
        self.k_w = nn.Linear(embed_size, head_dim)
        self.v_w = nn.Linear(embed_size, head_dim)
    
    # x - input
    def forward(self, x):

        # compute q, k, v
        q = self.q_w(x) # creates q by multiplying x by q_w
        k = self.k_w(x)
        v = self.v_w(x)

        # scores(scaling) - compute attention
        # score = QK^T / sqrt(d_k) where d_k is the dimension of the key vectors = head_dim
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5

        # weights - apply softmax
        # weights = softmax(scores)
        weights = torch.softmax(scores, dim=-1)

        # output - apply weights to v
        # output = weights @ v
        output = torch.matmul(weights, v)
        
        return output



class MultiHeadAttention(nn.Module):

    # embed_size - the size of the input vectors
    # num_heads - the number of attention heads
    # if it's a singular head, then num_heads = 1, then head_dim = embed_size
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.q_w = nn.Linear(embed_size, embed_size) # creates a weight matrix internally
        self.k_w = nn.Linear(embed_size, embed_size)
        self.v_w = nn.Linear(embed_size, embed_size)

        self.out_w = nn.Linear(embed_size, embed_size)
    
    # x - input
    # encoder_output - for cross-attention, the output of the encoder
    # mask - for masked self-attention, the mask will have 0s in positions that should be masked and 1s elsewhere
    def forward(self, x, encoder_output=None, mask=None):
        
        # compute q, k, v
        q = self.q_w(x) # creates q by multiplying x by q_w
        if encoder_output is not None: # for cross-attention
            k = self.k_w(encoder_output)
            v = self.v_w(encoder_output)
        else:
            k = self.k_w(x)
            v = self.v_w(x)

        # reshape into multiple heads
        batch_size, seq_length, _ = x.shape

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) # (batch_size, num_heads, seq_length, head_dim) - swapped middle values
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # scores(scaling) - compute attention
        # score = QK^T / sqrt(d_k) where d_k is the dimension of the key vectors = head_dim
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5

        # apply mask if provided (for Masked self-attention in the decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) # set masked positions to -inf so that softmax will give them 0 weight

        # weights - apply softmax
        # weights = softmax(scores)
        weights = torch.softmax(scores, dim=-1)

        # output - apply weights to v
        # output = weights @ v
        output = torch.matmul(weights, v)

        # output shape: (batch, num_heads, seq_length, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_size)
        # now shape: (batch, seq_length, embed_size) — heads concatenated back together

        # apply final linear transformation
        output = self.out_w(output)

        return output

class EncoderLayer(nn.Module):

    def __init__(self, embed_size, num_heads, ff_hidden_dim):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_size, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_size)
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # self-attention
        attn_output = self.self_attn(x)
        x = self.norm1(x + attn_output) # add & norm

        # feed-forward
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output) # add & norm

        return x

class DecoderLayer(nn.Module):

    def __init__(self, embed_size, num_heads, ff_hidden_dim):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_size, num_heads)
        self.cross_attn = MultiHeadAttention(embed_size, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_size)
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

    def forward(self, x, encoder_output, mask=None):
        # masked self-attention
        attn_output = self.self_attn(x, mask=mask)
        x = self.norm1(x + attn_output) # add & norm

        # cross-attention
        cross_attn_output = self.cross_attn(x, encoder_output=encoder_output)
        x = self.norm2(x + cross_attn_output) # add & norm

        # feed-forward
        ff_output = self.ff(x)
        x = self.norm3(x + ff_output) # add & norm

        return x

class Transformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, num_heads, ff_hidden_dim, num_layers, max_seq_length):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_seq_length, embed_size)
        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_size, num_heads, ff_hidden_dim) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(embed_size, num_heads, ff_hidden_dim) for _ in range(num_layers)])
        self.out_linear = nn.Linear(embed_size, tgt_vocab_size)

    def forward(self, src_seq, tgt_seq, mask=None):
        # encode
        x = self.src_embedding(src_seq)
        positions = torch.arange(0, src_seq.size(1)).unsqueeze(0).to(src_seq.device) # (1, seq_length)
        x = x + self.pos_embedding(positions) # add positional encoding
        for layer in self.encoder_layers:
            x = layer(x)
        encoder_output = x

        # decode
        y = self.tgt_embedding(tgt_seq)
        positions = torch.arange(0, tgt_seq.size(1)).unsqueeze(0).to(tgt_seq.device) # (1, seq_length)
        y = y + self.pos_embedding(positions) # add positional encodings
        for layer in self.decoder_layers:
            y = layer(y, encoder_output, mask=mask)
        
        output = self.out_linear(y)
        return output



# SANITY CHECK

if __name__ == "__main__":
    # Tiny test: can the transformer learn simple number mappings?
    # Source: [1, 2, 3] → Target: [4, 5, 6]
    
    src_vocab_size = 10
    tgt_vocab_size = 10
    embed_size = 64
    num_heads = 4
    ff_hidden_dim = 128
    num_layers = 2
    max_seq_length = 10

    model = Transformer(src_vocab_size, tgt_vocab_size, embed_size, num_heads, ff_hidden_dim, num_layers, max_seq_length)
    
    # Create a simple mask for decoder (lower triangular)
    def create_mask(size):
        mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
        return mask
    
    # Dummy training data
    src = torch.tensor([[1, 2, 3]])  # input sequence
    tgt = torch.tensor([[4, 5, 6]])  # target sequence
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    for epoch in range(200):
        mask = create_mask(tgt.size(1))
        output = model(src, tgt, mask)
        # output shape: (1, 3, 10) → reshape for loss
        loss = loss_fn(output.view(-1, tgt_vocab_size), tgt.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Test
    mask = create_mask(tgt.size(1))
    output = model(src, tgt, mask)
    predicted = output.argmax(dim=-1)
    print(f"Source: {src.tolist()}")
    print(f"Target: {tgt.tolist()}")
    print(f"Predicted: {predicted.tolist()}")
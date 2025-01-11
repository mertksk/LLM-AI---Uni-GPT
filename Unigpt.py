!wget https://raw.githubusercontent.com/mertksk/LLM-AI---Uni-GPT/refs/heads/main/data.txt

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

# hyperparameters
batch_size = 64  # number of independent sequences to process in parallel
context_length = 256  # maximum context length for predictions
max_iterations = 5000
eval_iterations = 500
learning_rate = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
latent_size = 384
attention_size = 6
num_layers = 6
dropout = 0.2
use_mixed_precision = device.type == 'cuda'  # Use mixed precision on GPU
num_latents = 64  # number of latent vectors
latent_dim = latent_size  # dimension of latent vectors
# ------------

torch.manual_seed(1337)

with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Unique characters in the text
characters = sorted(list(set(text)))
vocab_size = len(characters)
# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(characters)}
itos = {i: ch for i, ch in enumerate(characters)}
encode = lambda s: [stoi[c] for c in s]  # encoder: takes a string, outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: takes a list of integers, outputs a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, the rest will be validation
train_data = data[:n]
val_data = data[n:]

# Data loading function
def get_batch(split):
    # Generate a small batch of data with inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i + context_length] for i in ix])
    y = torch.stack([data[i + 1:i + context_length + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, latent_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, 4 * latent_size),
            nn.ReLU(),
            nn.Linear(4 * latent_size, latent_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class PerceiverCrossAttention(nn.Module):
    def __init__(self, d_model, latent_dim, num_latents, num_heads=1):
        super().__init__()
        self.num_latents = num_latents
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        # Latents are learnable parameters, independent of input length
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # Projection layers for Q (from latents), K and V (from input)
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(d_model, latent_dim)
        self.v_proj = nn.Linear(d_model, latent_dim)

        if num_heads > 1:
            assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"
            self.head_dim = latent_dim // num_heads
            self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        B, N, _ = x.size()

        # Expand latents for batch dimension
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, M, latent_dim)

        # Compute Q, K, V
        Q = self.q_proj(latents)  # (B, M, latent_dim)
        K = self.k_proj(x)        # (B, N, latent_dim)
        V = self.v_proj(x)        # (B, N, latent_dim)

        if self.num_heads > 1:
            Q = Q.reshape(B, self.num_latents, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, M, head_dim)
            K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)               # (B, heads, N, head_dim)
            V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)               # (B, heads, N, head_dim)

            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, heads, M, N)

            # Softmax over the last dimension (N)
            attn = F.softmax(scores, dim=-1)

            # Compute attention output
            out = torch.matmul(attn, V)  # (B, heads, M, head_dim)

            # Combine heads
            out = out.transpose(1, 2).reshape(B, self.num_latents, self.latent_dim)  # (B, M, latent_dim)
            out = self.out_proj(out)

        else:
            scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.latent_dim ** 0.5)  # (B, M, N)
            attn = F.softmax(scores, dim=-1)  # (B, M, N)
            out = torch.matmul(attn, V)       # (B, M, latent_dim)

        return out

class Block(nn.Module):
    """ Transformer block with Perceiver Cross-Attention """

    def __init__(self, latent_size, attention_size):
        super().__init__()
        self.perceiver = PerceiverCrossAttention(
            d_model=latent_size,
            latent_dim=latent_dim,
            num_latents=num_latents,
            num_heads=attention_size
        )
        self.ffwd = FeedForward(latent_size)
        self.ln1 = nn.LayerNorm(latent_size)
        self.ln2 = nn.LayerNorm(latent_size)
        
        # Additional projection to match dimensions
        self.output_proj = nn.Linear(num_latents * latent_dim, context_length * latent_size)

    def forward(self, x):
        B, T, C = x.shape
        
        # Apply layer norm before perceiver
        x_norm = self.ln1(x)
        
        # Pass through perceiver
        latent_output = self.perceiver(x_norm)  # (B, num_latents, latent_dim)
        
        # Reshape and project back to original sequence length
        latent_flat = latent_output.reshape(B, -1)  # (B, num_latents * latent_dim)
        output = self.output_proj(latent_flat)  # (B, context_length * latent_size)
        output = output.reshape(B, T, C)  # (B, T, C)
        
        # Add residual connection
        x = x + output
        
        # Feed forward part remains the same
        x = x + self.ffwd(self.ln2(x))
        return x

class UniGPT(nn.Module):

    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, latent_size)
        self.position_embedding_table = nn.Embedding(context_length, latent_size)
        self.blocks = nn.Sequential(*[Block(latent_size, attention_size=attention_size) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(latent_size)  # Final layer norm
        self.lm_head = nn.Linear(latent_size, vocab_size)

        # Better initialization, not covered in the original GPT video, but important
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last context_length tokens
            idx_cond = idx[:, -context_length:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

model = UniGPT()
m = model.to(device)
# Print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Create gradient scaler for mixed precision
scaler = GradScaler(enabled=use_mixed_precision)

for iter in range(max_iterations):
    if iter % eval_iterations == 0 or iter == max_iterations - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

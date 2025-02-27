#!/usr/bin/env python
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

def load_model_from_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def testModel():
    with open(args.work_dir + '/vocab_mappings.pkl', 'rb') as f:
        vocab_data = pickle.load(f)

    stoi = vocab_data['stoi']
    itos = vocab_data['itos']
    vocab_size = vocab_data['vocab_size']

    encode = lambda s: [stoi.get(c, stoi['<unk>']) for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    class Head(nn.Module):
        """ one head of self-attention """

        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # input of size (batch, time-step, channels)
            # output of size (batch, time-step, head size)
            B,T,C = x.shape
            k = self.key(x)   # (B,T,hs)
            q = self.query(x) # (B,T,hs)
            # compute attention scores ("affinities")
            wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
            wei = F.softmax(wei, dim=-1) # (B, T, T)
            wei = self.dropout(wei)
            # perform the weighted aggregation of the values
            v = self.value(x) # (B,T,hs)
            out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
            return out

    class MultiHeadAttention(nn.Module):
        """ multiple heads of self-attention in parallel """

        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(head_size * num_heads, n_embd)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out

    class FeedFoward(nn.Module):
        """ a simple linear layer followed by a non-linearity """

        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        """ Transformer block: communication followed by computation """

        def __init__(self, n_embd, n_head):
            # n_embd: embedding dimension, n_head: the number of heads we'd like
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedFoward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

    class GPTLanguageModel(nn.Module):

        def __init__(self):
            super().__init__()
            # each token directly reads off the logits for the next token from a lookup table
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd) # final layer norm
            self.lm_head = nn.Linear(n_embd, vocab_size)

            # better init, not covered in the original GPT video, but important, will cover in followup video
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

            # idx and targets are both (B,T) tensor of integers
            tok_emb = self.token_embedding_table(idx) # (B,T,C)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
            x = tok_emb + pos_emb # (B,T,C)
            x = self.blocks(x) # (B,T,C)
            x = self.ln_f(x) # (B,T,C)
            logits = self.lm_head(x) # (B,T,vocab_size)

            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)

            return logits, loss

        def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -block_size:]
                # get the predictions
                logits, loss = self(idx_cond)
                # focus only on the last time step
                logits = logits[:, -1, :] # becomes (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1) # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            return idx

        def generate_top_3(self, idx):
            # idx is (B, T) array of indices in the current context
            # Crop to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # Get the predictions
            logits, loss = self(idx_cond)

            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # Get the top 3 most probable tokens
            top_probs, top_indices = torch.topk(probs, 3, dim=-1)  # (B, 3)

            return top_indices  # Return the top 3 tokens (B, 3)
        
    model = GPTLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = load_model_from_checkpoint(model, optimizer, args.work_dir + '/model_checkpoint.pth', device)

    with open(args.test_output, 'w', encoding='utf-8') as output_file:
        with open(args.test_data, 'r', encoding='utf-8') as test_file:
            for line in test_file:
                input_text = line.strip()
                input_indices = torch.tensor(encode(input_text), dtype=torch.long, device=device).unsqueeze(0)
                generated_indices = model.generate_top_3(input_indices)
                generated_text = decode(generated_indices[0].tolist())
                output_file.write(generated_text + '\n')

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'test':
        testModel()
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))

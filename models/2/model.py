
import torch
import torch.nn as nn
import torch.nn.functional as F

document = ""
with open("../../data/sailors_home.txt", 'r', encoding='utf-8') as f:
            document = f.read()

class Tokenizer():
    def __init__(self):
        self.special_tokens = {"start_of_sequence": "<sos>", "unknown": "<unk>", "padding": "<pad>"}
        self.chars = sorted(list(set(document)) + list(self.special_tokens.values())) 
        self.stoi_dict = self.stoi()
        self.itos_dict = self.itos()
    
    def get_special_tokens(self):
         return self.special_tokens

    def get_vocab_size(self):
         return len(self.chars)

    def stoi(self):
        return { ch:i for i,ch in enumerate(self.chars) }
    def itos(self):
        return { i:ch for i,ch in enumerate(self.chars) }

    def encoder(self, input_string):
        encoded_list = []
        for i in input_string:
            if(i in self.chars):
                encoded_list.append(self.stoi_dict[i])
            else:
                encoded_list.append(self.stoi_dict['<unk>'])
        return encoded_list

    def decoder(self, tokens_list):
        decoded_string = ''
        for token in tokens_list:
            decoded_string = decoded_string + self.itos_dict[token]
        return decoded_string

tokenizer = Tokenizer()

block_size = 64 #context length
batch_size = 32
n_embd = 64
dropout = 0.4
n_head = 8
n_layer = 48
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = tokenizer.get_vocab_size()
learning_rate = 6 * 1e-6

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
            B,T,C = x.shape

            k = self.key(x)   # (B,T,C)
            q = self.query(x) # (B,T,C)

            # compute attention scores ("affinities")
            wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
            wei = torch.masked_fill(wei, torch.tril(torch.ones(T, T)) == 0, float("-inf") )
            wei = F.softmax(wei, dim = -1) # (B, T, T)
            wei = self.dropout(wei)
            
            # perform the weighted aggregation of the values
            v = self.value(x) # (B,T,C)
            masked_self_attention_scores = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
            return masked_self_attention_scores

class MultiHeadAttention(nn.Module):
        """ multiple heads of self-attention in parallel """

        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(n_embd, n_embd)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = torch.cat([h.forward(x) for h in self.heads], dim=-1)
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
            out = self.net(x);
            return out

class Block(nn.Module):
        """ Transformer block: communication followed by computation """

        def __init__(self):
            super().__init__()
            head_size = n_embd // n_head
            self.multi_head_masked_self_attention = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedFoward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x + self.multi_head_masked_self_attention.forward(self.ln1(x)) # x + is the residual connection
            x = x + self.ffwd(self.ln2(x))
            return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # I think It should not be an embedding.

        self.mul_head = MultiHeadAttention(n_head, n_embd // n_head)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

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
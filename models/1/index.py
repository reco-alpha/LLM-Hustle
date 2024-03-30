'''
Model Size: 2M
Data: sailor's_home.txt
Tokenizer: Character level.

Hyperparameters
    block_size = 64 #context length
    batch_size = 64
    n_embd = 64
    dropout = 0.2
    n_head = 8
    n_layer = 48
    learning_rate = 
    1000 iters: 3 * e-3 (other models' losses are not improving with lower lr 6*e-6)
    500 iters: 3 * 1e-6

Training

'''

import torch
import torch.nn.functional as F
from model import BigramLanguageModel, FeedFoward, Block, Head, MultiHeadAttention, Tokenizer
import sys
sys.path.append("../../") 

from utils import Utils 
u = Utils()

document = ""
with open("../../data/sailors_home.txt", 'r', encoding='utf-8') as f:
            document = f.read()

tokenizer = Tokenizer()
mode = 'pre-train'

#hyperparams
block_size = 64 #context length
batch_size = 32
n_embd = 64
dropout = 0.4
n_head = 8
n_layer = 48
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = tokenizer.get_vocab_size()
learning_rate = 3 * 1e-3
encoded_document = tokenizer.encoder(document)

def get_batch():
  xs = torch.randint((len(encoded_document) - block_size), (batch_size,)) # offsets | index
  X = torch.stack([torch.tensor(encoded_document[xi: xi + block_size]) for xi in xs])
  Y = torch.stack([torch.tensor(encoded_document[xi + 1 : xi + block_size + 1]) for xi in xs])
  return X, Y

def Inference(prompt, max_new_token, model):
  model.eval()
  for i in range(max_new_token):
    # cut the prompt to the block_size
    a = prompt.view(-1).tolist()[-block_size:]
    trimmed_prompt = torch.tensor(a).view(1, -1)
    logits, loss = model.forward(trimmed_prompt, targets = None)
    target_logit = logits[:, -1, :]
    probs = F.softmax(target_logit, dim=1).view(-1)
    pred_index = probs.argmax()
    prompt = torch.tensor(prompt.view(-1).tolist() + [pred_index]).view(1, -1)  
  return prompt.view(-1).tolist()

#Initializing Model
model = u.load_model(model_signature=BigramLanguageModel, mode=mode)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
params = 0

for p in model.parameters():
 params+=p.numel()
print(params/1e6, 'M parameters')

dummy_losses = []

# model.train()
# for i in range(1):
#   for iter in range(2000):
#       # sample a batch of data
#       xb, yb = get_batch()

#       # evaluate the loss
#       logits, loss = model.forward(xb, yb)
#       if(iter % 200 == 0):
#           print("loss, iter", (loss, iter))
#           dummy_losses.append((loss.item(), mode))
#       optimizer.zero_grad(set_to_none=True)
#       loss.backward()
#       optimizer.step()

#   # saving model & losses
#   u.save_model(model, mode=mode)
#   u.save_losses(dummy_losses)


# Running
# prompt = torch.tensor(tokenizer.encoder("Are you going to visit it?"))
# preds = tokenizer.decoder(Inference(prompt, 200, model))
# print(preds)




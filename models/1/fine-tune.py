
import torch
import torch.nn.functional as F
from model import BigramLanguageModel, FeedFoward, Block, Head, MultiHeadAttention, Tokenizer
import json
import sys
sys.path.append("../../") 

from utils import Utils 
u = Utils()
tokenizer = Tokenizer()
special_tokens = tokenizer.get_special_tokens()

learning_rate = 3 * 1e-3
block_size = 64 #context length
mode = 'fine-tune'

# Open the file and load the JSON data
with open("../../data/sailors_home_fine_tune.json", 'r') as file:
    fine_tune_data = json.load(file)


'''
List of X & Y tuples.
X: Entire example.
Y: Offset by 'ONE' token & ends with '<sos>' special token.
'''
def prepare_data():
   
    data = []
    for pair in fine_tune_data["chapter_1"]:
        example = ""
        prompt = pair["prompt"]
        response = pair["response"]
        example = f'''
        {prompt}\n\n{response}        
        '''

        sos_token = tokenizer.stoi()[special_tokens["start_of_sequence"]]
        pad_token = tokenizer.stoi()[special_tokens["padding"]]
        encoded_example = [sos_token] + tokenizer.encoder(example) + [sos_token]
        encoded_example_length = len(encoded_example)

        X = encoded_example[:encoded_example_length - 1]
        Y = encoded_example[1:]

        no_of_complete_sequences = int(len(X) // block_size)

        x = torch.stack([ torch.tensor(X[i * block_size : (i* block_size) +block_size]) for i in range(no_of_complete_sequences)])
        y = torch.stack([ torch.tensor(Y[i * block_size : (i* block_size) +block_size]) for i in range(no_of_complete_sequences)])

        if(len(X) > no_of_complete_sequences * block_size):
            remaining_x = torch.tensor(X[ - (len(X) - no_of_complete_sequences * block_size) :] + [pad_token for i in range(block_size - len(X) + no_of_complete_sequences * block_size )]).view(1, -1)
            remaining_y = torch.tensor(Y[ - (len(Y) - no_of_complete_sequences * block_size) :] + [pad_token for i in range(block_size - len(Y) + no_of_complete_sequences * block_size )]).view(1, -1)
            x = torch.concat((x, remaining_x), dim = 0)
            y = torch.concat((y, remaining_y), dim = 0)
        data.append((x, y))
    return data

        
# data = prepare_data()

model = u.load_model(model_signature=BigramLanguageModel, mode=mode)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

dummy_losses = []

# model.train()
# for i in range(1000):
#   for iter in range(len(data)):
#       # sample a batch of data
#       xb, yb = data[iter][0], data[iter][1]

#       # evaluate the loss
#       logits, loss = model.forward(xb, yb)
#       if(iter % len(data) == 0):
#           print("loss, iter", (loss, i))
#           dummy_losses.append((loss.item(), mode))

#       optimizer.zero_grad(set_to_none=True)
#       loss.backward()
#       optimizer.step()

# u.save_model(model, mode=mode)
# u.save_losses(dummy_losses)


def Inference(prompt, max_new_token, model):
  sos_token = tokenizer.stoi()['<sos>']
  model.eval()
  i= 0
  while(i<max_new_token):
    # cut the prompt to the block_size
    a = prompt.view(-1).tolist()[-block_size:]
    trimmed_prompt = torch.tensor(a).view(1, -1)
    logits, loss = model.forward(trimmed_prompt, targets = None)
    i+=1
    target_logit = logits[:, -1, :]
    probs = F.softmax(target_logit, dim=1).view(-1)
    pred_index = probs.argmax()
    prompt = torch.tensor(prompt.view(-1).tolist() + [pred_index]).view(1, -1) 

    if(pred_index == sos_token ):
       return prompt.view(-1).tolist()
     
  return prompt.view(-1).tolist()


# Running
prompt = torch.tensor(tokenizer.encoder("Describe Ned Franks' plans for the future."))
max_new_tokens = 400
preds = tokenizer.decoder(Inference(prompt, max_new_tokens, model))
print(preds)
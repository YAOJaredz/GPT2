from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch 
import math
from tqdm import tqdm
import numpy as np


def predict_next(model,tokenizer,input_ids,t):
    # extract the target word for matching
    target = t[0][-1]

    # run the model, get logits
    outputs = model.forward(input_ids)
    next_token_logits = outputs.logits[0, -1, :]

    # caluculate the probability
    prob = torch.softmax(next_token_logits,dim=0)
    target_prob = prob[target].item()

    # make judgements for accuracy
    guess = torch.argmax(next_token_logits)
    if guess == target:
        acc = True
    else:
        acc = False

    return [tokenizer.decode(target), target_prob, acc]

def analyze_sent(model, tokenizer, sent):
    model.eval()
    # tokenize the input sentence
    input_ids = tokenizer.encode(sent, return_tensors='pt')

    words_prob = []
    for i in tqdm(range(1,input_ids.shape[-1])):
        if i < 1024:
            words_prob.append(predict_next(model,tokenizer,input_ids[:,:i],input_ids[:,:i+1]))
        else:
            words_prob.append(predict_next(model,tokenizer,input_ids[:,i-1023:i],input_ids[:,i-1022:i+1]))
    print(words_prob)

    for i in words_prob:
        i[1] = -math.log(i[1])
    # print(words_prob)

    # calculate the accruracy on this sentence
    count = 0
    for i in words_prob:
        if i[2] == True:
            count += 1
    print(count/(input_ids.shape[-1]-1))

def analyze(
    text_path,
    gpt2_type = 'gpt2',
    model = GPT2LMHeadModel.from_pretrained('gpt2'), 
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    ):
    # Import the model: gpt2
    with open (text_path,'r') as file:
        sent = file.read()
    sent = 'One fish two fish red fish blue fish.'
    print("Calculating the probability for the text")
    analyze_sent(model,tokenizer, sent)

if __name__=='__main__':
    # input the file as string into sent 
    # sent only stores one sentence
    # model = GPT2LMHeadModel.from_pretrained('model')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(torch.load('server/next_pred-50.pt',map_location=torch.device('cpu')))
    # model = torch.load('model/trained_model.pt')
    # text_path = '/Users/jared/Desktop/Ezzyat_Memory_Lab/GPT-2/Speech2Text/Tunnel_text.txt'
    text_path = 'text.txt'
    analyze(text_path=text_path, model = model)

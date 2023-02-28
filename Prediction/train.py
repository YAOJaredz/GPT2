import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import random
import torch
import logging
import os
import csv
from prediction import analyze
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import warnings

def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = custom_formatwarning
warnings.warn("achtung")

# load the training dataset
class load_dataset(Dataset):
    
    def __init__(self, path, truncate=False, gpt2_type="gpt2", max_length=768):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.txt = []

        
        with open(path, 'r') as file:
            for line in iter(file.readline, ''):
                self.txt.append(torch.tensor(
                    self.tokenizer.encode(f'{line}<|endoftext|>')[:max_length]
                    ))
                print(line)
                
        if truncate:
            self.txt = self.txt[:20000]
        self.text_count = len(self.txt)
        
    def __len__(self):
        return self.text_count

    def __getitem__(self, item):
        return self.txt[item]


def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

def train(
    dataset,
    model,
    tokenizer,
    device='cuda',
    batch_size=16,
    epochs=4,
    lr=2e-5,
    max_seq_len=400,
    warmup_steps=5000,
    gpt2_type="gpt2",
    output_dir=".",
    output_prefix="wreckgar",
    test_mode=False,
    save_model_on_epoch=False,
):

    acc_steps = 100

    # model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        Warning(f"Training epoch {epoch}")
        for idx, entry in enumerate(train_dataloader):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            # input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            print(f'{idx}it - loss: -{loss}')
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    # torch.save(model,'model/trained_model.pt')
    print('Training Finished')
    return model

if __name__=='__main__':
    gpt2_type = 'gpt2'
    path = 'dataset/Gateway-preprocessed.txt'
    dataset = load_dataset(path)
    print(len(dataset))
    model = train(
    dataset,
    GPT2LMHeadModel.from_pretrained(gpt2_type),
    GPT2Tokenizer.from_pretrained(gpt2_type),
    device='cpu',
    batch_size=4,
    epochs=1,
    lr=2e-4,
    max_seq_len=140,
    warmup_steps=50,
    gpt2_type=gpt2_type,
    output_dir="model",
    output_prefix="next_pred",
    save_model_on_epoch=True
    )
    # model.eval()
    # analyze(text_path='text.txt', model=model)
    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    # model.load_state_dict(torch.load('model/next_pred-1.pt'))
    # model = torch.load('model/trained_model.pt')
    # analyze(text_path='text.txt', model = model)
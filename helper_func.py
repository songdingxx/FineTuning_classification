import torch
import pandas as pd
import random
import numpy as np
import datetime

from torch.utils.data import TensorDataset, random_split, DataLoader

def choose_device():
    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device

def read_sentiment_file(filename):
    df = pd.read_csv(filename)
    text = df.text.values
    labels = df.sentiment.values
    
    return text, labels

def tokenize_text(text, tokenizer):
    input_ids = []
    attention_masks = []
    for sent in text:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    return input_ids, attention_masks

def generate_dataset(text, labels, tokenizer):
    input_ids, attention_masks = tokenize_text(text, tokenizer)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

def generate_train_val_dataset(dataset, ratio):
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size

    return random_split(dataset, [train_size, val_size])

def generate_dataloader(dataset, sampler, batch_size=32):
    return DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size)

def init_tokenizer(model_type, desc):
    tokenizer = model_type.from_pretrained(desc, do_lower_case=True)
    return tokenizer

def init_model(model_type, desc):
    model = model_type.from_pretrained(
        desc,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    model.cuda()
    return model

def fix_random_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))
import os
import sys

import argparse
import logging

import torch
from torch.utils.data import (
    TensorDataset, DataLoader, 
    RandomSampler, SequentialSampler
)
from transformers import (
    BertForSequenceClassification, BertTokenizer, 
    AdamW, get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
import sklearn.metrics as metrics

# import dependencies for Debugging andd Profiling
from smdebug.pytorch import get_hook
from smdebug.pytorch import modes


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

hook = get_hook(create_if_not_exists=True)
logger.info(f"HOOK {hook}")


# initialize the tokenizer
logger.info("INITIALIZE BERT TOKENIZER")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def net():
    logger.info("INITIALIZE BERT MODEL MODEL FOR FINETUNING")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                          num_labels=4,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    
    return model


def data_reader(data):
    logger.info("READ DATA FILES")
    with open(data+"/lyrics.txt", "r") as f:
        lyrics = f.read()
        
    with open(data+"/labels.txt", "r") as f:
        labels = f.read()
        
    # remove empty
    lyrics = lyrics.split("\n")
    labels = labels.split("\n")
    lyrics.remove('')
    labels.remove('')
    
    return lyrics, labels


def encode_data(lyrics, labels, max_length):
    logger.info("ENCODE DATA")
    data = pd.DataFrame({"lyrics": lyrics, "quadrant": labels})
    data["quadrant"] = pd.to_numeric(data["quadrant"]) # labels to numbers

    encoded_data = tokenizer.batch_encode_plus(
        data.lyrics.values,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = encoded_data["input_ids"]
    attention_masks = encoded_data["attention_mask"]
    labels = torch.tensor(data.quadrant.values)
    
    return input_ids, attention_masks, labels


def create_data_loaders(train_data, valid_data, test_data, batch_size, max_length):
    # get the data sets
    logger.info("CREATING DATA LOADERS")
#     train_data = os.path.join(train_data, "train")
#     valid_data = os.path.join(valid_data, "valid")
#     test_data = os.path.join(test_data, "test")
    
    # read the data
    train_lyrics, train_labels = data_reader(train_data)
    valid_lyrics, valid_labels = data_reader(valid_data)
    test_lyrics, test_labels = data_reader(test_data)
    
    # encode the data
    train_input_ids, train_attention_masks, train_labels = encode_data(train_lyrics, train_labels, max_length)
    valid_input_ids, valid_attention_masks, valid_labels = encode_data(valid_lyrics, valid_labels, max_length)
    test_input_ids, test_attention_masks, test_labels = encode_data(test_lyrics, test_labels, max_length)
    
    # datasets
    train_set = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    valid_set = TensorDataset(valid_input_ids, valid_attention_masks, valid_labels)
    test_set = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    
    
    # create data loaders
    train_loader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size)
    valid_loader = DataLoader(valid_set, sampler=SequentialSampler(valid_set), batch_size=batch_size)
    test_loader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader


def train(model, train_loader, valid_loader, epochs, optimizer, scheduler, device):
    logger.info("BEGIN TRAINING")
    for i in range(epochs):
        if hook:
            hook.set_mode(modes.TRAIN)
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
                  }
            
            optimizer.zero_grad()
            output = model(**inputs)
            loss = output[0]
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        
        if hook:
            hook.set_mode(modes.EVAL)   
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = tuple(b.to(device) for b in batch)

                inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2]
                }

                output = model(**inputs)
                loss = output[0]
                val_loss += loss.item()
                
        logger.info(f"Epoch: {i}, Train loss: {train_loss/len(train_loader):.3f}, Val loss: {val_loss/len(valid_loader):.3f}")
    logger.info("COMPLETE TRAINING")


def test(model, test_loader):
    logger.info("BEGIN TESTING")
    if hook:
        hook.set_mode(modes.EVAL)
    model.to("cpu")
    model.eval()

    test_loss = 0
    y_pred, y_true = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2]
            }


            outputs = model(**inputs)
            loss = outputs[0]
            logits = outputs[1]
            test_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs["labels"].cpu().numpy()
            y_pred.append(logits)
            y_true.append(label_ids)

    predictions = np.concatenate(y_pred, axis=0)
    true_vals = np.concatenate(y_true, axis=0)
    
    logger.info(metrics.classification_report(list(true_vals), [_.argmax(0) for _ in predictions]))
    logger.info(f"Test Loss: {test_loss/len(test_loader):.4f}")
    logger.info("COMPLETE TESTING")
    


def save_model(model, model_dir):
    logger.info("SAVING MODEL WEIGHTS")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    

def main(args):
    # set up device for gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(F"DEVICE {device}")
    
    # move model to gpu
    model = net()
    model.to(device)
    
    # optimizer and linear scheduler
    # https://stackoverflow.com/questions/60120043/optimizer-and-scheduler-for-bert-fine-tuning
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=args.batch_size*args.epochs)
    
    # data loaders
    logger.info(f"TRAIN DATA DIRECTORY {args.train_dir}")
    logger.info(f"VALID DATA DIRECTORY {args.valid_dir}")
    logger.info(f"TEST DATA DIRECTORY {args.test_dir}")
    train_loader, valid_loader, test_loader = create_data_loaders(args.train_dir, args.valid_dir, args.test_dir, args.batch_size, args.max_length)
    
    # train model
    train(model, train_loader, valid_loader, args.epochs, optimizer, scheduler, device)
    
    # test model
    test(model, test_loader)
    
    # save model
    save_model(model, args.model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        metavar="N",
        help="input max length for encoders (default: 128)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for datasets (default: 64)",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        metavar="N",
        help="number of epochs to train (default: 4)",
    )
    
    parser.add_argument(
        "--lr",
        type=float, 
        default=2e-5, 
        metavar="LR", 
        help="learning rate (default: 2e-5)",
    )
    
    # container specific arguments
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--valid-dir", type=str, default=os.environ["SM_CHANNEL_VALID"])
    parser.add_argument("--test-dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    
    args = parser.parse_args()
    
    main(args)
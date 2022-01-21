import os
import logging
import sys

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import json


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        
def net():
    logger.info("INITIALIZE BERT MODEL MODEL FOR FINETUNING")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                          num_labels=4,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    
    return model


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"DEVICE: {device}")
    
    logger.info("LOADING MODEL WEIGHTS")
    model = net().to(device)
    
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    return model



def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
    
        encoded_data = tokenizer.encode(
            data,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoded_data["input_ids"]
        attention_masks = encoded_data["attention_mask"]

        return input_ids, attention_masks


def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_ids, attention_masks = input_data
    input_ids.to(device)
    attention_masks.to(device)

    with torch.no_grad():
        return model(input_ids, token_type_ids=None, attention_mask=attention_masks)[0]

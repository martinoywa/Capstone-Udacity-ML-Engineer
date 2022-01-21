import os
import logging
import sys

import torch
from transformers import BertForSequenceClassification


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

        
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
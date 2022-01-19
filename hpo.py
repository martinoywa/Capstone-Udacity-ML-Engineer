import argparse
import logging

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
    

def main(args):
    # set up device for gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(F"DEVICE {device}")
    
    # move model to gpu
    model = net()
    model.to(deviceice)

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
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    
    args = parser.parse_args()
    
    main(args)
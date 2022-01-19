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


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


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
    with open(train_data+"/lyrics.txt", "r") as f:
        lyrics = f.read()
        
    with open(train_data+"/labels.txt", "r") as f:
        labels = f.read()
    
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


def create_dataloaders(data, batch_size, max_length):
    # get the data sets
    logger.info("CREATING DATA LOADERS")
    train_data = os.path.join(data, "train")
    valid_data = os.path.join(data, "valid")
    test_data = os.path.join(data, "test")
    
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
    
    return train_loader, valid_loader_ test_loader
    

def main(args):
    # set up device for gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(F"DEVICE {device}")
    
    # move model to gpu
    model = net()
    model.to(deviceice)
    
    # optimizer and linear scheduler
    # https://stackoverflow.com/questions/60120043/optimizer-and-scheduler-for-bert-fine-tuning
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=args.batch_size*args.epochs)
    
    # data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size, args.max_length)


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
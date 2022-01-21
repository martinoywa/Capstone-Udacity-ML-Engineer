import os
import logging
import sys

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import json

import re


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
MAX_LEN= 128

# preprocess the lyrics column
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re
contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = {"you've", 'can', 'through', 'those', 'and', 'below', 'd', 're', 'few', "wouldn't", 'weren', 'themselves', 'while', "haven't", 'you', 'some', 'after', 'nor', "needn't", 'until', 'yourselves', 'off', 'so', 'them', 'how', "don't", 'we', 'hasn', 'this', 'but', 'hadn', 'against', 'will', 'here', 'yourself', 'which', 'a', 'where', 've', 'such', 'won', 'whom', 'its', 'myself', 'who', 'once', 'other', 'above', 'an', 'more', 'there', 'mightn', 'each', 'your', 'my', "you'll", 'between', 'haven', 'with', 'now', 'yours', 'y', 'very', "you're", 'same', 'he', 'her', 'further', "shan't", 'under', 'out', 'mustn', 'does', 'just', 'before', "isn't", 'that', 'our', 'himself', 'didn', 'what', 'i', 'on', 'o', 'his', 'they', 'ourselves', 'shouldn', 'own', 'm', "you'd", 'into', 'the', 'too', "should've", 'been', 'do', "aren't", 'have', 'll', 'in', 'am', "she's", 'doing', 'for', 'these', 'than', 'at', 't', 'because', 'has', 'about', 'no', 'herself', "shouldn't", 'hers', "didn't", 'be', 'over', 'when', 'it', "it's", 'if', 'their', "hadn't", 'isn', 'up', 'having', 'aren', 'couldn', 'is', 'ours', 'ma', 'had', 'him', 'as', 'ain', "mustn't", 'itself', 'was', 'me', 'theirs', "couldn't", 'being', 'she', 'why', 'shan', 'wasn', 'any', 'of', 's', 'only', 'down', 'both', "weren't", 'are', "hasn't", 'needn', 'most', 'not', 'all', "that'll", "mightn't", 'then', 'don', 'did', 'to', 'or', 'wouldn', 'during', 'were', 'doesn', 'by', 'should', 'again', "doesn't", "wasn't", 'from', "won't"}

def text_preprocessing(text):
    text = text.lower() # lower text
    text = replace_contractions(text) # remove contactions
    text = "".join("".join(text).replace("\n", " ").replace("\r", " ")) # remove \n and \r
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace symbols with space
    text = BAD_SYMBOLS_RE.sub('', text) # replace bad characters with nothing
    text = re.sub(r'[0-9]', '', text) # remove residual numbers
    text = text.strip()
    text = " ".join([word for word in text.split() if word not in STOPWORDS]) # remove stopwords
    
    return text


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
    """An input_fn that loads a pickled tensor"""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        logger.info(f"INPUT DATA: {data}")
        
        encoded_data = tokenizer.encode_plus(
            data,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        logger.info(f"ENCODED DATA: {encoded_data}")
        input_ids = encoded_data["input_ids"]
        attention_masks = encoded_data["attention_mask"]

        return input_ids, attention_masks
    

def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    logger.info(f"ENCODED INPUT DATA: {input_data}")
    input_ids, attention_masks = input_data
    input_ids.to(device)
    attention_masks.to(device)

    with torch.no_grad():
        logger.info("MAKE PREDICTION")
        output = model(input_ids, attention_mask=attention_masks)[0]
        _, pred = torch.max(output, dim=1)
        return pred.tolist()[0]

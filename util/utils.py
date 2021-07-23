import os
import torch
import yaml
import time
import argparse
import numpy as np
import random
import torch.nn.functional as F


def load_args(config):
    with open(config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg['timestamp'] = str(time.strftime("%Y-%m_%d-%H_%M_%S", time.localtime()))

    # cfg to args
    parser = argparse.ArgumentParser()
    for key, value in cfg.items():
        parser.add_argument('--{}'.format(key), type=type(value), default=value)
    args = parser.parse_args()

    device = 'cpu'
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print("Using device:{0} ".format(torch.cuda.current_device()))
        device = 'cuda'
    device = torch.device(device)
    args.device = device
    
    args.train = 'data/{}/train.csv'.format(args.dataset)
    args.test = 'data/{}/test.csv'.format(args.dataset)
    args.valid = 'data/{}/valid.csv'.format(args.dataset)

    return args

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def compute_logProb(X, score):
    logProb = F.softmax(score, dim=-1)  # bsz, seq_len, vocab_size
    logProb_X = torch.zeros(X.shape[0], X.shape[1])
    for i in range(X.shape[0]):   # bsz
        for j in range(X.shape[1]):   # seq_len
            logProb_X[i][j] = logProb[i][j][X[i][j]]
    return torch.log(logProb_X)

def init_tokenizer(gpt_path):
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
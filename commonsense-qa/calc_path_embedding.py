import argparse
import os
import time
import random
import numpy as np 
import logging
import sys
import json
import math
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
# from torch.utils.tensorboard import SummaryWriter
from transformers import *

from utils.data_helper import DataHelper
from utils.path_embedding import save_path_embedding
from modeling.generator import *

torch.set_num_threads(4)

# for REPRODUCIBILITY
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = logging.getLogger()

def run_generating(args):

    # ----------------------------------------------------- #
    log_path = os.path.join('./saved_models/pretrain_generator', 'run.log')

    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path, 'w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info('args: {}'.format(args))

    # ----------------------------------------------------- #
    # load data & init model and optimizer

    logger.info('Loading data & model')

    config = GPT2Config.from_pretrained(args.generator_type, cache_dir='../cache/')
    datahelper = DataHelper(args)

    path_embedding_file = os.path.join('./path_embeddings/', args.data_dir, 'path_embedding.pickle')

    # self define lm head gpt2
    gpt = GPT2Model.from_pretrained(args.generator_type, cache_dir='../cache/')
    config.vocab_size = len(datahelper.gpt_tokenizer)
    gpt.resize_token_embeddings(len(datahelper.gpt_tokenizer))
    pretrain_generator_ckpt = os.path.join('./saved_models/pretrain_generator', 'model.ckpt')
    generator = Generator(gpt, config, max_len=args.output_len).to(args.device)
    generator.load_state_dict(torch.load(pretrain_generator_ckpt, map_location=args.device))

    save_path_embedding(datahelper, generator, path_embedding_file, args)
    print('Finish.')

def main():
    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--gen_dir', type=str, default='./checkpoints/pretrain_generator')
    parser.add_argument('--gen_id', type=int)
    parser.add_argument('--sparsity', type=float, default=1.0)

    # model
    parser.add_argument('--generator_type', type=str)
    parser.add_argument('--text_encoder', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--run_id', type=int)
    parser.add_argument('--output_len', type=int)
    parser.add_argument('--context_len', type=int)
  
    # gpu option
    parser.add_argument('--gpu_device', type=str, default='0')

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu_device) if torch.cuda.is_available() else 'cpu')
    # ----------------------------------------------------- #

    run_generating(args)

if __name__ == '__main__':
    main()
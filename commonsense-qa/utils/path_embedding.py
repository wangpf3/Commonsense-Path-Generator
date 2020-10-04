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
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

def _get_path_embedding_greedy(dataset, generator, args, tokenizer=None, output_file=None):
    data_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=args.batch_size)
    generator.eval()
    epoch_iterator = tqdm(dataloader, desc="Path Generation")
    path_embeddings = []
    for step, context in enumerate(epoch_iterator):

        # questions, contexts, answers, choices = batch
        context = context[0].to(args.device)

        with torch.no_grad():
            batch_size, num_choice, num_context, context_len = context.size()
            context = context.view(-1, context_len)
            context_embedding, generated_paths = generator(context, train=False, return_path=True)
            context_embedding = context_embedding.view(batch_size, num_choice, num_context, -1)

        if not output_file is None:
            for path in generated_paths:
                path = tokenizer.decode(path.tolist(), skip_special_tokens=True)
                path = ' '.join(path.replace('<PAD>', '').split())
                output_file.write(path+'\n')

        path_embeddings.extend(context_embedding.tolist())

    path_embeddings = torch.tensor(path_embeddings, dtype=torch.float)
    return path_embeddings

def save_path_embedding(datahelper, generator, save_file, args):
    path_embeddings_dict = {}
    path_embeddings_dict['train'] = _get_path_embedding_greedy(datahelper.trainset, generator, args)
    path_embeddings_dict['dev'] = _get_path_embedding_greedy(datahelper.devset, generator, args)
    path_embeddings_dict['test'] = _get_path_embedding_greedy(datahelper.testset, generator, args)

    with open(save_file, 'wb') as handle:
        pickle.dump(path_embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


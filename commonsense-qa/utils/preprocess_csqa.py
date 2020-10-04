import os
import pickle
import torch
import json
from collections import defaultdict, OrderedDict
import random
from tqdm import tqdm, trange
from transformers import *

class PreprocessData_Ground(object):
    """docstring for PreprocessData"""
    def __init__(self, data_name, gpt_tokenizer_type, context_len):
        super(PreprocessData_Ground, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_tokenizer_type, cache_dir='../cache/')
        data_dir = os.path.join('./data', data_name)
        self.ground_path = os.path.join(data_dir, 'ground_token_context{}_{}.pkl'.format(context_len, gpt_tokenizer_type))
        self.context_len = context_len

        self.tokenizer.add_tokens(['<PAD>'])
        self.tokenizer.add_tokens(['<SEP>'])
        self.tokenizer.add_tokens(['<END>'])
        self.PAD = self.tokenizer.convert_tokens_to_ids('<PAD>')
        self.SEP = self.tokenizer.convert_tokens_to_ids('<SEP>')
        self.END = self.tokenizer.convert_tokens_to_ids('<END>')

        if not os.path.exists(self.ground_path):
            train_context_path = os.path.join(data_dir, 'grounded', 'train.grounded.jsonl')
            train_contexts = self.load_context(train_context_path)
            dev_context_path = os.path.join(data_dir, 'grounded', 'dev.grounded.jsonl')
            dev_contexts = self.load_context(dev_context_path)
            test_context_path = os.path.join(data_dir, 'grounded', 'test.grounded.jsonl')
            test_contexts = self.load_context(test_context_path)

            token_dataset = {}
            token_dataset['train'] = train_contexts
            token_dataset['dev'] = dev_contexts
            token_dataset['test'] = test_contexts

            with open(self.ground_path, 'wb') as handle:
                pickle.dump(token_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_context(self, data_path):
        data_context = []
        question_context = []
        with open(data_path, 'r') as fr:
            for _id, line in enumerate(tqdm(fr)):
                obj = json.loads(line)
                qc_list = obj['qc']
                ac_list = obj['ac']
                choice_context = []

                choice = obj['ans'].replace('_', ' ')

                for qc in qc_list[:5]:
                    qc = qc.replace('_', ' ')

                    context = choice + '<SEP>' + qc
                    context = self.tokenizer.encode(context, add_special_tokens=False)[:self.context_len]
                    context += [self.PAD] * (self.context_len - len(context))

                    choice_context.append(context)
                num_context = len(choice_context)
                for _ in range(5 - num_context):
                    _input = [self.PAD] * self.context_len
                    choice_context.append(_input)
                question_context.append(choice_context)
                if (_id + 1) % 5 == 0:
                    data_context.append(question_context)
                    question_context = []
        data_context = torch.tensor(data_context, dtype=torch.long)
        return data_context 


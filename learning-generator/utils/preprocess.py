import os
import pickle
import torch
import json
from collections import defaultdict, OrderedDict
import random
from tqdm import tqdm, trange

class PreprocessData(object):
    """docstring for PreprocessData"""
    def __init__(self, data_dir, tokenizer=None):
        super(PreprocessData, self).__init__()
        train_path = os.path.join(data_dir, 'train.txt')
        dev_path = os.path.join(data_dir, 'dev.txt')
        test_path = os.path.join(data_dir, 'test.txt')
        rel2text_path = os.path.join(data_dir, 'relation2text.json')
        relation_vocab_path = os.path.join(data_dir, 'relation_vocab.pkl')
        self.token_path = os.path.join(data_dir, 'token_gpt.pkl')

        self.tokenizer = tokenizer

        self.tokenizer.add_tokens(['<PAD>'])
        self.tokenizer.add_tokens(['<SEP>'])
        self.tokenizer.add_tokens(['<END>'])
        self.PAD = self.tokenizer.convert_tokens_to_ids('<PAD>')
        self.SEP = self.tokenizer.convert_tokens_to_ids('<SEP>')
        self.END = self.tokenizer.convert_tokens_to_ids('<END>')
        # self.load_relation_vocab(relation_vocab_path)

        if not os.path.exists(self.token_path):

            token_dataset = {}
            token_dataset['train'] = self.text2token_cnpt(train_path)
            token_dataset['dev'] = self.text2token_cnpt(dev_path)
            token_dataset['test'] = self.text2token_cnpt(test_path)

            with open(self.token_path, 'wb') as handle:
                pickle.dump(token_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_relation2text(self, data_path):
        with open(data_path, 'r') as fr:
            self.relation2text = json.load(fr)
        rel2id = {}
        id2rel = []
        for rel in self.relation2text:
            rel2id[rel] = len(id2rel)
            id2rel.append(rel)
        self.num_relation = len(id2rel)
        self.r2i = rel2id
        self.i2r = id2rel

    def load_relation_vocab(self, data_path):
        with open(data_path, 'rb') as fr:
            rel_vocab = pickle.load(fr)
        self.r2i = rel_vocab['r2i']
        self.i2r = rel_vocab['i2r']
        self.num_relation = len(self.i2r)
        for rel in self.i2r:
            # self.tokenizer.add_tokens(['<' + rel + '>'])
            rel = '<' + rel.replace('<', '').replace('>', '') + '>'
            self.tokenizer.add_tokens([rel])

    def text2token_cnpt(self, data_path):
        input_list = []
        # cnt_line = 0
        max_context_len = 16
        max_label_len = 31
        with open(data_path, 'r') as fr:
            for line in tqdm(fr, desc=data_path):
                line_split = line.strip().split('\t')
                current_idx = 0
                text = ''
                for _idx, element in enumerate(line_split[1:]):

                    if _idx % 2 != 0:
                        ent_words = element.replace('_', ' ')
                        text += ent_words
                    else:
                        text += ' ' + element + ' '
                _input = self.tokenizer.encode(text)[:max_label_len]
                _input += [self.PAD] * (max_label_len - len(_input))
                context = line_split[-1].replace('_', ' ') + '<SEP>' + line_split[0].replace('_', ' ')
                context = self.tokenizer.encode(context)[:max_context_len]
                context += [self.PAD] * (max_context_len - len(context))
                _input = (context + _input + [self.END])

                input_list.append(_input)

        return input_list


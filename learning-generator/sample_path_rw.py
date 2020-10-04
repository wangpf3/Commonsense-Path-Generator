import networkx as nx
import itertools
import math
import random
import json
from tqdm import tqdm
import sys
import time
import timeit
import nltk
import json
import pickle
import os
import numpy as np

def load_rel_weight(data_dir):
    freq_path = os.path.join(data_dir, 'relation_freq.pkl')
    with open(freq_path, 'rb') as handle:
        rel_weight = pickle.load(handle)
    return rel_weight

def load_kg(data_dir):
    print("loading cpnet....")
    data_path = os.path.join(data_dir, 'conceptnet_graph.nx')
    kg_full = nx.read_gpickle(data_path)

    kg_simple = nx.DiGraph()
    for u, v, data in kg_full.edges(data=True):
        kg_simple.add_edge(u, v)

    return kg_full, kg_simple

def random_walk(start_node, kg_full, kg_simple, max_len=3):


    edges_before_t_iter = 0
    # while True:
    #     curr_node = random.randint(0, nr_nodes-1) 
    #     if curr_node in kg_simple:
    #         break
    curr_node = start_node
    num_sampled_nodes = 1
    path = [curr_node]
    node_visited = set()
    relation_visited = set()
    node_visited.add(curr_node)
    while num_sampled_nodes != max_len:
        edges = [n for n in kg_simple[curr_node]]
        iteration_node = 0
        flag_valid = False
        while True:
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            if not chosen_node in node_visited:
                rel_list = kg_full[curr_node][chosen_node]
                rel_list = list(set([rel_list[item]['rel'] for item in rel_list]))
                iteration_rel = 0
                while True:
                    index_of_rel = np.random.choice(len(rel_list), 1)[0]
                    chosen_rel = kg_full[curr_node][chosen_node][index_of_rel]['rel']
                    if not ((chosen_rel) % 40) in relation_visited:
                        flag_valid = True
                        break
                    else:
                        iteration_rel += 1
                        if iteration_rel >= 3:
                            break
                if flag_valid:
                    break
                else:
                    iteration_node += 1
                    if iteration_node >= 3:
                        return []
            else:
                iteration_node += 1
                if iteration_node >= 3:
                    return []
        node_visited.add(chosen_node)
        relation_visited.add(chosen_rel % 40)
        path.append(chosen_rel)
        path.append(chosen_node)

        curr_node = chosen_node
        num_sampled_nodes += 1

    return path 

def load_vocab(data_dir):
    rel_path = os.path.join(data_dir, 'relation_vocab.pkl')
    ent_path = os.path.join(data_dir, 'entity_vocab.pkl')

    with open(rel_path, 'rb') as handle:
        rel_vocab = pickle.load(handle)

    with open(ent_path, 'rb') as handle:
        ent_vocab = pickle.load(handle)

    return rel_vocab['i2r'], rel_vocab['r2i'], ent_vocab['i2e'], ent_vocab['e2i']

def path2str(path, i2r, i2e, r2i):
    str2w = []
    _idx = 0
    ent = i2e[path[_idx]]
    str2w.append(ent)
    _idx += 1
    while _idx < len(path):
        rel = i2r[path[_idx]]
        _idx += 1
        ent = i2e[path[_idx]]
        _idx += 1
        str2w.append(rel)
        str2w.append(ent)

    str2w = '\t'.join(str2w) + '\n'
    return str2w

if __name__ == '__main__':
    data_dir = './data/conceptnet/'
    output_dir = './data/sample_path'
    num_paths = [10, 8, 6]
    path_lens = [2, 3, 4] # 1,2,3 hop

    print('loading relation weight and vocab')
    # relation_freq = load_rel_weight(data_dir)
    i2r, r2i, i2e, e2i = load_vocab(data_dir)
    print('num of entities: {}'.format(len(i2e)))
    print('num of relations: {}'.format(len(i2r)))
    # for rel in relation_freq:
    #     print('{}: {:.3f}'.format(i2r[rel], relation_freq[rel]))

    print('loading kg')
    kg_full, kg_simple = load_kg(data_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'sample_path.txt')

    nr_nodes = len(kg_full.nodes())
    visited_concept = dict()
    num_visited_concept = len(visited_concept)
    not_visited_idx = []
    for cnpt_idx in range(nr_nodes):
        if not i2e[cnpt_idx] in visited_concept:
            not_visited_idx.append(cnpt_idx)
    print('not visited: {}'.format(len(not_visited_idx)))

    with open(output_path, 'w') as fw:
        for curr_node in tqdm(not_visited_idx, desc='generating paths'):
            if not curr_node in kg_simple:
                continue 
            for _id, _len in enumerate(path_lens):
                visited_path = set()
                for pid in range(num_paths[_id]):
                    num_try = 0
                    while True:
                        path = random_walk(curr_node, kg_full, kg_simple, _len)
                        if len(path) > 0:
                            str2w = path2str(path, i2r, i2e, r2i)
                            if str2w not in visited_path:
                                fw.write(str2w)
                                visited_path.add(str2w)
                                break
                        num_try += 1
                        if num_try >10:
                            break

    fw.close()
    print('finish!')

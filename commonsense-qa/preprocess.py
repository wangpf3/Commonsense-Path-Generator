import os
import argparse
from multiprocessing import cpu_count
from utils.convert_csqa import convert_to_entailment
from utils.convert_obqa import convert_to_obqa_statement
from utils.tokenization_utils import tokenize_statement_file, make_word_vocab
from utils.conceptnet import extract_english, construct_graph
from utils.embedding import glove2npy, load_pretrained_embeddings
from utils.grounding import create_matcher_patterns, ground
from utils.paths import find_paths, score_paths, prune_paths, find_relational_paths_from_paths, generate_path_and_graph_from_adj
from utils.graph import generate_graph, generate_adj_data_from_grounded_concepts, coo_to_normalized
from utils.triples import generate_triples_from_adj

input_paths = {
    'csqa': {
        'train': './data/csqa/train.jsonl',
        'dev': './data/csqa/dev.jsonl',
        'test': './data/csqa/test_rand_split_no_answers.jsonl',
    },
    'obqa': {
        'train': './data/obqa/raw_data/train.jsonl',
        'dev': './data/obqa/raw_data/dev.jsonl',
        'test': './data/obqa/raw_data/test.jsonl',
    },
    'cpnet': {
        'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
    'glove': {
        'txt': './data/glove/glove.6B.300d.txt',
    },
    'numberbatch': {
        'txt': './data/transe/numberbatch-en-19.08.txt',
    },
    'transe': {
        'ent': './data/transe/glove.transe.sgd.ent.npy',
        'rel': './data/transe/glove.transe.sgd.rel.npy',
    },
}

output_paths = {
    'cpnet': {
        'csv': './data/cpnet/conceptnet.en.csv',
        'vocab': './data/cpnet/concept.txt',
        'patterns': './data/cpnet/matcher_patterns.json',
        'unpruned-graph': './data/cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': './data/cpnet/conceptnet.en.pruned.graph',
    },
    'glove': {
        'npy': './data/glove/glove.6B.300d.npy',
        'vocab': './data/glove/glove.vocab',
    },
    'numberbatch': {
        'npy': './data/transe/nb.npy',
        'vocab': './data/transe/nb.vocab',
        'concept_npy': './data/transe/concept.nb.npy'
    },
    'csqa': {
        'statement': {
            'train': './data/csqa/statement/train.statement.jsonl',
            'dev': './data/csqa/statement/dev.statement.jsonl',
            'test': './data/csqa/statement/test.statement.jsonl',
            'vocab': './data/csqa/statement/vocab.json',
        },
        'statement-with-ans-pos': {
            'train': './data/csqa/statement/train.statement-with-ans-pos.jsonl',
            'dev': './data/csqa/statement/dev.statement-with-ans-pos.jsonl',
            'test': './data/csqa/statement/test.statement-with-ans-pos.jsonl',
        },
        'tokenized': {
            'train': './data/csqa/tokenized/train.tokenized.txt',
            'dev': './data/csqa/tokenized/dev.tokenized.txt',
            'test': './data/csqa/tokenized/test.tokenized.txt',
        },
        'grounded': {
            'train': './data/csqa/grounded/train.grounded.jsonl',
            'dev': './data/csqa/grounded/dev.grounded.jsonl',
            'test': './data/csqa/grounded/test.grounded.jsonl',
        },
        'paths': {
            'raw-train': './data/csqa/paths/train.paths.raw.jsonl',
            'raw-dev': './data/csqa/paths/dev.paths.raw.jsonl',
            'raw-test': './data/csqa/paths/test.paths.raw.jsonl',
            'scores-train': './data/csqa/paths/train.paths.scores.jsonl',
            'scores-dev': './data/csqa/paths/dev.paths.scores.jsonl',
            'scores-test': './data/csqa/paths/test.paths.scores.jsonl',
            'pruned-train': './data/csqa/paths/train.paths.pruned.jsonl',
            'pruned-dev': './data/csqa/paths/dev.paths.pruned.jsonl',
            'pruned-test': './data/csqa/paths/test.paths.pruned.jsonl',
            'adj-train': './data/csqa/paths/train.paths.adj.jsonl',
            'adj-dev': './data/csqa/paths/dev.paths.adj.jsonl',
            'adj-test': './data/csqa/paths/test.paths.adj.jsonl',
        },
        'graph': {
            'train': './data/csqa/graph/train.graph.jsonl',
            'dev': './data/csqa/graph/dev.graph.jsonl',
            'test': './data/csqa/graph/test.graph.jsonl',
            'adj-train': './data/csqa/graph/train.graph.adj.pk',
            'adj-dev': './data/csqa/graph/dev.graph.adj.pk',
            'adj-test': './data/csqa/graph/test.graph.adj.pk',
            'nxg-from-adj-train': './data/csqa/graph/train.graph.adj.jsonl',
            'nxg-from-adj-dev': './data/csqa/graph/dev.graph.adj.jsonl',
            'nxg-from-adj-test': './data/csqa/graph/test.graph.adj.jsonl',
        },
        'triple': {
            'train': './data/csqa/triples/train.triples.pk',
            'dev': './data/csqa/triples/dev.triples.pk',
            'test': './data/csqa/triples/test.triples.pk',
        },
    },
    'obqa': {
        'statement': {
            'train': './data/obqa/statement/train.statement.jsonl',
            'dev': './data/obqa/statement/dev.statement.jsonl',
            'test': './data/obqa/statement/test.statement.jsonl',
            'train-fairseq': './data/obqa/fairseq/official/train.jsonl',
            'dev-fairseq': './data/obqa/fairseq/official/valid.jsonl',
            'test-fairseq': './data/obqa/fairseq/official/test.jsonl',
            'vocab': './data/obqa/statement/vocab.json',
        },
        'tokenized': {
            'train': './data/obqa/tokenized/train.tokenized.txt',
            'dev': './data/obqa/tokenized/dev.tokenized.txt',
            'test': './data/obqa/tokenized/test.tokenized.txt',
        },
        'grounded': {
            'train': './data/obqa/grounded/train.grounded.jsonl',
            'dev': './data/obqa/grounded/dev.grounded.jsonl',
            'test': './data/obqa/grounded/test.grounded.jsonl',
        },
        'paths': {
            'raw-train': './data/obqa/paths/train.paths.raw.jsonl',
            'raw-dev': './data/obqa/paths/dev.paths.raw.jsonl',
            'raw-test': './data/obqa/paths/test.paths.raw.jsonl',
            'scores-train': './data/obqa/paths/train.paths.scores.jsonl',
            'scores-dev': './data/obqa/paths/dev.paths.scores.jsonl',
            'scores-test': './data/obqa/paths/test.paths.scores.jsonl',
            'pruned-train': './data/obqa/paths/train.paths.pruned.jsonl',
            'pruned-dev': './data/obqa/paths/dev.paths.pruned.jsonl',
            'pruned-test': './data/obqa/paths/test.paths.pruned.jsonl',
            'adj-train': './data/obqa/paths/train.paths.adj.jsonl',
            'adj-dev': './data/obqa/paths/dev.paths.adj.jsonl',
            'adj-test': './data/obqa/paths/test.paths.adj.jsonl',
        },
        'graph': {
            'train': './data/obqa/graph/train.graph.jsonl',
            'dev': './data/obqa/graph/dev.graph.jsonl',
            'test': './data/obqa/graph/test.graph.jsonl',
            'adj-train': './data/obqa/graph/train.graph.adj.pk',
            'adj-dev': './data/obqa/graph/dev.graph.adj.pk',
            'adj-test': './data/obqa/graph/test.graph.adj.pk',
            'nxg-from-adj-train': './data/obqa/graph/train.graph.adj.jsonl',
            'nxg-from-adj-dev': './data/obqa/graph/dev.graph.adj.jsonl',
            'nxg-from-adj-test': './data/obqa/graph/test.graph.adj.jsonl',
        },
        'triple': {
            'train': './data/obqa/triples/train.triples.pk',
            'dev': './data/obqa/triples/dev.triples.pk',
            'test': './data/obqa/triples/test.triples.pk',
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['common', 'csqa'], choices=['common', 'csqa', 'obqa', 'make_word_vocab'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'common': [
            {'func': glove2npy, 'args': (input_paths['glove']['txt'], output_paths['glove']['npy'], output_paths['glove']['vocab'])},
            {'func': glove2npy, 'args': (input_paths['numberbatch']['txt'], output_paths['numberbatch']['npy'], output_paths['numberbatch']['vocab'], True)},
            {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'])},
            {'func': load_pretrained_embeddings,
             'args': (output_paths['numberbatch']['npy'], output_paths['numberbatch']['vocab'], output_paths['cpnet']['vocab'], False, output_paths['numberbatch']['concept_npy'])},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['unpruned-graph'], False)},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['pruned-graph'], True)},
            {'func': create_matcher_patterns, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
        ],
        'csqa': [
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['train'], output_paths['csqa']['statement']['train'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['test'], output_paths['csqa']['statement']['test'])},
            {'func': ground, 'args': (output_paths['csqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['csqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'],
                                                                        output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['csqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'],
                                                                        output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['csqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'],
                                                                        output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-test'], args.nprocs)},
            {'func': generate_triples_from_adj, 'args': (output_paths['csqa']['graph']['adj-train'], output_paths['csqa']['grounded']['train'],
                                                         output_paths['cpnet']['vocab'], output_paths['csqa']['triple']['train'])},
            {'func': generate_triples_from_adj, 'args': (output_paths['csqa']['graph']['adj-dev'], output_paths['csqa']['grounded']['dev'],
                                                         output_paths['cpnet']['vocab'], output_paths['csqa']['triple']['dev'])},
            {'func': generate_triples_from_adj, 'args': (output_paths['csqa']['graph']['adj-test'], output_paths['csqa']['grounded']['test'],
                                                         output_paths['cpnet']['vocab'], output_paths['csqa']['triple']['test'])},
            {'func': generate_path_and_graph_from_adj, 'args': (output_paths['csqa']['graph']['adj-train'], output_paths['cpnet']['pruned-graph'], output_paths['csqa']['paths']['adj-train'], output_paths['csqa']['graph']['nxg-from-adj-train'], args.nprocs)},
            {'func': generate_path_and_graph_from_adj, 'args': (output_paths['csqa']['graph']['adj-dev'], output_paths['cpnet']['pruned-graph'], output_paths['csqa']['paths']['adj-dev'], output_paths['csqa']['graph']['nxg-from-adj-dev'], args.nprocs)},
            {'func': generate_path_and_graph_from_adj, 'args': (output_paths['csqa']['graph']['adj-test'], output_paths['cpnet']['pruned-graph'], output_paths['csqa']['paths']['adj-test'], output_paths['csqa']['graph']['nxg-from-adj-test'], args.nprocs)},
        ],

        'obqa': [
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['train'], output_paths['obqa']['statement']['train'], output_paths['obqa']['statement']['train-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['dev'], output_paths['obqa']['statement']['dev'], output_paths['obqa']['statement']['dev-fairseq'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['test'], output_paths['obqa']['statement']['test'], output_paths['obqa']['statement']['test-fairseq'])},
            {'func': ground, 'args': (output_paths['obqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['obqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'],
                                                                        output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['obqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'],
                                                                        output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['obqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'],
                                                                        output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-test'], args.nprocs)},
            {'func': generate_triples_from_adj, 'args': (output_paths['obqa']['graph']['adj-train'], output_paths['obqa']['grounded']['train'],
                                                         output_paths['cpnet']['vocab'], output_paths['obqa']['triple']['train'])},
            {'func': generate_triples_from_adj, 'args': (output_paths['obqa']['graph']['adj-dev'], output_paths['obqa']['grounded']['dev'],
                                                         output_paths['cpnet']['vocab'], output_paths['obqa']['triple']['dev'])},
            {'func': generate_triples_from_adj, 'args': (output_paths['obqa']['graph']['adj-test'], output_paths['obqa']['grounded']['test'],
                                                         output_paths['cpnet']['vocab'], output_paths['obqa']['triple']['test'])},
            {'func': generate_path_and_graph_from_adj, 'args': (output_paths['obqa']['graph']['adj-train'], output_paths['cpnet']['pruned-graph'], output_paths['obqa']['paths']['adj-train'], output_paths['obqa']['graph']['nxg-from-adj-train'], args.nprocs)},
            {'func': generate_path_and_graph_from_adj, 'args': (output_paths['obqa']['graph']['adj-dev'], output_paths['cpnet']['pruned-graph'], output_paths['obqa']['paths']['adj-dev'], output_paths['obqa']['graph']['nxg-from-adj-dev'], args.nprocs)},
            {'func': generate_path_and_graph_from_adj, 'args': (output_paths['obqa']['graph']['adj-test'], output_paths['cpnet']['pruned-graph'], output_paths['obqa']['paths']['adj-test'], output_paths['obqa']['graph']['nxg-from-adj-test'], args.nprocs)},
        ],

        'exp': [
            {'func': convert_to_entailment,
             'args': (input_paths['csqa']['train'], output_paths['csqa']['statement-with-ans-pos']['train'], True)},
            {'func': convert_to_entailment,
             'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement-with-ans-pos']['dev'], True)},
            {'func': convert_to_entailment,
             'args': (input_paths['csqa']['test'], output_paths['csqa']['statement-with-ans-pos']['test'], True)},
        ],

        'make_word_vocab': [
            {'func': make_word_vocab, 'args': ((output_paths['csqa']['statement']['train'],), output_paths['csqa']['statement']['vocab'])},
            {'func': make_word_vocab, 'args': ((output_paths['obqa']['statement']['train'],), output_paths['obqa']['statement']['vocab'])},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()

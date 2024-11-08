import torch
import argparse
import random
import collections
import math

import numpy as np
import os
import pickle

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def read_triples_QTO(filenames, nrelation, datapath):
    adj_list = [[] for i in range(nrelation)]
    edges_all = set() # all edges of train + valid + test
    edges_vt = set() # all edges of valid + test
    for filename in filenames:
        with open(filename) as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                adj_list[int(r)].append((int(h), int(t)))
    for filename in ['valid.txt', 'test.txt']:
        with open(os.path.join(datapath, filename)) as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                edges_all.add((int(h), int(r), int(t)))
                edges_vt.add((int(h), int(r), int(t)))
    with open(os.path.join(datapath, "train.txt")) as f:
        for line in f.readlines():
            h, r, t = line.strip().split('\t')
            edges_all.add((int(h), int(r), int(t)))

    return adj_list, edges_all, edges_vt


def read_triples(filenames, nrelation, datapath, ent2id, rel2id):
    adj_list = [[] for i in range(nrelation)]
    edges_all = set() # all edges of train + valid + test
    edges_vt = set() # all edges of valid + test
    for filename in filenames:
        with open(filename) as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                h = ent2id[h]
                t = ent2id[t]
                r = rel2id[r]
                adj_list[int(r)].append((int(h), int(t)))
    for filename in ['valid', 'test']:
        with open(os.path.join(datapath, filename)) as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                h = ent2id[h]
                t = ent2id[t]
                r = rel2id[r]
                edges_all.add((int(h), int(r), int(t)))
                edges_vt.add((int(h), int(r), int(t)))
    with open(os.path.join(datapath, "train")) as f:
        for line in f.readlines():
            h, r, t = line.strip().split('\t')
            h = ent2id[h]
            t = ent2id[t]
            r = rel2id[r]
            edges_all.add((int(h), int(r), int(t)))

    return adj_list, edges_all, edges_vt

    
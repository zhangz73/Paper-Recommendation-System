import time, sys
import json
import numpy as np
import math
from joblib import Parallel, delayed
from tqdm import tqdm
import networkx as nx

PATH = "../Graph/"
EDGELIST_FILE = PATH + "PaperAuthorAffiliationInterestKG.edgelist"
PREFIX_LST = ["A", "P", "Aff", "Int"]

OUTPUT_PATH = "../KG_pykg2vec/"
OUTPUT_NAME = "PaperAuthorAffiliationInterestKG"

all_edges = []
with open(EDGELIST_FILE, "r") as f:
    for line in tqdm(f):
        curr = line.split(";")[:2]
        node1 = curr[0]
        node2 = curr[1]
        attr1, attr2 = None, None
        for prefix in PREFIX_LST:
            if node1.startswith(prefix):
                attr1 = prefix
            if node2.startswith(prefix):
                attr2 = prefix
        edge = node1 + "\t" + attr1 + "+==+to+==+" + attr2 + "\t" + node2
        all_edges.append(edge)

train_size = int(len(all_edges) * 0.8)
valid_size = int(len(all_edges) * 0.1)
test_size = len(all_edges) - train_size - valid_size
idx_train = np.random.choice(len(all_edges), train_size, replace=False)

print("Constructing training set...")
edges_train = [all_edges[i] for i in tqdm(idx_train)]
print("Factoring out validation & test sets...")
idx_train_set = set(idx_train)
edges_rest = [all_edges[i] for i in tqdm(range(len(all_edges))) if i not in idx_train_set]
idx_valid = np.random.choice(len(edges_rest), valid_size, replace=False)
print("Constructing validation set...")
idx_valid_set = set(idx_valid)
edges_valid = [edges_rest[i] for i in tqdm(idx_valid)]
print("Constructing test set...")
edges_test = [edges_rest[i] for i in tqdm(range(len(edges_rest))) if i not in idx_valid_set]

with open(OUTPUT_PATH + OUTPUT_NAME + "-all.txt", "w") as f:
    for line in tqdm(all_edges):
        f.write(line + "\n")
with open(OUTPUT_PATH + OUTPUT_NAME + "-train.txt", "w") as f:
    for line in tqdm(edges_train):
        f.write(line + "\n")
with open(OUTPUT_PATH + OUTPUT_NAME + "-valid.txt", "w") as f:
    for line in tqdm(edges_valid):
        f.write(line + "\n")
with open(OUTPUT_PATH + OUTPUT_NAME + "-test.txt", "w") as f:
    for line in tqdm(edges_test):
        f.write(line + "\n")

import time, sys
import json
import numpy as np
import math
from tqdm import tqdm

PATH = "../KG_pykg2vec/"
INPUT_FILE = PATH + "PaperAuthorAffiliationInterestKG-all.txt"
OUTPUT_FILE = PATH + "PaperAuthorAffiliationInterestKG-sample.txt"
RECOVER_FILE = "../TransE/nodes_margin=10_dim=20_batch=200_loss=L2_lr=0.1_num-itr=100000.txt"


NUM_AUTHOR = 50000
author_list = []
edge_list = []
with open(INPUT_FILE, "r") as f:
    for line in tqdm(f):
        line = line.strip("\n")
        arr = line.split("\t")
        edge_list.append(line)
        if arr[0].startswith("A"):
            author_list.append(arr[0])
        if arr[2].startswith("A"):
            author_list.append(arr[2])
author_list = list(set(author_list))

#author_sample_idx = np.random.choice(len(author_list), NUM_AUTHOR, replace=False)
#author_sample = set([author_list[x] for x in author_sample_idx])
author_sample = []
with open(RECOVER_FILE, "r") as f:
    for line in tqdm(f):
        arr = line.split("\t")
        if arr[0].startswith("A"):
            author_sample.append(arr[0])
author_sample = set(author_sample)

edge_sample = []
paper_delete = []
affiliation_preserve = []
interest_preserve = []
for line in tqdm(edge_list):
    line = line.strip("\n")
    arr = line.split("\t")
    if arr[1] == "A+==+to+==+P":
        if arr[0] not in author_sample:
            paper_delete.append(arr[2])
    elif arr[1] == "A+==+to+==+Aff":
        if arr[0] in author_sample:
            affiliation_preserve.append(arr[2])
    elif arr[1] == "A+==+to+==+Int":
        if arr[0] in author_sample:
            interest_preserve.append(arr[2])
paper_delete = set(paper_delete)
#affiliation_preserve = set(affiliation_preserve)
interest_preserve = set(interest_preserve)
for line in tqdm(edge_list):
    line = line.strip("\n")
    arr = line.split("\t")
    if arr[1] == "P+==+to+==+Aff":
        if arr[0] not in paper_delete:
            affiliation_preserve.append(arr[2])
affiliation_preserve = set(affiliation_preserve)

for line in tqdm(edge_list):
    line = line.strip("\n")
    arr = line.split("\t")
    remove = False
    if arr[0].startswith("A") and arr[0] not in author_sample:
        remove = True
    if not remove and arr[0].startswith("P") and arr[0] in paper_delete:
        remove = True
    if not remove and arr[2].startswith("A") and arr[2] not in author_sample:
        remove = True
    if not remove and arr[2].startswith("P") and arr[2] in paper_delete:
        remove = True
    if not remove and arr[2].startswith("Aff") and arr[2] not in affiliation_preserve:
        remove = True
    if not remove and arr[2].startswith("Int") and arr[2] not in interest_preserve:
        remove = True
    if not remove:
        edge_sample.append(line)

with open(OUTPUT_FILE, "w") as f:
    for line in tqdm(edge_sample):
        f.write(line + "\n")

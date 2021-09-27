import time, sys
import json
import numpy as np
import math
from tqdm import tqdm
from joblib import Parallel, delayed

N_CPU = 1
MAX_PATH_LEN = int(sys.argv[1])

PATH = "../KG_pykg2vec/"
INPUT_FILE = PATH + "PaperAuthorAffiliationInterestKG-sample.txt"
EDGES_VEC_FILE = "../TransE/edges_margin=10_dim=20_batch=200_loss=L2_lr=0.1_num-itr=100000.txt"
NODES_VEC_FILE = "../TransE/nodes_margin=10_dim=20_batch=200_loss=L2_lr=0.1_num-itr=100000.txt"
PATH_FILE = f"../A2PPaths/A2PPaths-sample-{MAX_PATH_LEN}.txt"
VEC_FILE = f"../A2PPaths/A2PPaths-Vec-sample-{MAX_PATH_LEN}.txt"

def DFS_single(src_of_interest, graph):
    all_path = []
    for src in tqdm(src_of_interest):
        stack = [src]
        stack_path = [src]
        stack_visited = [set([])]
        #visited = set([src])
        while len(stack) > 0:
            point = stack.pop()
            visited = stack_visited.pop()
            visited.add(point)
            path = stack_path.pop()
            path_len = (len(path.split("\t")) + 1) / 2
            if path_len < MAX_PATH_LEN:
                for tup in graph[point]:
                    if tup[0] not in visited:
                        curr_path = path + "\t" + tup[1] + "\t" + tup[0]
                        stack_path.append(curr_path)
                        stack.append(tup[0])
                        curr_visited = set([tup[0]])
                        for p in visited:
                            curr_visited.add(p)
                        stack_visited.append(curr_visited)
                        if tup[0].startswith("P"):
                            all_path.append(curr_path)
    return all_path

def get_path():
    graph = {}
    print("Begin Extracting Paths...")
    print("   Loading Graph...")
    with open(INPUT_FILE, "r") as f:
        for line in tqdm(f):
            line = line.strip("\n")
            arr = line.split("\t")
            node1 = arr[0]
            node2 = arr[2]
            edge = arr[1]
            if node1 not in graph:
                graph[node1] = []
            if node2 not in graph:
                graph[node2] = []
            graph[node1].append((node2, edge))
            graph[node2].append((node1, edge))
    print("   DFS...")
    src_of_interest = [x for x in graph if x.startswith("A") and not x.startswith("Aff-")]
    
    if N_CPU > 1:
        all_path = []
        batch_size = int(math.ceil(len(src_of_interest) / N_CPU))
        results = Parallel(n_jobs=N_CPU)(delayed(DFS_single)(
            src_of_interest[(i * batch_size):min((i + 1) * batch_size, len(src_of_interest))], graph
        ) for i in range(N_CPU))
        
        for res in results:
            all_path += res
    
        print("   Write to file...")
        with open(PATH_FILE, "w") as f:
            for path in tqdm(all_path):
                f.write(path + "\n")
    else:
        with open(PATH_FILE, "w") as f:
            for src in tqdm(src_of_interest):
                stack = [src]
                stack_path = [src]
                stack_visited = [set([])]
                #visited = set([src])
                while len(stack) > 0:
                    point = stack.pop()
                    visited = stack_visited.pop()
                    visited.add(point)
                    path = stack_path.pop()
                    path_len = (len(path.split("\t")) + 1) / 2
                    if path_len < MAX_PATH_LEN:
                        for tup in graph[point]:
                            if tup[0] not in visited:
                                curr_path = path + "\t" + tup[1] + "\t" + tup[0]
                                stack_path.append(curr_path)
                                stack.append(tup[0])
                                curr_visited = set([tup[0]])
                                for p in visited:
                                    curr_visited.add(p)
                                stack_visited.append(curr_visited)
                                if tup[0].startswith("P"):
                                    f.write(curr_path + "\n")
    print("Done Extracting Paths!")

def get_vec():
    print("Begin Translation Into Vectors...")
    edges_vec = {}
    nodes_vec = {}
    print("   Loading Edges...")
    with open(EDGES_VEC_FILE, "r") as f:
        for line in tqdm(f):
            line = line.strip("\n")
            arr = line.split("\t")
            edges_vec[arr[0]] = arr[1]
    print("   Loading Nodes...")
    with open(NODES_VEC_FILE, "r") as f:
        for line in tqdm(f):
            line = line.strip("\n")
            arr = line.split("\t")
            nodes_vec[arr[0]] = arr[1]
    all_path = []
    print("   Translating...")
    with open(PATH_FILE, "r") as f:
        with open(VEC_FILE, "w") as f2:
            for line in tqdm(f):
                line = line.strip("\n")
                arr = line.split("\t")
                path = ""
                for i in range(len(arr)):
                    if i % 2 == 0:
                        path += nodes_vec[arr[i]]
                    else:
                        path += edges_vec[arr[i]]
                    if i < len(arr) - 1:
                        path += "\t"
                #all_path.append(path)
                f2.write(path + "\n")
#    print("   Writing to file...")
#    with open(VEC_FILE, "w") as f:
#        for path in tqdm(all_path):
#            f.write(path + "\n")
    print("Done Translation Into Vectors!")

get_path()
get_vec()

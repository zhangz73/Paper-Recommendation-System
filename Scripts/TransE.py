import time, sys
import json
import numpy as np
import math
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

FNAME = "../KG_pykg2vec/PaperAuthorAffiliationInterestKG-sample.txt"

def load_edgelist(fname):
    print("Begin loading " + fname + "...")
    edge_list_arr = []
    nodes_embeddings = {}
    edges_embeddings = {}
    with open(fname, "r") as f:
        for line in tqdm(f):
            if len(line.strip("\n")) > 0:
                arr = line.strip("\n").split("\t")
                edge_list_arr.append((arr[0], arr[1], arr[2]))
                if arr[0] not in nodes_embeddings:
                    nodes_embeddings[arr[0]] = 0
                if arr[2] not in nodes_embeddings:
                    nodes_embeddings[arr[2]] = 0
                if arr[1] not in edges_embeddings:
                    edges_embeddings[arr[1]] = 0
    print("End loading " + fname + "!")
    return edge_list_arr, nodes_embeddings, edges_embeddings

def ranking_loss(h1, t1, h2, t2, r, margin, loss_metric):
    if loss_metric == "L2":
        d1 = torch.sum(torch.square(h1 + r - t1))
        d2 = torch.sum(torch.square(h2 + r - t2))
    else:
        d1 = torch.sum(torch.abs(h1 + r - t1))
        d2 = torch.sum(torch.abs(h2 + r - t2))
    return torch.relu(margin + d1 - d2)

def training(edge_list_arr, nodes_embeddings, edges_embeddings, margin, embedding_dim, batch_size = 1000, loss_metric = "L2", lr=0.01, num_itr=100):
    ## Initialize
    print("Initialize parameters...")
    print("   Nodes:")
    for node in tqdm(nodes_embeddings):
        val = (torch.rand(embedding_dim) - 0.5) * 6 / float(np.sqrt(embedding_dim))
        val = val / torch.sqrt(torch.sum(torch.square(val)))
        val.requires_grad = True
        nodes_embeddings[node] = val
    print("   Edges:")
    for edge in tqdm(edges_embeddings):
        val = (torch.rand(embedding_dim) - 0.5) * 6 / float(np.sqrt(embedding_dim))
        val.requires_grad = True
        edges_embeddings[edge] = val
    loss_arr = []
    nodes_lst = list(nodes_embeddings.keys())
    node_indices_all = np.random.choice(len(nodes_lst), 2 * batch_size * num_itr, replace=True).reshape((num_itr, 2 * batch_size))
    print("Training...")
    for i in tqdm(range(num_itr)):
        #print("   Iteration # = " + str(i + 1) + ":")
        #print("        Normalizing edges:")
        for edge in edges_embeddings:
            val = edges_embeddings[edge].data
            val = val / torch.sqrt(torch.sum(torch.square(val)))
            edges_embeddings[edge].data = val
        loss_sum = torch.zeros(1)
        batch = np.random.choice(len(edge_list_arr), batch_size, replace=True)
        #print("        Training on batches:")
        #node_indices_all = np.random.choice(len(nodes_lst), 2 * batch_size, replace=False)
        for j in range(batch_size):
            idx = batch[j]
            node_indices = node_indices_all[i, (2 * j) : (2 * j + 2)]
            tup1 = edge_list_arr[idx]
            #node_indices = np.random.choice(len(nodes_lst), 2, replace=False)
            tup2 = (nodes_lst[node_indices[0]], tup1[1], nodes_lst[node_indices[1]])
            #t_batch.append((t1, t2))
            h1 = nodes_embeddings[tup1[0]]
            t1 = nodes_embeddings[tup1[2]]
            h2 = nodes_embeddings[tup2[0]]
            t2 = nodes_embeddings[tup2[2]]
            r = edges_embeddings[tup1[1]]
            curr_loss = ranking_loss(h1, t1, h2, t2, r, margin, loss_metric)
            loss_sum += curr_loss
        loss_sum.backward()
        for node in nodes_embeddings:
            val = nodes_embeddings[node]
            if val.grad is not None:
#                nodes_embeddings[node].data = val.data - lr * val.grad
#                nodes_embeddings[node].grad.detach()
#                nodes_embeddings[node].grad.zero_()
                val.data = val.data - lr * val.grad
                val.grad.detach()
                val.grad = None
                nodes_embeddings[node] = val
        for edge in edges_embeddings:
            val = edges_embeddings[edge]
            if val.grad is not None:
                val.data = val.data - lr * val.grad
                val.grad.detach()
                val.grad = None
                edges_embeddings[edge] = val
        loss_arr.append(loss_sum.data / batch_size)
        
    print("Done Training!")
    return nodes_embeddings, edges_embeddings, loss_arr

def dump_results(nodes_embeddings, edges_embeddings, loss_arr, nodes_output_fname, edges_output_fname, loss_output_fname):
    print("Writing nodes to file...")
    with open(nodes_output_fname, "w") as f:
        for node in tqdm(nodes_embeddings):
            f.write(node + "\t" + ",".join([str(float(x)) for x in nodes_embeddings[node].data]) + "\n")
    print("Writing edges to file...")
    with open(edges_output_fname, "w") as f:
        for edge in tqdm(edges_embeddings):
            f.write(edge + "\t" + ",".join([str(float(x)) for x in edges_embeddings[edge].data]) + "\n")
    print("Printing loss over time...")
    plt.plot(np.arange(len(loss_arr)) + 1, loss_arr)
    plt.xlabel("Iteration #")
    plt.ylabel("Loss")
    plt.title("Loss Over Iterations")
    plt.savefig(loss_output_fname)
    plt.clf()
    print("Jobs Done!")

def workflow(margin, embedding_dim, batch_size = 1000, loss_metric = "L2", lr=0.01, num_itr=1000):
    prefix = "../TransE/"
    suffix = f"margin={margin}_dim={embedding_dim}_batch={batch_size}_loss={loss_metric}_lr={lr}_num-itr={num_itr}"
    nodes_output_fname = prefix + "nodes_" + suffix + ".txt"
    edges_output_fname = prefix + "edges_" + suffix + ".txt"
    loss_output_fname = prefix + "loss_" + suffix + ".png"
    edge_list_arr, nodes_embeddings, edges_embeddings = load_edgelist(FNAME)
    nodes_embeddings, edges_embeddings, loss_arr = training(edge_list_arr, nodes_embeddings, edges_embeddings, margin, embedding_dim, batch_size, loss_metric, lr, num_itr)
    dump_results(nodes_embeddings, edges_embeddings, loss_arr, nodes_output_fname, edges_output_fname, loss_output_fname)
    
workflow(10, 20, batch_size = 200, lr = 1e-3, num_itr = 50000)

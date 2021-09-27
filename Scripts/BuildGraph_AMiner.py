import time, sys
import json
import numpy as np
import math
from joblib import Parallel, delayed
from tqdm import tqdm
import networkx as nx

PATH = "../Data/"
AUTHOR_FILE = PATH + "Aminer-Author.txt"
COAUTHOR_FILE = PATH + "Aminer-Coauthor.txt"
PAPER_FILE = PATH + "Aminer-Paper.txt"
AUTHOR2PAPER_FILE = PATH + "Aminer-Author2Paper.txt"
AUTHOR2AFFILIATIONS_FILE = PATH + "Aminer-Author2Affiliations.txt"
AUTHOR2INTERESTS_FILE = PATH + "Aminer-Author2Interests.txt"
PAPER2AFFILIATIONS_FILE = PATH + "Aminer-Paper2Affiliations.txt"
PAPER2ABSTRACT_FILE = PATH + "Aminer-Paper2Abstract.txt"

def load_author(G):
    dct = {}
    print("Loading Authors...")
    with open(AUTHOR_FILE, "r") as f:
        idx = 0
        for line in tqdm(f):
            if len(line) > 1 and line[0] == '#':
                line = line.replace("\n", "").strip()
                if line.startswith("#index"):
                    idx = "A" + line.strip("#index").strip()
                    dct[idx] = {}
                elif line.startswith("#n"):
                    name = line.strip("#n").strip()
                    #dct[idx]["name"] = name
                elif line.startswith("#a"):
                    pass
                    #affiliations = line.strip("#a").strip().split(";")
                    #dct[idx]["affiliations"] = affiliations
                elif line.startswith("#pc"):
                    pc = float(line.strip("#pc").strip())
                    dct[idx]["publish_cnt"] = pc
                elif line.startswith("#cn"):
                    cn = float(line.strip("#cn").strip())
                    dct[idx]["citation_cnt"] = cn
                elif line.startswith("#hi"):
                    hi = float(line.strip("#hi").strip())
                    dct[idx]["H_idx"] = hi
                elif line.startswith("#pi"):
                    pi = float(line.strip("#pi").strip())
                    dct[idx]["P_idx"] = pi
                elif line.startswith("#upi"):
                    upi = float(line.strip("#upi").strip())
                    dct[idx]["UP_idx"] = upi
                elif line.startswith("#t"):
                    pass
                    #t = line.strip("#t").strip().split(";")
                    #dct[idx]["interests"] = t

    print("   Populating Author Nodes...")
    G.add_nodes_from([(k, dct[k]) for k in dct.keys()])
    print("Done Loading Authors!")
    return G

def load_coauthor(G):
    lst = []
    print("Begin Loading Co-Authors...")
    with open(COAUTHOR_FILE, "r") as f:
        for line in tqdm(f):
            if len(line) > 1 and line[0] == '#':
                line = line.strip("#").replace("\n", "").strip()
                x = [str(int(x)) for x in line.split()]
                lst.append(("A" + x[0], "A" + x[1], int(x[2])))
    print("   Populating Co-Author Edges...")
    print("   Total Co-authors = " + str(len(lst)))
    G.add_weighted_edges_from(lst)
    print("Done Loading Co-Authors!")
    return G

def load_paper(G):
    dct = {}
    print("Begin Loading Paper...")
    with open(PAPER_FILE, "r") as f:
        for line in tqdm(f):
            if len(line) > 1 and line[0] == '#':
                line = line.replace("\n", "").strip()
                if line.startswith("#index"):
                    idx = "P" + line.strip("#index").strip()
                    dct[idx] = {"citations":[]}
                elif line.startswith("#*"):
                    name = line.strip("#*").strip()
                    #dct[idx]["title"] = name
                elif line.startswith("#@"):
                    pass
                    #authors = line.strip("#@").strip().split(";")
                    #dct[idx]["authors"] = authors
                elif line.startswith("#o"):
                    pass
                    #affiliations = line.strip("#o").strip().split(";")
                    #dct[idx]["affliations"] = affiliations
                elif line.startswith("#t"):
                    year = line.strip("#t").strip()
                    if len(year) == 0:
                        year = None
                    else:
                        year = int(year)
                    dct[idx]["year"] = year
                elif line.startswith("#c"):
                    c = line.strip("#c").strip()
                    dct[idx]["venue"] = c
                elif line.startswith("#!"):
                    abstract = line.strip("#!").strip()
                    #dct[idx]["abstract"] = abstract
                elif line.startswith("#%"):
                    id = "P" + line.strip("#%").strip()
                    dct[idx]["citations"].append(id)
    
    print("   Adding Citation Edges...")
    citations = [(k, v) for k in dct.keys() for v in dct[k]["citations"]]
    print("   Total citations = " + str(len(citations)))
    G.add_edges_from(citations)
    
    print("   Populating Paper Nodes...")
    for k in dct.keys():
        dct[k].pop("citations", None)
    G.add_nodes_from([(k, dct[k]) for k in dct.keys()])
    
    print("Done Loading Papers!")
    return G

def load_author2paper(G):
    lst = []
    print("Begin Loading Author2Paper...")
    with open(AUTHOR2PAPER_FILE, "r") as f:
        for line in tqdm(f):
            if len(line) > 1:
                line = line.replace("\n", "").strip()
                x = [str(int(x)) for x in line.split()]
                lst.append(("A" + x[1], "P" + x[2], int(x[3])))
    
    print("   Populating Author-Paper Edges...")
    print("   Total Author2Paper = " + str(len(lst)))
    G.add_weighted_edges_from(lst)
    print("Done Loading Author2Paper!")
    return G

def load_author2affiliations(G):
    lst = []
    print("Begin Loading Author2Affiliations...")
    with open(AUTHOR2AFFILIATIONS_FILE, "r") as f:
        for line in tqdm(f):
            if len(line) > 1:
                line = line.replace("\n", "").strip()
                left = line.split()[0].strip("#")
                right = line[len(left):].strip().split(";")
                lst += [(left, "Aff-" + x) for x in right]
    print("   Populating Author-Affiliations Edges...")
    print("   Total Author2Affiliations = " + str(len(lst)))
    G.add_edges_from(lst)
    print("Done Loading Author2Affiliations!")
    return G

def load_author2interests(G):
    lst = []
    print("Begin Loading Author2Interests...")
    with open(AUTHOR2INTERESTS_FILE, "r") as f:
        for line in tqdm(f):
            if len(line) > 1:
                line = line.replace("\n", "").strip()
                left = line.split()[0].strip("#")
                right = line[len(left):].strip().split(";")
                lst += [(left, "Int-" + x) for x in right]
    print("   Populating Author-Interests Edges...")
    print("   Total Author2Interests = " + str(len(lst)))
    G.add_edges_from(lst)
    print("Done Loading Author2Interests!")
    return G

def load_paper2affiliations(G):
    lst = []
    print("Begin Loading Paper2Affiliations...")
    with open(PAPER2AFFILIATIONS_FILE, "r") as f:
        for line in tqdm(f):
            if len(line) > 1:
                line = line.replace("\n", "").strip()
                left = line.split()[0].strip("#")
                right = line[len(left):].strip().split(";")
                lst += [(left, "Aff-" + x) for x in right]
    print("   Populating Paper-Affiliations Edges...")
    print("   Total Paper2Affiliations = " + str(len(lst)))
    G.add_edges_from(lst)
    print("Done Loading Paper2Affiliations!")
    return G

def load_paper2abstract(G):
    print("Begin Loading Paper2Abstract...")
    with open(PAPER2ABSTRACT_FILE, "r") as f:
        for line in tqdm(f):
            if len(line) > 1:
                line = line.strip("#").replace("\n", "").strip()
                paper = line.split()[0]
                abstract = line[len(paper):].strip()
                G.nodes[paper]["abstract"] = abstract
    print("Done Loading Paper2Abstract!")
    return G

def load_graph():
    G = nx.Graph()
    G = load_author(G)
    G = load_coauthor(G)
    G = load_paper(G)
    G = load_author2paper(G)
    G = load_author2affiliations(G)
    G = load_author2interests(G)
    G = load_paper2affiliations(G)
    G = load_paper2abstract(G)
    return G

G = load_graph()
print(nx.info(G))
joblib.dump(G, "../Graph/PaperAuthorAffiliationInterestKG.joblib")

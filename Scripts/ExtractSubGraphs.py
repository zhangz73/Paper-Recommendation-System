import time, sys
import json
import numpy as np
import math
from tqdm import tqdm

PATH = "../Data/"
PAPER_FILE = PATH + "Aminer-Paper.txt"
AUTHOR_FILE = PATH + "Aminer-Author.txt"

def load_graph(INPUT_FILE, OUTPUT_FILE, attr, prefix):
    print("Loading Sub-Graph " + INPUT_FILE + " --> " + OUTPUT_FILE + "...")
    dct = {}
    with open(INPUT_FILE, "r") as f:
        idx = 0
        for line in tqdm(f):
            if len(line) > 1 and line[0] == '#':
                line = line.replace("\n", "").strip()
                if line.startswith("#index"):
                    idx = prefix + line.strip("#index").strip()
                elif line.startswith(attr):
                    all_attr = line.strip(attr).strip()
                    dct[idx] = all_attr
    print("Writing Results...")
    with open(OUTPUT_FILE, "w") as f:
        for k in tqdm(dct.keys()):
            f.write("#" + k + " " + dct[k] + "\n")
    print("End Extracting Sub-Graph!")

load_graph(AUTHOR_FILE, PATH + "Aminer-Author2Affiliations.txt", "#a", "A")
load_graph(AUTHOR_FILE, PATH + "Aminer-Author2Interests.txt", "#t", "A")
load_graph(PAPER_FILE, PATH + "Aminer-Paper2Affiliations.txt", "#o", "P")

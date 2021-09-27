import gensim.models as g
from tqdm import tqdm

PATH = "../Data/"
MODEL = "../Model/WikiDoc2Vec-DBOW.bin"
PAPER_FILE = PATH + "Aminer-Paper.txt"
OUTPUT_FILE = PATH + "Aminer-AbstractEmbedding.txt"

#inference hyper-parameters
start_alpha=0.01
infer_epoch=20

print("Begin loading model...")
m = g.Doc2Vec.load(MODEL)

print("Begin computing abstract embeddings...")
with open(PAPER_FILE, "r") as f:
    dct = {}
    for line in tqdm(f):
        if len(line) > 1 and line[0] == '#':
            line = line.replace("\n", "").strip()
            if line.startswith("#index"):
                idx = "P" + line.strip("#index").strip()
            elif line.startswith("#!"):
                abstract = line.strip("#!").strip().split()
                dct[idx] = m.infer_vector(abstract, alpha=start_alpha, steps=infer_epoch)

print("Writing results to file...")
with open(OUTPUT_FILE, "w") as f:
    for k in tqdm(dct.keys()):
        f.write("#" + k + " " + str(dct[k]) + "\n")

print("Jobs Done!")

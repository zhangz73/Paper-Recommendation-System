import time, sys
import json
import numpy as np
import gc
import math
from joblib import Parallel, delayed
from tqdm import tqdm
#import multiprocessing as m
from joblib import wrap_non_picklable_objects
import multiprocessing
from multiprocessing import Pool, Manager, Queue, Process
from contextlib import closing

n_cpu = 10

manager = multiprocessing.Manager()

authors_id_map = {}
authors_org_map = {}
fos_name_map = {}
publisher_map = {}
year_map = {}
reference_map = {}

authors_id_prob = {}
authors_org_prob = {}
fos_name_prob = {}
publisher_prob = {}
year_prob = {}
authors_id_co = {}
authors_org_co = {}
fos_name_co = {}
publisher_co = {}
year_co = {}
doc2vec_map = {}
lines = []

def get_attribute_data(line, attr_arr):    
    line = line.strip(",")
    d = json.loads(line)

    data = []
    if attr_arr[0] in d:
        if len(attr_arr) == 1:
            if isinstance(d[attr_arr[0]], list):
                data = d[attr_arr[0]]
            else:
                data = [d[attr_arr[0]]]
        else:
            if isinstance(d[attr_arr[0]], list):
                for j in range(len(d[attr_arr[0]])):
                    if attr_arr[1] in d[attr_arr[0]][j]:
                        data.append(d[attr_arr[0]][j][attr_arr[1]])
            else:
                if attr_arr[1] in d[attr_arr[0]]:
                    data = d[attr_arr[0]][attr_arr[1]]

    data = list(set(data))
    ret = [str(x) for x in data]
    return ret

def get_prob(attr):
    print("   loading prob on " + attr + "...")
    with open("training_data_v2/" + attr + "_relation_prob_train.txt", "r") as f:
        attr_prob = {}
        for line in f:
            left = line.split(":")[:-1]
            left = ":".join(left)
            right = line.split(":")[-1].strip("\n")
            attr_prob[left] = float(right)
    return attr_prob

def get_co_relation(attr):
    print("   loading co-relation on " + attr + "...")
    with open("training_data_v2/" + attr + "_co_relation_train.txt", "r") as f:
        attr_co = {}
        prev = ""
        for line in f:
            cline = prev + line
            left = cline.split(":")[:-1]
            left = ":".join(left)
            right = cline.split(":")[-1].strip("\n")
            try:
                indices = set(map(int, right.split(",")))
                attr_co[left] = indices
                prev = ""
            except:
                prev = prev + line
    return attr_co

def get_doc2vec():
    print("   loading doc2vec...")
    with open("AbstractDoc2Vecs.txt", "r") as f:
        doc2vec_map = {}
        for line in tqdm(f):
            left = int(line.split(":")[0])
            right = line.split(":")[1].strip("\n")
            vec = list(map(float, right.split(",")))
            doc2vec_map[left] = np.array(vec)
    return doc2vec_map

def compute_prob(attr_curr, attr_ref, attr_prob):
    prob_sum = 0
    for attr1 in attr_curr:
        for attr2 in attr_ref:
            key = str(attr1) + "+==+" + str(attr2)
            if key in attr_prob:
                prob_sum += attr_prob[key]
            #prob_sum += attr_prob[str(attr1) + "+==+" + str(attr2)]
    return prob_sum

def subspace_single(attr_curr, attr_co):
    if len(attr_curr) == 0 or attr_curr[0] not in attr_co:
        return set([])
    ret = attr_co[attr_curr[0]]
    for i in range(1, len(attr_curr)):
        attr = attr_curr[i]
        if attr not in attr_co:
            return set([])
        ret = ret.intersection(attr_co[attr])
    return ret

def compute_subspace(attr_curr, attr_ref, attr_co):
    subspace = subspace_single(attr_ref, attr_co)
    subspace = attr_curr.intersection(subspace)
    return len(subspace)
#    subspace = attr_curr
#    for attr in attr_ref:
#        if attr not in attr_co:
#            return 0
#        subspace = subspace.intersection(attr_co[attr])
#    return len(subspace)

def compute_cosine(vec_curr, vec_ref):
    #np_vec_curr = np.array(vec_curr)
    #np_vec_ref = np.array(vec_ref)
    return np.dot(vec_curr, vec_ref) / np.sqrt(np.sum(vec_curr ** 2) * np.sum(vec_ref ** 2))

def dump_to_file(from_ret, to_ret, authors_id_subspace_ret, authors_org_subspace_ret, fos_name_subspace_ret, publisher_subspace_ret, year_subspace_ret):
    print("Dumping intermediate results to file...")
    with open("dataframe_train_full_subspace.csv", "a") as f:
        for i in tqdm(range(len(from_ret))):
            line = str(from_ret[i]) + "," + str(to_ret[i]) + "," +\
                str(authors_id_subspace_ret[i]) + "," +\
                str(authors_org_subspace_ret[i])+ "," +\
                str(fos_name_subspace_ret[i]) + "," +\
                str(publisher_subspace_ret[i]) + "," +\
                str(year_subspace_ret[i]) + "\n"
            f.write(line)

def build_DF_single(lo, hi):
    from_ret = []
    to_ret = []
    authors_id_prob_ret = []
    authors_org_prob_ret = []
    fos_name_prob_ret = []
    publisher_prob_ret = []
    year_prob_ret = []
    authors_id_subspace_ret = []
    authors_org_subspace_ret = []
    fos_name_subspace_ret = []
    publisher_subspace_ret = []
    year_subspace_ret = []
    cosine_ret = []
    label_ret = []
    
#    lo = lo_hi[0]
#    hi = lo_hi[1]
    
    for i in tqdm(range(lo, hi)):
        line = lines[i]
        line = line.strip("\n")
        curr_id = int(line.split(":")[0])
        ref_list = list(map(int, line.split(":")[1].split(",")))
        
        authors_id_ref_self = []
        authors_org_ref_self = []
        fos_name_ref_self = []
        publisher_ref_self = []
        year_ref_self = []
        
        for ref_self in reference_map[curr_id]:
            if ref_self in reference_map:
                authors_id_ref_self += authors_id_map[ref_self]
                authors_org_ref_self += authors_org_map[ref_self]
                fos_name_ref_self += fos_name_map[ref_self]
                publisher_ref_self += publisher_map[ref_self]
                year_ref_self += year_map[ref_self]
        authors_id_ref_self = subspace_single(list(set(authors_id_ref_self)), authors_id_co)
        authors_org_ref_self = subspace_single(list(set(authors_org_ref_self)), authors_org_co)
        fos_name_ref_self = subspace_single(list(set(fos_name_ref_self)), fos_name_co)
        publisher_ref_self = subspace_single(list(set(publisher_ref_self)), publisher_co)
        year_ref_self = subspace_single(list(set(year_ref_self)), year_co)
        
        for ref in ref_list:
            if ref != curr_id and ref in reference_map:
                authors_id_ref = authors_id_map[ref]
                authors_org_ref = authors_org_map[ref]
                fos_name_ref = fos_name_map[ref]
                publisher_ref = publisher_map[ref]
                year_ref = year_map[ref]
                
#                authors_id_p = compute_prob(authors_id_map[curr_id], authors_id_ref, authors_id_prob)
#                authors_org_p = compute_prob(authors_org_map[curr_id], authors_org_ref, authors_org_prob)
#                fos_name_p = compute_prob(fos_name_map[curr_id], fos_name_ref, fos_name_prob)
#                publisher_p = compute_prob(publisher_map[curr_id], publisher_ref, publisher_prob)
#                year_p = compute_prob(year_map[curr_id], year_ref, year_prob)
                
                if ref in reference_map[curr_id]:
                    #label_ret.append(1)
                    authors_id_s = len(authors_id_ref_self)
                    authors_org_s = len(authors_org_ref_self)
                    fos_name_s = len(fos_name_ref_self)
                    publisher_s = len(publisher_ref_self)
                    year_s = len(year_ref_self)
                else:
                    #label_ret.append(0)
                    authors_id_s = compute_subspace(authors_id_ref_self, authors_id_ref, authors_id_co)
                    authors_org_s = compute_subspace(authors_org_ref_self, authors_org_ref, authors_org_co)
                    fos_name_s = compute_subspace(fos_name_ref_self, fos_name_ref, fos_name_co)
                    publisher_s = compute_subspace(publisher_ref_self, publisher_ref, publisher_co)
                    year_s = compute_subspace(year_ref_self, year_ref, year_co)

#                if curr_id in doc2vec_map and ref in doc2vec_map:
#                    cosine = compute_cosine(doc2vec_map[curr_id], doc2vec_map[ref])
#                else:
#                    cosine = 0
                from_ret.append(curr_id)
                to_ret.append(ref)
                
#                authors_id_prob_ret.append(authors_id_p)
#                authors_org_prob_ret.append(authors_org_p)
#                fos_name_prob_ret.append(fos_name_p)
#                publisher_prob_ret.append(publisher_p)
#                year_prob_ret.append(year_p)
                
                authors_id_subspace_ret.append(authors_id_s)
                authors_org_subspace_ret.append(authors_org_s)
                fos_name_subspace_ret.append(fos_name_s)
                publisher_subspace_ret.append(publisher_s)
                year_subspace_ret.append(year_s)
                
                #cosine_ret.append(cosine)
#        if (i - lo) % 200000 == 0:
#            dump_to_file(from_ret, to_ret, authors_id_subspace_ret, authors_org_subspace_ret, fos_name_subspace_ret, publisher_subspace_ret, year_subspace_ret)
#            from_ret = []
#            to_ret = []
#            authors_id_subspace_ret = []
#            authors_org_subspace_ret = []
#            fos_name_subspace_ret = []
#            publisher_subspace_ret = []
#            year_subspace_ret = []
#            gc.collect()
    #dump_to_file(from_ret, to_ret, authors_id_subspace_ret, authors_org_subspace_ret, fos_name_subspace_ret, publisher_subspace_ret, year_subspace_ret)
            
    return from_ret, to_ret, authors_id_prob_ret, authors_org_prob_ret, fos_name_prob_ret, publisher_prob_ret, year_prob_ret, authors_id_subspace_ret, authors_org_subspace_ret, fos_name_subspace_ret, publisher_subspace_ret, year_subspace_ret, cosine_ret, label_ret

def build_DF(train=True):
    global lines, authors_id_map, authors_org_map, fos_name_map, publisher_map, year_map, reference_map, authors_id_prob, authors_org_prob, fos_name_prob, publisher_prob, year_prob, authors_id_co, authors_org_co, fos_name_co, publisher_co, year_co, doc2vec_map
    
    print("Begin constructing dataframe...")
    if train:
        fname = "ref_graph_map_train.txt"
    else:
        fname = "ref_graph_map_test.txt"
    nrow = 0
    cnt = 0
    with open(fname, "r") as f:
        lines = f.readlines()
    
    from_ret = []
    to_ret = []
    authors_id_prob_ret = []
    authors_org_prob_ret = []
    fos_name_prob_ret = []
    publisher_prob_ret = []
    year_prob_ret = []
    authors_id_subspace_ret = []
    authors_org_subspace_ret = []
    fos_name_subspace_ret = []
    publisher_subspace_ret = []
    year_subspace_ret = []
    cosine_ret = []
    label_ret = []
        
    
#    arg_lst = []
#    results = []
#    procs = []
#    q = Queue()
#    for i in range(n_cpu):
#        arg_lst.append(((i * batch_size), min((i + 1) * batch_size, len(lines))))
#    for i in range(n_cpu):
#        p = Process(target=build_DF_single, args=(arg_lst[i], q))
#        procs.append(p)
    
    if train:
        fname = "dblp.v12.train.json"
    else:
        fname = "dblp.v12.test.json"

    print("Begin loading data...")
    cnt = 0
    with open(fname, "r") as f:
        authors_id_map = {}
        authors_org_map = {}
        fos_name_map = {}
        publisher_map = {}
        year_map = {}
        reference_map = {}
        for line in tqdm(f):
            line = line.strip(",").strip("\n")
            if len(line) > 1:
                d = json.loads(line)
                curr_id = int(d["id"])
                authors_id_map[curr_id] = list(map(str, get_attribute_data(line, ["authors", "id"])))
                authors_org_map[curr_id] = get_attribute_data(line, ["authors", "org"])
                fos_name_map[curr_id] = get_attribute_data(line, ["fos", "name"])
                publisher_map[curr_id] = get_attribute_data(line, ["publisher"])
                year_map[curr_id] = list(map(str, get_attribute_data(line, ["year"])))
                if "references" in d:
                    reference_map[curr_id] = list(map(int, d["references"]))
                else:
                    reference_map[curr_id] = []

    print("Begin loading pretrained data...")
    #authors_id_prob = get_prob("authors_id")
    #authors_org_prob = get_prob("authors_org")
    #fos_name_prob = get_prob("fos_name")
    #publisher_prob = get_prob("publisher")
    #year_prob = get_prob("year")
    authors_id_co = get_co_relation("authors_id")
    authors_org_co = get_co_relation("authors_org")
    fos_name_co = get_co_relation("fos_name")
    publisher_co = get_co_relation("publisher")
    year_co = get_co_relation("year")
    #doc2vec_map = get_doc2vec()
    
#    for p in procs:
#        p.start()
#    for p in procs:
#        ret = q.get()
#        results.append(ret)
#    for p in procs:
#        p.join()
    
#    print("Combining results...")
#    for res in results:
#        from_ret_single, to_ret_single, authors_id_prob_ret_single, authors_org_prob_ret_single, fos_name_prob_ret_single, publisher_prob_ret_single, year_prob_ret_single, authors_id_subspace_ret_single, authors_org_subspace_ret_single, fos_name_subspace_ret_single, publisher_subspace_ret_single, year_subspace_ret_single, cosine_ret_single, label_ret_single = res
#        from_ret += from_ret_single
#        to_ret += to_ret_single
#        authors_id_prob_ret += authors_id_prob_ret_single
#        authors_org_prob_ret += authors_org_prob_ret_single
#        fos_name_prob_ret += fos_name_prob_ret_single
#        publisher_prob_ret += publisher_prob_ret_single
#        year_prob_ret += year_prob_ret_single
#        authors_id_subspace_ret += authors_id_subspace_ret_single
#        authors_org_subspace_ret += authors_org_subspace_ret_single
#        fos_name_subspace_ret += fos_name_subspace_ret_single
#        publisher_subspace_ret += publisher_subspace_ret_single
#        year_subspace_ret += year_subspace_ret_single
#        cosine_ret += cosine_ret_single
#        label_ret += label_ret_single
#    with open("dataframe_train_full_subspace.csv", "a") as f:
#    f.write("From,To,Authors_org_subspace,Fos_name_subspace,Publisher_subspace,Year_subspace\n")
    if n_cpu > 1:
        #with open(fname, "a") as f:
            #f.write("From,To,Authors_org_subspace,Fos_name_subspace,Publisher_subspace,Year_subspace\n")
        
        batch_size = int(math.ceil(len(lines) / n_cpu / 2))
        offset = batch_size * n_cpu
        for j in range(1, 2):
            from_ret = []
            to_ret = []
            authors_id_prob_ret = []
            authors_org_prob_ret = []
            fos_name_prob_ret = []
            publisher_prob_ret = []
            year_prob_ret = []
            authors_id_subspace_ret = []
            authors_org_subspace_ret = []
            fos_name_subspace_ret = []
            publisher_subspace_ret = []
            year_subspace_ret = []
            cosine_ret = []
            label_ret = []
            results = Parallel(n_jobs = n_cpu, backend="multiprocessing")(delayed(build_DF_single)(
                offset * j + (i * batch_size), min(offset * j + (i + 1) * batch_size, len(lines))
            ) for i in range(n_cpu))
            
            print("Combining results...")
            for res in results:
                from_ret_single, to_ret_single, authors_id_prob_ret_single, authors_org_prob_ret_single, fos_name_prob_ret_single, publisher_prob_ret_single, year_prob_ret_single, authors_id_subspace_ret_single, authors_org_subspace_ret_single, fos_name_subspace_ret_single, publisher_subspace_ret_single, year_subspace_ret_single, cosine_ret_single, label_ret_single = res
                from_ret += from_ret_single
                to_ret += to_ret_single
                authors_id_prob_ret += authors_id_prob_ret_single
                authors_org_prob_ret += authors_org_prob_ret_single
                fos_name_prob_ret += fos_name_prob_ret_single
                publisher_prob_ret += publisher_prob_ret_single
                year_prob_ret += year_prob_ret_single
                authors_id_subspace_ret += authors_id_subspace_ret_single
                authors_org_subspace_ret += authors_org_subspace_ret_single
                fos_name_subspace_ret += fos_name_subspace_ret_single
                publisher_subspace_ret += publisher_subspace_ret_single
                year_subspace_ret += year_subspace_ret_single
                cosine_ret += cosine_ret_single
                label_ret += label_ret_single
                
            print("Start Dumping to file...")
            if train:
                fname = "dataframe_train_full_subspace.csv"
            else:
                fname = "dataframe_test_full_subspace.csv"
            with open(fname, "a") as f:
                for i in range(len(from_ret)):
                    line = str(from_ret[i]) + "," + str(to_ret[i]) + "," +\
                        str(authors_id_subspace_ret[i]) + "," +\
                        str(authors_org_subspace_ret[i])+ "," +\
                        str(fos_name_subspace_ret[i]) + "," +\
                        str(publisher_subspace_ret[i]) + "," +\
                        str(year_subspace_ret[i]) + "\n"
                    f.write(line)
#    else:
#        from_ret, to_ret, authors_id_prob_ret, authors_org_prob_ret, fos_name_prob_ret, publisher_prob_ret, year_prob_ret, authors_id_subspace_ret, authors_org_subspace_ret, fos_name_subspace_ret, publisher_subspace_ret, year_subspace_ret, cosine_ret, label_ret = build_DF_single(0, len(lines))
    
    
#build_DF(train=False)
build_DF(train=True)
#d = get_doc2vec()
#print(d.keys()[0], d[d.keys()[0]])

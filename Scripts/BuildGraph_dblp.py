import time, sys
import json
import numpy as np
import math
from joblib import Parallel, delayed
from tqdm import tqdm

class KnowledgeGraphNode:
    def __init__(self, local_identifier, category):
        assert category in ["author", "topic", "publisher", "organization", "year", "paper"]
        self.local_identifier = local_identifier
        self.category = category
        self.global_identifier = category + "_" + str(local_identifier)
        self.owns_list = set([])
        self.attr_list = set([])
        self.cite_list = set([])
    
    def add_to_owns_list(self, paper_to_add):
        assert type(paper_to_add) == str
        if self.category != "paper":
            self.owns_list.add(paper_to_add)
    
    def add_to_attr_list(self, attr_to_add):
        assert type(attr_to_add) == str
        if self.category == "paper":
            self.attr_list.add(attr_to_add)
    
    def add_to_cite_list(self, paper_to_add):
        assert type(paper_to_add) == str
        if self.category == "paper":
            self.cite_list.add(paper_to_add)

    def get_local_identifier(self):
        return self.local_identifier
    
    def get_global_identifier(self):
        return self.global_identifier
    
    def get_category(self):
        return self.category
    
    def get_owns_list(self):
        return [x for x in self.owns_list]

    def get_attr_list(self):
        return [x for x in self.attr_list]
    
    def get_cite_list(self):
        return [x for x in self.cite_list]
    
    def has_own(self, entity):
        return entity in self.owns_list
    
    def has_cite(self, paper):
        return paper in self.cite_list
    
    def has_attr(self, attr):
        return attr in self.attr_list

class KnowledgeGraph:
    def __init__(self, src_filename, embedding_dim = 10):
        self.nodes = {}
        self.src_filename = src_filename
        self.create_graph()
        self.nodes_embeddings = {}
        for node in self.nodes:
            self.nodes_embeddings[node] = np.ones(embedding_dim)
        self.edges_embeddings = {"paper_paper":np.ones(embedding_dim), "paper_author": np.ones(embedding_dim), "paper_topic":np.ones(embedding_dim), "paper_publisher":np.ones(embedding_dim), "paper_organization":np.ones(embedding_dim), "paper_year":np.ones(embedding_dim)}
    
    def get_nodes(self):
        return self.nodes
    
    def set_nodes_embeddings(self, nodes_embeddings):
        for node in nodes_embeddings:
            if node in self.nodes_embeddings:
                self.nodes_embeddings[node] = nodes_embeddings[node]
    
    def set_edges_embeddings(self, edges_embeddings):
        for edge in edges_embeddings:
            if edge in self.edges_embeddings:
                self.edges_embeddings[edge] = edges_embeddings[edge]
    
    def get_nodes_embeddings(self):
        return self.nodes_embeddings
    
    def get_edges_embeddings(self):
        return self.edges_embeddings
    
    def create_graph(self):
        if len(self.nodes) == 0:
            print("Creating Knowledge Graph...")
            with open(self.src_filename, "r") as f:
                for line in tqdm(f):
                    line = line.strip(",").replace("\n", "")
                    if len(line) > 1:
                        d = json.loads(line)
                        curr_id = str(d["id"])
                        authors_id_arr = list(map(str, self.get_attribute_data(line, ["authors", "id"])))
                        authors_org_arr = self.get_attribute_data(line, ["authors", "org"])
                        fos_name_arr = self.get_attribute_data(line, ["fos", "name"])
                        publisher_arr = self.get_attribute_data(line, ["publisher"])
                        year_arr = list(map(str, self.get_attribute_data(line, ["year"])))
                        
                        if "references" in d:
                            reference_arr = list(map(str, d["references"]))
                        else:
                            reference_arr = []
                        
                        self.paper_attr_edge(curr_id, authors_id_arr, "author")
                        self.paper_attr_edge(curr_id, authors_org_arr, "organization")
                        self.paper_attr_edge(curr_id, fos_name_arr, "topic")
                        self.paper_attr_edge(curr_id, publisher_arr, "publisher")
                        self.paper_cite_edge(curr_id, reference_arr)
            print("Done Creating Knowledge Graph!")
        else:
            print("Knowledge Graph Is Already Created!")
    
    def paper_attr_edge(self, paper_id, attr_arr, attr_name):
        assert attr_name in ["author", "topic", "publisher", "organization", "year"]
        paper_identifier = "paper_" + paper_id
        if paper_identifier not in self.nodes:
            node = KnowledgeGraphNode(paper_id, "paper")
            self.nodes[paper_identifier] = node
        for attr in attr_arr:
            global_identifier = attr_name + "_" + attr
            if global_identifier not in self.nodes:
                node = KnowledgeGraphNode(attr, attr_name)
                self.nodes[global_identifier] = node
            self.nodes[global_identifier].add_to_owns_list(paper_identifier)
            self.nodes[paper_identifier].add_to_attr_list(global_identifier)
    
    def paper_cite_edge(self, paper_id, ref_arr):
        central_paper_identifier = "paper_" + paper_id
        if central_paper_identifier not in self.nodes:
            node = KnowledgeGraphNode(central_paper_id, "paper")
            self.nodes[central_paper_identifier] = node
        for ref_paper_id in ref_arr:
            ref_paper_identifier = "paper_" + ref_paper_id
            if ref_paper_identifier not in self.nodes:
                node = KnowledgeGraphNode(ref_paper_id, "paper")
                self.nodes[ref_paper_identifier] = node
            self.nodes[central_paper_identifier].add_to_cite_list(ref_paper_identifier)
    
    def get_attribute_data(self, line, attr_arr):
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

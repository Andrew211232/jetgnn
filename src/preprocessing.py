import uproot
import numpy as np
import awkward as ak
import torch
from torch_geometric.data import Data
import os
import random

training_files = {
    "HToBB": [
        'data/raw/JetClass_5M/HToBB_120.root:tree',
        'data/raw/JetClass_5M/HToBB_122.root:tree',
        'data/raw/JetClass_5M/HToBB_124.root:tree',
    ],
    "HToCC": [
        'data/raw/JetClass_5M/HToCC_120.root:tree',
        'data/raw/JetClass_5M/HToCC_122.root:tree',
        'data/raw/JetClass_5M/HToCC_124.root:tree',
    ],
    "HToGG": [
        'data/raw/JetClass_5M/HToGG_120.root:tree',
        'data/raw/JetClass_5M/HToGG_122.root:tree',
        'data/raw/JetClass_5M/HToGG_124.root:tree',
    ],
    "HToWW2Q1L": [
        'data/raw/JetClass_5M/HToWW2Q1L_120.root:tree',
        'data/raw/JetClass_5M/HToWW2Q1L_122.root:tree',
        'data/raw/JetClass_5M/HToWW2Q1L_124.root:tree',
    ],
    "HToWW4Q": [
        'data/raw/JetClass_5M/HToWW4Q_120.root:tree',
        'data/raw/JetClass_5M/HToWW4Q_122.root:tree',
        'data/raw/JetClass_5M/HToWW4Q_124.root:tree',
    ],
    "TTBar": [
        'data/raw/JetClass_5M/TTBar_120.root:tree',
        'data/raw/JetClass_5M/TTBar_122.root:tree',
        'data/raw/JetClass_5M/TTBar_124.root:tree',
    ],
    "TTBarLep": [
        'data/raw/JetClass_5M/TTBarLep_120.root:tree',
        'data/raw/JetClass_5M/TTBarLep_122.root:tree',
        'data/raw/JetClass_5M/TTBarLep_124.root:tree',
    ],
    "WToQQ": [
        'data/raw/JetClass_5M/WToQQ_120.root:tree',
        'data/raw/JetClass_5M/WToQQ_122.root:tree',
        'data/raw/JetClass_5M/WToQQ_124.root:tree',
    ],
    "ZJetsToNuNu": [
        'data/raw/JetClass_5M/ZJetsToNuNu_120.root:tree',
        'data/raw/JetClass_5M/ZJetsToNuNu_122.root:tree',
        'data/raw/JetClass_5M/ZJetsToNuNu_124.root:tree',
    ],
    "ZtoQQ": [
        'data/raw/JetClass_5M/ZToQQ_120.root:tree',
        'data/raw/JetClass_5M/ZToQQ_122.root:tree',
        'data/raw/JetClass_5M/ZToQQ_124.root:tree',
    ]
}
validation_files = {
    "HToBB": ['data/raw/JetClass_5M/HToBB_123.root:tree'],
    "HToCC": ['data/raw/JetClass_5M/HToCC_123.root:tree'],
    "HToGG": ['data/raw/JetClass_5M/HToGG_123.root:tree'],
    "HToWW2Q1L": ['data/raw/JetClass_5M/HToWW2Q1L_123.root:tree'],
    "HToWW4Q": ['data/raw/JetClass_5M/HToWW4Q_123.root:tree'],
    "TTBar": ['data/raw/JetClass_5M/TTBar_123.root:tree'],
    "TTBarLep": ['data/raw/JetClass_5M/TTBarLep_123.root:tree'],
    "WToQQ": ['data/raw/JetClass_5M/WToQQ_123.root:tree'],
    "ZJetsToNuNu": ['data/raw/JetClass_5M/ZJetsToNuNu_123.root:tree'],
    "ZtoQQ": ['data/raw/JetClass_5M/ZToQQ_123.root:tree']
}
testing_files = {
    "HToBB": ['data/raw/JetClass_5M/HToBB_121.root:tree'],
    "HToCC": ['data/raw/JetClass_5M/HToCC_121.root:tree'],
    "HToGG": ['data/raw/JetClass_5M/HToGG_121.root:tree'],
    "HToWW2Q1L": ['data/raw/JetClass_5M/HToWW2Q1L_121.root:tree'],
    "HToWW4Q": ['data/raw/JetClass_5M/HToWW4Q_121.root:tree'],
    "TTBar": ['data/raw/JetClass_5M/TTBar_121.root:tree'],
    "TTBarLep": ['data/raw/JetClass_5M/TTBarLep_121.root:tree'],
    "WToQQ": ['data/raw/JetClass_5M/WToQQ_121.root:tree'],
    "ZJetsToNuNu": ['data/raw/JetClass_5M/ZJetsToNuNu_121.root:tree'],
    "ZtoQQ": ['data/raw/JetClass_5M/ZToQQ_121.root:tree']
}

def process_chunk(chunk):

    # Targets
    target_names = [f for f in chunk.fields if f.startswith("label_")]
    target_matrix = np.column_stack([ak.to_numpy(chunk[name]) for name in target_names])
    targets = np.argmax(target_matrix, axis=-1)

    # Features
    feature_names = sorted([f for f in chunk.fields if f.startswith("part_")]) # includes only the low level features we want
    graphs = []

    for jet in range(len(targets)):
        particle_features = [ak.to_numpy(chunk[name][jet]) for name in feature_names] # gets the features of each particle
        x = np.stack(particle_features, axis=-1)

        amount_of_particles = x.shape[0] # used for iteration over the set of particles
        this_node, adjacent_node = np.where(
            ~np.eye(amount_of_particles, dtype=bool)
        )
        edge_index = torch.tensor(np.stack([this_node, adjacent_node]), dtype=torch.long)
        particle_graph = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index = edge_index,
            y = torch.tensor([targets[jet]], dtype=torch.long))
        graphs.append(particle_graph)

    return graphs

def make_chunks(files, step=1000):
    iterators = {}

    for cls, file in files.items():
        iterators[cls] = uproot.iterate(
            file,
            filter_name=["part_*", "label_*"],
            step_size=step,
        )
    while True:

        graph_list = []
        for cls, iterator in list(iterators.items()):
            try:
                chunk = next(iterator)
            except StopIteration:
                del iterators[cls]
                continue
            graph = process_chunk(chunk)
            graph_list.extend(graph)
        if len(graph_list) == 0:
            break

        random.shuffle(graph_list)

        yield graph_list

def get_mean_std(files):

    total = 0
    sum_x = None
    sum_x2 = None

    for graph_list in make_chunks(files): # optional to add float64 dtypes
        for graph in graph_list:
            x = graph.x
            if sum_x is None:
                x_features = x.shape[1]
                sum_x = torch.zeros(x_features, dtype=torch.float64)
                sum_x2 = torch.zeros(x_features, dtype=torch.float64)
            x=x.to(torch.float64)
            sum_x += x.sum(dim=0)
            sum_x2 += (x ** 2).sum(dim=0)
            total += x.shape[0]
    mean = sum_x / total
    std = torch.sqrt(sum_x2 / total - mean**2)
    std[std == 0] = 1

    return mean, std


def preprocess_and_save(files, output_dir, step=1000):
    os.makedirs(output_dir, exist_ok=True)
    for category, file_list in files.items():
        iterator = uproot.iterate(
            file_list, filter_name=["part_*", "label_*"], step_size=step,
        )
        chunk_number = 0
        for chunk in iterator:
            graphs = process_chunk(chunk)
            save_path = os.path.join(output_dir, f"{category}_{chunk_number}.pt")
            torch.save(graphs, save_path)
            chunk_number += 1
    print(f"finished preprocessing {output_dir}")

mean, std = get_mean_std(training_files)
torch.save({
    "mean": mean,
    "std": std
}, "/home/lustre/anmckeev/JetGNN/data/processed/normalization.pt")

preprocess_and_save(training_files, "/home/lustre/anmckeev/JetGNN/data/processed/processed_training")
preprocess_and_save(validation_files, "/home/lustre/anmckeev/JetGNN/data/processed/processed_validation")
preprocess_and_save(testing_files, "/home/lustre/anmckeev/JetGNN/data/processed/processed_testing")

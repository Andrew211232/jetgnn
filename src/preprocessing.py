import uproot
import numpy as np
import awkward as ak
import torch
from torch_geometric.data import Data
import os
import random

training_files = {
    "HToBB": [
        'JetClass_values/HToBB_120.root:tree',
        'JetClass_values/HToBB_122.root:tree',
        'JetClass_values/HToBB_124.root:tree',
    ],
    "HToCC": [
        'JetClass_values/HToCC_120.root:tree',
        'JetClass_values/HToCC_122.root:tree',
        'JetClass_values/HToCC_124.root:tree',
    ],
    "HToGG": [
        'JetClass_values/HToGG_120.root:tree',
        'JetClass_values/HToGG_122.root:tree',
        'JetClass_values/HToGG_124.root:tree',
    ],
    "HToWW2Q1L": [
        'JetClass_values/HToWW2Q1L_120.root:tree',
        'JetClass_values/HToWW2Q1L_122.root:tree',
        'JetClass_values/HToWW2Q1L_124.root:tree',
    ],
    "HToWW4Q": [
        'JetClass_values/HToWW4Q_120.root:tree',
        'JetClass_values/HToWW4Q_122.root:tree',
        'JetClass_values/HToWW4Q_124.root:tree',
    ],
    "TTBar": [
        'JetClass_values/TTBar_120.root:tree',
        'JetClass_values/TTBar_122.root:tree',
        'JetClass_values/TTBar_124.root:tree',
    ],
    "TTBarLep": [
        'JetClass_values/TTBarLep_120.root:tree',
        'JetClass_values/TTBarLep_122.root:tree',
        'JetClass_values/TTBarLep_124.root:tree',
    ],
    "WToQQ": [
        'JetClass_values/WToQQ_120.root:tree',
        'JetClass_values/WToQQ_122.root:tree',
        'JetClass_values/WToQQ_124.root:tree',
    ],
    "ZJetsToNuNu": [
        'JetClass_values/ZJetsToNuNu_120.root:tree',
        'JetClass_values/ZJetsToNuNu_122.root:tree',
        'JetClass_values/ZJetsToNuNu_124.root:tree',
    ],
    "ZtoQQ": [
        'JetClass_values/ZToQQ_120.root:tree',
        'JetClass_values/ZToQQ_122.root:tree',
        'JetClass_values/ZToQQ_124.root:tree',
    ]
}
validation_files = {
    "HToBB": ['JetClass_values/HToBB_123.root:tree'],
    "HToCC": ['JetClass_values/HToCC_123.root:tree'],
    "HToGG": ['JetClass_values/HToGG_123.root:tree'],
    "HToWW2Q1L": ['JetClass_values/HToWW2Q1L_123.root:tree'],
    "HToWW4Q": ['JetClass_values/HToWW4Q_123.root:tree'],
    "TTBar": ['JetClass_values/TTBar_123.root:tree'],
    "TTBarLep": ['JetClass_values/TTBarLep_123.root:tree'],
    "WToQQ": ['JetClass_values/WToQQ_123.root:tree'],
    "ZJetsToNuNu": ['JetClass_values/ZJetsToNuNu_123.root:tree'],
    "ZtoQQ": ['JetClass_values/ZToQQ_123.root:tree']
}
testing_files = {
    "HToBB": ['JetClass_values/HToBB_121.root:tree'],
    "HToCC": ['JetClass_values/HToCC_121.root:tree'],
    "HToGG": ['JetClass_values/HToGG_121.root:tree'],
    "HToWW2Q1L": ['JetClass_values/HToWW2Q1L_121.root:tree'],
    "HToWW4Q": ['JetClass_values/HToWW4Q_121.root:tree'],
    "TTBar": ['JetClass_values/TTBar_121.root:tree'],
    "TTBarLep": ['JetClass_values/TTBarLep_121.root:tree'],
    "WToQQ": ['JetClass_values/WToQQ_121.root:tree'],
    "ZJetsToNuNu": ['JetClass_values/ZJetsToNuNu_121.root:tree'],
    "ZtoQQ": ['JetClass_values/ZToQQ_121.root:tree']
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
preprocess_and_save(training_files, "/home/lastc/projects/JetGNN/data/processed/processed_training")
preprocess_and_save(validation_files, "/home/lastc/projects/JetGNN/data/processed/processed_validation")
preprocess_and_save(testing_files, "/home/lastc/projects/JetGNN/data/processed/processed_testing")

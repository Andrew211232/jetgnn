import uproot
import numpy as np
import awkward as ak
import torch
from torch_geometric.data import Data
import os
import random
from pathlib import Path

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
    "ZToQQ": [
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
    "ZToQQ": ['data/raw/JetClass_5M/ZToQQ_123.root:tree']
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
    "ZToQQ": ['data/raw/JetClass_5M/ZToQQ_121.root:tree']
}
class_labels = [
    "HToBB",
    "HToCC",
    "HToGG",
    "HToWW2Q1L",
    "HToWW4Q",
    "TTBar",
    "TTBarLep",
    "WToQQ",
    "ZJetsToNuNu",
    "ZToQQ"
]

def process_chunk(chunk):
    # Targets
    target_matrix = np.column_stack([ak.to_numpy(chunk[f"label_{name}"]) for name in class_labels])
    targets = np.argmax(target_matrix, axis=-1)

    # Features
    feature_names = sorted([f for f in chunk.fields if f.startswith("part_")]) # includes only the low level features we want
    graphs = []

    for jet in range(len(targets)):
        particle_features = [ak.to_numpy(chunk[name][jet]) for name in feature_names] # gets the features of each particle
        check_length = [len(feature) for feature in particle_features]
        if len(set(check_length)) != 1:
            raise ValueError(f"Inconsistent particle feature lengths for jet {jet}: {check_length}")
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

def get_mean_std(files, step=1000):

    total = 0
    sum_x = None
    sum_x2 = None

    for file_list in files.values():
        iterator = uproot.iterate(
            file_list,
            filter_name=["part_*"],
            step_size=step
        )
        for chunk in iterator:
            feature_names = sorted([f for f in chunk.fields if f.startswith("part_")])
            if len(feature_names) != 16:
                raise ValueError(f"There were not 16 features in Jet, only {len(feature_names)}")
            particle_count = ak.to_numpy(ak.num(chunk[feature_names[0]], axis=-1))
            for name in feature_names[1:]:
                counts = ak.to_numpy(ak.num(chunk[name], axis=-1))
                if not np.array_equal(particle_count, counts):
                    raise ValueError("Particle counts are inconsistent")
            if sum_x is None:
                sum_x = np.zeros(len(feature_names), dtype=torch.float64)
                sum_x2 = np.zeros(len(feature_names), dtype=torch.float64)
            for index, value in enumerate(feature_names):
                values = ak.to_numpy(ak.flatten(chunk[value], axis=None)).astype(np.float64, copy=False)
                sum_x[index] += np.sum(values, dtype=np.float64)
                sum_x2[index] += np.sum(np.square(values), dtype=np.float64)
            total += particle_count.sum()
    if total == 0:
        raise ValueError("No graphs in list, check input files")
    mean = sum_x / total
    variance = np.maximum(sum_x2 / total - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-12] = 1.0

    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


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

project_dir = Path(__file__).resolve().parent.parent
processed_dir = project_dir / "data" / "processed"
if __name__ == "__main__":
    mean, std = get_mean_std(training_files)
    processed_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mean": mean,
        "std": std,
        "class_names": class_labels
    }, processed_dir / "normalization.pt")

    preprocess_and_save(training_files, processed_dir / "processed_training")
    preprocess_and_save(validation_files, processed_dir / "processed_validation")
    preprocess_and_save(testing_files, processed_dir / "processed_testing")

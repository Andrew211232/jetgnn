import torch
import glob, os
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import random

def load_processed_chunks(directory, include_category=False, shuffle=False):

    files = sorted(glob.glob(
        os.path.join(directory, "*.pt")
    ))
    if shuffle:
        random.shuffle(files)
    if not files:
        raise FileExistsError("No files to load")
    for filename in files:
        graphs = torch.load(filename, weights_only=False)
        if include_category:
            category = os.path.basename(filename).rsplit("_", 1)[0]
            yield category, graphs
        else:
            yield graphs

class EarlyStopping: # I could also implement this directly into the code
    def __init__(self, patience, delta_min):
        self.patience = patience
        self.delta_min = delta_min
        self.count = 0
        self.best_model = float("inf")
        self.early_stop = False
    def __call__(self, validation_loss):
        if validation_loss < self.best_model - self.delta_min:
            self.best_model = validation_loss
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())
    def forward(self, x):
        return (x-self.mean) / self.std

class MessageLayer(nn.Module):
    def __init__(self, hidden_layer=64):
        super().__init__()
        self.message_network = nn.Sequential(
            nn.Linear(2*hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
        )
        self.update_network = nn.Sequential(
            nn.Linear(2*hidden_layer, hidden_layer),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(hidden_layer) # Later I can try Mean Aggregation rather than Sum Aggregation
    def forward(self, x, edge_index):
        this_node = edge_index[0] # Gets this nodes data from input tensor
        adjacent_node = edge_index[1] # Gets other nodes data from input tensor
        this_node_features = x[this_node] # gets the features from x for this node
        adjacent_node_features = x[adjacent_node] # gets the features from for other nodes
        edge_features = torch.cat([this_node_features, adjacent_node_features], dim=1) # combines features into one list
        node_messages = self.message_network(edge_features) # passes said list into the linear/ReLU transformations
        incoming_update = torch.zeros_like(x).index_add(0, adjacent_node, node_messages) # creates a x-sized tensor and adds info from the adjacent node
        update_node = torch.cat([x, incoming_update], dim=1) # adds the new info to x
        updated_node = self.update_network(update_node) # passes x through more linear/ReLU layers
        return self.norm(x + updated_node)


class JetClassGNN(nn.Module):
    def __init__(self, input_mean, input_std, starting_size = 16, hidden_layer_size = 64, classes = 10, dropout_rate = 0.3):
        super().__init__()
        self.input_to_hidden = nn.Linear(starting_size, hidden_layer_size)
        self.send_messages_1 = MessageLayer(hidden_layer_size)
        self.send_messages_2 = MessageLayer(hidden_layer_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_to_classes = nn.Linear(hidden_layer_size, classes)
        self.normalize = Normalize(input_mean, input_std)

    def forward(self, data):
        x = self.normalize(data.x)
        edge_index = data.edge_index
        batch = data.batch
        x = torch.relu(self.input_to_hidden(x))
        x = self.send_messages_1(x, edge_index)
        x = self.dropout(x)
        x = self.send_messages_2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.hidden_to_classes(x)

        return x

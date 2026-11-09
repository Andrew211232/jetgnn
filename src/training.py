import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import models

gpu_or_cpu = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
normalization = torch.load("data/processed/normalization.pt", weights_only=True)
mean = normalization["mean"]
std = normalization["std"]

random.seed(12345)
np.random.seed(12345)
torch.manual_seed(12345)
EPOCHS = 1
#mean, std = preprocessing.get_mean_std(training_files)
model = models.JetClassGNN(mean, std)
model = model.to(gpu_or_cpu)
best_model = models.EarlyStopping(patience=5, delta_min=1*e^4)

optimizer = torch.optim.AdamW(model.parameters(), lr=-4, weight_decay=3e-3)
loss_function = nn.CrossEntropyLoss()

train_losses = []
train_accuracies = []
validation_losses = []
validation_accuracies = []

for epoch in range(EPOCHS):
    # Training Phase
    model.train()
    train_loss = 0
    train_graphs = 0
    train_correct = 0
    train_total = 0
    training_graph_list = models.load_processed_chunks("training_files")
    for graph in training_graph_list:
        if not graph:
            continue
        loader = DataLoader(graph, batch_size = 5, shuffle=True)
        for batch in loader:
            batch = batch.to(gpu_or_cpu)
            optimizer.zero_grad()
            output = model(batch)
            prediction = output.argmax(dim=1)
            targets = batch.y.view(-1)
            loss = loss_function(output, targets)
            loss.backward()
            optimizer.step()
            train_graphs += batch.num_graphs
            train_loss += loss.item() * batch.num_graphs
            train_correct += (prediction == targets).sum().item()
            train_total += len(targets)
    train_loss /= train_graphs
    train_accuracies.append(train_correct / train_total)

    # Validation Stage
    model.eval()
    validation_loss = 0
    correct = 0
    total = 0
    validation_graphs = 0
    validation_graph_list = models.load_processed_chunks("validation_files")
    with torch.no_grad():
        for graph in validation_graph_list:
            if not graph:
                continue
            loader = DataLoader(graph, batch_size = 256, shuffle=False)
            for batch in loader:
                batch = batch.to(gpu_or_cpu)
                output = model(batch)
                targets = batch.y.view(-1)
                validation_loss += loss_function(output, targets).item() * batch.num_graphs
                validation_graphs += batch.num_graphs
                prediction = output.argmax(dim=1)
                correct += (prediction == targets).sum().item()
                total += len(targets)
        validation_loss /= validation_graphs
    print(f"Epoch {epoch}   ", f"training loss: {train_loss:.4f}   ", f"validation loss: {validation_loss:.4f}   ", f"validation accuracy: {correct/total:.4f}")
    train_losses.append(train_loss)
    validation_losses.append(validation_loss)
    validation_accuracies.append(correct / total)

# Testing Phase
testing_loss = 0
test_correct = 0
test_total = 0
test_graphs = 0
test_predictions = []
test_targets = []
testing_graph_list = models.load_processed_chunks("testing_files")
with torch.no_grad():
    for graph in testing_graph_list:
        if not graph:
            continue
        loader = DataLoader(graph, batch_size = 256, shuffle=False)
        for batch in loader:
            batch = batch.to(gpu_or_cpu)
            output = model(batch)
            targets = batch.y.view(-1)
            testing_loss += loss_function(output, targets).item() * batch.num_graphs
            test_graphs += batch.num_graphs
            prediction = output.argmax(dim=1)
            test_predictions.extend(prediction.detach().cpu().numpy())
            test_targets.extend(targets.detach().cpu().numpy())
            test_correct += (prediction == targets).sum().item()
            test_total += len(targets)
    testing_loss /= test_graphs
print(
    f"Test loss: {testing_loss:.4f}    "
    f"Test accuracy: {test_correct/test_total:.4f}"
)

# Training Loss vs Validation Loss
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend()
plt.grid()
plt.show()

# Training and Validation Accuracy
plt.figure(figsize=(8,5))
plt.plot(validation_accuracies, label='Validation')
plt.plot(train_accuracies, label='Training')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.grid()
plt.show()

# Confusion Matrix
# cm = confusion_matrix(test_targets, test_predictions)
# cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
# plt.figure(figsize=(8,8))
# plt.imshow(cm_norm, vmin=0, vmax=1)
# plt.colorbar()
# classes = ["HToBB", "HToCC", "HToGG", "HToWW2Q1L", "HToWW4Q", "TTBar", "TTBarLep", "WToQQ", "ZJetsToNuNu", "ZtoQQ"]
# plt.xticks(range(10), classes, rotation=45)
# plt.yticks(range(10), classes)
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.show()

# Per-Class Accuracy
# per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
# for name, acc in zip(classes, per_class_accuracy):
#     print(f"{name:15s}: {acc:.3f}")
# plt.figure(figsize=(10,5))
# plt.bar(classes, per_class_accuracy)
# plt.xticks(rotation=45)
# plt.ylabel("Accuracy")
# plt.ylim(0,1)
# plt.title("Per Class Accuracy")
# plt.grid(axis="y")
# plt.show()

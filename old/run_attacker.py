import numpy as np
import torch
import torch.nn as nn
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score
from utils import EarlyStopper
from itertools import product


class AttackerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttackerNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)#,  # Binary classification
        )

    def forward(self, x):
        return self.model(x)


def run_attacker(x_train, t_train, x_test, t_test, hyperparameters):
    n_hidden = hyperparameters["n_hidden"]
    learning_rate = hyperparameters["lr"]
    batch_size = hyperparameters["batch_size"]

    # Prepare DataLoader
    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(t_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(t_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and train the model
    model = AttackerNetwork(input_size=x_train.shape[1], hidden_size=n_hidden).to(device)
    if torch.cuda.is_available():
        device = "gpu"
    else:
        device = "cpu"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    early_stopper = EarlyStopper(patience=10, min_delta=0)
    for epoch in range(1000):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # compute test loss
        model.eval()
        total_test_loss = 0
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.long())
            total_test_loss += loss

        #if epoch % 10 == 0:
        #    print(f"Epoch {epoch}, Train Loss: {total_loss:.4f}, Test Loss : {total_test_loss:.4f}")

        # todo change that
        if early_stopper.early_stop(total_test_loss):
            #print(f"Epoch {epoch}, Train Loss: {total_loss:.4f}, Test Loss : {total_test_loss:.4f}")
            break

    model.eval()
    predictions, groundtruth = [], []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        predicted_class = np.argmax(outputs.detach().numpy(), axis=1)
        predictions.extend(predicted_class)
        groundtruth.extend(labels.numpy())
    bacc_train = balanced_accuracy_score(groundtruth, predictions)

    predictions, groundtruth = [], []
    for inputs, labels in test_loader:
        outputs = model(inputs.to(device))
        predicted_class = np.argmax(outputs.detach().numpy(), axis=1)
        predictions.extend(predicted_class)
        groundtruth.extend(labels.to(device).numpy())
    bacc_test = balanced_accuracy_score(groundtruth, predictions)

    # estimate epsilon
    tpr = np.mean(np.bitwise_and(groundtruth, predictions))
    fpr = np.mean(np.bitwise_and(np.logical_not(groundtruth), predictions))
    e = np.log(tpr / fpr)

    #print("BAcc: %f (Train), %f (Val)" % (bacc_train, bacc_test))

    return bacc_train, bacc_test, e

def load_splits(path, name):
    with open(path + "/" + name + ".X_train", "rb") as file:
        X_train = pickle.load(file)
    with open(path + "/" + name + ".T_train", "rb") as file:
        T_train = pickle.load(file)
    with open(path + "/" + name + ".X_test", "rb") as file:
        X_test = pickle.load(file)
    with open(path + "/" + name + ".T_test", "rb") as file:
        T_test = pickle.load(file)

    return X_train, X_test, T_train, T_test

dataset = "ml1m"
method = "random_dp"
compute_baseline = False
if method.endswith("_dp"):
    epsilons = [3, 2, 1, 0.1]
else:
    epsilons = [np.inf]
betas = [0.8, 0.6, 0.4, 0.2, 0]
results = []
configs = list(product(epsilons, betas))
if compute_baseline:
    configs = [(np.inf, 1)] + configs
for i, (e, b) in enumerate(configs):
    print("=============================================================")
    print("Run %s e=%f, b=%f (%d/%d)" % (method, e, b, i+1, len(configs)))
    print("=============================================================")

    # NoDP Baseline
    if b == 1 and e == np.inf:
        filename = dataset + ".train"
    elif b == 0:
        # FullDP Baseline
        if method.endswith("_dp"):
            filename = dataset + ".train_e" + str(e) + "_b" + str(b) + "_random_dp"
        else:
            filename = dataset + ".train_b" + str(b) + "_random_del"
    else:
        if method.endswith("_dp"):
            filename = dataset + ".train_e" + str(e) + "_b" + str(b) + "_" + method + ""
        else:
            filename = dataset + ".train_b" + str(b) + "_" + method + ""


    #filename_template = dataset + ".train_e" + str(e) + "_b" + str(b) + "_" + method
    X_train, X_test, T_train, T_test = load_splits(path="attacker/" + dataset, name=filename)

    # select best params
    with open("attacker/" + dataset + "/" + filename + ".hypertuning", "rb") as f:
        hypertuning_results = pickle.load(f)
        best_val_bacc = -np.inf
        best_config = hypertuning_results[0]
        for r in hypertuning_results:
            if r["avg_bacc_val"] > best_val_bacc:
                best_val_bacc = r["avg_bacc_val"]
                best_config = r
        print(best_config)

    #best_params = {"n_hidden": 128, "batch_size": 64, "lr": 0.001}

    bacc_train, bacc_test, e_est = run_attacker(x_train=X_train, t_train=T_train, x_test=X_test, t_test=T_test,
                                                hyperparameters=best_config)


    print(bacc_train, bacc_test, e_est)

    res = {"e": e, "b": b, "train_bacc": bacc_train, "test_bacc": bacc_test, "e_est": e_est}
    results.append(res)

with open("results/" + dataset + "/attacker." + method, "wb") as f:
    pickle.dump(results, f)
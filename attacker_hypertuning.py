import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from sklearn.metrics import balanced_accuracy_score, auc, roc_curve
from sklearn.preprocessing import Normalizer
from copy import deepcopy
#from utils import obfuscation_per_user_random, sampling_procedure, obfuscation_per_user_ister_leaky, obfuscation_per_user_ister, sample_blur_random, obfuscation_per_user_entropy, obfuscation_per_user_coeffs, obfuscation_per_user_popularity
from datetime import datetime as dt
import os
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

# todo mapping works, we need it for sparse matrix,
def _transform_dataframe_to_sparse(interactions_df, gender_map, max_items, max_users):
    df = interactions_df.copy()
    df["gender"] = [gender_map[user_id] for user_id in df["user_id"]]
    item_map = {b: a for a, b in enumerate(df["item_id"].unique())}
    n_users = max_users
    n_items = max_items
    df["item_id"] = df["item_id"].map(item_map)
    sparse_interaction_matrix = sp.csr_matrix((df["rating"].values, (df["user_id"].values, df["item_id"].values)), (n_users, n_items))

    gender_per_user_df = df[["user_id", "gender"]].drop_duplicates()
    gender_values = (gender_per_user_df["gender"] == "M").astype(int).values

    return sparse_interaction_matrix, gender_values, item_map

def generate_dense_matrix(trainset, user_attributes, n_items, n_users):
    T = np.array([int(user_attributes[uid]) for uid in range(n_users)])
    X = np.zeros((n_users, n_items))
    for uid, iid, r in trainset:
        X[int(uid), int(iid)] = r
    return X, T


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

        # compute validation loss
        model.eval()
        total_test_loss = 0
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.long())
            total_test_loss += loss

        #if epoch % 10 == 0:
        #    print(f"Epoch {epoch}, Train Loss: {total_loss:.4f}, Test Loss : {total_test_loss:.4f}")

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

def tune_attacker(data, target):
    n_hidden_space = [32, 64, 128, 256]
    batch_size_space = [32, 64, 128]
    lr_space = [0.0001, 0.001, 0.01]
    configs = list(product(n_hidden_space, batch_size_space, lr_space))

    results = []
    for i, (n_hidden, batch_size, lr) in enumerate(configs):
        print("Run config %d/%d (n_hidden: %d, batch_size: %d, lr: %f)" % (i+1, len(configs), n_hidden, batch_size, lr))
        print("=====================")
        cv = StratifiedKFold(n_splits=5)
        bacc_val_c = []
        for f, (train, val) in enumerate(cv.split(data, target)):
            #print("Fold %d ..." % (f+1))
            x_train, t_train = data[train], target[train]
            x_val, t_val = data[val], target[val]
            bacc_train, bacc_val, _ = run_attacker(x_train=x_train, t_train=t_train, x_test=x_val, t_test=t_val, hyperparameters={"n_hidden": n_hidden, "batch_size": batch_size, "lr": lr})
            bacc_val_c.append(bacc_val)
        bacc_val_c = np.mean(bacc_val_c)
        print("BAcc: %f (Val)" % bacc_val_c)
        results.append({"n_hidden": n_hidden, "batch_size": batch_size, "lr": lr, "avg_bacc_val": bacc_val_c})
        print()

    best_val_bacc = -np.inf
    best_config = results[0]
    for r in results:
        if r["avg_bacc_val"] > best_val_bacc:
            best_val_bacc = r["avg_bacc_val"]
            best_config = r
    print(best_config)
    return results, best_config

def save_splits(path, name, X_train, T_train, X_test, T_test):
    os.makedirs(path, exist_ok=True)
    with open(path + "/" + name + ".X_train", "wb") as file:
        pickle.dump(X_train, file)
    with open(path + "/" + name + ".T_train", "wb") as file:
        pickle.dump(T_train, file)
    with open(path + "/" + name + ".X_test", "wb") as file:
        pickle.dump(X_test, file)
    with open(path + "/" + name + ".T_test", "wb") as file:
        pickle.dump(T_test, file)


dataset = "ml1m"
PATH = "custom_datasets_prepared_rp/" + dataset + "/"
method = "random_dp"

epsilons = [3, 2, 1, 0.1]
betas = [0.8, 0.6, 0.4, 0.2, 0.0] #[1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
configs = list(product(epsilons, betas))
for i, (e, b) in enumerate(configs):
    print("=============================================================")
    print("Run e=%f, b=%f (%d/%d)" % (e, b, i+1, len(configs)))
    print("=============================================================")
    filename = dataset + ".train_e" + str(e) + "_b" + str(b) + "_" + method + ".rating"

    trainset = pd.read_csv(PATH + filename, sep="\t", header=None).to_records(index=False).tolist()
    user_attr_df = pd.read_csv(PATH + dataset + ".userlist", sep="\t")
    user_attr_map = user_attr_df.set_index("user_id:token")["attr:token"].to_dict()
    itemlist = pd.read_csv(PATH + dataset + ".itemlist", header=None).squeeze().values.tolist()

    n_users, n_items = len(user_attr_map), len(itemlist)
    R, T = generate_dense_matrix(trainset=trainset, user_attributes=user_attr_map, n_users=n_users, n_items=n_items)

    X_train, X_test, t_train, t_test = train_test_split(R, T, test_size=0.2, stratify=T)
    path = "attacker/" + dataset
    name = filename.replace(".rating", "")
    save_splits(path=path, name=name, X_train=X_train, T_train=t_train, X_test=X_test, T_test=t_test)

    hypertuning_results, best_config = tune_attacker(X_train, t_train)
    os.makedirs("attacker/" + dataset, exist_ok=True)
    with open("attacker/" + dataset + "/" + filename.replace(".rating", "") + ".hypertuning", "wb") as f:
        pickle.dump(hypertuning_results, f)
    print()
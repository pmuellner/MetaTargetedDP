import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from sklearn.metrics import balanced_accuracy_score, auc, roc_curve
from sklearn.preprocessing import Normalizer
from copy import deepcopy
from utils import obfuscation_per_user_random, sampling_procedure, obfuscation_per_user_ister_leaky, obfuscation_per_user_ister, sample_blur_random, obfuscation_per_user_entropy, obfuscation_per_user_coeffs, obfuscation_per_user_popularity
from datetime import datetime as dt
import os

def construct_fname(attacker, method, epsilon, beta=None):
    if beta:
        name = method + "_" + attacker + "_e" + str(epsilon) + "_b" + str(beta) + "_t" + dt.now().strftime('%Y%m%d%H%M')
    else:
        name = method + "_" + attacker + "_e" + str(epsilon) + "_t" + dt.now().strftime('%Y%m%d%H%M')
    return name


def _train_attacker(model, train_loader, epochs=50, lr=0.001):
#def _train_attacker(model, train_loader, epochs=200, lr=0.0005):
    if torch.cuda.is_available():
        device = "gpu"
    else:
        device = "cpu"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch == epochs-1:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
        #print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    model.eval()

    return model

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
    n_users = max_users#df["user_id"].nunique()
    n_items = max_items#df["item_id"].nunique()
    df["item_id"] = df["item_id"].map(item_map)
    sparse_interaction_matrix = sp.csr_matrix((df["rating"].values, (df["user_id"].values, df["item_id"].values)), (n_users, n_items))

    gender_per_user_df = df[["user_id", "gender"]].drop_duplicates()
    gender_values = (gender_per_user_df["gender"] == "M").astype(int).values

    return sparse_interaction_matrix, gender_values, item_map

def attacker_logreg(dataset, data_df, gender_map, n_items, n_users):
    T = np.array([int(gender_map[uid]) for uid in range(n_users)])

    X = np.zeros((n_users, n_items))
    for _, row in data_df.iterrows():
        uid = row["user_id"]
        iid = row["item_id"]
        r = row["rating"]
        X[int(uid), int(iid)] = r

    cv = StratifiedKFold(n_splits=5)
    random_state = np.random.RandomState(0)

    accuracy_train = []
    accuracy_test = []
    roc_auc = []

    coeffs = []
    for train, test in cv.split(X, T):
        x_train, t_train = X[train], T[train]
        x_test, t_test = X[test], T[test]

        avg = np.mean(x_train)
        x_train -= avg
        x_test -= avg

        if dataset == "ml1m":
            model = LogisticRegression(penalty="l2", random_state=random_state, max_iter=250, solver="liblinear", C=0.01)
        elif dataset == "mc":
            model = LogisticRegression(penalty="l2", random_state=random_state, max_iter=250, solver="liblinear", C=10)
        elif dataset == "el":
            model = LogisticRegression(penalty="l2", random_state=random_state, max_iter=250, solver="liblinear", C=0.1)
        elif dataset == "bx":
            model = LogisticRegression(penalty="l2", random_state=random_state, max_iter=250, solver="liblinear", C=10)
        else:
            print("Error wrong dataset!")
            exit()

        model.fit(x_train, t_train)
        coeffs.append(model.coef_[0])

        preds = model.predict(x_train)
        accuracy_train.append(balanced_accuracy_score(t_train, preds))

        preds = model.predict(x_test)
        accuracy_test.append(balanced_accuracy_score(t_test, preds))

        probs = model.predict_proba(x_test)
        preds = probs[:, 1]
        fpr, tpr, threshold = roc_curve(t_test, preds)
        roc_auc.append(auc(fpr, tpr))

    return np.mean(accuracy_train), np.mean(accuracy_test), np.mean(roc_auc)

def attacker_neuralnet(data_df, gender_map, n_items, n_users):
    # T = np.array([int(gender_map[uid] == "M") for uid in range(n_users)])
    T = np.array([int(gender_map[uid]) for uid in range(n_users)])

    X = np.zeros((n_users, n_items))
    for _, row in data_df.iterrows():
        uid = row["user_id"]
        iid = row["item_id"]
        r = row["rating"]
        X[int(uid), int(iid)] = r

    cv = StratifiedKFold(n_splits=5)
    random_state = np.random.RandomState(0)

    accuracy_train = []
    accuracy_test = []

    coeffs = []
    for train, test in cv.split(X, T):
        x_train, t_train = X[train], T[train]
        x_test, t_test = X[test], T[test]

        avg = np.mean(x_train)
        x_train -= avg
        x_test -= avg

        input_size = x_train.shape[1]
        # Prepare DataLoader
        batch_size = 64
        train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(t_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(t_test))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize and train the model
        attacker_model = AttackerNetwork(input_size=input_size, hidden_size=128).to(device)
        model = _train_attacker(attacker_model, train_loader)
        model.eval()
        predictions, groundtruth = [], []
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            predicted_class = np.argmax(outputs.detach().numpy(), axis=1)
            predictions.extend(predicted_class)
            groundtruth.extend(labels.to(device).numpy())
        bacc = balanced_accuracy_score(groundtruth, predictions)
        accuracy_test.append(bacc)

        predictions, groundtruth = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predicted_class = np.argmax(outputs.detach().numpy(), axis=1)
            predictions.extend(predicted_class)
            groundtruth.extend(labels.numpy())
        bacc = balanced_accuracy_score(groundtruth, predictions)
        accuracy_train.append(bacc)

    return np.mean(accuracy_train), np.mean(accuracy_test), np.nan



dataset = "bx"
epsilon = 2
betas = [0.8, 0.6, 0.4, 0.2, 0.0]
attacker = attacker_logreg
#attacker = attacker_neuralnet

if dataset == "ml1m":
    PATH = "data/ml1m/original/"
    user_gender_map = pd.read_csv(PATH + "/ml1m.usergendermap", header=None, names=["user_id", "gender"], index_col="user_id").squeeze().to_dict()
    user_gender_map = {uid: int(user_gender_map[uid] == "M") for uid in user_gender_map.keys()}
    trainset = pd.read_csv(PATH + "ml1m.train.rating", sep="\t", header=None).to_records(index=False).tolist()
    original_trainset_df = pd.DataFrame.from_records(trainset, columns=["user_id", "item_id", "rating"])
    ister_scores = pd.read_csv(PATH + "/ml1m.ister_scores", header=None, names=["user_id", "score"], index_col="user_id").squeeze().to_dict()
    rmin, rmax = 1, 5
elif dataset == "bx":
    PATH = "data/bx/"
    user_gender_map = pd.read_csv(PATH + "/bx.usergendermap", header=None, names=["user_id", "gender"], index_col="user_id").squeeze().to_dict()
    trainset = pd.read_csv(PATH + "bx.train.rating", sep="\t", header=None).to_records(index=False).tolist()
    original_trainset_df = pd.DataFrame.from_records(trainset, columns=["user_id", "item_id", "rating"])
    ister_scores = pd.read_csv(PATH + "/bx.ister_scores", header=None, names=["user_id", "score"], index_col="user_id").squeeze().to_dict()
    rmin, rmax = 1, 10
elif dataset == "mc":
    PATH = "data/mc/"
    user_gender_map = pd.read_csv(PATH + "/mc.usergendermap", header=None, names=["user_id", "gender"], index_col="user_id").squeeze().to_dict()
    trainset = pd.read_csv(PATH + "mc.train.rating", sep="\t", header=None).to_records(index=False).tolist()
    original_trainset_df = pd.DataFrame.from_records(trainset, columns=["user_id", "item_id", "rating"])
    ister_scores = pd.read_csv(PATH + "/mc.ister_scores", header=None, names=["user_id", "score"], index_col="user_id").squeeze().to_dict()
    rmin, rmax = 1, 5
elif dataset == "el":
    PATH = "data/el/"
    user_gender_map = pd.read_csv(PATH + "/el.usergendermap", header=None, names=["user_id", "gender"], index_col="user_id").squeeze().to_dict()
    trainset = pd.read_csv(PATH + "el.train.rating", sep="\t", header=None).to_records(index=False).tolist()
    original_trainset_df = pd.DataFrame.from_records(trainset, columns=["user_id", "item_id", "rating"])
    ister_scores = pd.read_csv(PATH + "/el.ister_scores", header=None, names=["user_id", "score"], index_col="user_id").squeeze().to_dict()
    rmin, rmax = 1, 5
else:
    print("No Dataset!")
    exit()

max_users = original_trainset_df["user_id"].max()+1
max_items = original_trainset_df["item_id"].max()+1

obfuscated_trainset = sampling_procedure(trainset, beta=1)
obfuscated_trainset_df = pd.DataFrame.from_records(obfuscated_trainset, columns=["user_id", "item_id", "rating"])
acc_train, acc_test, roc_auc = attacker(dataset=dataset, data_df=obfuscated_trainset_df, gender_map=user_gender_map, n_users=max_users, n_items=max_items)
print("[No DP Sampling] Beta %.2f: %.4f (%.4f), %.4f (ROC AUC)" % (1, acc_test, acc_train, roc_auc))
print()
#fname = construct_fname(method="baseline", epsilon=np.inf, beta=1.0)

baseline_train_acc = acc_train
baseline_test_acc = acc_test
deletion_train_acc, ister_train_acc, rand_train_acc = [], [], []
deletion_test_acc, ister_test_acc, rand_test_acc = [], [], []
for beta in betas:
    obfuscated_trainset = sampling_procedure(trainset, beta=beta)
    obfuscated_trainset_df = pd.DataFrame.from_records(obfuscated_trainset, columns=["user_id", "item_id", "rating"])
    acc_train, acc_test, roc_auc = attacker(dataset=dataset, data_df=obfuscated_trainset_df, gender_map=user_gender_map, n_users=max_users, n_items=max_items)
    print("[No DP Sampling] Beta %.2f: %.4f (%.4f), %.4f (ROC AUC)" % (beta, acc_test, acc_train, roc_auc))
    deletion_train_acc.append(acc_train)
    deletion_test_acc.append(acc_test)

    obfuscated_trainset = obfuscation_per_user_random(trainset, beta=beta, epsilon=epsilon, coin_flip=True, rmin=rmin, rmax=rmax)
    obfuscated_trainset_df = pd.DataFrame.from_records(obfuscated_trainset, columns=["user_id", "item_id", "rating"])
    acc_train, acc_test, roc_auc = attacker(dataset=dataset, data_df=obfuscated_trainset_df, gender_map=user_gender_map, n_users=max_users, n_items=max_items)
    print("[Random] Beta %.2f: %.4f (%.4f), %.4f (ROC AUC)" % (beta, acc_test, acc_train, roc_auc))
    rand_train_acc.append(acc_train)
    rand_test_acc.append(acc_test)

    obfuscated_trainset = obfuscation_per_user_ister(trainset, beta=beta, epsilon=epsilon, scores=ister_scores, user_gender_map=user_gender_map, coin_flip=True, rmin=rmin, rmax=rmax)
    obfuscated_trainset_df = pd.DataFrame.from_records(obfuscated_trainset, columns=["user_id", "item_id", "rating"])
    acc_train, acc_test, roc_auc = attacker(dataset=dataset, data_df=obfuscated_trainset_df, gender_map=user_gender_map, n_users=max_users, n_items=max_items)
    print("[I_ster] Beta %.2f: %.4f (%.4f), %.4f (ROC AUC)" % (beta, acc_test, acc_train, roc_auc))
    ister_train_acc.append(acc_train)
    ister_test_acc.append(acc_test)

    print()

model_name = construct_fname(attacker="logreg", method="baseline", beta=1.0, epsilon=np.inf)
os.makedirs("attack_results/" + dataset + "/baseline/" + model_name)
with open("attack_results/" + dataset + "/baseline/" + model_name + "/train_bacc.json", "w") as file:
    json.dump(baseline_train_acc, file)
with open("attack_results/" + dataset + "/baseline/" + model_name + "/test_bacc.json", "w") as file:
    json.dump(baseline_test_acc, file)

model_name = construct_fname(attacker="logreg", method="deletion", epsilon=epsilon, beta=None)
os.makedirs("attack_results/" + dataset + "/deletion/" + model_name)
with open("attack_results/" + dataset + "/deletion/" + model_name + "/train_bacc.json", "w") as file:
    json.dump(deletion_train_acc, file)
with open("attack_results/" + dataset + "/deletion/" + model_name + "/test_bacc.json", "w") as file:
    json.dump(deletion_test_acc, file)
with open("attack_results/" + dataset + "/deletion/" + model_name + "/betas.json", "w") as file:
    json.dump(betas, file)

model_name = construct_fname(attacker="logreg", method="random_dp", epsilon=epsilon, beta=None)
os.makedirs("attack_results/" + dataset + "/random_dp/" + model_name)
with open("attack_results/" + dataset + "/random_dp/" + model_name + "/train_bacc.json", "w") as file:
    json.dump(deletion_train_acc, file)
with open("attack_results/" + dataset + "/random_dp/" + model_name + "/test_bacc.json", "w") as file:
    json.dump(deletion_test_acc, file)
with open("attack_results/" + dataset + "/random_dp/" + model_name + "/betas.json", "w") as file:
    json.dump(betas, file)

model_name = construct_fname(attacker="logreg", method="ister_dp", epsilon=epsilon, beta=None)
os.makedirs("attack_results/" + dataset + "/ister_dp/" + model_name)
with open("attack_results/" + dataset + "/ister_dp/" + model_name + "/train_bacc.json", "w") as file:
    json.dump(deletion_train_acc, file)
with open("attack_results/" + dataset + "/ister_dp/" + model_name + "/test_bacc.json", "w") as file:
    json.dump(deletion_test_acc, file)
with open("attack_results/" + dataset + "/ister_dp/" + model_name + "/betas.json", "w") as file:
    json.dump(betas, file)


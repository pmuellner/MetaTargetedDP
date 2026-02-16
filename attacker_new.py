import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from data.sampling import random_sampling_dp, item_stereotypicality_sampling_dp
from itertools import product as cartesian
import torch.nn as nn
import torch
from sklearn.metrics import balanced_accuracy_score
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import EarlyStopper
import pickle as pkl

class AttackerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttackerNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        return self.model(x)



def train_attacker(parameters, train_data, train_target, val_data, val_target, test_data=None, test_target=None):
    def _evaluate(net, data_loader):
        net.eval()
        predictions, groundtruth = [], []

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            predicted_class = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            predictions.extend(predicted_class)
            groundtruth.extend(labels.cpu().numpy())
        return balanced_accuracy_score(groundtruth, predictions)

    n_hidden = parameters["n_hidden"]
    batch_size = parameters["batch_size"]
    lr = parameters["lr"]
    weight_decay = parameters["weight_decay"]

    if test_data is not None or test_target is not None:
        testing = True
    else:
        testing = False

    # Prepare DataLoader
    train_dataset = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_target))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(torch.from_numpy(val_data).float(), torch.from_numpy(val_target))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    if testing:
        test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_target))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and train the model
    model = AttackerNetwork(input_size=train_data.shape[1], hidden_size=n_hidden).to(device)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)

    model.train()
    early_stopper = EarlyStopper(patience=20, min_delta=0)
    train_bacc_per_epoch, val_bacc_per_epoch, test_bacc_per_epoch = [], [], []
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

        # compute train loss
        train_bacc = _evaluate(net=model, data_loader=train_loader)
        train_bacc_per_epoch.append(train_bacc)

        # compute val loss
        val_bacc = _evaluate(net=model, data_loader=val_loader)
        val_bacc_per_epoch.append(val_bacc)

        if testing:
            # compute test loss
            test_bacc = _evaluate(net=model, data_loader=test_loader)
            test_bacc_per_epoch.append(test_bacc)


        #print("Epoch %d, Train BAcc: %f, Val BAcc: %f" % (epoch, train_bacc, val_bacc))

        if early_stopper.early_stop(1 - val_bacc):
            break


    max_val_idx = np.argmax(val_bacc_per_epoch)
    best_train_bacc = train_bacc_per_epoch[max_val_idx]
    best_val_bacc = val_bacc_per_epoch[max_val_idx]
    if testing:
        best_test_bacc = test_bacc_per_epoch[max_val_idx]
    else:
        best_test_bacc = np.inf

    #print("Best Epoch %d, Train BAcc: %f, Val BAcc: %f, Test BAcc: %f" % (max_val_idx, best_train_bacc, best_val_bacc, best_test_bacc))

    return {"n_hidden": n_hidden, "batch_size": batch_size, "lr": lr, "best_epoch": max_val_idx,
            "train_bacc": best_train_bacc, "val_bacc": best_val_bacc, "test_bacc": best_test_bacc}


def aggregate_results(list_of_results):
    averages, stdev = dict(), dict()
    for key in list_of_results[0].keys():
        if key.endswith("_bacc"):
            averages[key] = np.mean([r[key] for r in list_of_results])
            stdev[key] = np.std([r[key] for r in list_of_results])
    return averages, stdev

def generate_vector(target, n_users):
    return np.array([int(target[uid]) for uid in range(n_users)])

def generate_dense_matrix(data, n_items, n_users):
    X = np.zeros((n_users, n_items))
    for uid, iid, r in data:
        X[int(uid), int(iid)] = r
    return X

def generate_records(X):
    records = []
    for uid, iid in zip(*np.where(X > 0)):
        rating = X[uid, iid]
        records.append((uid, iid, int(rating)))
    return records

def sampling_dp(method, X, t, rmin, rmax, b, e):
    data = generate_records(X)
    X_df = pd.DataFrame.from_records(data, columns=["user_id:token", "item_id:token", "rating:token"])

    if method == "ister_dp":
        t_train_df = pd.DataFrame()
        t_train_df["user_id:token"] = list(range(len(t)))
        t_train_df["attr:token"] = t
        dp_df = item_stereotypicality_sampling_dp(dataset_df=X_df, user_info_df=t_train_df, rmin=rmin, rmax=rmax, beta=b, epsilon=e)
    else:
        dp_df = random_sampling_dp(dataset_df=X_df, rmin=rmin, rmax=rmax, beta=b, epsilon=e)

    return dp_df


if __name__ == "__main__":
    hypertuning = True
    DATASET = "bx"
    METHOD = "random_dp"
    PATH = "custom_datasets_prepared_rp/" + DATASET + "/"

    print("=============================================================================")
    print("Dataset: %s, Sampling Method: %s, Hypertuning: %s" % (DATASET, METHOD, "yes" if hypertuning else "no"))
    print("=============================================================================")

    if DATASET == "bx":
        rmin, rmax = 1, 10
    else:
        rmin, rmax = 1, 5

    best_params = {'lr': 0.0001, 'n_hidden': 32, 'batch_size': 64}

    dataset_df = pd.read_csv(PATH + DATASET + ".dataset.rating", sep="\t", header=None, names=["user_id", "item_id", "rating"])
    users_df = pd.read_csv(PATH + DATASET + ".userlist", sep="\t")

    n_users = dataset_df["user_id"].nunique()
    n_items = dataset_df["item_id"].nunique()

    attr = users_df.set_index("user_id:token")["attr:token"].to_dict()
    t = generate_vector(target=attr, n_users=dataset_df["user_id"].nunique())
    data = dataset_df.to_records(index=False).tolist()
    X = generate_dense_matrix(data=data, n_users=n_users, n_items=n_items)

    epsilons = [0.1]#, 3]
    betas = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    results = []
    for cid, (e, b) in enumerate(list(cartesian(epsilons, betas))):
        print("--------------------------------------")
        print("Epsilon: %f, Beta: %f" % (e, b))

        results_per_config = []
        n_runs = 10

        for rid in range(n_runs):
            print("Run %d/%d: " % (rid+1, n_runs), end='')
            X_trainval, X_test, t_trainval, t_test = train_test_split(X, t, test_size=0.2, stratify=t)
            X_train, X_val, t_train, t_val = train_test_split(X_trainval, t_trainval, test_size=0.25, stratify=t_trainval)

            if hypertuning:
                print("Hypertuning ", end='')
                lr_space = [0.00001, 0.0001]
                n_hidden_space = [16, 32, 64]
                batch_size_space = [64]
                weight_decay_space = [0.01, 0.001, 0.0001]
                best_val_bacc = -np.inf
                for lr, n_hidden, batch_size, weight_decay in list(cartesian(lr_space, n_hidden_space, batch_size_space, weight_decay_space)):
                    print(". ", end='')
                    X_train_dp_df = sampling_dp(method=METHOD, X=X_train, t=t_train, rmin=rmin, rmax=rmax, b=b, e=e)
                    data_dp = X_train_dp_df.to_records(index=False).tolist()
                    X_train_dp = generate_dense_matrix(data=data_dp, n_users=X_train.shape[0], n_items=X_train.shape[1])

                    result = train_attacker(train_data=X_train_dp, train_target=t_train,
                                            val_data=X_val, val_target=t_val,
                                            parameters={"lr": lr, "n_hidden": n_hidden, "batch_size": batch_size, "weight_decay": weight_decay})

                    if best_val_bacc < result["val_bacc"]:
                        best_val_bacc = result["val_bacc"]
                        best_params = {"lr": lr, "n_hidden": n_hidden, "batch_size": batch_size, "weight_decay": weight_decay}

                print(best_params)
                # only do hypertuning on first run
                hypertuning = False

            print("\n")

            X_train_dp_df = sampling_dp(method=METHOD, X=X_train, t=t_train, rmin=rmin, rmax=rmax, b=b, e=e)
            data_dp = X_train_dp_df.to_records(index=False).tolist()
            X_train_dp = generate_dense_matrix(data=data_dp, n_users=X_train.shape[0], n_items=X_train.shape[1])

            result = train_attacker(train_data=X_train_dp, train_target=t_train,
                                    val_data=X_val, val_target=t_val,
                                    test_data=X_test, test_target=t_test,
                                    parameters=best_params)
            results_per_config.append(result)

        averages, stdevs = aggregate_results(results_per_config)
        averages.update({"e": e, "b": b})
        averages.update({"per_run": results_per_config})
        print("\n")
        print(averages)
        print()
        results.append(averages)

        # activate hypertuning for next e and b
        hypertuning = True

    with open("results/" + DATASET + "/attacker." + METHOD, "wb") as f:
        pkl.dump(results, f)

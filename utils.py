import pandas as pd
from MetaMF import *
from torch.utils.data import Dataset
from copy import deepcopy
from collections import defaultdict



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.bigger_better = False

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def read_raw_dataset(dataset_name, path):
    trainset = pd.read_csv(path + "/" + dataset_name + ".train.rating", sep="\t", header=None).to_records(index=False).tolist()
    valset = pd.read_csv(path + "/" + dataset_name + ".val.rating", sep="\t", header=None).to_records(index=False).tolist()
    testset = pd.read_csv(path + "/" + dataset_name + ".test.rating", sep="\t", header=None).to_records(index=False).tolist()

    return trainset, valset, testset

def read_dp_trainset(dataset_name, epsilon, beta, method, path):
    if method == "baseline":
        trainset_dp = pd.read_csv(path + "/" + dataset_name + ".train_e" + str(epsilon) + "_b" + str(beta) + "_random_dp.rating",sep="\t", header=None).to_records(index=False).tolist()
    else:
        trainset_dp = pd.read_csv(path + "/" + dataset_name + ".train_e" + str(epsilon) + "_b" + str(beta) + "_" + method + ".rating", sep="\t", header=None).to_records(index=False).tolist()
    return trainset_dp

def read_useranditemlist(dataset_name, path):
    users_df = pd.read_csv(path + "/" + dataset_name + ".userlist", sep="\t")
    userlist = users_df["user_id:token"].squeeze().values.tolist()
    gendermap = dict(users_df.to_records(index=False))
    itemlist = pd.read_csv(path + "/" + dataset_name + ".itemlist", header=None).squeeze().values.tolist()

    return userlist, gendermap, itemlist


def batchtoinput(batch, use_cuda):
    users = []
    items = []
    ratings = []
    for example in batch:
        users.append(example[0])
        items.append(example[1])
        ratings.append(example[2])
    users = torch.tensor(users, dtype=torch.int64)
    items = torch.tensor(items, dtype=torch.int64)
    ratings = torch.tensor(ratings, dtype=torch.float32)
    if use_cuda:
        users = users.cuda()
        items = items.cuda()
        ratings = ratings.cuda()
    return users, items, ratings


def getbatches(traindata, batch_size, use_cuda, shuffle):
    dataset = traindata.copy()
    if shuffle:
        random.shuffle(dataset)
    for batch_i in range(0, int(np.ceil(len(dataset) / batch_size))):
        start_i = batch_i * batch_size
        batch = dataset[start_i:start_i + batch_size]
        yield batchtoinput(batch, use_cuda)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def get_eval(ratlist, predlist):
    mae = np.mean(np.abs(ratlist - predlist))
    mse = np.mean(np.square(ratlist - predlist))
    return mae, mse


def evaluate(model, data, batch_size, use_cuda, save_predictions=False):
    results = []
    model.eval()
    groundtruth, estimation = [], []
    for users, items, ratings in getbatches(data, batch_size, use_cuda, False):
        predictions = model(users, items)
        estimation.extend(predictions.tolist())
        groundtruth.extend(ratings.tolist())

        if save_predictions:
            results_batch = list(zip(users.tolist(), items.tolist(), ratings.tolist(), predictions.tolist()))
            results.extend(results_batch)

    mae, mse = get_eval(np.array(groundtruth), np.array(estimation))

    if save_predictions:
        return mae, mse, results
    else:
        return mae, mse


def topk_items_per_user(predictions, k=10):
    predictions_per_user = defaultdict(list)
    topk_predictions = []
    for uid, iid, r_real, r_est in predictions:
        predictions_per_user[uid].append((r_real, r_est))

    for uid, preds_uid in predictions_per_user.items():
        topk_preds = sorted(preds_uid, key=lambda t: t[1], reverse=True)[:k]
        topk_predictions.append(topk_preds)

    return topk_predictions

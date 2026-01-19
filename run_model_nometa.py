import random
import sys

import numpy as np
import torch
import pandas as pd
from MetaMF import *
from datetime import datetime as dt
#import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime as dt
import os.path
from utils import *
import json
import os
import pickle
#from opacus import PrivacyEngine
from torch.utils.data import DataLoader
from itertools import product as cartesian_product

if __name__ == "__main__":
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False

    print("CUDA available? " + str(use_cuda))
    if use_cuda:
        print("Current device: %d" % torch.cuda.current_device())

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)  # set random seed for cpu
    torch.cuda.manual_seed(1)  # set random seed for current gpu
    torch.cuda.manual_seed_all(1)  # set random seed for all gpus



    # run experiment(s)
    dataset = "ml1m"
    PATH = "custom_datasets_prepared_rp/" + dataset
    traindata, valdata, testdata = read_raw_dataset(dataset, path=PATH)
    userlist, gendermap, itemlist = read_useranditemlist(dataset, path=PATH)


    # todo
    #traindata = traindata[:1000]
    #valdata = valdata[:1000]
    #testdata = testdata[:1000]

    learning_rate = 0.001
    regularization_factor = 0.0001
    batch_size = 64
    model_name = "NoMetaMF"
    method = "random_del"
    if method.endswith("_dp"):
        DP = True
    else:
        DP = False

    results = []
    if DP:
        betas = [0, 0.2, 0.4, 0.6, 0.8]
        epsilons = [3, 2, 1, 0.1]
        configs = cartesian_product(epsilons, betas)
    elif method.endswith("_del"):
        betas = [0, 0.2, 0.4, 0.6, 0.8]
        configs = betas
    else:
        # baseline
        betas = [1.0]
        configs = betas
    for c in configs:
        if DP:
            e, b = c
            print("==================================")
            print("MetaMF")
            print("Started %s with b: %f, e: %f" % (method, b, e))

            traindata_new = read_dp_trainset(dataset_name=dataset, path=PATH, epsilon=e, beta=b, method=method)
        else:
            b = c
            print("==================================")
            print("MetaMF")
            print("Started %s with b: %f" % (method, b))

            traindata_new = read_del_trainset(dataset_name=dataset, path=PATH, beta=b, method=method)

        train_dataloader = DataLoader(traindata_new, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(valdata, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(testdata, batch_size=batch_size, shuffle=True)

        train_loss, validation_loss = [], []
        net = MetaMF(len(userlist), len(itemlist))

        # disable meta learning
        if model_name == "NoMetaMF":
            net.disable_meta_learning()
            print("Meta Learning disabled!")

        # initialize parameters of neural network
        net.apply(weights_init)
        if use_cuda:
            net.cuda()

        print("==================================")
        starttime = dt.now()

        # model training
        hyperparameters = {"lr": learning_rate, "lambda": regularization_factor, "batch_size": batch_size,
                           "n_epochs": 1000}
        optimizer = optim.Adam(net.parameters(), lr=hyperparameters["lr"], weight_decay=hyperparameters["lambda"])
        batch_size = hyperparameters["batch_size"]
        n_epochs = hyperparameters["n_epochs"]

        train_maes, val_maes, test_maes, all_predictions = [], [], [], []
        early_stopper = EarlyStopper(patience=50, min_delta=0)
        for epoch in range(n_epochs):
            net.train()
            error = 0
            num = 0
            for users, items, ratings in train_dataloader:
                optimizer.zero_grad()
                if use_cuda:
                    pred = net(users.cuda(), items.cuda())
                    loss = net.loss(pred, ratings.cuda())
                else:
                    pred = net(users, items)
                    loss = net.loss(pred, ratings)

                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5)
                optimizer.step()
                error += loss.detach().cpu().numpy() * len(users)
                num += len(users)
            train_loss.append(error / num)

            # evaluate train error
            train_mae, train_mse = evaluate(net, traindata, batch_size, use_cuda)
            train_maes.append(train_mae)

            # evaluate val error
            val_mae, val_mse = evaluate(net, valdata, batch_size, use_cuda)
            val_maes.append(val_mae)

            # evaluate test error
            test_mae, test_mse, predictions = evaluate(net, testdata, batch_size, use_cuda, save_predictions=True)
            test_maes.append(test_mae)
            all_predictions.append(predictions)

            print(
                "Epoch %d/%d (MAE/MSE) - Training Loss: %.4f/%.4f, Validation Loss %.4f/%.4f, Test Loss %.4f/%.4f, Time Elapsed %s" %
                (epoch + 1, n_epochs, train_mae, train_mse, val_mae, val_mse, test_mae, test_mse, dt.now() - starttime))
            print()

            if early_stopper.early_stop(val_mae):
                break

        min_val_idx = np.argmin(val_maes)
        all_predictions = all_predictions[min_val_idx]

        recommendations = topk_items_per_user(all_predictions, k=10)
        if DP:
            res = {"e": e, "b": b, "best_valid_score": val_maes[min_val_idx],
                   "train_scores_per_epoch": train_maes, "test_mae": test_maes[min_val_idx],
                   "predictions": recommendations}
        else:
            res = {"b": b, "best_valid_score": val_maes[min_val_idx],
                   "train_scores_per_epoch": train_maes, "test_mae": test_maes[min_val_idx],
                   "predictions": recommendations}
        results.append(res)

    os.makedirs("results/" + dataset, exist_ok=True)
    file = open("results/" + dataset + "/" + model_name + "." + method, "wb")
    pickle.dump(results, file)
    file.close()
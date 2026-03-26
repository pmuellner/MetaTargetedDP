import pandas as pd
import numpy as np
import os

from sampling import random_sampling_dp, random_sampling_del
from sampling import item_stereotypicality_sampling_dp, item_stereotypicality_sampling_del, hybrid_dp, hybrid_del

def str_to_float(s):
    if str.isnumeric(s):
        return float(s)
    else:
        return np.nan

def k_core_pruning(df, k=5):
    pruned_df = df.copy()
    while True:
        # Count interactions
        ratings_per_user = pruned_df.groupby("user_id").size()
        ratings_per_item = pruned_df.groupby("item_id").size()

        # Filter valid users and items
        valid_users = ratings_per_user[ratings_per_user >= k].index
        valid_items = ratings_per_item[ratings_per_item >= k].index

        new_pruned_df = pruned_df[pruned_df["user_id"].isin(valid_users) & pruned_df["item_id"].isin(valid_items)]

        # Stop when no more rows are removed
        if len(pruned_df) == len(new_pruned_df):
            break

        pruned_df = new_pruned_df

    return pruned_df.reset_index(drop=True)

def preprocess_data(dataset_name):
    if dataset_name == "ml1m":
        df = pd.read_csv("ml1m/ratings.dat", sep="::", header=None)
        df.columns = ["user_id", "item_id", "rating", "timestamp"]
        df.drop(columns=["timestamp"], inplace=True)

        users_df = pd.read_csv("ml1m/users.dat", sep="::", header=None)
        users_df.columns = ["user_id:token", "gender:token", "age:token", "occupation:token", "zip_code:token"]

        users_df["gender:token"] = (users_df["gender:token"] == "M").astype(int)
        users_df.rename(columns={"gender:token": "attr:token"}, inplace=True)

        items_df = pd.read_csv("ml1m/movies.dat", sep="::", header=None, encoding='latin-1')
        items_df.columns = ["item_id", "title_year", "genres"]

        items_df = items_df[items_df["item_id"].isin(df["item_id"].unique())]
        items_df["movie_title"] = items_df["title_year"].apply(lambda s: s[:-7])
        items_df["release_year"] = items_df["title_year"].apply(lambda s: s[-6:])
        items_df["release_year"] = items_df["release_year"].str.replace("(", "")
        items_df["release_year"] = items_df["release_year"].str.replace(")", "")
        items_df.drop(columns=["title_year"], inplace=True)
        items_df.rename(columns={"item_id": "item_id:token", "movie_title": "movie_title:token_seq",
                                 "release_year": "release_year:token", "genres": "genre:token_seq"}, inplace=True)

    elif dataset_name == "bx":
        users_df = pd.read_csv("bx/Users.csv", sep=";").dropna()
        users_df.columns = ["user_id", "age"]
        users_df["age"] = users_df["age"].apply(str_to_float)
        df = pd.read_csv("bx/Ratings.csv", sep=";")
        df.columns = ["user_id", "item_id", "rating"]
        items_df = pd.read_csv("bx/Books.csv", sep=";")
        items_df.columns = ["item_id", "title", "author", "year", "publisher"]
        items_df.drop(columns=["title", "author", "year", "publisher"], inplace=True)

        # remove implicit ratings
        df = df[df["rating"] > 0]

        # merging
        merged_df = pd.merge(left=df, right=users_df, left_on="user_id", right_on="user_id")
        merged_df = pd.merge(left=merged_df, right=items_df, left_on="item_id", right_on="item_id")
        merged_df.dropna(inplace=True)

        # pruning
        merged_df = k_core_pruning(merged_df, k=5)

        threshold = users_df["age"].median()
        print("BX Age Threshold: %f" % threshold)
        merged_df["age"] = (merged_df["age"] >= threshold).astype(int)

        users_df = merged_df[["user_id", "age"]].copy()
        users_df.rename(columns={"user_id": "user_id:token", "age": "age:token"}, inplace=True)
        items_df = merged_df[["item_id"]].copy()
        items_df.rename(columns={"item_id": "item_id:token"}, inplace=True)
        df = merged_df[["user_id", "item_id", "rating"]].copy()

        users_df.rename(columns={"age:token": "attr:token"}, inplace=True)
        users_df.drop_duplicates(inplace=True)
        items_df.drop_duplicates(inplace=True)
    else:
        print("Wrong dataset!")
        return -1

    user_map = {old: new for new, old in enumerate(df["user_id"].unique())}
    item_map = {old: new for new, old in enumerate(df["item_id"].unique())}

    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)
    users_df["user_id:token"] = users_df["user_id:token"].map(user_map)
    items_df["item_id:token"] = items_df["item_id:token"].map(item_map)

    return df, users_df, items_df

def print_stats(dataset_df):
    if "user_id:token" in dataset_df.columns and "item_id:token":
        n_users = dataset_df["user_id:token"].nunique()
        n_items = dataset_df["item_id:token"].nunique()
        n_ratings_per_user = dataset_df.groupby("user_id:token").size().mean()
        n_ratings_per_item = dataset_df.groupby("item_id:token").size().mean()
    else:
        n_users = dataset_df["user_id"].nunique()
        n_items = dataset_df["item_id"].nunique()
        n_ratings_per_user = dataset_df.groupby("user_id").size().mean()
        n_ratings_per_item = dataset_df.groupby("item_id").size().mean()

    n_ratings= len(dataset_df)
    sparsity = 1 - n_ratings / (n_users*n_items)

    print("No. Users: %d, No. Items: %d, No. Ratings: %d, Sparsity: %.2f%%" % (n_users, n_items,n_ratings, 100*sparsity))
    print("No. Ratings per User: %.2f, No. Ratings per Item: %.2f" % (n_ratings_per_user, n_ratings_per_item))
    print()

if __name__ == '__main__':
    # read raw data
    DATASET = "ml1m"
    if DATASET == "ml1m":
        print("=== ML-1M ===")
        df, users_df, items_df = preprocess_data("ml1m")
        print_stats(df)
        attr = "gender:token"
        rmin, rmax = 1, 5
    elif DATASET == "bx":
        print("=== BX ===")
        df, users_df, items_df = preprocess_data("bx")
        print_stats(df)
        attr = "age:token"
        rmin, rmax = 1, 10
    else:
        df = pd.DataFrame()
        users_df = pd.DataFrame()
        items_df = pd.DataFrame()
        attr = ""


    df = df.sample(frac=1)
    profile_size = df.groupby("user_id").size()

    print("data splits ...")
    # at least 11 ratings per set (1 support, 10 query)
    MIN_RATINGS = 1
    # split the data into training, validation and test set
    trainset, valset, testset = [], [], []
    for user_id, n in profile_size.items():
        # splits: 60% trainset, 20% valset, 20% testset
        n_val = int(np.ceil(n * 0.2))
        n_test = int(np.ceil(n * 0.2))
        n_train = n - n_val - n_test

        if n_train < MIN_RATINGS or n_val < MIN_RATINGS or n_test < MIN_RATINGS:
            continue

        all_ratings_uid = df[df["user_id"] == user_id].to_records(index=False)
        valset.extend(all_ratings_uid[:n_val])
        testset.extend(all_ratings_uid[n_val:n_val + n_test])
        trainset.extend(all_ratings_uid[n_val + n_test:])

    train_df = pd.DataFrame.from_records(trainset, columns=["user_id:token", "item_id:token", "rating:float"])
    val_df = pd.DataFrame.from_records(valset, columns=["user_id:token", "item_id:token", "rating:float"])
    test_df = pd.DataFrame.from_records(testset, columns=["user_id:token", "item_id:token", "rating:float"])
    all_items = set(train_df["item_id:token"].unique())

    print_stats(pd.concat([train_df, val_df, test_df]))

    print(users_df.groupby("attr:token").size() / len(users_df))

    # save splits without DP (beta=1)
    PATH = "../custom_datasets_prepared_rp/" + DATASET
    os.makedirs(PATH, exist_ok=True)
    train_df.to_csv("../custom_datasets_prepared_rp/" + DATASET + "/" + DATASET + ".train.rating", sep="\t", index=False, header=False)
    val_df.to_csv("../custom_datasets_prepared_rp/" + DATASET + "/" + DATASET + ".val.rating", sep="\t", index=False, header=False)
    test_df.to_csv("../custom_datasets_prepared_rp/" + DATASET + "/" + DATASET + ".test.rating", sep="\t", index=False, header=False)
    pd.concat([train_df, val_df, test_df]).to_csv("../custom_datasets_prepared_rp/" + DATASET + "/" + DATASET + ".dataset.rating", sep="\t", index=False, header=False)
    users_df.to_csv("../custom_datasets_prepared_rp/" + DATASET + "/" + DATASET + ".userlist", sep="\t", index=False, columns=["user_id:token", "attr:token"])

    items_df.to_csv("../custom_datasets_prepared_rp/" + DATASET + "/" + DATASET + ".itemlist", sep="\t", index=False, header=False, columns=["item_id:token"])

    betas = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    epsilons = [0.1, 1, 2, 3]
    print("dp data ...")
    for b in betas:
        for e in epsilons:
            print("random_sampling_dp (e=%f, b=%f)" % (e, b))
            train_dp_df = random_sampling_dp(dataset_df=train_df, beta=b, epsilon=e, rmin=rmin, rmax=rmax)
            train_dp_df.to_csv(PATH + "/" + DATASET + ".train_e" + str(e) + "_b" + str(b) + "_random_dp" + ".rating", sep="\t", index=False, header=False)

            print("ister_sampling_dp (e=%f, b=%f)" % (e, b))
            train_dp_df = item_stereotypicality_sampling_dp(dataset_df=train_df, user_info_df=users_df, beta=b, epsilon=e, rmin=rmin, rmax=rmax)
            train_dp_df.to_csv(PATH + "/" + DATASET + ".train_e" + str(e) + "_b" + str(b) + "_ister_dp" + ".rating", sep="\t", index=False, header=False)
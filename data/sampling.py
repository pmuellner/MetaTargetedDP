import pandas as pd
import numpy as np

def _apply_dp(dataset_df, item_pool, privacy_budget, rmin=1, rmax=5):
    prob_keep = 1 / (np.exp(privacy_budget) + 1) + (np.exp(privacy_budget) - 1) / (np.exp(privacy_budget) + 1)

    new_dataset = []
    for user_id, dataset_user_df in dataset_df.groupby("user_id:token"):
        n_keep = np.ceil(len(dataset_user_df) * prob_keep).astype(int)
        keep_data = dataset_user_df.loc[np.random.choice(dataset_user_df.index, replace=False, size=n_keep)]
        new_dataset.extend(keep_data.to_records(index=False).tolist())

        neg_items = set(item_pool).difference(dataset_user_df["item_id:token"].unique())
        added_iids = np.random.choice(list(neg_items), replace=False, size=min(len(dataset_user_df) - n_keep, len(neg_items)))
        rating_range = list(range(rmin, rmax + 1))
        add_data = [(user_id, item_id, np.random.choice(rating_range, replace=True)) for item_id in added_iids]
        new_dataset.extend(add_data)

    return new_dataset


def random_sampling_dp(dataset_df, beta, epsilon, rmin, rmax):
    item_pool = dataset_df["item_id:token"].unique()

    new_dataset = []
    for user_id, user_data_df in dataset_df.groupby("user_id:token"):
        n_public = np.ceil(len(user_data_df) * beta).astype(int)
        public_idxs = np.random.choice(user_data_df.index, size=n_public, replace=False)
        private_idxs = list(set(user_data_df.index).difference(public_idxs))

        public_data = user_data_df.loc[public_idxs].to_records(index=False)
        private_data = _apply_dp(dataset_df=user_data_df.loc[private_idxs], item_pool=item_pool, privacy_budget=epsilon,
                                    rmin=rmin, rmax=rmax)

        new_dataset.extend(public_data)
        new_dataset.extend(private_data)
    return pd.DataFrame.from_records(new_dataset, columns=["user_id:token", "item_id:token", "rating:token"])

def item_stereotypicality_sampling_dp(dataset_df, user_info_df, epsilon, rmin, rmax, beta=None, threshold=None):
    if beta is not None and threshold is not None:
        print("Please select either beta or threshold!")
        return pd.DataFrame()

    item_pool = dataset_df["item_id:token"].unique()

    # compute igi score
    merged_df = pd.merge(dataset_df, user_info_df, left_on="user_id:token", right_on="user_id:token")
    item_attr_interactions_df = merged_df[["item_id:token", "attr:token"]]
    item_attr_dist = item_attr_interactions_df.groupby(["item_id:token", "attr:token"]).size()
    attr_dist = item_attr_interactions_df.groupby("attr:token").size()
    igi_score = item_attr_dist / attr_dist

    # compute i_ster score
    ister_scores = dict()
    for iid in dataset_df["item_id:token"].unique():
        if 0 not in igi_score.loc[iid] or 1 not in igi_score.loc[iid]:
            ister_scores[iid] = 0
        else:
            diff = igi_score.loc[iid][0] - igi_score.loc[iid][1]
            ister_scores[iid] = diff / max(igi_score.loc[iid][0], igi_score.loc[iid][1])

    new_dataset = []
    for user_id, user_data_df in dataset_df.groupby("user_id:token"):
        if beta is not None:
            n_public = np.ceil(len(user_data_df) * beta).astype(int)
        else:
            n_public = np.inf

        user_item_scores = [(iid, ister_scores[iid]) for iid in user_data_df["item_id:token"].unique()]
        if user_info_df[user_info_df["user_id:token"] == user_id]["attr:token"].values[0] == 1:
            if threshold:
                public_iids = [iid for iid, score in user_item_scores if score >= threshold]
            else:
                public_iids = [iid for iid, _ in sorted(user_item_scores, key=lambda t: t[1], reverse=True)[:n_public]]
        else:
            if threshold:
                public_iids = [iid for iid, score in user_item_scores if score <= threshold]
            else:
                public_iids = [iid for iid, _ in sorted(user_item_scores, key=lambda t: t[1], reverse=False)[:n_public]]

        public_data_df = user_data_df[user_data_df["item_id:token"].isin(public_iids)]
        public_data = public_data_df.to_records(index=False)

        private_idxs = list(set(user_data_df.index).difference(public_data_df.index))
        private_data = _apply_dp(dataset_df=user_data_df.loc[private_idxs], item_pool=item_pool, privacy_budget=epsilon,
                                 rmin=rmin, rmax=rmax)

        new_dataset.extend(public_data)
        new_dataset.extend(private_data)

    return pd.DataFrame.from_records(new_dataset, columns=["user_id:token", "item_id:token", "rating:float"])
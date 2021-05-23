import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn import metrics

# from scipy.sparse import data

from utils import load_config
from als_model import implicit_als_cg
from data_processor import DataProcessor
import pdb


def create_interaction_matrix(df, pur_col="pur_count"):
    """
    create user-merchant interaction matrix

    :param df: input dataframe with aggregated clicks info
    :type df: pd.DataFrame
    :param pur_col: column name with purchase metric, defaults to 'pur_count'
    :type pur_col: str, optional
    """
    users = sorted(df["user_id"].unique().tolist())  # get all users in the dataset
    merchants = sorted(
        df["merchant_id"].unique().tolist()
    )  # get all merchants in the dataset
    pur_counts = list(df[pur_col])  # get all purchase counts

    user2row = dict(zip(users, [i for i in range(len(users))]))
    merchant2col = dict(zip(merchants, [i for i in range(len(merchants))]))

    rows = df["user_id"].map(user2row)
    cols = df["merchant_id"].map(merchant2col)
    df_mat = sparse.csr_matrix(
        (pur_counts, (rows, cols)), shape=(len(users), len(merchants))
    )

    # compute sparsity
    mat_size = len(users) * len(merchants)
    num_interactions = len(df_mat.nonzero()[0])
    sparsity = (1 - num_interactions / mat_size) * 100
    print("sparsity in the interaction matrix = {:.2f}".format(sparsity))
    return df_mat, user2row, merchant2col


def prepare_train_test_data():
    """
    time based split for training and test data
    """
    config = load_config()
    data_processor = DataProcessor()
    df_train = data_processor.prepare_clicks_data(end_date=config["test_split_date"])
    df_test = data_processor.prepare_clicks_data(start_date=config["test_split_date"])

    df_mat, user2row, merchant2col = create_interaction_matrix(
        df_train, pur_col="norm_pur_count"
    )
    df_test["user_id"] = df_test["user_id"].map(user2row)
    df_test["merchant_id"] = df_test["merchant_id"].map(merchant2col)
    df_test = df_test[
        ~((df_test["user_id"].isna()) | (df_test["merchant_id"].isna()))
    ].copy()

    df_test["user_id"] = df_test["user_id"].astype(int)
    df_test["merchant_id"] = df_test["merchant_id"].astype(int)

    df_test = df_test.groupby("user_id")["merchant_id"].agg(list).reset_index()
    test_dict = dict(zip(df_test["user_id"], df_test["merchant_id"]))
    return df_mat, test_dict


def compute_auc_metric(truths, preds):
    fpr, tpr, thresholds = metrics.roc_curve(truths, preds)
    this_auc = metrics.auc(fpr, tpr)
    return this_auc


def run_train_validation():
    """
    run training and validation for the model
    compare model performance with baseline popularity model
    """
    train_data, test_data = prepare_train_test_data()
    alpha_val = 25
    train_data = (train_data * alpha_val).astype("double")
    user_embeds, mer_embeds = implicit_als_cg(train_data, iterations=20, features=16)

    # baseline popularity preds
    pop_preds = np.array(train_data.sum(axis=0)).squeeze()
    model_aucs, pop_aucs = [], []

    for k, v in test_data.items():
        user_embed = user_embeds[k, :].toarray()
        preds = user_embed.dot(mer_embeds.toarray().T).squeeze()
        truths = [1 if i in v else 0 for i in range(mer_embeds.shape[0])]
        model_auc, pop_auc = (
            compute_auc_metric(truths, preds),
            compute_auc_metric(truths, pop_preds),
        )
        # print("model={}, pop={}".format(model_auc, pop_auc))
        model_aucs.append(model_auc)
        pop_aucs.append(pop_auc)
    # compute mean auc
    model_mean_auc, pop_mean_auc = np.mean(model_aucs), np.mean(pop_aucs)
    print(model_mean_auc)
    print(pop_mean_auc)


if __name__ == "__main__":
    run_train_validation()

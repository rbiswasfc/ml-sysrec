from json import load
from gensim.models.word2vec import LineSentence
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn import metrics

from utils import load_config
from als_model import implicit_als_cg
from data_processor import DataProcessor
from gensim.models import Word2Vec

import pdb


class Merchant2VecModel:
    """
    Model to create merchant embeddings
    """

    def __init__(self, embed_size=64, window=5, num_rec=5):
        """
        initialize embedding model

        :param embed_size: embedding dimension, defaults to 64
        :type embed_size: int, optional
        :param window: max window size to define merchant co-occurrence, defaults to 5
        :type window: int, optional
        :param num_rec: number of recommendation to produce, defaults to 5
        :type num_rec: int, optional
        """
        self.embed_size = embed_size
        self.window = window
        self.model = None
        self.num_rec = num_rec
        self.config = load_config()
        self.data_processor = DataProcessor()
        self.user_list = self.data_processor.create_user_list()
        self.user_recent_click_dict = None
        self.popular_merchants = None

    def prepare_corpus(self, final_training):
        """
        prepare corpus for training of merchant embeddings

        :return: training corpus
        :rtype: List[List(str)]
        """
        if final_training:
            df_seq = self.data_processor.create_user_click_sequence()
        else:
            df_seq = self.data_processor.create_user_click_sequence(
                end_date=self.config["test_split_date"]
            )
        sentences = df_seq["merchant_seq"].values.tolist()
        sentences = [list(map(str, sent)) for sent in sentences]
        return sentences

    def train(self, final_training=False):
        """
        training of word to vector model

        :param final_training: flag to indicate training with all data, defaults to False
        :type final_training: bool, optional
        """
        # initialize the model
        self.model = Word2Vec(
            min_count=3,  # consider a merchant if merchant is present more than this threshold
            window=self.window,
            vector_size=self.embed_size,
            alpha=0.01,  # learning rate
            min_alpha=0.001,  # minimum learning rate
            negative=20,  # number of random negative sampling
        )
        # build vocab
        corpus = self.prepare_corpus(final_training)
        self.model.build_vocab(corpus)

        # training
        self.model.train(corpus, total_examples=self.model.corpus_count, epochs=50)

        # init sims (Precompute L2-normalized embeddings)
        self.model.init_sims(replace=True)

    def get_most_popular_merchants(self):
        """
        get top most popular merchants. In the case where user does not have sufficient click history,
        most popular items will be recommended to the user.
        """
        if self.model:
            return self.model.wv.index_to_key[: self.num_rec]
        else:
            print("train the model before performing this step")
            return None

    def generate_predictions(self, user_id, eval_date=None):
        """
        generate next mechants for users based on merchant similarity scores
        
        :param user_id: user id
        :type user_id: int
        :param eval_date: lead generation date, defaults to None
        :type eval_date: str ('%Y-%m-%d'), optional
        """
        # get recent purchase of users
        if not self.user_recent_click_dict:
            self.user_recent_click_dict = self.data_processor.get_last_n_clicks(
                self.window, eval_date
            )
        if not self.popular_merchants:
            self.popular_merchants = self.get_most_popular_merchants()

        if user_id not in self.user_list:
            return "error! user not found..."

        this_user_clicks = self.user_recent_click_dict.get(user_id, [])
        # print(this_user_clicks)

        click_threshold = 2
        # not sufficient info to predict from model
        if len(this_user_clicks) <= click_threshold:
            return self.popular_merchants

        this_user_clicks = list(map(str, this_user_clicks))
        pred_merchants = self.model.wv.most_similar(
            positive=this_user_clicks, topn=self.num_rec
        )
        pred_merchants = [int(pred[0]) for pred in pred_merchants]
        return pred_merchants

    def generate_batch_predictions(self):
        pass

    def save_model(self):
        """
        save a trained model
        """
        if self.model:
            self.model.save(self.config["model_path"])

    def load_model(self):
        """
        load a pre-trained model
        """
        try:
            self.model = Word2Vec.load(self.config["model_path"])
        except Exception as e:
            print(e)
            print("error in model loading!")


# def create_interaction_matrix(df, pur_col="pur_count"):
#     """
#     create user-merchant interaction matrix

#     :param df: input dataframe with aggregated clicks info
#     :type df: pd.DataFrame
#     :param pur_col: column name with purchase metric, defaults to 'pur_count'
#     :type pur_col: str, optional
#     """
#     users = sorted(df["user_id"].unique().tolist())  # get all users in the dataset
#     merchants = sorted(
#         df["merchant_id"].unique().tolist()
#     )  # get all merchants in the dataset
#     pur_counts = list(df[pur_col])  # get all purchase counts

#     user2row = dict(zip(users, [i for i in range(len(users))]))
#     merchant2col = dict(zip(merchants, [i for i in range(len(merchants))]))

#     rows = df["user_id"].map(user2row)
#     cols = df["merchant_id"].map(merchant2col)
#     df_mat = sparse.csr_matrix(
#         (pur_counts, (rows, cols)), shape=(len(users), len(merchants))
#     )

#     # compute sparsity
#     mat_size = len(users) * len(merchants)
#     num_interactions = len(df_mat.nonzero()[0])
#     sparsity = (1 - num_interactions / mat_size) * 100
#     print("sparsity in the interaction matrix = {:.2f}".format(sparsity))
#     return df_mat, user2row, merchant2col


# def prepare_train_test_data():
#     """
#     time based split for training and test data
#     """
#     config = load_config()
#     data_processor = DataProcessor()
#     df_train = data_processor.prepare_clicks_data(end_date=config["test_split_date"])
#     df_test = data_processor.prepare_clicks_data(start_date=config["test_split_date"])

#     df_mat, user2row, merchant2col = create_interaction_matrix(df_train)
#     df_test["user_id"] = df_test["user_id"].map(user2row)
#     df_test["merchant_id"] = df_test["merchant_id"].map(merchant2col)
#     df_test = df_test[
#         ~((df_test["user_id"].isna()) | (df_test["merchant_id"].isna()))
#     ].copy()

#     df_test["user_id"] = df_test["user_id"].astype(int)
#     df_test["merchant_id"] = df_test["merchant_id"].astype(int)

#     df_test = df_test.groupby("user_id")["merchant_id"].agg(list).reset_index()
#     test_dict = dict(zip(df_test["user_id"], df_test["merchant_id"]))
#     return df_mat, test_dict


# def compute_auc_metric(truths, preds):
#     fpr, tpr, thresholds = metrics.roc_curve(truths, preds)
#     this_auc = metrics.auc(fpr, tpr)
#     return this_auc


# def run_train_validation():
#     """
#     run training and validation for the model
#     compare model performance with baseline popularity model
#     """
#     train_data, test_data = prepare_train_test_data()
#     alpha_val = 5
#     train_data = (train_data * alpha_val).astype("double")
#     user_embeds, mer_embeds = implicit_als_cg(train_data, iterations=10, features=32)

#     # baseline popularity preds
#     pop_preds = np.array(train_data.sum(axis=0)).squeeze()
#     model_aucs, pop_aucs = [], []

#     for k, v in test_data.items():
#         user_embed = user_embeds[k, :].toarray()
#         preds = user_embed.dot(mer_embeds.toarray().T).squeeze()
#         truths = [1 if i in v else 0 for i in range(mer_embeds.shape[0])]
#         model_auc, pop_auc = (
#             compute_auc_metric(truths, preds),
#             compute_auc_metric(truths, pop_preds),
#         )
#         # print("model={}, pop={}".format(model_auc, pop_auc))
#         model_aucs.append(model_auc)
#         pop_aucs.append(pop_auc)
#     # compute mean auc
#     model_mean_auc, pop_mean_auc = np.mean(model_aucs), np.mean(pop_aucs)
#     print(model_mean_auc)
#     print(pop_mean_auc)

if __name__ == "__main__":
    config = load_config()
    model = Merchant2VecModel()
    model.train()
    preds = model.generate_predictions(
        user_id=33576, eval_date=config["test_split_date"]
    )
    print(preds)


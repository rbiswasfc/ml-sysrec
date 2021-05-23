import itertools
import numpy as np
import pandas as pd
from data_loader import DataLoader


class DataProcessor:
    """
    class to process and prepare data for sys-rec model 
    """

    def __init__(self, data_dir="./data"):
        """
        initialize the data processor

        :param data_dir: path to data directory, defaults to './data'
        :type data_dir: str, optional
        """
        self.data_loader = DataLoader(data_dir)

    def create_user_list(self):
        """
        maintain a list of user base
        """
        df_users = self.data_loader.load_user_data()
        user_list = df_users["user_id"].unique().tolist()
        return user_list

    def filter_clicks_data(self, start_date, end_date):
        """
        filter clicks data with respect to time
        
        :param start_date: click record start date
        :type start_date: str ('%Y-%m-%d)/ None
        :param end_date: click record end date
        :type end_date: str ('%Y-%m-%d)/ None
        :return: filtered clicks data
        :rtype: pd.DataFrame
        """
        df_clicks = self.data_loader.load_clicks_data()
        if start_date:
            df_clicks = df_clicks[df_clicks["created_at"] > start_date].copy()
        if end_date:
            df_clicks = df_clicks[df_clicks["created_at"] <= end_date].copy()
        return df_clicks

    def create_user_click_sequence(self, start_date=None, end_date=None):
        """
        create user click history sequence

        :param start_date: click record start date, defaults to None
        :type start_date: str ('%Y-%m-%d'), optional
        :param end_date: click record end date, defaults to None
        :type end_date: str ('%Y-%m-%d'), optional
        """
        df_clicks = self.filter_clicks_data(start_date, end_date)
        df_stores = self.data_loader.load_store_data()
        df_clicks = df_clicks = pd.merge(
            df_clicks, df_stores[["store_id", "merchant_id"]], on="store_id", how="left"
        )
        df_clicks = df_clicks.sort_values(by=["user_id", "created_at"])

        def _get_sequence(click_history):
            """
            get time ordered sequence of mechant_ids that user clicked

            :param click_history: click data from a user
            :type pur_history: list
            :return: list of non-repeated merchant ids 
            :rtype: list
            """
            seq = [x[0] for x in itertools.groupby(click_history)]
            return seq

        df_seq = (
            df_clicks.groupby(["user_id"])["merchant_id"]
            .agg(_get_sequence)
            .reset_index()
            .rename(columns={"merchant_id": "merchant_seq"})
        )
        return df_seq

    def get_last_n_clicks(self, n=5, eval_date=None):
        """
        get ids of last n merchants visited by the users at a specific evaluation date.
        If evaluation date is not specified, entire clicks data will be considered.
        If user does not have 'n' previous clicks, all previous clicked merchants will be returned.

        :param n: number of mechants to get, defaults to 5
        :type n: int, optional
        :param eval_date: evaluation date, defaults to None
        :type eval_date: str ('%Y-%m-%d'), optional
        """
        df_seq = self.create_user_click_sequence(end_date=eval_date)
        df_seq["merchant_seq"] = df_seq["merchant_seq"].apply(lambda x: x[-n:])
        seq_dict = dict(zip(df_seq["user_id"], df_seq["merchant_seq"]))
        return seq_dict

    # def prepare_clicks_data_for_als(self, start_date=None, end_date=None):
    #     """
    #     prepare aggregated clicks data

    #     :param start_date: record start date, defaults to None
    #     :type start_date: str ('%Y-%m-%d), optional
    #     :param end_date: record end date, defaults to None
    #     :type end_date: str ('%Y-%m-%d), optional
    #     :return: aggregated clicks data
    #     :rtype: pd.DataFrame
    #     """
    #     df_clicks = self.data_loader.load_clicks_data()
    #     if start_date:
    #         df_clicks = df_clicks[df_clicks["created_at"] > start_date].copy()
    #     if end_date:
    #         df_clicks = df_clicks[df_clicks["created_at"] <= end_date].copy()

    #     df_stores = self.data_loader.load_store_data()
    #     df_clicks = df_clicks = pd.merge(
    #         df_clicks, df_stores[["store_id", "merchant_id"]], on="store_id", how="left"
    #     )
    #     df_input = (
    #         df_clicks.groupby(["user_id", "merchant_id"])["id"]
    #         .agg("count")
    #         .reset_index()
    #         .rename(columns={"id": "pur_count"})
    #     )
    #     df_input["max_count"] = df_input.groupby("merchant_id")["pur_count"].transform(
    #         np.max
    #     )
    #     df_input["min_count"] = df_input.groupby("merchant_id")["pur_count"].transform(
    #         np.min
    #     )
    #     df_input["norm_pur_count"] = (df_input["pur_count"] - df_input["min_count"]) / (
    #         df_input["max_count"] - df_input["min_count"] + 1e-6
    #     )
    #     df_input = df_input.drop(columns=["min_count", "max_count"])
    #     df_input["norm_pur_count"] = df_input["norm_pur_count"].clip(0, 1)
    #     return df_input


if __name__ == "__main__":
    data_processor = DataProcessor()
    df_x = data_processor.create_user_click_sequence(end_date="2021-02-28")
    seq_data = data_processor.get_last_n_clicks()
    print(df_x.sample(5))

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

    def prepare_clicks_data(self, start_date=None, end_date=None):
        """
        prepare aggregated clicks data

        :param start_date: record start date, defaults to None
        :type start_date: str ('%Y-%m-%d), optional
        :param end_date: record end date, defaults to None
        :type end_date: str ('%Y-%m-%d), optional
        :return: aggregated clicks data
        :rtype: pd.DataFrame
        """
        df_clicks = self.data_loader.load_clicks_data()
        if start_date:
            df_clicks = df_clicks[df_clicks["created_at"] > start_date].copy()
        if end_date:
            df_clicks = df_clicks[df_clicks["created_at"] <= end_date].copy()

        df_stores = self.data_loader.load_store_data()
        df_clicks = df_clicks = pd.merge(
            df_clicks, df_stores[["store_id", "merchant_id"]], on="store_id", how="left"
        )
        df_input = (
            df_clicks.groupby(["user_id", "merchant_id"])["id"]
            .agg("count")
            .reset_index()
            .rename(columns={"id": "pur_count"})
        )
        df_input["max_count"] = df_input.groupby("merchant_id")["pur_count"].transform(
            np.max
        )
        df_input["min_count"] = df_input.groupby("merchant_id")["pur_count"].transform(
            np.min
        )
        df_input["norm_pur_count"] = (df_input["pur_count"] - df_input["min_count"]) / (
            df_input["max_count"] - df_input["min_count"] + 1e-6
        )
        df_input = df_input.drop(columns=["min_count", "max_count"])
        df_input["norm_pur_count"] = df_input["norm_pur_count"].clip(0, 1)
        return df_input


if __name__ == "__main__":
    data_processor = DataProcessor()
    df_x = data_processor.prepare_clicks_data()
    print(df_x.sample(5))

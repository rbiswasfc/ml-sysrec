import os
import numpy as np
import pandas as pd


class DataLoader:
    """
    class to help loading of required data  
    """

    def __init__(self, data_dir):
        """
        initialize data loader

        :param data_dir: path to data directory
        :type data_dir: str
        """
        self.data_dir = data_dir

    def load_clicks_data(self):
        """
        load click data 
        """
        df = pd.read_parquet(os.path.join(self.data_dir, "clicks.parquet"))
        df["created_at"] = pd.to_datetime(df["created_at"])
        return df

    def load_store_data(self):
        """
        load store meta-data
        """
        df = pd.read_parquet(os.path.join(self.data_dir, "stores.parquet"))
        df = df.rename(columns={"id": "store_id"})
        return df

    def load_user_data(self):
        """
        load user information 
        """
        df = pd.read_parquet(os.path.join(self.data_dir, "users.parquet"))
        df = df.rename(columns={"id": "user_id"})
        return df


if __name__ == "__main__":
    data_loader = DataLoader("./data")
    df_users = data_loader.load_user_data()
    print(df_users.head())


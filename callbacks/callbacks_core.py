import os
import pandas as pd
import pdb
import dash_table
import dash_html_components as html
import plotly.express as px


from dash.dependencies import Input, Output
import dash_core_components as dcc
from app import app, cache
from utils import load_config
from data_loader import DataLoader
from model_training import Merchant2VecModel, build_sys_rec_model

config = load_config()
TIME_OUT = 3600


@cache.memoize(timeout=TIME_OUT)
def restore_model():
    model = Merchant2VecModel()
    # check if pretrained model exists, otherwise build the model
    if not os.path.exists(config["model_path"]):
        build_sys_rec_model()
    # load pre-trained model
    model.load_model()
    _ = model.generate_predictions(0)
    return model


model = restore_model()

# get merchant-store mappings
data_loader = DataLoader(config["data_dir"])
df_stores = data_loader.load_store_data()
df_clicks = data_loader.load_clicks_data()
df_clicks = df_clicks = pd.merge(
    df_clicks, df_stores[["store_id", "merchant_id"]], on="store_id", how="left"
)

df_counts = (
    df_clicks.groupby(["user_id", "merchant_id"])["id"]
    .agg("count")
    .reset_index()
    .rename(columns={"id": "click_count"})
)

df_merchant_store = (
    df_stores.groupby("merchant_id")["store_id"]
    .agg(lambda x: ", ".join(list(map(str, x))))
    .reset_index()
)

df_merchant_text = (
    df_stores.groupby("merchant_id")["display_text"]
    .agg(lambda x: ", ".join(list(filter(None, x))))
    .reset_index()
)


def make_table(df, page_size=10):
    """
    utility function to create dash tables

    :param df: dataframe from which table will be created
    :type df: pd.DataFrame
    :param page_size: number of rows to display in one page, defaults to 10
    :type page_size: int, optional
    :return: dash table created from dataframe
    :rtype: html.Div
    """
    table = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in df.columns],
        data=df.to_dict("records"),
        style_as_list_view=True,
        style_header={"backgroundColor": "rgb(30, 30, 30)"},
        style_cell={
            "backgroundColor": "rgb(50, 50, 50)",
            "color": "white",
            "textAlign": "center",
        },
        sort_action="native",
        page_size=page_size,
    )
    layout = html.Div(table, className="container", style={"margin-top": "2rem"})
    return layout


@app.callback(
    [Output("rec-output-table", "children"), Output("user-click-plot", "children")],
    [Input("user-id-input", "value")],
)
def make_recommendation_table(user_id):
    if not user_id:
        return html.Div(""), html.Div("")

    # get model predictions for this user id
    recs = model.generate_predictions(user_id=user_id)
    # if len(recs) == 0:  # exception from model prediction func
    #    print("not user")
    #    return html.Div("")

    df = pd.DataFrame()
    df["merchant_id"] = recs
    df["merchant_id"] = df["merchant_id"].astype(int)
    df = pd.merge(df, df_merchant_store, on="merchant_id", how="left")
    df = pd.merge(df, df_merchant_text, on="merchant_id", how="left")
    df = df.rename(
        columns={
            "merchant_id": "Merchant",
            "store_id": "Stores",
            "display_text": "Display",
        }
    )
    table = make_table(df)

    # user past behavior
    df_counts_user = df_counts[df_counts["user_id"] == user_id].copy()
    df_counts_user["merchant_id"] = df_counts_user["merchant_id"].apply(
        lambda x: "merchant_{}".format(x)
    )
    df_counts_user = df_counts_user.sort_values(by="click_count", ascending=False)
    if len(df_counts_user) == 0:
        bar_plot = html.Div("NA")
    else:
        top_k = 15
        df_counts_user = df_counts_user.iloc[:top_k].copy()
        fig = px.bar(
            df_counts_user,
            y="click_count",
            x="merchant_id",
            title="User Clicks History",
            labels={"click_count": "# Clicks", "merchant_id": "Merchant"},
            template="seaborn",
        )
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
        fig.update_layout(margin={"l": 5, "b": 75, "t": 25, "r": 0})
        bar_plot = dcc.Graph(
            figure=fig, className="container", style={"maxWidth": "650px"}
        )

    return table, bar_plot

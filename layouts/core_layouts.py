from data_loader import DataLoader
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from pandas.core.frame import DataFrame
from utils import load_config


def make_predict_page():
    layout = html.Div(
        id="predict-page-layout",
        children=[
            html.Div("Next Merchant Recommendations", className="inline-header col-12"),
            dbc.Row(
                dcc.Input(
                    id="user-id-input",
                    type="number",
                    placeholder="Enter User Id...",
                    # value=161890,
                ),
                justify="center",
            ),
            dcc.Loading(html.Div(id="rec-output-table"), color="black"),
            html.Div("User Past Clicks Plot", className="inline-header col-12"),
            dcc.Loading(html.Div(id="user-click-plot"), color="black"),
        ],
        style={"textAlign": "center"},
    )

    return layout


def make_batch_predict_page():
    layout = html.Div(
        id="batch-predict-page-layout",
        children=[
            html.Div("Merchant Leads", className="inline-header col-12"),
            dbc.Row(
                dbc.Button(
                    "Generate",
                    id="btn-lead-generate",
                    color="secondary",
                    block=True,
                    # value=161890,
                    className="col-4",
                ),
                justify="center",
                # style={"maxWidth": "650px", "textAlign": "center"},
            ),
            dcc.Loading(html.Div(id="lead-output-table"), color="black"),
        ],
        style={"textAlign": "center"},
    )

    return layout


def make_visualization_page():
    data_loader = DataLoader(load_config()["data_dir"])
    df_stores = data_loader.load_store_data()
    all_merchants = df_stores["merchant_id"].unique().tolist()
    options = [{"label": idx, "value": idx} for idx in all_merchants]

    layout = html.Div(
        id="visualization-page-layout",
        children=[
            html.Div("Merchant Clicks Time-series", className="inline-header col-12"),
            dbc.Row(
                html.Div(
                    dcc.Dropdown(
                        id="merchant-dropdown",
                        options=options,
                        placeholder="select merchant id ...",
                    ),
                    className="col-6",
                ),
                justify="center",
            ),
            dcc.Loading(html.Div(id="visualization-output"), color="black"),
        ],
        style={"textAlign": "center"},
    )

    return layout

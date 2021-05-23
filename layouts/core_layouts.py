import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from layouts.layout_utils import make_header, make_dropdown


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

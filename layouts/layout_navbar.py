import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


def get_navigation_tabs():
    tabs = dbc.Tabs(
        id="navigation-tab",
        active_tab="Predict",
        children=[
            dbc.Tab(label="Predict", tab_id="Predict"),
            dbc.Tab(label="Batch-Prediction", tab_id="Batch-Prediction"),
            dbc.Tab(label="Visualization", tab_id="Visualization"),
        ],
        className="tabs-modifier",
    )

    return tabs


def get_navigation_bar():
    tabs = get_navigation_tabs()
    nav_bar = dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        html.Img(src="/assets/sys-rec-logo.jpeg", height="60px"),
                        dbc.NavbarBrand(
                            " Recommender System ", className="ml-2 title-modifier"
                        ),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                dbc.Col([tabs], className="ml-auto", width="auto"),
            ]
        ),
        color="dark",
        dark=True,
        className="mb-5 banner-modifier",
    )

    return nav_bar

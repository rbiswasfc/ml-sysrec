import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app
from layouts.layout_navbar import get_navigation_bar


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    section = html.Div(
        children=[get_navigation_bar(), html.Div(id="current-tab-content"),]
    )
    return section

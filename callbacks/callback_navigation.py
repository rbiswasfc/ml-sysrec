import dash
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app

from layouts.core_layouts import make_predict_page

# from layouts.layout_leads import make_leads_page
# from layouts.layout_squad import make_squad_page


@app.callback(
    Output("current-tab-content", "children"), [Input("navigation-tab", "active_tab")]
)
def render_content(tab):
    print("current tab = {}".format(tab))
    if tab == "Predict":
        return make_predict_page()
    elif tab == "Batch-Prediction":
        return "this is batch prediction page"
    elif tab == "Visualization":
        return "this is visualization page"
    else:
        return html.Div("Page not found!")

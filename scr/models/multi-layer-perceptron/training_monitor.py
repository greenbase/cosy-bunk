"""
Implements a monitoring dashboard for hyperparameter tuning.

Dashboard tracks error in approximation of joint positions and accuracy of
postures approximated correctly.
"""
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id="loss"),dcc.Interval(id="interval-component-loss",interval=10*1000,n_intervals=0)],style={"display":"inline-block"}),
    html.Div([
        dcc.Graph(id="accuracy"),
        dcc.Interval(id="interval-component-acc",interval=10*1000,n_intervals=0)
        ],
        style={"display":"inline-block"}),
])


@app.callback(
    Output(component_id='accuracy', component_property='figure'),
    Input("interval-component-acc", "n_intervals")
)
def update_accuracy(n):
    with open("model_metrics.csv","r") as csv_file:
        df=pd.read_csv(csv_file,header=0,usecols=["epoch","accuracy"],index_col="epoch")
    assert df.shape[1]==1 , f"Shape of accuracy df is {df.shape}"
    fig=px.line(df,y="accuracy",)
    # fig.update_layout(margin=dict(pad=10))
    return fig

@app.callback(
    Output(component_id="loss", component_property="figure"),
    Input("interval-component-loss", "n_intervals")
)
def update_loss(n):
    with open("model_metrics.csv","r") as csv_file:
        df=pd.read_csv(csv_file,header=0,usecols=["epoch","distance_avg_mm"],index_col="epoch")
    assert df.shape[1]==1, f"Shape of loss df is {df.shape}"
    fig=px.line(df,y="distance_avg_mm",)
    # fig.update_layout(margin=dict(pad=10))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
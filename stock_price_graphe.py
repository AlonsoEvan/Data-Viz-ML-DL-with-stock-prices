import numpy as np
import pandas as pd
import math
import datetime
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output




app = dash.Dash()

app.layout = html.Div(children=[
    html.Div(children='''
        Symbol to graph:
    '''),
    dcc.Input(id='input', value='', type='text'),
    html.Div(id='output-graph'),
])

@app.callback(
    Output(component_id='output-graph', component_property='children'),
    [Input(component_id='input', component_property='value')]
)

def update_value(input_data):
	df = pd.read_csv("data/prices-split-adjusted.csv")
	df.reset_index(inplace = True)
	df.set_index("date", inplace = True)
	df =df[df['symbol'] == input_data]
	df = df.drop(['symbol', 'index'],  axis=1)


	return dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': df.index, 'y': df.close, 'type': 'line', 'name': input_data},
            ],
            'layout': {
                'title': input_data
            }
        }
    )



if __name__ == '__main__':
    app.run_server(debug=True)  

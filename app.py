import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split
import plotly.express as px
from plotly.tools import mpl_to_plotly
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate

from plotly.subplots import make_subplots

import Tool

np.random.seed(6313)




# load data
url = 'https://raw.githubusercontent.com/sshikha78/Solar-Radiation-Prediction/main/SolarPrediction.csv'
df = pd.read_csv(url)
df['Datetime'] = pd.to_datetime(df['Data'] + ' ' + df['Time'])
df.sort_values(by='Datetime', inplace=True)
df = df.drop(['UNIXTime'], axis=1)
print(df.head().to_string())
print(df.info())
df['TimeSunRise'] = pd.to_datetime(df['Data'] + ' ' + df['TimeSunRise']).astype(np.int64)
df['TimeSunSet'] = pd.to_datetime(df['Data'] + ' ' + df['TimeSunSet']).astype(np.int64)
df = df.resample('H', on='Datetime').mean()
date = pd.date_range(start='9/1/2016',
                     periods=len(df),
                     freq='H')
df.index = date
df['Date'] = date
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)

external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/solar/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-OlHCZiXlLQ2ygA+VLylPNI8WYIZXXtcWTAL/ZoOXYx2j1MYjbT8T6DzU6tn0U6/K',
        'crossorigin': 'anonymous'
    },
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/litera/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-l79IMV7i9dGnfBaxh3q4ZsUbN7DYg+LLd8FGz3q3rksA2fX9Iv6AMlUZpF07ZnRz',
        'crossorigin': 'anonymous'
    }
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Solar Radiation Prediction "

# define the layout for each tab
tab1_layout = html.Div([
    html.H1("Data Distribution"),
    html.Div([
        dcc.Dropdown(
            id='target-selector',
            options=[
                {'label': 'Radiation', 'value': 'Radiation'},
                {'label': 'Seasonal Difference (s=24)', 'value': 's_diff'},
                {'label': 'Non-Seasonal Difference (order=1)', 'value': 'ns_diff'}
            ],
            value='Radiation'
        ),

        html.Button('Submit', id='submit-button', n_clicks=0)
    ]),
    html.Div(
        dcc.Graph(id="Radiation-chart",
                  style={'height': '500px'},
                  config={"displayModeBar": False})
    ),
], style={'padding': '50px'})



import dash_table


tab3_layout = html.Div([
    html.H1("Correlation Matrix"),
    html.Div([
        dcc.Dropdown(
            id='corr-target-selector',
            options=[
                {'label': 'Temperature', 'value': 'Temperature'},
                {'label': 'Pressure', 'value': 'Pressure'},
                {'label': 'Humidity', 'value': 'Humidity'},
                {'label': 'Wind Direction(Degrees)', 'value': 'WindDirection(Degrees)'},
                {'label': 'Speed', 'value': 'Speed'},
                {'label': 'Radiation', 'value': 'Radiation'}
            ],
            value='Temperature'
        )
    ], style={'width': '50%', 'display': 'inline-block'}),

    html.Div(
        id='correlation-matrix-container',
        children=[
            dash_table.DataTable(
                id='correlation-matrix',
                style_table={'overflowX': 'scroll'},
            )
        ]
    )
])
@app.callback(
    Output('correlation-matrix', 'data'),
    Output('correlation-matrix', 'columns'),
    Input('corr-target-selector', 'value')
)
def update_correlation_matrix(variable):
    corr_matrix = df.corr()
    corr_matrix = corr_matrix[variable].sort_values(ascending=False).reset_index()
    corr_matrix.columns = ['Variable', 'Correlation']
    columns = [{'name': i, 'id': i} for i in corr_matrix.columns]
    data = corr_matrix.to_dict('records')
    return data, columns


tab4_layout = html.Div([
    html.H1("STL Decomposition"),
    html.Div([
        dcc.Dropdown(
            id='stl-target-selector',
            options=[
                {'label': 'Radiation', 'value': 'Radiation'},
                {'label': 'Seasonal Difference (s=24)', 'value': 's_diff'},
                {'label': 'Non-Seasonal Difference (order=1)', 'value': 'ns_diff'}
            ],
            value='Radiation'
        ),
    ], style={'width': '50%', 'display': 'inline-block', 'padding': '20px'}),

    html.Div([
        dcc.Graph(id="stl-orig-chart",
                  style={'height': '500px'},
                  config={"displayModeBar": False})
    ], style={'width': '50%', 'display': 'inline-block', 'padding': '20px'}),

    html.Div([
        dcc.Graph(id="stl-decomp-chart",
                  style={'height': '500px'},
                  config={"displayModeBar": False})
    ], style={'width': '50%', 'display': 'inline-block', 'padding': '20px'}),
], style={'padding': '20px'})


@app.callback(
    Output("stl-orig-chart", "figure"),
    Output("stl-decomp-chart", "figure"),
    Input('stl-target-selector', 'value')
)
def update_stl_plots(variable):
    if variable == 's_diff':
        s = 24
        data = Tool.seasonal_differencing(df['Radiation'], seasons=s)
    elif variable == 'ns_diff':
        s = 24
        data = Tool.seasonal_differencing(df['Radiation'], seasons=s)
        data = Tool.non_seasonal_differencing(data, 1)
    else:
        data = df['Radiation']
    data.index = df.index
    stl = STL(data, period=24)
    res = stl.fit()

    # Convert index to datetime
    df.index = pd.to_datetime(df.index)

    fig1 = {
        "data": [
            {
                "x": df.index,
                "y": data,
                "type": "lines",
                "hovertemplate": "%{y:.2f}<extra></extra>",
                "name": "Original Data",
                "line": {"color": "#636EFA"}
            },
        ],
        "layout": {
            "title": {
                "text": "Original Data - " + variable,
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"title": "Date"}
        },
    }

    fig2 = {
        "data": [
            {
                "x": df.index,
                "y": res.trend,
                "type": "lines",
                "hovertemplate": "%{y:.2f}<extra></extra>",
                "name": "Trend",
                "line": {"color": "#EF553B"}
            },
            {
                "x": df.index,
                "y": res.seasonal,
                "type": "lines",
                "hovertemplate": "%{y:.2f}<extra></extra>",
                "name": "Seasonal",
                "line": {"color": "#00CC96"}
            },
            {
                "x": df.index,
                "y": res.resid,
                "type": "lines",
                "hovertemplate": "%{y:.2f}<extra></extra>",
                "name": "Residual",
                "line": {"color": "#AB63FA"}
            },
        ],
        "layout": {
            "title": {
                "text": "STL Decomposition - " + variable,
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"title": "Date"}
        },
    }

    return fig1, fig2


# define the callbacks for each tab
@app.callback(
    Output("Radiation-chart", "figure"),
    Input('submit-button', 'n_clicks'),
    State('target-selector', 'value')
)
def update_charts(n_clicks, variable):
    if variable == 's_diff':
        s = 24
        filtered_data = Tool.seasonal_differencing(df['Radiation'], seasons=s)
    elif variable == 'ns_diff':
        s = 24
        filtered_data = Tool.seasonal_differencing(df['Radiation'], seasons=s)
        filtered_data = Tool.non_seasonal_differencing(filtered_data, 24)
    else:
        filtered_data = df['Radiation']
    figure = {
        "data": [
            {
                "x": df['Date'],
                "y": filtered_data,
                "type": "lines",
                "hovertemplate": "%{y:.2f}<extra></extra>",
                "name": variable
            },
        ],
        "layout": {
            "title": {
                "text": "Data - " + variable,
                "x": 0.05,
                "xanchor": "left",
            },
        },
    }

    return figure



# Add a new tab for rolling mean and variance
tab5_layout = html.Div([
        html.H1("Rolling Mean & Variance"),
        html.Div([
            dcc.Dropdown(
                id='target-selection_roll',
                options=[
                    {'label': 'Radiation', 'value': 'Radiation'},
                    {'label': 'Seasonal Difference (s=24)', 'value': 's_diff'},
                    {'label': 'Non-Seasonal Difference (order=1)', 'value': 'ns_diff'}
                ],
                value='Radiation'
            ),
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '20px'}),
        html.Button('Submit', id='submit-button_roll', n_clicks=0),
        html.Div([
            dcc.Graph(id="rolling-mean-variance-chart",
                      style={'height': '500px'},
                      config={"displayModeBar": False})
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '20px'}),
    ], style={'padding': '20px'})


# Callback function to update rolling mean and variance charts
@app.callback(
    Output("rolling-mean-variance-chart", "figure"),
    Input('submit-button_roll', 'n_clicks'),
    State('target-selection_roll', 'value')

)
def update_rolling_plots(n_clicks, variable):
    window_size = 24
    if variable == 's_diff':
        s = 24
        data = Tool.seasonal_differencing(df['Radiation'], seasons=s)
        data = data[s:]
    elif variable == 'ns_diff':
        s = 24
        data = Tool.seasonal_differencing(df['Radiation'], seasons=s)
        data = Tool.non_seasonal_differencing(data, 1)
        data = data[s+1:]
    else:
        data = df['Radiation']
    data.index = df.index
    rolling_mean = data.rolling(window=window_size).mean()
    rolling_var = data.rolling(window=window_size).var()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=rolling_mean, mode='lines', name='Rolling Mean', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=rolling_var, mode='lines', name='Rolling Variance', line=dict(color='red')))
    fig.update_layout(title=f'Rolling Mean & Variance - {variable}', xaxis_title='Date')

    return fig

# Add a new tab for ACF-PACF plots

# tab2_layout = html.Div([
#     html.H1("ACF & PACF"),
#     html.Div([
#         dcc.Dropdown(
#             id='target-selector2',
#             options=[
#                 {'label': 'Radiation', 'value': 'Radiation'},
#                 {'label': 'Seasonal Difference (s=24)', 'value': 's_diff'},
#                 {'label': 'Non-Seasonal Difference (order=1)', 'value': 'ns_diff'}
#             ],
#             value='Radiation'
#         ),
#         dcc.Slider(
#             id='lag-slider',
#             min=0,
#             max=50,
#             value=20,
#             marks={str(i): str(i) for i in range(0, 51, 5)},
#             step=None
#         ),
#         html.Button('Submit', id='submit-button2', n_clicks=0)
#     ], style={'width': '50%', 'margin': 'auto', 'padding': '20px'}),
#     dcc.Graph(id="acf-pacf-graph", style={'height': '500px'})
# ])
#
#
#
#
# # Callback function to update ACF-PACF graph
# @app.callback(
#     Output("acf-pacf-graph", "figure"),
#     Input('submit-button2', 'n_clicks'),
#     Input("lag-slider", "value"),
#     State('target-selector2', 'value')
# )
# def update_tab2(n_clicks, lags, variable):
#     if variable == 's_diff':
#         s = 24
#         filtered_data = Tool.seasonal_differencing(df['Radiation'], seasons=s)
#         filtered_data = filtered_data[s:]
#     elif variable == 'ns_diff':
#         s = 24
#         filtered_data = Tool.seasonal_differencing(df['Radiation'], seasons=s)
#         filtered_data = Tool.non_seasonal_differencing(filtered_data, 24)
#         filtered_data = filtered_data[s+1:]
#     else:
#         filtered_data = df['Radiation']
#     fig = Tool.ACF_PACF_Plot(filtered_data, lags=int(lags), method_name=f"{variable} - {lags} Lags")
#     return fig
#


app.layout = html.Div(
    style={
        'backgroundColor': 'rgb(250, 240, 230)',
        'height': '100%',
    },
    children=[
        dcc.Tab(label='Introduction', value='tab0', children=[
            html.Div(
                [ html.H1("Solar Radiation Prediction", style={'font-size': '3.5em', 'color': 'black'}),
                  html.P("Welcome to the Solar Radiation Prediction Dashboard. This dashboard uses time series forecasting to predict solar radiation. "
                         "Solar radiation is a key factor in the efficient operation of solar energy systems, and accurate prediction of "
                         "solar radiation is crucial for effective grid management and energy policy planning. "
                         "This project uses historical weather data, including"
                         " temperature, humidity, and wind speed, to predict solar radiation levels for the next 24 hours."
                         " The dashboard visualizes the predicted solar radiation levels alongside the actual solar radiation "
                         "levels, allowing us to compare the accuracy of the predictions.", style={'font-size': '1.5em', 'color': 'blue','font-style': 'italic'}),
                  html.P("Developed by Shikha Sharma.", style={'font-size': '1.5em', 'color': 'green','font-style': 'italic'}),
                ],
                style={'width': '80%', 'margin': 'auto', 'text-align': 'center'}
            ),
            dcc.Tab(label='Data Distribution', value='tab1', children=[tab1_layout]),
            # dcc.Tab(label='ACF-PACF', value='tab2', children=[tab2_layout]),
            dcc.Tab(label='Correlation Matrix', value='tab3', children=[tab3_layout]),
            dcc.Tab(label='STL Decomposition', value='tab4', children=[tab4_layout]),
            dcc.Tab(label='Rolling Mean & Variance', value='tab5', children=[tab5_layout]),
            # dcc.Tab(label='Polynomial Function', value='tab4', children=[tab4_layout]),
            # dcc.Tab(label='Sinusoidal Function', value='tab5', children=[tab5_layout]),
            # dcc.Tab(label='Neural Network', value='tab6', children=[tab6_layout]),
        ]),
    ]
)

if __name__ == '__main__':
    app.run_server(debug=False)
import dash
from dash import dcc
from dash import html
from dash import dash_table
from datetime import datetime as dt
import yfinance as yf
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
from sklearn.svm import *
import plotly.graph_objs as go
import plotly.express as px
from model import prediction

# model
# from model import prediction


def get_stock_price_fig(df):

    fig = px.line(df,
                  x="Date",
                  y=["Close", "Open"],
                  title="Closing and Openning Price vs Date")

    return fig


def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df,
                     x="Date",
                     y="EWA_20",
                     title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig


app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Roboto&display=swap"
    ])
server = app.server
# html layout of site
app.layout = html.Div(
    [
        html.Div(
            [
                # Navigation
                html.P("Welcome to the Stock Dash App!", className="start"),
                html.Div([
                    html.P("Input stock code: "),
                    html.Div([
                        dcc.Input(id="dropdown_tickers", type="text"),
                        html.Button("Submit", id='submit'),
                    ],
                             className="form")
                ],
                         className="input-place"),
                html.Div([
                    dcc.DatePickerRange(id='my-date-picker-range',
                                        min_date_allowed=dt(1995, 8, 5),
                                        max_date_allowed=dt.now(),
                                        initial_visible_month=dt.now(),
                                        end_date=dt.now().date()),
                ],
                         className="date"),
                html.Div([
                    html.Button(
                        "Stock Price", className="stock-btn", id="stock"),
                    html.Button("Indicators",
                                className="indicators-btn",
                                id="indicators"),
                    dcc.Input(id="n_days",
                              type="text",
                              placeholder="number of days"),
                    html.Button(
                        "Forecast", className="forecast-btn", id="forecast")
                ],
                         className="buttons"),
                # here
            ],
            className="nav"),

        # content
        html.Div(
            [
                html.Div(
                    [  # header
                        html.Img(id="logo"),
                        html.P(id="ticker")
                    ],
                    className="header"),
                html.Div(id="description", className="decription_ticker"),
                html.Div([], id="graphs-content"),
                html.Div(id='pricetable'),
                html.Div([], id="main-content"),
                html.Div([], id="forecast-content"),
                #html.Div(id="predi"),
            ],
            className="content"), 
    ],
    className="container")


# callback for company info
@app.callback([
    Output("description", "children"),
    Output("logo", "src"),
    Output("ticker", "children"),
    Output("stock", "n_clicks"),
    Output("indicators", "n_clicks"),
    Output("forecast", "n_clicks")
], [Input("submit", "n_clicks")], [State("dropdown_tickers", "value")])
def update_data(n, val):  # inpur parameter(s)
    if n == None:
        return "Hey there! Please enter a legitimate stock code to get details.", "https://melmagazine.com/wp-content/uploads/2019/07/Screen-Shot-2019-07-31-at-5.47.12-PM.png", "Stocks", None, None, None
        # raise PreventUpdate
    else:
        if val == None:
            raise PreventUpdate
        else:
            ticker = yf.Ticker(val)
            inf = ticker.info
            df = pd.DataFrame().from_dict(inf, orient="index").T
            df[['logo_url', 'shortName', 'longBusinessSummary']]
            return df['longBusinessSummary'].values[0], df['logo_url'].values[
                0], df['shortName'].values[0], None, None, None


# callback for stocks graphs
@app.callback([
    #Output("graphs-content", "children"),
    Output('description.children', 'children', allow_duplicate=True),
    Output("pricetable", "children")
], [
    Input("stock", "n_clicks"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    State("dropdown_tickers", "value")
])
def stock_price(n, start_date, end_date, val):
    try:
        if n == None:
            return [""], None
            #raise PreventUpdate
        if val == None:
            raise PreventUpdate
        else:
            if start_date != None:
                df = yf.download(val, str(start_date), str(end_date))
            else:
                df = yf.download(val)

        df.reset_index(inplace=True)
        df = pd.DataFrame(df)
        df["Date"] = df["Date"].astype(str)
        split_data = df['Date'].str.split("_", n=1, expand=True)
        df['Dates'] = split_data[0]
        #df.head(15)
        df[['Open', 'High', 'Close']] = df[['Open', 'High', 'Close']].round(2)
        new_data = df[['Dates', 'Open', 'High', 'Close'
        ]].tail(50)
        fig = get_stock_price_fig(df)
        return [dcc.Graph(figure=fig)], dash_table.DataTable(data=new_data.to_dict('records'))
    except:
        return "Please enter the correct stock code or Try again later", None

#@app.callback






@app.callback(
    Output("main-content", "children"),
    [Input("indicators", "n_clicks"),
     Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date')],
    [State("dropdown_tickers", "value")]
)
def indicators(n, start_date, end_date, val):
    if n is None:
        return [""]
    if val is None:
        return [""]

    if start_date is None:
        df_more = yf.download(val)
    else:
        df_more = yf.download(val, str(start_date), str(end_date))

    df_more.reset_index(inplace=True)

    if n > 0:
        fig = get_more(df_more)
        return dcc.Graph(figure=fig)
    else:
        return [""]

# callback for forecast
@app.callback([Output("forecast-content", "children"),
               #Output("predi", "children")
               ],
              [Input("forecast", "n_clicks")],
              [State("n_days", "value"),
               State("dropdown_tickers", "value")])
def forecast(n, n_days, val):
    if n == None:
        return [""]
    if val == None:
        raise PreventUpdate
    fig = prediction(val, int(n_days) + 1)
    #df2 = prediction(val, int(n_days) + 1).df2
    return [dcc.Graph(figure=fig)] #df2 dash_table.DataTable(data=df2.to_dict('records'))


if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
    
    # # callback for indicators
# @app.callback([Output("main-content", "children")], [
#     Input("indicators", "n_clicks"),
#     Input('my-date-picker-range', 'start_date'),
#     Input('my-date-picker-range', 'end_date')
# ], [State("dropdown_tickers", "value")])
# def indicators(n, start_date, end_date, val):
#     if n == None:
#         return [""]
#     if val == None:
#         return [""]

#     if start_date == None:
#         df_more = yf.download(val)
#     else:
#         df_more = yf.download(val, str(start_date), str(end_date))

#     df_more.reset_index(inplace=True)
    
#     fig = get_more(df_more)
#     return [dcc.Graph(figure=fig)]

# callback for indicators
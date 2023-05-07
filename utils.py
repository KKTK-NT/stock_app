import os
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from sbi_access import get_sbi_holdings
from dotenv import load_dotenv

load_dotenv()

username = os.getenv("SBI_USERNAME")
password = os.getenv("SBI_PASSWORD")

st.set_page_config(layout="wide")

# 銘柄リストを取得する関数
def get_symbol_list(file):
    with open(file, "r") as f:
        symbols = [line.strip() for line in f.readlines()]
    return symbols
    
# スクレイピングを行う関数
def get_stock_info():
    stock_table, balance = get_sbi_holdings(username, password)
    asset = round(stock_table["評価額"].sum()/10000)
    balance = round(balance/10000)
    return stock_table, asset, balance

# スクレイピングしたデータをキャッシュする
@st.cache_data
def cached_stock_info():
    return get_stock_info()

@st.cache_data
def fetch_data(symbols, interval, start_date, end_date):
    data = {}
    for symbol in symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        data[symbol] = stock_data
    return data

def display_charts(data, symbols, interval):
    for symbol in symbols:
        st.subheader(f"{symbol} {interval} Chart")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data[symbol].index, y=data[symbol]['Close'],
                    mode='lines',
                    name='Close Price'))
        fig.update_layout(
            title=f"{symbol} {interval} Chart",
            xaxis_title="Date",
            yaxis_title="Close Price"
        )

        st.plotly_chart(fig)

def calculate_mape(actuals, predictions):
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    return mape

def display_prophet(figures, target_symbols):
    for target_symbol in target_symbols:
        st.image(figures[target_symbol])
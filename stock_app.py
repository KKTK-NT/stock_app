import os
import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sbi_access import get_sbi_holdings
from algo_trade import algo_trade
import csv
import requests
from io import StringIO
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
def cached_data():
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

stock_table, asset, balance = cached_data()

# ページのリスト
pages = [
    "保有銘柄情報",
    "アルゴリズムトレード"
]

# サイドバーにページ選択ボックスを追加
page = st.sidebar.radio("ページを選択", pages)

# 保有銘柄情報ページ
if page == "保有銘柄情報":
    st.title("保有銘柄情報")

    # スクレイピングが初回と「Update」ボタン押下時のみになるように修正
    update_button = False
    update_button = st.button("Update")

    if stock_table is None:
        # キャッシュされたデータを取得
        stock_table, asset, balance = cached_data()

    # Streamlitアプリのメインページに保有銘柄を表示
    st.write(stock_table[["code","name","評価額","損益（％）"]])
    st.write("評価額合計（万円）："+str(asset))
    st.write("現金残高等（万円）："+str(balance))
    st.write("資産合計額（万円）："+str(balance+asset))
    st.write("https://site.sbisec.co.jp/account/assets")

    # サイドバーから「保有銘柄情報」ページトップのコンテンツに移動
    interval_mapping = {
        'Daily': '1d',
        '1 Hour': '1h',
        '1 Minute': '1m',
    }
    
    st.header("Settings")
    selected_interval = st.selectbox("Select Timeframe", list(interval_mapping.keys()))
    interval = interval_mapping[selected_interval]

    initial_symbols = ""
    for i, stock in enumerate(stock_table["code"]):
        if i > 0:
            initial_symbols += ","
        initial_symbols += str(stock)

    # initial_symbolsから重複を削除
    unique_symbols = ','.join(list(set(initial_symbols.split(','))))

    stock_input = st.text_input("Enter Stock Symbols (comma-separated)", unique_symbols)
    stocks = [stock.strip().upper() for stock in stock_input.split(",")]

    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=30))
    end_date = st.date_input("End Date", datetime.today())
    
    stock_data = fetch_data(stocks, interval, start_date, end_date)
    display_charts(stock_data, stocks, interval)

    if update_button:
        # キャッシュをクリアして、再度スクレイピングを行う
        st.cache_data.clear()
        stock_table, asset, balance = cached_data()

# アルゴリズムトレードページ
elif page == "アルゴリズムトレード":
    st.title("アルゴリズムトレード")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    shift = 1 # n 期先予測

    start_cal = st.button("計算開始")
    if start_cal:

        # ここにアルゴリズムトレードに関連するコードを追加
        symbols = get_symbol_list("feat_symbols.txt")
        target_symbols = get_symbol_list("target_symbols.txt")

        predictions, actuals, future_predictions = algo_trade(symbols, target_symbols, start_date, end_date, shift)

        # クロスバリデーションのもっともよかったモデルの予測と実績の対比と、
        # 一週間分の将来予測を時系列グラフにして表示
        for target_symbol in target_symbols:
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=actuals[target_symbol].index,
                                    y=actuals[target_symbol]['actual'],
                                    mode='lines',
                                    name='Actual',
                                    line=dict(color='black')))
            fig.add_trace(go.Scatter(x=predictions[target_symbol].index,
                                    y=predictions[target_symbol]['prediction'],
                                    mode='lines',
                                    name='Predicted',
                                    line=dict(color='red', dash='3px,2px', width=1.4)))

            # Add future predictions to the plot
            future_dates = future_predictions[target_symbol].index
            fig.add_trace(go.Scatter(x=future_dates,
                                    y=future_predictions[target_symbol]['prediction'],
                                    mode='lines',
                                    name='Forecasted',
                                    line=dict(color='green', dash='3px,2px', width=1.4)))

            fig.update_layout(title=f"{target_symbol} Predictions vs Actuals vs Forecasted",
                            xaxis_title="Date",
                            yaxis_title="Close Price")

            fig.update_layout(title=f"{target_symbol} Predictions vs Actuals vs Forecasted",
                            xaxis_title="Date",
                            yaxis_title="Close Price")
            st.plotly_chart(fig)

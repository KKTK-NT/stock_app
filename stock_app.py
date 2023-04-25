import os
import numpy as np
import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp
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

def calculate_mape(actuals, predictions):
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    return mape

def display_prediction(target_symbols, predictions, actuals, future_predictions):
    # クロスバリデーションのもっともよかったモデルの予測と実績の対比と、
    # 一週間分の将来予測を時系列グラフにして表示
    for target_symbol in target_symbols:
        # 時系列プロットとy-yプロットのカラムを作成
        col1, col2 = st.columns([2, 1])

        # 時系列プロット
        fig = go.Figure()

        # 時系列プロットの幅を900に設定
        fig.update_layout(width=850)

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
        col1.plotly_chart(fig)

        # y-yプロット
        fig_yy = go.Figure()

        # y-yプロットの幅を300に設定
        fig_yy.update_layout(width=350)

        fig_yy.add_trace(go.Scatter(x=actuals[target_symbol]['actual'],
                                    y=predictions[target_symbol]['prediction'],
                                    mode='markers',
                                    name='Predicted vs Actual',
                                    marker=dict(color='blue', size=5)))

        # y=yの点線を追加
        fig_yy.add_shape(
            type='line',
            x0=min(actuals[target_symbol]['actual']),
            x1=max(actuals[target_symbol]['actual']),
            y0=min(actuals[target_symbol]['actual']),
            y1=max(actuals[target_symbol]['actual']),
            yref='y',
            xref='x',
            line=dict(color='black', dash='dot')
        )

        # MAPEの計算
        mape = calculate_mape(actuals[target_symbol]['actual'], predictions[target_symbol]['prediction'])

        fig_yy.update_layout(title=f"{target_symbol} Y-Y Plot (MAPE: {mape:.2f}%)",
                            xaxis_title="Actual",
                            yaxis_title="Predicted")
        col2.plotly_chart(fig_yy)


            
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

    update_button = False
    update_button = st.button("情報更新")

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
        '1 Hour': '1h'
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
        st.experimental_rerun()

# アルゴリズムトレードページ
elif page == "アルゴリズムトレード":
    st.title("アルゴリズムトレード")
    
    years = 5 # 学習データの期間
    shift = 5 # n 期先予測

    symbols = get_symbol_list("feat_symbols.txt")
    target_symbols = get_symbol_list("target_symbols.txt")

    start_cal = st.button("計算開始")

    if start_cal:
        predictions, actuals, future_predictions = algo_trade(symbols, target_symbols, years, shift)
        display_prediction(target_symbols, predictions, actuals, future_predictions)
        st.cache_data.clear()
        st.experimental_rerun()
    else:
        predictions, actuals, future_predictions = algo_trade(symbols, target_symbols, years, shift)
        display_prediction(target_symbols, predictions, actuals, future_predictions)
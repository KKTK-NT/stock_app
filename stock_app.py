import utils
import streamlit as st
from datetime import datetime, timedelta
    
stock_table, asset, balance = utils.cached_stock_info()

st.title("保有銘柄情報")

update_button = False
update_button = st.button("情報更新")

if stock_table is None:
    # キャッシュされたデータを取得
    stock_table, asset, balance = utils.cached_stock_info()

# Streamlitアプリのメインページに保有銘柄を表示
st.write(stock_table[["code","name","評価額","損益（％）"]])
st.write("評価額合計（万円）："+str(asset))
st.write("現金残高等（万円）："+str(balance))
st.write("資産合計額（万円）："+str(balance+asset))
st.write("https://site.sbisec.co.jp/account/assets")

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

stock_data = utils.fetch_data(stocks, interval, start_date, end_date)
utils.display_charts(stock_data, stocks, interval)

if update_button:
    st.cache_data.clear()
    stock_table, asset, balance = utils.cached_stock_info()
    st.experimental_rerun()
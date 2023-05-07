import streamlit as st
import utils
import algo_trade

st.title("アルゴリズムトレード")

years = 5 # 学習データの期間

symbols = utils.get_symbol_list("feat_symbols.txt")
target_symbols = utils.get_symbol_list("target_symbols.txt")

@st.cache_data
def cached_main_process(symbols, target_symbols, years):
    return algo_trade.main_process(symbols, target_symbols, years)

start_cal = st.button("計算開始")

features, figures = cached_main_process(symbols, target_symbols, years)
utils.display_prophet(figures, target_symbols)

if start_cal:
    st.cache_data.clear()
    features, figures = cached_main_process(symbols, target_symbols, years)
    utils.display_prophet(figures, target_symbols)
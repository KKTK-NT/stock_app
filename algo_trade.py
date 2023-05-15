import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import List
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from s4_reg.core import s4regressor as regressor
from prophet import Prophet
import matplotlib.pyplot as plt
import io

if not os.path.exists('./data'):
    os.makedirs('./data')

def download_stock_data(symbols: List[str], target_symbols: List[str], start_date: str, end_date: str):
    all_symbols = list(symbols + target_symbols)
    data = pd.DataFrame()
    remove_symbols = []

    # Download JPY=X data first
    jpy_data = yf.download("JPY=X", start=start_date, end=end_date, interval='1d')
    jpy_data = jpy_data["Adj Close"]

    for symbol in list(set(all_symbols)):
        file_name = f"./data/{symbol}.csv"
        if os.path.isfile(file_name):
            stock_data = pd.read_csv(file_name, index_col=0, parse_dates=True)
            stock_data.index = pd.to_datetime(stock_data.index)
            last_date = stock_data.index[-1]
            if last_date >= pd.Timestamp(datetime.now()):
                new_data = yf.download(symbol, start=last_date.date() + timedelta(days=1), end=end_date, interval='1d')
                if not new_data.empty and not pd.isna(new_data['Adj Close'].iloc[-1]):
                    # Convert US stocks to JPY
                    if symbol[-2:] != ".T" and symbol != "JPY=X":
                        stock_data = stock_data.mul(jpy_data, axis=0)
                    stock_data = stock_data.append(new_data)
                    stock_data = stock_data[~stock_data.index.duplicated(keep='last')]
                    with open(file_name, mode='w') as f:
                        stock_data.to_csv(f)
                else:
                    remove_symbols.append(symbol)
        else:
            stock_data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
            if not stock_data.empty:
                # Convert US stocks to JPY
                if symbol[-2:] != ".T" and symbol != "JPY=X":
                    stock_data = stock_data.mul(jpy_data, axis=0)
                with open(file_name, mode='w') as f:
                    stock_data.to_csv(f)
            else:
                remove_symbols.append(symbol)

        try:
            data[symbol] = stock_data['Adj Close']
        except Exception as e:
            remove_symbols.append(symbol)
            pass

    for item in list(set(remove_symbols)):
        if item in symbols:
            symbols.remove(item)
        if item in target_symbols:
            target_symbols.remove(item)

    with open("./feat_symbols.txt", "w") as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")
    with open("./target_symbols.txt", "w") as f:
        for symbol in target_symbols:
            f.write(f"{symbol}\n")

    return data, symbols, target_symbols

def prepare_data(data):
    prepared_data = data.copy()

    # NoneをNaNに置換
    prepared_data = prepared_data.replace(to_replace=['None', 'null', 'nan', 'NA'], value=np.nan)

    # 欠損値NaNを過去最新の値で埋める
    prepared_data = prepared_data.fillna(method="ffill")
    prepared_data = prepared_data.fillna(method="bfill")
    prepared_data = prepared_data.dropna(axis=1)            
    
    return prepared_data

def feature_selection(X, y):
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
    feature_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
    feature_selector.fit(X.values, y.values.ravel())
    selected_features = X.columns[feature_selector.support_].tolist()
    return selected_features

def load_selected_features(target_symbol, lag):
    file_name = f"./feature_selection/{target_symbol}_lag{lag}_selected_features.txt"
    if os.path.isfile(file_name):
        with open(file_name, "r") as f:
            selected_features = [line.strip() for line in f.readlines()]
        return selected_features
    return None

def save_selected_features(target_symbol, selected_features, lag):
    file_name = f"./feature_selection/{target_symbol}_lag{lag}_selected_features.txt"
    with open(file_name, "w") as f:
        for feature in selected_features:
            f.write(f"{feature}\n")

def make_s4_features(
    data,
    target,
    seq_len_target=180,
    pred_len_target=1,
    d_model_target=2048,
    seq_len_others=10,
    pred_len_others=1,
    d_model_others=10
    ):
    
    data = data.reset_index()
    data['date'] = pd.to_datetime(data['Date'])
    data = data.drop(['Date'],axis=1)

    assert seq_len_target > seq_len_others
       
    model_target = regressor(
        dataset = data,
        target = target,
        size = [seq_len_target, pred_len_target],
        features = 'S',
        d_model = d_model_target,
        device = 'cpu'
    )
    
    feat_df_target = model_target.get_features(data)
    feat_df_target = feat_df_target.drop([target], axis=1)
    
    model_others = regressor(
        dataset = data,
        target = target,
        size = [seq_len_others, pred_len_others],
        features = 'MS',
        d_model = d_model_others,
        device = 'cpu'
    )
    
    feat_df_others = model_others.get_features(data).iloc[seq_len_target-seq_len_others:,:]
    feat_df_others = feat_df_others.drop([target], axis=1)
    feat_df_others.columns = [f'exog_feat_{i+1}' for i in range(len(feat_df_others.columns))]

    features = pd.concat([
                          feat_df_target,
                          feat_df_others
                          ], axis=1)
    
    return features
    
def make_fig_prophet(
    data,
    target,
    days=500,
    periods=20
    ):
    
    data = data[target].reset_index()
    data['Date'] = pd.to_datetime(data['Date'])

    prophet_train = data.rename(columns={target:'y', 'Date':'ds'})
    model_prophet = Prophet()
    model_prophet.fit(prophet_train)
    future = model_prophet.make_future_dataframe(periods=periods)
    forecast = model_prophet.predict(future)
    
    model_prophet.plot(forecast)
    xmin = datetime.today() - timedelta(days=days)
    xmax = datetime.today() + timedelta(days=periods)
    holidays = int(days/30.5*8)
    ymin = np.min(forecast['yhat'].iloc[-days+holidays-periods:])
    ymax = np.max(forecast['yhat'].iloc[-days+holidays-periods:])
    ave  = (ymin + ymax) / 2.0
    plt.title(f'{target}')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin-ave*0.15, ymax+ave*0.15])
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    return buf.getvalue()
    
def get_feature(data, target_symbols):
    features = {}
    figures = {}

    for target_symbol in target_symbols:
        features[target_symbol] = make_s4_features(data, target_symbol)
        figures[target_symbol] = make_fig_prophet(data, target_symbol)
 
    return features, figures

def main_process(symbols, target_symbols, years):
    end_date = datetime.now().replace(microsecond=0, second=0, minute=0)
    start_date = end_date - timedelta(days=years*365)
    data, symbols, target_symbols = download_stock_data(symbols, target_symbols, start_date, end_date)
    prepared_data = prepare_data(data)

    features, figures = get_feature(prepared_data, target_symbols)

    return features, figures
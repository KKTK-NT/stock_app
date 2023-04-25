import copy
import yfinance as yf
import pandas as pd
import numpy as np
# import lightgbm as lgb
import streamlit as st
import os
from sklearn.model_selection import TimeSeriesSplit
# from sklearn.model_selection import GridSearchCV
from datetime import datetime, timedelta
from typing import List
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from autogluon.tabular import TabularPredictor

if not os.path.exists('./data'):
    os.makedirs('./data')

def download_stock_data(symbols: List[str], target_symbols: List[str], start_date: str, end_date: str):
    all_symbols = list(symbols + target_symbols)
    data = pd.DataFrame()
    remove_symbols = []
    for symbol in list(set(all_symbols)):
        file_name = f"./data/{symbol}.csv"
        if os.path.isfile(file_name):
            stock_data = pd.read_csv(file_name, index_col=0, parse_dates=True)
            stock_data.index = pd.to_datetime(stock_data.index)
            last_date = stock_data.index[-1]
            if last_date >= pd.Timestamp(datetime.now()):
                new_data = yf.download(symbol, start=last_date.date() + timedelta(days=1), end=end_date, interval='1d')
                if not new_data.empty and pd.isna(new_data['Adj Close'].iloc[-1]):
                    stock_data = stock_data.append(new_data)
                    stock_data = stock_data[~stock_data.index.duplicated(keep='last')]
                    with open(file_name, mode='w') as f:
                        stock_data.to_csv(f)
                else:
                    remove_symbols.append(symbol)
            else:
                pass
        else:
            stock_data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
            if not stock_data.empty and pd.isna(stock_data['Adj Close'].iloc[-1]):
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

def prepare_data(data, symbols, target_symbols, shift=1):
    prepared_data = {}
    for target_symbol in target_symbols:
        stock_data = pd.DataFrame(data[target_symbol])
        stock_data[f'{target_symbol}_lag_{shift}'] = data[target_symbol].shift(shift, fill_value=0)
        for symbol in symbols:
            if symbol != target_symbol:
                stock_data[f'{symbol}_lag_{shift}'] = data[symbol].shift(shift, fill_value=0)
        prepared_data[target_symbol] = stock_data
    
    for key in prepared_data.keys():
        # shiftの先頭行を排除
        prepared_data[key] = prepared_data[key].iloc[shift:,:]

        # NoneをNaNに置換
        prepared_data[key] = prepared_data[key].replace(to_replace=['None', 'null', 'nan', 'NA'], value=np.nan)

        # 欠損値NaNを過去最新の値で埋める
        prepared_data[key] = prepared_data[key].fillna(method="ffill")
        prepared_data[key] = prepared_data[key].dropna(axis=1)            

        if prepared_data[key].isnull().any().any():
            st.write(f"{key} : NaN values found.")
    
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

# def fit_lgb(X, y, target_symbol):
#     X = X.drop([target_symbol], axis=1)
#     tscv = TimeSeriesSplit(n_splits=5)
#     model = lgb.LGBMRegressor(random_state=42)
#     # Hyperparameter grid
#     param_grid = {
#         'n_estimators': [1, 3, 10],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'num_leaves': [3, 10],
#         'min_child_samples': [1, 3, 10]
#     }

#     # Grid search with cross-validation
#     grid = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
#     grid.fit(X, y)
    
#     return grid.best_estimator_

def fit_gluon(X, target_symbol):
    predictor = TabularPredictor(label=target_symbol, path=f'C:/Users/rodin/work/stock_trade/', problem_type='regression')
    predictor.fit(train_data=X, presets='best_quality', time_limit=10)
    
    return predictor

def train_and_test(data, symbols, target_symbols, original_data, future_days=1):
    predictions = {}
    actuals = {}
    future_predictions = {}

    for target_symbol in target_symbols:
        X = data[target_symbol].drop([target_symbol], axis=1)
        y = data[target_symbol][target_symbol]

        # Perform feature selection using Boruta
        selected_features = load_selected_features(target_symbol, future_days)
        if not selected_features:
            selected_features = feature_selection(X, y)
            save_selected_features(target_symbol, selected_features, future_days)
        
        X = X[selected_features]
        X[target_symbol] = y

        # Split data for final evaluation
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # predictor = fit_lgb(X_train, y_train, target_symbol)        # lgb predictor
        predictor = fit_gluon(X_train, target_symbol)               # gluon predictor
        
        predictions[target_symbol] = pd.DataFrame(predictor.predict(X_test).values, index=y_test.index, columns=["prediction"])

        actuals[target_symbol] = pd.DataFrame(y_test.values, index=y_test.index, columns=["actual"])

        future_dates = [data[target_symbol].index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]

        # Initialize future_df with columns for all symbols
        future_df = pd.DataFrame(index=future_dates, columns=X.columns).drop([target_symbol], axis=1)

        for column in future_df.columns:
            symbol, _, lag = column.split('_')
            lag = int(lag)
            future_df.loc[:, column] = original_data[symbol].iloc[-lag:].values
    
        # Predict future prices for the target symbol and save the results
        future_predictions[target_symbol] = pd.DataFrame(predictor.predict(future_df[selected_features]).values, index=future_dates[:len(future_df)], columns=["prediction"])

    return predictions, actuals, future_predictions

@st.cache_data
def algo_trade(symbols, target_symbols, years, shift):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    data, symbols, target_symbols = download_stock_data(symbols, target_symbols, start_date, end_date)
    original_data = copy.deepcopy(data)
    prepared_data = prepare_data(data, symbols, target_symbols, shift=shift)
    
    predictions, actuals, future_predictions = train_and_test(prepared_data, symbols, target_symbols, original_data, future_days=shift)

    return predictions, actuals, future_predictions


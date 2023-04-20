import copy
import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from datetime import date, timedelta
from typing import List

def download_stock_data(symbols: List[str], target_symbols: List[str], start_date: str, end_date: str):
    all_symbols = list(symbols + target_symbols)
    data = pd.DataFrame()
    for symbol in all_symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
        if not stock_data.empty:  # データが存在しない銘柄を除外
            data[symbol] = stock_data['Adj Close']
    return data

def prepare_data(data, symbols, target_symbols, shift=1):
    prepared_data = {}
    for target_symbol in target_symbols:
        stock_data = pd.DataFrame(data[target_symbol])
        stock_data[f'{target_symbol}_lag_{shift}'] = data[target_symbol].shift(shift)
        for symbol in symbols:
            if symbol != target_symbol:
                stock_data[f'{symbol}_lag_{shift}'] = data[symbol].shift(shift)
        prepared_data[target_symbol] = stock_data.dropna()
    return prepared_data

def train_and_test(data, symbols, target_symbols, original_data, future_days=1):
    predictions = {}
    actuals = {}
    future_predictions = {}

    for target_symbol in target_symbols:
        X = data[target_symbol].drop([target_symbol], axis=1)
        y = data[target_symbol][target_symbol]

        tscv = TimeSeriesSplit(n_splits=3)

        model = lgb.LGBMRegressor(random_state=42)

        # Hyperparameter grid
        param_grid = {
            'n_estimators': [1, 3, 20, 50, 100],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'num_leaves': [3, 15, 31, 63],
            'min_child_samples': [10, 20, 30]
        }

        # Grid search with cross-validation
        grid = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X, y)

        best_model = grid.best_estimator_

        # Split data for final evaluation
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
        # best_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=20)
        best_model.fit(X_train, y_train)

        predictions[target_symbol] = pd.DataFrame(best_model.predict(X_test), index=y_test.index, columns=["prediction"])
        actuals[target_symbol] = pd.DataFrame(y_test.values, index=y_test.index, columns=["actual"])

        future_dates = [data[target_symbol].index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]

        # Initialize future_df with columns for all symbols
        future_df = pd.DataFrame(index=future_dates, columns=X_train.columns)

        # Fill in future_df with recent values from original_data
        future_df.loc[:, f'{target_symbol}_lag_{future_days}'] = original_data[target_symbol].iloc[-future_days:].values
        for symbol in symbols:
            future_df.loc[:, f'{symbol}_lag_{future_days}'] = original_data[symbol].iloc[-future_days:].values
        
        # Predict future prices for the target symbol and save the results
        future_predictions[target_symbol] = pd.DataFrame(best_model.predict(future_df), index=future_dates[:len(future_df)], columns=["prediction"])

    return predictions, actuals, future_predictions


def algo_trade(symbols, target_symbols, start_date, end_date, shift):
    data = download_stock_data(symbols, target_symbols, start_date, end_date)
    original_data = copy.deepcopy(data)
    prepared_data = prepare_data(data, symbols, target_symbols, shift=shift)
    
    predictions, actuals, future_predictions = train_and_test(prepared_data, symbols, target_symbols, original_data, future_days=shift)

    return predictions, actuals, future_predictions

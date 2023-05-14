import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from collections import OrderedDict


class StockTradingEnv(gym.Env):
    def __init__(self, stock_data, exogenous_data_dict, max_shares=500):
        super().__init__()

        self.stock_data = stock_data.reset_index(drop=True)
        self.exogenous_data_dict = exogenous_data_dict
        self.current_step = None
        self.positions = defaultdict(int)
        self.max_shares = max_shares
        self.initial_investment = 1000000
        self.cash = self.initial_investment
        self.sell_history = []
        self.transaction_fee_rate = 0.0055

        n_stocks = len(stock_data.columns)
        n_features = len(next(iter(exogenous_data_dict.values())).columns) + 1  # Stock price column is added to the features

        space_dict = {
            f"T_{stock.replace('.', '_')}": spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32) for stock in stock_data.columns
        }
        self.observation_space = spaces.Dict(space_dict)

        self.action_space = spaces.MultiDiscrete([2] * (len(self.stock_data.columns) + 1))
        
    def reset(self, seed=None, options={}):
        self.current_step = self._reset_data()
        self.positions.clear()
        self.cash = self.initial_investment
        self.state = self._get_observation()
        return (self.state, {})

    def step(self, action):
        if self._is_done():
            return self._get_observation(), 0, True, {}
        
        assert len(action) == self.action_space.shape[0], "Invalid action shape."
        
        trade_amounts = action[:-1] * 2 - 1
        trade_type = action[-1]
        
        info = {}
        for i, stock in enumerate(self.stock_data.columns):
            trade_amount = trade_amounts[i]
            stock_price = self.stock_data.loc[self.current_step, stock]
            if trade_amount > 0 and self.positions[stock] < self.max_shares:
                shares_to_buy = min(trade_amount, self.max_shares - self.positions[stock])
                cost = shares_to_buy * stock_price * (1 + self.transaction_fee_rate)
                if cost <= self.cash:
                    self.positions[stock] += shares_to_buy
                    self.cash -= cost
            elif trade_amount < 0 and self.positions[stock] > 0:
                shares_to_sell = min(abs(trade_amount), self.positions[stock])
                sell_value = shares_to_sell * stock_price
                transaction_fee = sell_value * self.transaction_fee_rate
                self.positions[stock] -= shares_to_sell
                self.sell_history.append(sell_value)
                self.cash += sell_value - transaction_fee
                info[stock] = {'sell_value': sell_value}

        self.current_step = self._next_step()
        reward = self._get_reward()
        done = self._is_done()
        truncated = False
        self.state = self._get_observation()
        return self.state, reward, done, truncated, {}

    def _reset_data(self):
        return 0

    def _is_done(self):
        return self.current_step >= len(self.stock_data) - 1

    def _next_step(self):
        return self.current_step + 1

    def _get_reward(self):
        total_value = self.cash
        for stock, shares in self.positions.items():
            total_value += shares * self.stock_data.loc[self.current_step, stock]
        return total_value - self.initial_investment

    def _get_observation(self):
        obs_dict = OrderedDict()
        for stock in self.stock_data.columns:
            stock_price = self.stock_data.loc[self.current_step, stock]
            exog_data = self.exogenous_data_dict[stock]
            obs_array = np.concatenate(([stock_price], exog_data.iloc[self.current_step].values), axis=0)
            obs_key = f"T_{stock.replace('.', '_')}"
            obs_dict[obs_key] = obs_array

        return obs_dict


import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from collections import OrderedDict

class StockTradingEnv(gym.Env):
    def __init__(self, stock_data, exogenous_data_dict, max_shares=3000, min_shares=10):
        super().__init__()

        self.stock_data = stock_data.reset_index(drop=True)
        self.exogenous_data_dict = exogenous_data_dict
        self.current_step = 0
        self.positions = defaultdict(int)
        self.max_shares = max_shares
        self.initial_investment = 1000000
        self.cash = self.initial_investment
        self.sell_history = []
        self.transaction_fee_rate = 0.00055
        self.shares_amounts = [500, 300, 200, 100, 50, 10, 0, -10, -50, -100, -200, -300, -500]
        self.min_reward = -1.e9
        self.max_reward = 1.e9

        n_stocks = len(stock_data.columns)
        n_features = len(next(iter(exogenous_data_dict.values())).columns) + 1

        space_dict = {
            f"T_{stock.replace('.', '_')}": spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32) for stock in stock_data.columns
        }
        space_dict["cash"] = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        space_dict["avail_actions"] = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.shares_amounts),), dtype=np.float32)
        space_dict["action_mask"] = spaces.Box(low=0, high=1, shape=(n_stocks, len(self.shares_amounts)), dtype=np.float32)

        self.observation_space = spaces.Dict(space_dict)
        self.min_shares = min_shares

        self.action_space = spaces.MultiDiscrete([len(self.shares_amounts)]*n_stocks)
        
    def reset(self, seed=None, options={}):
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = self._reset_data()
        
        for stock in self.stock_data.columns:
            self.positions[stock] = 0
            
        self.cash = self.initial_investment
        self.state = self._get_observation()
        
        return self.state, {}

    def step(self, action):
        if self._is_done():
            return self._get_observation(), 0, True, {}

        info = {}
        for i, stock in enumerate(self.stock_data.columns):
            trade_action = self.shares_amounts[action[i]]
            stock_price = self.stock_data.loc[self.current_step, stock]
            cost = 0
            sell_value = 0
            if trade_action > 0 and self.positions[stock] < self.max_shares:  # Buy
                shares_to_buy = min(trade_action, self.max_shares - self.positions[stock])
                cost_judge = shares_to_buy * stock_price * (1 + self.transaction_fee_rate)
                if cost_judge <= self.cash:
                    self.positions[stock] += shares_to_buy
                    self.cash -= cost_judge
                    cost += cost_judge
            elif trade_action < 0:  # Sell
                shares_to_sell = min(abs(trade_action), self.positions[stock])
                self.positions[stock] -= shares_to_sell
                sell_value += shares_to_sell * stock_price * (1 - self.transaction_fee_rate)
                self.sell_history.append(sell_value)
                self.cash += sell_value - cost

        self.current_step += 1
        self.state = self._get_observation()

        reward = self._get_reward()
        done = self._is_done()
        truncated = False
        info = {'current_step': self.current_step, 'positions': self.positions, 'cash': self.cash}

        return self.state, reward, done, truncated, info

    def _reset_data(self):
        return 0

    def _is_done(self):
        return self.current_step >= len(self.stock_data) - 1

    def _get_reward(self):
        total_value = self.cash
        for stock, shares in self.positions.items():
            total_value += shares * self.stock_data.loc[self.current_step, stock]

        reward = total_value - self.initial_investment

        if total_value < self.initial_investment:
            penalty = self.initial_investment - total_value
            reward -= penalty
            
        reward = (reward - self.min_reward) / (self.max_reward - self.min_reward)
        reward = reward * 2 - 1

        return reward

    def _get_observation(self):
        obs_dict = OrderedDict()
        for stock in self.stock_data.columns:
            stock_price = self.stock_data.loc[self.current_step, stock]
            exog_data = self.exogenous_data_dict[stock]
            # print(np.shape(self.exogenous_data_dict[stock]))
            obs_array = np.concatenate(([stock_price], exog_data.iloc[self.current_step].values), axis=0)
            obs_key = f"T_{stock.replace('.', '_')}"
            obs_dict[obs_key] = obs_array

        obs_dict["cash"] = np.array([self.cash])
        obs_dict["avail_actions"] = np.array([500, 300, 200, 100, 50, 10, 0, -10, -50, -100, -200, -300, -500])
        obs_dict["action_mask"] = self._get_action_mask()
        return obs_dict

    def _get_action_mask(self):
        action_mask = np.ones((len(self.stock_data.columns), len(self.shares_amounts)), dtype=np.float32)
        for j, stock in enumerate(self.stock_data.columns):
            for i, shares in enumerate(self.shares_amounts):
                if shares > 0:  # Buying
                    if self.cash < (self.min_shares * self.stock_data.loc[self.current_step, stock] * (1 + self.transaction_fee_rate)):
                        action_mask[j, i] = 0  # Not enough cash to buy
                elif shares < 0:  # Selling
                    if self.positions[stock] < self.min_shares:
                        action_mask[j, i] = 0  # Not enough shares to sell
                if self.positions[stock] + shares > self.max_shares:  # Buying more than max_shares
                    action_mask[j, i] = 0
                        
        return action_mask

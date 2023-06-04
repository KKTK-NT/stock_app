import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from collections import OrderedDict


class StockTradingEnv(gym.Env):
    def __init__(self, stock_data, exogenous_data_dict):
        super().__init__()

        self.stock_data = stock_data.reset_index(drop=True)
        self.exogenous_data_dict = exogenous_data_dict
        self.positions = defaultdict(int)

        self.current_step = None
        self.cash = self.initial_investment = 4000000
        self.transaction_fee_rate = 0.00055
        self.shares_amounts = [500, 300, 200, 100, 50, 10, 0, -10, -50, -100, -200, -300, -500]
        self.min_reward = -1.e6
        self.max_reward = 1.e6
        self.wrong_action_penalty = 0
        self.correct_action_bonus = 0
        self.coeff_penalty = 10.00
        self.coeff_bonus = 1.00
        self.coeff_reward = 0.001
        
        self.rewards = []
        self.penalties = []
        self.bonuses = []
        self.increase_percentages = []

        n_features = len(next(iter(exogenous_data_dict.values())).columns) + 1

        space_dict = {
            f"T_{stock.replace('.', '_')}": spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32) for stock in stock_data.columns
        }
        space_dict["cash"] = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)  # Add cash to the observation space
        space_dict["positions"] = spaces.Box(low=0, high=np.inf, shape=(len(stock_data.columns),), dtype=np.float32)  # Add cash to the observation space
        self.observation_space = spaces.Dict(space_dict)
        self.action_space = spaces.MultiDiscrete([len(self.shares_amounts)] * len(self.stock_data.columns))

    def reset(self, seed=None, options={}):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.positions.clear()
        self.cash = self.initial_investment
        self.state = self._get_observation()
        self.wrong_action_penalty = 0
        self.correct_action_bonus = 0
        
        self.rewards = []
        self.penalties = []
        self.bonuses = []
        self.increase_percentages = []
        
        return (self.state, {})

    def _is_done(self):
        return self.current_step >= len(self.stock_data) - 1

    def _next_step(self):
        return self.current_step + 1

    def _get_observation(self):
        obs_dict = OrderedDict()
        pos = []
        for stock in self.stock_data.columns:
            stock_price = self.stock_data.loc[self.current_step, stock]
            exog_data = self.exogenous_data_dict[stock]
            obs_array = np.concatenate(([stock_price], exog_data.iloc[self.current_step].values), axis=0)
            obs_key = f"T_{stock.replace('.', '_')}"
            obs_dict[obs_key] = obs_array
            pos.append(self.positions[stock])

        obs_dict["cash"] = np.array([self.cash])  # Add cash to the observation
        obs_dict["positions"] = np.array(pos)  # Add positions to the observation

        return obs_dict

    def step(self, action):
        if self._is_done():
            return self._get_observation(), 0, True, {}, {}
        
        self.wrong_action_penalty = 0
        self.correct_action_bonus = 0
        info = {}
        for i, stock in enumerate(self.stock_data.columns):
            min_transaction_unit = 10 if stock == "1557.T" else (100 if stock.endswith(".T") else 10)
            trade_action = self.shares_amounts[action[i]]
            stock_price = self.stock_data.loc[self.current_step, stock]
            cost, sell_value, penalty, bonus = self._process_trade(stock, trade_action, stock_price, min_transaction_unit)
            info[stock] = {'cost': cost, 'sell_value': sell_value}
            self.wrong_action_penalty += penalty
            self.correct_action_bonus += bonus

        total_value = self._calculate_total_value()
        info['total_asset_value'] = total_value
        Action_penalty = self.wrong_action_penalty * self.coeff_penalty / len(self.stock_data.columns)
        Action_bonus = self.correct_action_bonus * self.coeff_bonus / len(self.stock_data.columns)
        self.current_step = self._next_step()
        reward = self._get_reward(Action_penalty, Action_bonus)
        done = self._is_done()
        self.state = self._get_observation()
        
        self.rewards.append(reward)
        self.penalties.append(Action_penalty)
        self.bonuses.append(Action_bonus)
        increase_percentage = (info['total_asset_value'] - self.initial_investment) / self.initial_investment * 100
        self.increase_percentages.append(increase_percentage)

        if done:
            print("--- Information of episodes ---")
            # print("Wrong/Correct/Hold: {:.2f} {:.2f} {:.2f}".format(
            #     np.mean(self.penalties), 
            #     np.mean(self.bonuses), 
            #     1 - np.mean(self.bonuses) - np.mean(self.penalties)
            # ))
            print("Wrong/Correct: {:.2f} {:.2f}".format(
                np.mean(self.penalties), 
                np.mean(self.bonuses),
            ))
            print("Average reward : {:.2f}".format(np.mean(self.rewards) / self.coeff_reward))
            print("Average increase percentage : {:.2f}".format(np.mean(self.increase_percentages)))
            
        return self.state, reward, done, False, info

    def _process_trade(self, stock, trade_action, stock_price, min_transaction_unit):
        cost = 0
        sell_value = 0
        penalty = 0
        bonus = 0

        if trade_action == 0:  # Hold, do nothing
            pass

        elif trade_action > 0:  # Buy
            shares_to_buy = trade_action
            cost_judge = shares_to_buy * stock_price * (1 + self.transaction_fee_rate)
            if cost_judge <= self.cash and shares_to_buy >= min_transaction_unit:  # Check if the agent can afford the purchase and meets the minimum transaction unit
                self.positions[stock] += shares_to_buy
                self.cash -= cost_judge
                cost = cost_judge
                bonus += 1
            else:
                penalty += 1

        else:  # Sell
            shares_to_sell = -trade_action
            if self.positions[stock] >= shares_to_sell and shares_to_sell >= min_transaction_unit:  # Check if the agent has enough shares to sell and meets the minimum transaction unit
                sell_value = shares_to_sell * stock_price * (1 - self.transaction_fee_rate)
                transaction_fee = sell_value * self.transaction_fee_rate
                self.positions[stock] -= shares_to_sell
                self.cash += sell_value - transaction_fee
                bonus += 1
            else:
                penalty += 1

        return cost, sell_value, penalty, bonus

    def _calculate_total_value(self):
        total_value = self.cash
        for stock, shares in self.positions.items():
            total_value += shares * self.stock_data.loc[self.current_step, stock]
        return total_value
    
    def _get_reward(self, action_penalty, action_bonus):
        total_value = self._calculate_total_value()
        reward = total_value - self.initial_investment
        if total_value < self.initial_investment:
            penalty = self.initial_investment - total_value
            reward -= penalty
        reward = (reward - self.min_reward) / (self.max_reward - self.min_reward)
        reward = reward * 2 - 1
        return ( reward - (action_penalty - action_bonus)) * self.coeff_reward

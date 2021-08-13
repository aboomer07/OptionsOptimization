################################################################################
# Import Libraries
################################################################################

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from Optimizer import Optimize
from Option_Visual import Option, OptionStrat
from Params import get_params, vals
from Returns import Returns

################################################################################
# Define Optimization Results Evaluation Class
################################################################################

class Opt_Eval(Returns):
	def __init__(self, params, res):
		self.r = params['r'] #Risk Free Annual Interest Rate
		self.q = params['q'] #Dividend
		self.N = params['N'] #Time Steps
		self.M = params['M'] #Number of Simulations
		self.dt = params['dt']
		self.sigma = params['sigma']
		self.res = pd.DataFrame.from_dict(res)

		self.m = params['m'] #Mean Jump Size
		self.v = params['v'] #Standard Deviation of Jump Size
		self.lam = params['lam'] #Number of Jumps per Year

		self.rho = params['rho'] #Brownian correlation coefficient
		self.kappa = params['kappa'] #Rate of mean reversion
		self.theta = params['theta'] #Long run mean variance
		self.xi = params['xi'] #volatility of volatility
		self.v0 = params['v0']

	def get_combo(self, combo_dict):
		self.combo = combo_dict.copy()

	def monte_price(self, model, row, strike, leg):
		sim = self.simulate(model, row)
		if leg in ['lc', 'sc']:
			p = np.maximum(sim - strike, 0).mean()*np.exp(-self.r*row['T'])
		elif leg in ['lp', 'sp']:
			p = np.maximum(strike - sim, 0).mean()*np.exp(-self.r*row['T'])

		return(p)

	def combo_return(self, test_data):
		df = test_data.copy(deep=True)
		self.combo['Act_Price'] = np.empty(len(self.combo['Legs']))
		self.combo['Profit'] = np.empty(len(self.combo['Legs']))
		cols = [vals['strike_col'], vals['tick_col'],
			vals['type_col']]
		df['Unique'] = df[cols].apply(lambda x: str(x[cols[0]]) + "_" + x[cols[1]] + "_" + x[cols[2]], axis=1)

		conds = {}
		for i in range(len(self.combo['Legs'])):
			cond_str = str(self.combo['Strikes'][i]) + "_" + self.combo['Ticker'] + "_" + self.combo['Contracts'][i].type.lower()
			self.combo['Act_Price'][i] = df[df['Unique'] == cond_str]['Last'].values[0]
			payoff = self.combo['Act_Price'][i] - self.combo['Last'][i]
			if self.combo['Legs'][i][0] == 's':
				payoff = payoff * -1
			self.combo['Profit'][i] = payoff
		self.combo['Return'] = self.combo['Profit'].sum() / abs(self.combo['EnterCost'])

	def sharpe(self):
		returns = self.combo_return()
		std = np.std(returns)
		return(np.mean(returns - self.r)/std)

	def get_delta(self, model, ds = 1e-2):
		high, low = self.combo.copy(), self.combo.copy()
		high[vals['stock_col']] += ds
		low[vals['stock_col']] -= ds
		delta = 0
		for i in range(len(self.combo['Legs'])):
			high_price = self.monte_price(model, high, high['Strikes'][i], high['Legs'][i])
			low_price = self.monte_price(model, high, low['Strikes'][i], low['Legs'][i])
			delta += ((high_price - low_price)/(2 * ds))

		return(delta)

	def get_gamma(self, model, ds = 1e-2):
		high, low = self.combo.copy(), self.combo.copy()
		high[vals['stock_col']] += ds
		low[vals['stock_col']] -= ds
		gamma = 0
		for i in range(len(self.combo['Legs'])):
			high_price = self.monte_price(model, high, high['Strikes'][i], high['Legs'][i])
			low_price = self.monte_price(model, high, low['Strikes'][i], low['Legs'][i])
			gamma += ((high_price - (2 * self.combo['Last'][i]) + low_price)/(ds**2))

		return(gamma)

	def get_theta(self, model, dt=1e-1):
		high, low = self.combo.copy(), self.combo.copy()
		high['T'] += dt
		low['T'] -= dt
		theta = 0
		for i in range(len(self.combo['Legs'])):
			high_price = self.monte_price(model, high, high['Strikes'][i], high['Legs'][i])
			low_price = self.monte_price(model, high, low['Strikes'][i], low['Legs'][i])
			theta += -1*((high_price - low_price)/(2 * dt))

		return(theta)

	def get_vega(self):
		pass

	def total_risk(self, model, runs, type_):
		x = np.empty(runs)
		if type_ == 'Delta':
			for i in range(runs):
				x[i] = self.get_delta(model)
		elif type_ == 'Gamma':
			for i in range(runs):
				x[i] = self.get_gamma(model)
		elif type_ == 'Theta':
			for i in range(runs):
				x[i] = self.get_theta(model)

		return(list(x))

	def WriteCall(self):
		pass

	def BuyHold(self):
		pass
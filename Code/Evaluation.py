################################################################################
# Import Libraries
################################################################################

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################
# Define Optimization Results Evaluation Class
################################################################################

class Opt_Eval(Returns):
	def __init__(self, params, res, data):
		self.S0 = params['S0'] #Initial Stock Price
		self.T = params['T'] #Time to End of Simulation (Maturity Usually)
		self.r = params['r'] #Risk Free Annual Interest Rate
		self.q = params['q'] #Dividend
		self.sigma = params['sigma'] #Volatility
		self.N = params['N'] #Time Steps
		self.M = params['M'] #Number of Simulations
		self.dt = params['dt']
		self.res = res.copy(deep=True)
		self.data = data.copy(deep=True)

		self.m = params['m'] #Mean Jump Size
		self.v = params['v'] #Standard Deviation of Jump Size
		self.lam = params['lam'] #Number of Jumps per Year

		self.rho = params['rho'] #Brownian correlation coefficient
		self.kappa = params['kappa'] #Rate of mean reversion
		self.theta = params['theta'] #Long run mean variance
		self.xi = params['xi'] #volatility of volatility
		self.v0 = params['v0']
		
	def get_combo(self, combo_dict):
		self.combo_dict = combo_dict.copy()

	def monte_price(self, model, row, strike, leg):
		sim = self.simulate(model, row)
		if leg in ['lc', 'sc']:
			p = np.maximum(sim - strike, 0).mean()*np.exp(-self.r*row['T'])
		elif leg in ['lp', 'sp']:
			p = np.maximum(strike - sim, 0).mean()*np.exp(-self.r*row['T'])

		return(p)

	def combo_return(self):
		pass

	def sharpe(self):
		returns = self.combo_return()
		std = np.std(returns)
		return(np.mean(returns - self.r)/std)

	def get_delta(self, model, ds = 1e-2):
		high, low = self.combo_dict.copy(), self.combo_dict.copy()
		high[self.stock_col] += ds
		low[self.stock_col] -= ds
		delta = 0
		for i in range(len(self.combo_dict['Legs'])):
			high_price = self.monte_price(model, high, high['Strikes'][i], high['Legs'][i])
			low_price = self.monte_price(model, high, low['Strikes'][i], low['Legs'][i])
			delta += ((high_price - low_price)/(2 * ds))

		return(delta)

	def get_gamma(self, model, ds = 1e-2):
		high, low = self.combo_dict.copy(), self.combo_dict.copy()
		high[self.stock_col] += ds
		low[self.stock_col] -= ds
		gamma = 0
		for i in range(len(self.combo_dict['Legs'])):
			high_price = self.monte_price(model, high, high['Strikes'][i], high['Legs'][i])
			low_price = self.monte_price(model, high, low['Strikes'][i], low['Legs'][i])
			gamma += ((high_price - (2 * self.combo_dict['Last'][i]) + low_price)/(ds**2))

		return(gamma)

	def get_theta(self, model, dt=1e-1):
		high, low = self.combo_dict.copy(), self.combo_dict.copy()
		high['T'] += dt
		low['T'] -= dt
		theta = 0
		for i in range(len(self.combo_dict['Legs'])):
			high_price = self.monte_price(model, high, high['Strikes'][i], high['Legs'][i])
			low_price = self.monte_price(model, high, low['Strikes'][i], low['Legs'][i])
			theta += -1*((high_price - low_price)/(2 * dt))

		return(theta)

	def get_vega(self):
		pass

	def total_risk(self, model, runs):
		risk = {'Delta' : np.empty(runs), 'Gamma' : np.empty(runs),
		'Theta' : np.empty(runs)}
		for i in range(runs):
			risk['Delta'][i] = self.get_delta(model)
			risk['Gamma'][i] = self.get_gamma(model)
			risk['Theta'][i] = self.get_theta(model)
		return(risk)

	def WriteCall(self):
		pass

	def BuyHold(self):
		pass
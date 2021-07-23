################################################################################
# Import Libraries
################################################################################

import numpy as np
import pandas as pd
import datetime as dt
import cvxpy as cp
from mip import maximize, xsum, BINARY, Model, INTEGER
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm
from Returns import Returns

################################################################################
# Define Optimization Class
################################################################################

class Optimize(Returns):

	def __init__(self, params, data):
		self.m = params['m'] #Mean Jump Size
		self.v = params['v'] #Standard Deviation of Jump Size
		self.lam = params['lam'] #Number of Jumps per Year

		self.rho = params['rho'] #Brownian correlation coefficient
		self.kappa = params['kappa'] #Rate of mean reversion
		self.theta = params['theta'] #Long run mean variance
		self.xi = params['xi'] #volatility of volatility
		self.v0 = params['v0']

		self.r = params['r'] #Risk Free Annual Interest Rate
		self.q = params['q'] #Dividend
		self.sigma = params['sigma'] #Volatility
		self.opts = data[self.type_col].unique()
		self.strikes = sorted(data[self.strike_col].unique())
		self.N = params['N']
		self.M = params['M']
		self.T = data['T'].min()
		self.S0 = data[self.stock_col].min() #Initial Stock Price
		self.symbol = data[self.tick_col]
		self.sides = params['Sides']
		self.expiry = params['dates'][0]
		self.trade_date = min(data[self.dt_col].unique(), key=lambda x: abs(x - params['trade_dates'][0]))
		self.dt = params['dt']
		self.prices = {k : f.groupby(self.strike_col)[self.stock_col].apply(max).to_dict() for k, f in data[(data[self.exp_col] == self.expiry)&(data[self.dt_col] == self.trade_date)].groupby(self.type_col)}

	def exp_profit(self, S, K, type_):
		if type_ == 'call':
			return(np.maximum(S - K, 0))
		elif type_ == 'put':
			return(np.maximum(K - S, 0))

	def get_opt_prices(self):
		return(self.prices)

	def Run(self, model, data, legs, solver='ECOS', gamma=4, integer=False, optim='Utility'):
		As = []
		grped = data.groupby([self.tick_col, self.exp_col, 'T'], as_index=False).Underlying_Price.first()
		grped['Paths'] = grped.apply(lambda x: self.simulate(model, x)[-1, :], axis=1)
		data = data.merge(grped, on=['Ticker', self.exp_col, 'T', self.stock_col], how='left')
		data['As'] = np.where(data[self.type_col] == 'call', data.Paths - data.Strike, data.Strike - data.Paths)
		data['As'] = data['As'].apply(lambda x: np.maximum(x, 0))
		data['As'] = data['As']*np.exp(-self.r*(data['T'] - self.dt))
		data['As'] = (data['As'] - data.Last) / data.Last
		As = np.array(data.As.tolist()).T
		vec = np.array([1 + self.r]*self.M)
		w = cp.Variable(data.shape[0], integer=integer)
		mat_var = As@w + vec
		if optim == 'Utility':
			utils = cp.sum(cp.power(mat_var, gamma)/(1-gamma))/self.M
		else:
			utils = (cp.sum(mat_var) - cp.sum(cp.power(mat_var, 2)))/self.M

		objective = cp.Maximize(utils)
		constraints = [cp.norm(w, 1) <= legs]
		prob = cp.Problem(objective, constraints)

		a = dt.datetime.now()
		result = prob.solve(solver=solver, verbose=True)
		print((dt.datetime.now() - a).total_seconds())
		res = data.copy(deep=True)
		res['Results'] = w.value
		res['Delta'] = As.mean(axis=0)

		return(res)


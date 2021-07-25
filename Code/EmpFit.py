################################################################################
# Import Libraries and Scripts
################################################################################

import numpy as np
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from Params import get_params, vals
from DataPrep import get_data, rhood_daily, yahoo_daily
from Returns import Returns

################################################################################
# Define class to fit parameters to real data
################################################################################

class Fitter:

	def __init__(self, params):
		self.r = params['r'] #Risk Free Annual Interest Rate
		self.q = params['q'] #Dividend
		self.N = params['N'] #Time Steps
		self.M = params['M'] #Number of Simulations
		self.dt = (1/365)

	def gbm(self, x, row):
		sigma = x[0]
		N = round(row['T'] / self.dt)
		size = (N, self.M)
		mu = (self.r - self.q - sigma**2/2)*self.dt
		var = sigma*np.sqrt(self.dt)*np.random.normal(size=size)
		sim = np.cumsum(mu + var, axis=0)
		sim = row[vals['stock_col']] * np.exp(sim)
		return(sim[-1, :])

	def merton_jump(self, x, row):
		sigma = x[0]
		m = x[1]
		v = x[2]
		lam = x[3]
		N = round(row['T'] / self.dt)
		size = (N, self.M)

		mu = (self.r - self.q - sigma**2/2 - lam*(m+v**2*0.5))*self.dt
		var = sigma*np.sqrt(self.dt)*np.random.normal(size=size)

		geo = np.cumsum(mu + var, axis=0)
		poi_rv = np.multiply(np.random.poisson(lam*self.dt, size=size),
			np.random.normal(m, v, size=size)).cumsum(axis=0)
		sim = row[vals['stock_col']] * np.exp(geo + poi_rv)
		return(sim[-1, :])

	def svol(self, x, row):
		rho = x[0]
		kappa = x[1]
		theta = x[2]
		xi = x[3]
		N = round(row['T'] / self.dt)
		size = (N, self.M)
		sim = np.zeros(size)
		sim_vol = np.zeros(size)
		S_t = row[vals['stock_col']]
		v_t = 0.04

		for t in range(N):
			WT = np.random.multivariate_normal(np.array([0,0]), 
				cov = np.array([[1, rho], [rho, 1]]), 
				size=self.M) * np.sqrt(self.dt)

			S_t = S_t*(np.exp((self.r - 0.5*v_t)*self.dt + \
				np.sqrt(v_t)*WT[:,0])) 
			v_t = np.abs(v_t + kappa*(theta-v_t)*self.dt +\
				xi*np.sqrt(v_t)*WT[:,1])
			sim[t, :] = S_t
			sim_vol[t, :] = v_t

		return(sim[-1, :])

	def simulate(self, model, x, row):
		if model == 'BlackScholes':
			return(self.gbm(x, row))
		elif model == 'Merton':
			return(self.merton_jump(x, row))
		elif model == 'Heston':
			return(self.svol(x, row))

	def monte_price(self, x, model, data):
		grped = data.groupby([vals['exp_col'], 'T'], as_index=False)[vals['stock_col']].first()
		grped['Paths'] = grped.apply(lambda z: self.simulate(model, x, z), axis=1)
		data = data.merge(grped, on=[vals['exp_col'], 'T', vals['stock_col']], how='left')
		data['SimPrice'] = np.where(data[vals['type_col']] == 'call', data.Paths - data[vals['strike_col']], data[vals['strike_col']] - data.Paths)
		data['SimPrice'] = data['SimPrice'].apply(lambda z: np.maximum(z, 0))
		data['SimPrice'] = data['SimPrice']*np.exp(-self.r*data['T'])
		data['SimPrice'] = data['SimPrice'].apply(np.mean)
		return(np.linalg.norm(data.Last - data.SimPrice, 2))

	def fit(self, method, data, model, x_0, bounds, runs):
		res_list = []
		for run in range(runs):
			res = minimize(self.monte_price, method=method, x0=x_0, args=(model, data), bounds = bounds, tol=1e-30, options={"maxiter":500})
			res_list.append(res.x)

		return(res_list)





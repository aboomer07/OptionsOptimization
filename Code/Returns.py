################################################################################
# Import Libraries and Scripts
################################################################################

import numpy as np
import datetime as dt
import random
from Params import get_params, vals

################################################################################
# Make returns simulation class for different underlying models
################################################################################

#Next to-dos: Combine stochastic volatility with drift and var functions

class Returns: #Initialize
	def __init__(self, params):
		self.T = params['T'] #Time to End of Simulation (Maturity Usually)
		self.r = params['r'] #Risk Free Annual Interest Rate
		self.q = params['q'] #Dividend
		self.sigma = params['sigma'] #Volatility
		self.N = params['N'] #Time Steps
		self.M = params['M'] #Number of Simulations
		self.dt = params['dt']

		self.m = params['m'] #Mean Jump Size
		self.v = params['v'] #Standard Deviation of Jump Size
		self.lam = params['lam'] #Number of Jumps per Year

		self.rho = params['rho'] #Brownian correlation coefficient
		self.kappa = params['kappa'] #Rate of mean reversion
		self.theta = params['theta'] #Long run mean variance
		self.xi = params['xi'] #volatility of volatility
		self.v0 = params['v0']

	def gbm(self, row):
		N = round(row['T'] / self.dt)
		size = (N, self.M)
		mu = (self.r - self.q - self.sigma**2/2)*self.dt
		var = self.sigma*np.sqrt(self.dt)*np.random.normal(size=size)
		sim = np.cumsum(mu + var, axis=0)
		return(sim)

	def merton_jump(self, row):
		N = round(row['T'] / self.dt)
		size = (N, self.M)
		mu = self.dt * (self.r - self.q - self.sigma**2/2 - self.lam*(self.m+self.v**2*0.5))
		var = self.sigma*np.sqrt(self.dt)*np.random.normal(size=size)
		geo = np.cumsum(mu + var, axis=0)
		poi_rv = np.multiply(np.random.poisson(self.lam*self.dt, size=size),
			np.random.normal(self.m, self.v, size=size)).cumsum(axis=0)
		sim = geo + poi_rv
		return(sim)

	def svol(self, row, return_vol=False):
		N = round(row['T'] / self.dt)
		size = (N, self.M)
		sim = np.zeros(size)
		sim_vol = np.zeros(size)
		S_t = row[vals['stock_col']]
		v_t = self.v0
		for t in range(N):
			WT = np.random.multivariate_normal(np.array([0,0]), 
				cov = np.array([[1, self.rho],
								[self.rho, 1]]), size=self.M) * np.sqrt(self.dt)

			S_t = S_t*(np.exp((self.r - 0.5*v_t)*self.dt + \
				np.sqrt(v_t)*WT[:,0])) 
			v_t = np.abs(v_t + self.kappa*(self.theta-v_t)*self.dt +\
				self.xi*np.sqrt(v_t)*WT[:,1])
			sim[t, :] = S_t
			sim_vol[t, :] = v_t

		if return_vol:
			return(sim, sim_vol)

		return(sim)

	def simulate(self, model, row, return_vol=False):
		if model == 'BlackScholes':
			sim = self.gbm(row)
		elif model == 'Merton':
			sim = self.merton_jump(row)
		elif model == 'Heston':
			return(self.svol(row, return_vol=return_vol))

		sim = row[vals['stock_col']] * np.exp(sim)
		return(sim)

	def monte_price(self, model, strikes, x):
		sim = self.simulate(model, x, return_vol=False)
		price_dict = {'call':{}, 'put':{}}
		for strike in strikes:
			P = np.maximum(strike - sim, 0).mean()*np.exp(-self.r*self.T)
			price_dict['put'][str(strike)] = P

			C = np.maximum(sim - strike, 0).mean()*np.exp(-self.r*self.T)
			price_dict['call'][str(strike)] = C

		return(price_dict)




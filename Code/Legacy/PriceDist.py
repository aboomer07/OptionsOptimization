import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import datetime as dt
import time
from scipy.optimize import minimize
import pandas as pd

################################################################################
# Base Option Class
################################################################################

class Option:
	def __init__(self, S, K, T, r, sigma, q=0):
		self.S = S
		self.K = K
		self.T = T
		self.r = r 
		self.sigma = sigma
		self.q = q

	@property
	def params(self):
		return {'S': self.S, 
			'K': self.K, 
			'T': self.T, 
			'r':self.r,
			'q':self.q,
			'sigma':self.sigma}

	def d1(self):
		return((np.log(self.S/self.K) + (self.r -self.q + self.sigma**2/2)*self.T) / (self.sigma*np.sqrt(self.T)))

	def d2(self):
		return(self.d1() - self.sigma*np.sqrt(self.T))

	def BS_Price(self, type_):
		if type_ == 'C':
			return(self.S*np.exp(-self.q*self.T)*norm.cdf(self.d1()) - \
				self.K*np.exp(-self.r*self.T) * norm.cdf(self.d2()))
		elif type_ == 'P':
			return(self.K*np.exp(-self.r*self.T) * norm.cdf(-self.d2()) - \
					self.S*np.exp(-self.q*self.T)*norm.cdf(-self.d1()))

	def price(self, model, type_):
		if model == 'BS':
			return(self.BS_Price(type_))
		elif model == 'merton':
			r_old = self.r
			sigma_old = self.sigma
			p = 0
			for k in range(40):
				r_k = r_old - self.lam*(self.m-1) + (k*np.log(self.m)) / self.T
				sigma_k = np.sqrt(sigma_old**2 + (k*self.v**2) / self.T)
				k_fact = np.math.factorial(k)
				self.r = r_k
				self.sigma = sigma_k
				p += (np.exp(-self.m*self.lam*self.T) * (self.m*self.lam*self.T)**k / (k_fact)) * self.BS_Price(type_)
			self.r = r_old
			self.sigma = sigma_old
			return(p)


	def delta(self, type_):
		if type_ == 'C':
			return(norm.cdf(self.d1()))
		elif type_ == 'P':
			return(-norm.cdf(-self.d1()))

	def gamma(self):
		return(norm.pdf(self.d1())/(self.S*self.sigma*np.sqrt(self.T)))

	def vega(self):
		return(self.S*np.sqrt(self.T)*norm.pdf(self.d1()))

	def theta(self, type_):
		if type_ == 'C':
			p1 = -self.S*norm.pdf(self.d1())*self.sigma / (2 * np.sqrt(self.T))
			p2 = self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(self.d2())
			return(p1 - p2)
		elif type_ == 'P':
			p1 = -self.S*norm.pdf(self.d1())*self.sigma / (2 * np.sqrt(self.T))
			p2 = self.r*self.K*np.exp(-self.r*self.T)*N(-self.d2()) 
			return(p1 + p2)

	def rho(self, type_):
		if type_ == 'C':
			return(self.K*self.T*np.exp(-self.r*self.T)*norm.cdf(self.d2()))
		if type_ == 'P':
			return(-self.K*self.T*np.exp(-self.r*self.T)*N(-self.d2()))

	def implied_vol(self, opt_value, type_='C'):
		def obj(self, type_):
			return(abs(self.price(type_) - opt_value))

		res = minimize_scalar(self.obj(type_), 
			bounds=(0.01,6), method='bounded')
		return(res.x)

	def add_merton_params(self, m, v, lam):
		self.m = m
		self.v = v
		self.lam = lam

################################################################################
# Merton Jump Diffusion
################################################################################

S = 100 # current stock price
K = 100
strikes = np.arange(50, 150, 1)
T = 1 # time to maturity
r = 0.1 # risk free rate
v = 0.3 # standard deviation of jump
lam = 1 # intensity of jump i.e. number of jumps per annum
steps = 10000 # time steps
Npaths = 1 # number of paths to simulate
sigma = 0.3 # annaul standard deviation , for weiner process
T = 1
m = 0
m = np.exp(0+v**2*0.5)

def merton_jump_paths(S, T, r, sigma, lam, m, v, steps, Npaths):
	size=(steps, Npaths)0
	dt = T/steps 
	poi_rv = np.multiply(np.random.poisson( lam*dt, size=size),
		np.random.normal(m,v, size=size)).cumsum(axis=0)
	geo = np.cumsum(( (r -  sigma**2/2 -lam*(m+v**2*0.5))*dt + \
		sigma*np.sqrt(dt) * np.random.normal(size=size)), axis=0)

	return np.exp(geo+poi_rv)*S

j = merton_jump_paths(S, T, r, sigma, lam, m, v, steps, Npaths)

obj = Option(S, strikes[0], T, r, sigma)
obj.add_merton_params(m, v, lam)
plt.plot(strikes, obj.price('BS', 'C'), label='Black_Scholes')
plt.plot(strikes, (obj.price('merton', 'C')), label='Merton')
plt.legend()
plt.show()

setattr(Option, "d2", d2)

################################################################################
# Import Libraries
################################################################################

import sympy as sp
from sympy.stats import Exponential as exp
from sympy.stats import Normal, cdf, LogNormal, density

################################################################################
# Define functions
################################################################################

class Funcs:

	def __init__(self):

		#Initialize symbols
		self.S0, self.S_t, self.K, self.sigma, self.Time, self.rf, self.w, self.Delta, self.Theta, self.Gamma, self.Pi, self.U, self.gamma = sp.symbols('S0, S_t, K, sigma, Time, rf, w, Delta, Theta, Gamma, Pi, U, gamma')

		self.N = Normal('N', 0.0, 1.0) #Define a standard normal distribution
		self.N_dens = density(self.N)
		self.N_cdf = cdf(self.N) #Get the CDF of standard normal

		#Define lognormal distribution with black-scholes parameters
		self.LN = LogNormal('x', sp.log(self.S_t) + self.Time*(r - (self.sigma**2/2)), self.sigma*sp.sqrt(self.Time))

		self.gbm_dens = density(self.LN)(self.S0) #Get the density of lognormal gbm
		self.gbm_cdf = cdf(self.LN)(self.S0) #Get the cdf of lognormal gbm

	def d1(self):
		return((sp.ln(self.S_t/self.K) + (self.rf + sp.Rational(1, 2)*self.sigma**2)*self.Time) / (self.sigma*sp.sqrt(self.Time)))
	def d2(self):
		return(self.d1 - self.sigma*sp.sqrt(self.Time))

	def C(self):
		#Define call and put price functions according to black scholes
		return(self.S_t*N_cdf(self.d1) - N_cdf(self.d2)*self.K*sp.exp(-self.rf*self.Time))

	def P(self):
		P = -self.S_t*self.N_cdf(-self.d1()) + self.N_cdf(-self.d2())*self.K*sp.exp(-self.rf*self.Time)

	def C_lamb(self):
		return(sp.lambdify((self.S_t, self.K, self.sigma, self.Time, self.rf), self.C()))
	def P_lamb(self):
		return(sp.lambdify((self.S_t, self.K, self.sigma, self.Time, self.rf), self.P()))

	# #Define the payoff functions for the 4 possible option products
	# f_SC = sp.Min(K - self.S0, 0)
	# f_LC = sp.Max(self.S0 - K, 0)
	# f_SP = sp.Min(self.S0 - K, 0)
	# f_LP = sp.Max(K - self.S0, 0)

	# #Turn the payoff functions into lambda functions
	# f_SC_lamb = sp.lambdify((self.S0, K), f_SC)
	# f_LC_lamb = sp.lambdify((self.S0, K), f_LC)
	# f_SP_lamb = sp.lambdify((self.S0, K), f_SP)
	# f_LP_lamb = sp.lambdify((self.S0, K), f_LP)

	# #Define the expected value of the payoffs with lognormal distributed prices
	# E_SC = sp.Integral((K - self.S0) * gbm_dens, (self.S0, K, sp.oo)) + C
	# E_LC = sp.Integral((self.S0 - K) * gbm_dens, (self.S0, K, sp.oo)) - C
	# E_SP = sp.Integral((self.S0 - K) * gbm_dens, (self.S0, 0, K)) + P
	# E_LP = sp.Integral((K - self.S0) * gbm_dens, (self.S0, 0, K)) - P

	# #Define adjusted E[] if call/put premiums invested/borrowed at risk-free rate
	# E_SC_adj = sp.Integral((K - self.S0) * gbm_dens, (self.S0, K, sp.oo)) + (C * sp.exp(r*T))
	# E_LC_adj = sp.Integral((self.S0 - K) * gbm_dens, (self.S0, K, sp.oo)) - (C * sp.exp(r*T))
	# E_SP_adj = sp.Integral((self.S0 - K) * gbm_dens, (self.S0, 0, K)) + (P * sp.exp(r*T))
	# E_LP_adj = sp.Integral((K - self.S0) * gbm_dens, (self.S0, 0, K)) - (P * sp.exp(r*T))

	# #Turn the expected value of payoffs into python lambda functions
	# E_LC_lamb = sp.lambdify((self.S_t, self.K, self.sigma, T, r), E_LC)
	# E_LP_lamb = sp.lambdify((self.S_t, self.K, self.sigma, T, r), E_LP)
	# E_SC_lamb = sp.lambdify((self.S_t, self.K, self.sigma, T, r), E_SC)
	# E_SP_lamb = sp.lambdify((self.S_t, self.K, self.sigma, T, r), E_SP)

	# #Turn the adjusted expected value of payoffs into python lambda functions
	# E_LC_adj_lamb = sp.lambdify((self.S_t, self.K, self.sigma, T, r), E_LC_adj)
	# E_LP_adj_lamb = sp.lambdify((self.S_t, self.K, self.sigma, T, r), E_LP_adj)
	# E_SC_adj_lamb = sp.lambdify((self.S_t, self.K, self.sigma, T, r), E_SC_adj)
	# E_SP_adj_lamb = sp.lambdify((self.S_t, self.K, self.sigma, T, r), E_SP_adj)

	# #Define expected value of stock price according to gbm distribution
	# E_S = sp.Integral(self.S0 * gbm_dens, (self.S0, 0, sp.oo))
	# E_S_lamb = sp.lambdify((self.S_t, self.sigma, T, r), E_S)

	# #Get Greeks for Call and Put
	# deltaC = N_cdf(self.S0d1)
	# deltaP = N_cdf(-self.S0d1)
	# deltaC_lamb = sp.lambdify((self.S_t, K, self.sigma, T, r), deltaC)
	# deltaP_lamb = sp.lambdify((self.S_t, K, self.sigma, T, r), deltaP)

	# gammaC = N_dens(d1)/(self.S_t * self.sigma * sp.sqrt(T))
	# gammaP = gammaC
	# gammaC_lamb = sp.lambdify((self.S_t, K, self.sigma, T, r), gammaC)
	# gammaP_lamb = sp.lambdify((self.S_t, K, self.sigma, T, r), gammaP)

	# thetaC = (((self.S_t*N_dens(d1)*self.sigma)/(2*sp.sqrt(T))) + r*K*sp.exp(-r*T)*N_cdf(d2))
	# thetaP = (((self.S_t*N_dens(d1)*self.sigma)/(2*sp.sqrt(T))) - r*K*sp.exp(-r*T)*N_cdf(-d2))
	# thetaC_lamb = sp.lambdify((self.S_t, K, self.sigma, T, r), thetaC)
	# thetaP_lamb = sp.lambdify((self.S_t, K, self.sigma, T, r), thetaP)

	# vegaC = self.S_t * sp.sqrt(T) * N_dens(d1)
	# vegaP = vegaC
	# vegaC_lamb = sp.lambdify((self.S_t, K, self.sigma, T, r), vegaC)
	# vegaP_lamb = sp.lambdify((self.S_t, K, self.sigma, T, r), vegaP)
################################################################################
# Import Libraries and Scripts
################################################################################

import numpy as np
import pandas as pd

################################################################################
# Define parameter dictionaries
################################################################################

def get_params(updates):

	params = {'S0' : 100, # current stock price
				'T' : 1, # time to maturity
				'r' : 0.02, # risk free rate
				'm' : 0, # meean of jump size
				'v' : 0.3, # standard deviation of jump
				'q' : 0, #Dividend
				'lam' : 1, # intensity of jump i.e. number of jumps per annum
				'N' : 1, # time steps
				'M' : 250, # number of paths to simulate
				'sigma' : 0.2, # annaul standard deviation , for weiner process
				'kappa' : 4,
				'theta' : 0.02,
				'v0' :  0.02,
				'xi' : 0.9,
				'rho' : 0.9,
				'strikes' : np.arange(50, 150, 1),
				'stocks' : ['AAPL'],
				'dates' : ['2021-06-18'],
				'trade_dates' : [pd.to_datetime('2021-05-19')],
				'num_times' : 10,
				'Sides' : ['Long'],
				'file_source' : 'yahoo',
				'x0' : {'BlackScholes' : {'sigma':0.3},
					'Merton' : {'sigma':0.15, 'm':1, 'v':0.1, 'lambda':1},
					'Heston' : {'rho':-0.6, 'kappa':4, 'theta':0.02, 'xi':0.9}},
				'bounds' : {'BlackScholes' : {'sigma':(0.01, np.inf)},
					'Merton' : {'sigma':(0.01, np.inf) , 'm':(0.01, 2), 'v':(1e-2, np.inf) , 'lambda':(0.01, 5)},
					'Heston' : {'rho':(-0.999, 0.999), 'kappa':(0.01, np.inf), 'theta':(0.01, np.inf), 'xi':(0.01, np.inf)}}}

	params['dt'] = 1 / 365

	params.update(updates)

	return(params)

def settings(params):
	vals = {}
	vals['source'] = params['file_source']
	vals['stock_col'] = ['Underlying_Price', 'close_price'][vals['source'] == 'rhood']
	vals['tick_col'] = ['Ticker', 'symbol'][vals['source'] == 'rhood']
	vals['type_col'] = ['Type', 'Direction'][vals['source'] == 'rhood']
	vals['strike_col'] = ['Strike', 'strike_price'][vals['source'] == 'rhood']
	vals['exp_col'] = ['Expiry', 'expiration_date'][vals['source'] == 'rhood']
	vals['dt_col'] = ['Quote_Time', 'begins_at'][vals['source'] == 'rhood']
	vals['dt_form'] = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ'][vals['source'] == 'rhood']
	return(vals)

global vals
vals = settings(get_params({}))



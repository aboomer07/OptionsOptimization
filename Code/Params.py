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
				'stocks' : ['SPY'],
				'dates' : ['2021-06-18'],
				'trade_dates' : [pd.to_datetime('2021-05-19')],
				'num_times' : 10,
				'Sides' : ['Long'],
				'file_source' : 'yahoo',
				'x0' : {'BlackScholes' : [0.3],
						'Merton' : [0.15, 1, 0.1, 1],
						'Heston' : [-0.6, 4, 0.02, 0.9]},
				'bounds' : {'BlackScholes' : [(0.01, np.inf)],
						'Merton' : [(0.01, np.inf) , (0.01, 2), (1e-2, np.inf) , (0.01, 5)],
						'Heston' : [(-0.999, 0.999), (0.01, np.inf), (0.01, np.inf), (0.01, np.inf)]}}

	params['dt'] = 1 / 365

	params.update(updates)

	return(params)
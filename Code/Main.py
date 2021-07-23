################################################################################
# Import Libraries and Scripts
################################################################################

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cvxpy as cp
from cvxpy.settings import CPLEX
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm
from Params import get_params
from DataPrep import get_data, rhood_daily, yahoo_daily, fit_optim_test
import datetime as dt
from Returns import Returns
from Optimizer import Optimize
from Option_Visual import Option, OptionStrat
import xpress
import gurobipy
from EmpFit import Fitter
from tabulate import tabulate
import random
from itertools import product

################################################################################
# Set up folder structure
################################################################################

base_path = '/Users/andrewboomer/Desktop/M2_Courses/Thesis/Options/tradingbot'
code_path = base_path + '/Code'
out_path = base_path + '/Output'
report_path = base_path + '/Report'

################################################################################
# Get parameters
################################################################################

import_data = True

if import_data:
	data = get_data(get_params({}))

################################################################################
# Get Returns Simulations and Plot
################################################################################

data = data.reset_index().drop('index', axis=1)
curr_row = data[data.index == 0]
sim_params = get_params({'N' : 200})
obj = Returns(sim_params)

bs_sim = obj.simulate('BlackScholes', curr_row)
merton_sim = obj.simulate('Merton')
heston_sim = obj.simulate('Heston')

plt.plot(bs_sim)
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Merton Jump Diffusion Process')
plt.show()

strikes = sim_params['strikes']
bs_prices = obj.monte_price('BlackScholes', strikes)
merton_prices = obj.monte_price('Merton', strikes)
heston_prices = obj.monte_price('Heston', strikes)

plt.plot(strikes, bs_prices['call'].values(), label='BS Call Value')
plt.plot(strikes, merton_prices['call'].values(), label='Merton Call Value')
plt.plot(strikes, heston_prices['call'].valuees(), label='Heston Call Value')
plt.legend()
plt.show()

fdm_df = obj.fdm('Heston', strikes)
fig, ax = plt.subplots(1)
sns.lineplot(data=fdm_df[fdm_df['Partial'] == 'gamma'], x='strike_price', y='call', ax=ax, label='Call')
sns.lineplot(data=fdm_df[fdm_df['Partial'] == 'gamma'], x='strike_price', y='put', ax=ax, label='Put')
plt.show()

################################################################################
# Run a Sample Optimization
################################################################################

optim_params = get_params({'N' : 7, 'dt' : 7/365, 'M' : 50})
optim = Optimize(optim_params, data)

res = optim.Run('Merton', optim_data, 1, solver='CPLEX', integer=True, gamma=4)

combo = OptionStrat(res)

plot_params = {'title' : "Test Combo", 'file' : None}
combo.plot_profit(plot_params)

################################################################################
# Fit Parameters and Optimize
################################################################################

data = get_data(get_params({}))
fit_size = 10
min_dt = 7/365

fit_data, optim_data, test_data = fit_optim_test(data, fit_size, min_dt)

for model in ['BlackScholes', 'Merton']:

	# solvers = ['Powell','L-BFGS-B','SLSQP','trust-constr']
	a = dt.datetime.now()
	fit_params = get_params({'M' : 1000})
	fit_obj = Fitter(fit_params)
	x0 = fit_params['x0'][model]
	bounds = fit_params['bounds'][model]
	res = fit_obj.fit('SLSQP', fit_data, model, x0, bounds)
	print(str((dt.datetime.now() - a).total_seconds()))
	if model == 'BlackScholes':
		new = dict(zip(['sigma'], res.x))
	elif model == 'Merton':
		new = dict(zip(['sigma', 'm', 'v', 'lam'], res.x))
	elif model == 'Heston':
		new = dict(zip(['rho', 'kappa', 'theta', 'xi'], res.x))

	tex_out = tabulate(new.items(), headers=['Parameter', 'Value'], tablefmt='latex')

	with open(out_path + '/Params_' + model + '.tex', 'w') as f:
		f.write(tex_out)

	new.update({'N' : 7, 'dt' : 7/365, 'M' : 100})
	optim_params = get_params(new)
	optim = Optimize(optim_params, optim_data)

	b = dt.datetime.now()
	opt_res = optim.Run(model, optim_data, 4, solver='GUROBI', integer=True, gamma=4, optim='Utility')
	print((dt.datetime.now() - b).total_seconds())

	fit_combo = OptionStrat(opt_res, optim_params)

	tex_out = fit_combo.describ2e()

	with open(out_path + '/OptCombo' + model + '.tex', 'w') as f:
		f.write(tex_out)

	plot_params = {'title' : "Option Combination Optimization Plot: " + model, 'file' : out_path + '/Plot_' + model + '.png'}
	fit_combo.plot_profit(plot_params)

test_eval = Opt_Eval(get_params({'M': 10000, 'dt' :1/365}), opt_res, optim_data)
test_eval.get_combo({'Legs':['lc'], 'Underlying_Price':100, 'Strikes':[110], 'T':7/365, 'Last':[1]})
risk = test_eval.total_risk(model, 200)
tab_risk = {key: {'Mean':round(val.mean(), 3), 'Std': round(val.std(), 3)} for key, val in risk.items()}
print(tabulate(pd.DataFrame.from_dict(tab_risk).T, headers=['Risk', 'Mean', 'Std'], tablefmt='prettytable'))

################################################################################
# Import Libraries and Scripts
################################################################################

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cvxpy as cp
from cvxpy.settings import CPLEX
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm
from Params import get_params, vals
from DataPrep import get_data, rhood_daily, yahoo_daily, fit_optim_test
import datetime as dt
from Returns import Returns
from Optimizer import Optimize
from Option_Visual import Option, OptionStrat
from Evaluation import Opt_Eval
import xpress
import gurobipy
from EmpFit import Fitter
from tabulate import tabulate
import random
plt.style.use('seaborn')

################################################################################
# Set up folder structure
################################################################################

base_path = '/Users/andrewboomer/Desktop/M2_Courses/Thesis/Options/tradingbot'
code_path = base_path + '/OptionsOptimization/Code'
out_path = base_path + '/OptionsOptimization/Output'
report_path = base_path + '/OptionsOptimization/Report'

################################################################################
# Get Fitting, Optimization, and Testing Data
################################################################################

data = get_data(get_params({}))
fit_size = 10
min_dt = 7/365

fit_data, optim_data, test_data = fit_optim_test(data, fit_size, min_dt)

################################################################################
# Get Returns Simulations and Plot
################################################################################

curr_row = data[data.index == 0].to_dict(orient='records')[0]
sim_params = get_params({'N' : 200, 'M' : 1000})
obj = Returns(sim_params)

bs_sim = obj.simulate('BlackScholes', curr_row)
merton_sim = obj.simulate('Merton', curr_row)
heston_sim = obj.simulate('Heston', curr_row)

plt.plot(bs_sim)
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Black Scholes Process')
plt.savefig(out_path + "/BlackScholesSim.png")
plt.show()

plt.plot(merton_sim)
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Merton Jump Diffusion Process')
plt.savefig(out_path + "/MertonSim.png")
plt.show()

plt.plot(heston_sim)
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Heston Stochastic Volatility Process')
plt.savefig(out_path + "/HestonSim.png")
plt.show()

strikes = np.linspace(0.75*curr_row[vals['stock_col']], 1.25*curr_row[vals['stock_col']], 200)
bs_prices = obj.monte_price('BlackScholes', strikes, curr_row)
merton_prices = obj.monte_price('Merton', strikes, curr_row)
heston_prices = obj.monte_price('Heston', strikes, curr_row)

plt.plot(strikes, bs_prices['call'].values(), label='BS Call Value')
plt.plot(strikes, merton_prices['call'].values(), label='Merton Call Value')
plt.plot(strikes, heston_prices['call'].values(), label='Heston Call Value')
plt.axvline(x=curr_row[vals['stock_col']], label='Current Price', color='black',linestyle='dashed')
plt.legend()
plt.title("Comparison of Simulated Model Prices")
plt.savefig()
plt.show()

################################################################################
# Fit Parameters and Optimize
################################################################################

models = ['BlackScholes']
min_fit = 'SLSQP'

fit_res = dict(zip(models, ["" for i in models]))
optim_params = dict(zip(models, ["" for i in models]))

for model in models:

	fit_params = get_params({'M' : 500})
	fit_obj = Fitter(fit_params)
	x0 = list(fit_params['x0'][model].values())
	bounds = list(fit_params['bounds'][model].values())
	fit_res[model] = fit_obj.fit(min_fit, fit_data, model, x0, bounds, 5)
	means = [np.mean(k) for k in zip(*fit_res[model])]

	new = dict(zip(list(fit_params['x0'][model].keys()), means))

	tex_out = tabulate(new.items(), headers=['Parameter', 'Value'], tablefmt='latex')

	with open(out_path + '/Params_' + model + '.tex', 'w') as f:
		f.write(tex_out)

	new.update({'N' : 7, 'dt' : 7/365, 'M' : 50})
	optim_params[model] = get_params(new)

models = ['BlackScholes']
optim_res = dict(zip(models, ["" for i in models]))
combos = dict(zip(models, ["" for i in models]))
evals = dict(zip(models, ["" for i in models]))

for model in models:
	print("Running " + model)
	for max_legs in [1, 2, 3, 4]:
		print("With maximum legs in combo = " + str(max_legs))

		optim = Optimize(optim_params[model], optim_data)
		optim_res[model] = optim.Run(model, optim_data, max_legs, solver='GUROBI', integer=True, gamma=2, optim='Utility')

		combos[model] = OptionStrat(optim_res[model], optim_params[model])
		tex_out = combos[model].describe()
		with open(out_path + '/OptCombo' + model + "Legs_" + str(max_legs) + '.tex', 'w') as f:
			f.write(tex_out)

		plot_params = {'title' : "Option Combination Optimization Plot: " + model, 'file' : out_path + '/Plot_' + model + "Legs_" + str(max_legs) + '.png'}
		combos[model].plot_profit(plot_params)

		evals[model] = Opt_Eval(get_params({'M': 10000, 'dt' : 7/365}), optim_res[model], optim_data)
		evals[model].get_combo(combos[model].get_combo())
		evals[model].combo_return(test_data)
		print("Total Profit = " + str(evals[model].combo['Profit'].sum()))
	# risk = evals[model].total_risk(model, 200)
	# tab_risk = {key: {'Mean':round(val.mean(), 3), 'Std': round(val.std(), 3)} for key, val in risk.items()}
	# print(tabulate(pd.DataFrame.from_dict(tab_risk).T, headers=['Risk', 'Mean', 'Std'], tablefmt='prettytable'))

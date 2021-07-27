################################################################################
# Import Libraries and Scripts
################################################################################

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
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
import gurobipy as gp
from EmpFit import Fitter
from tabulate import tabulate
import random
from collections import defaultdict
from itertools import product, chain
from noisyopt import minimizeCompass
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
fit_size = 5
min_dt = 7/365

fit_data, optim_data, test_data = fit_optim_test(data, fit_size, min_dt)
while optim_data.shape[0] != test_data.shape[0]:
	fit_data, optim_data, test_data = fit_optim_test(data, fit_size, min_dt)
if optim_data.shape[0] == test_data.shape[0]:
	del data

################################################################################
# Get Returns Simulations and Plot
################################################################################

curr_row = data[data.index == 0].to_dict(orient='records')[0]
sim_params = get_params({'N' : 1, 'M' : 1000})
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

min_fit = 'trust-constr'
models = ['BlackScholes', 'Merton', 'Heston']
leg_list = [1, 2]
optim_sims = 4
fit_sims = 4
risk_sims = 200
paths = 75
gamma = 3
fit_params = get_params({'M' : 5000})
fit_obj = Fitter(fit_params)
eval_params = get_params({'M': 500, 'dt' : min_dt})

res_df = pd.DataFrame(models, columns=['Model'])
res_df['x0'] = res_df.apply(lambda x: fit_params['x0'][x['Model']], axis=1)
res_df['ParamNames'] = res_df.apply(lambda x: list(x['x0'].keys()), axis=1)
res_df['bounds'] = res_df.apply(lambda x: list(fit_params['bounds'][x['Model']].values()), axis=1)
res_df['FitResults'] = res_df.apply(lambda x: fit_obj.fit(min_fit, fit_data, x['Model'], list(x['x0'].values()), x['bounds'], fit_sims), axis=1)
res_df['FitMeans'] = res_df.apply(lambda x: [np.mean(k) for k in zip(*x['FitResults'])], axis=1)
res_df['FitStd'] = res_df.apply(lambda x: [np.std(k) for k in zip(*x['FitResults'])], axis=1)

fit_out = res_df.groupby('Model', as_index=False)[['ParamNames', 'FitMeans', 'FitStd']].first()
for index, row in fit_out.iterrows():
	tex_out = tabulate(list(zip(row['ParamNames'], row['FitMeans'], row['FitStd'])), headers=['Parameter', 'Mean', 'Std'], tablefmt='latex',
		floatfmt='.2f')

	with open(out_path + '/Params_' + row['Model'] + '.tex', 'w') as f:
		f.write(tex_out)

res_df['OptParams'] = res_df.apply(lambda x: get_params(dict(chain.from_iterable(d.items() for d in (x['x0'], {'N' : 365*min_dt, 'dt' : min_dt, 'M' : paths})))), axis=1)
res_df['Optimizer'] = res_df.apply(lambda x: Optimize(x['OptParams'], optim_data), axis=1)

leg_df = pd.DataFrame(leg_list, columns=['Max_Legs'])
leg_df['tmp'] = 1
res_df['tmp'] = 1
res_df = res_df.merge(leg_df, on='tmp', how='outer')
del leg_df

sim_df = pd.DataFrame(list(range(optim_sims)), columns=['SimNumber'])
sim_df['tmp'] = 1
res_df = res_df.merge(sim_df, on='tmp', how='outer')
del sim_df
res_df = res_df.drop('tmp', axis=1)

res_df['Results'] = res_df.apply(lambda x: x['Optimizer'].Run(x['Model'], optim_data, x['Max_Legs'], gamma=gamma), axis=1)
res_df['OptionStrat'] = res_df.apply(lambda x: OptionStrat(x['Results'], x['OptParams']), axis=1)
res_df['Opt_Eval'] = res_df.apply(lambda x: Opt_Eval(eval_params, x['Results']), axis=1)
for index, row in res_df.iterrows():
	row['Opt_Eval'].get_combo(row['OptionStrat'].get_combo())
	row['Opt_Eval'].combo_return(test_data)

res_df['Profit'] = res_df.apply(lambda x: x['Opt_Eval'].combo['Profit'], axis=1)
res_df['Return'] = res_df.apply(lambda x: x['Opt_Eval'].combo['Return'], axis=1)

sharpe_out = tabulate(res_df.groupby(['Model', 'Max_Legs'], as_index=False)['Return'].apply(lambda x: round(np.mean(x - 0.03)/np.std(x), 2)).values, headers=['Model', 'Max_Legs', 'Sharpe'], tablefmt='latex')

with open(out_path + '/Sharpe.tex', 'w') as f:
		f.write(sharpe_out)

res_df['Bounds'] = res_df['OptionStrat'].apply(lambda x: x.bounds())
res_df['LeftBound'] = res_df['Bounds'].apply(lambda x: x[0])
res_df['RightBound'] = res_df['Bounds'].apply(lambda x: x[1])

res_df['LeftBound'].hist(weights = np.ones_like(res_df.index) / len(res_df.index), label='Left Bound')
res_df['RightBound'].hist(weights = np.ones_like(res_df.index) / len(res_df.index), label='Right Bound')
plt.legend()
plt.show()

contracts = pd.concat([pd.DataFrame(res_df['Results'][i]) for i in range(len(res_df['Results']))], axis=0, ignore_index=True)[['Strike', 'Type', 'Results']]
sides = ['Short']*contracts.shape[0] + ['Long']*contracts.shape[0]
contracts = pd.concat([contracts, contracts], axis=0, ignore_index=True)
contracts['Side'] = sides
contracts['Num'] = np.where(((contracts['Results'] == -1)&(contracts['Side'] == 'Short'))|((contracts['Results'] == 1)&(contracts['Side'] == 'Long')), 1, np.nan)
contracts = contracts.groupby(['Strike', 'Type', 'Side'], as_index=False)['Num'].sum()
contracts['contract'] = contracts['Side'] + "_" + contracts['Type']

g = sns.FacetGrid(contracts, col="contract", col_wrap=2)
g.map(sns.lineplot, "Strike", "Num", alpha=.7)
g.add_legend()
plt.show()

res_df['Delta'] = res_df.apply(lambda x: x['Opt_Eval'].total_risk(x['Model'], risk_sims, 'Delta'), axis=1)
res_df['Gamma'] = res_df.apply(lambda x: x['Opt_Eval'].total_risk(x['Model'], risk_sims, 'Gamma'), axis=1)
res_df['Theta'] = res_df.apply(lambda x: x['Opt_Eval'].total_risk(x['Model'], risk_sims, 'Theta'), axis=1)

risk = pd.concat([res_df.groupby(['Model', 'Max_Legs'])[r].apply(lambda x: np.concatenate(x)) for r in ['Delta', 'Gamma', 'Theta']], axis=1)
risk = risk.loc[:, ~risk.columns.duplicated()]

risk['Delta_Mean'] = risk['Delta'].apply(lambda x: x.mean())
risk['Delta_Std'] = risk['Delta'].apply(lambda x: x.std())

risk['Gamma_Mean'] = risk['Gamma'].apply(lambda x: x.mean())
risk['Gamma_Std'] = risk['Gamma'].apply(lambda x: x.std())

risk['Theta_Mean'] = risk['Theta'].apply(lambda x: x.mean())
risk['Theta_Std'] = risk['Theta'].apply(lambda x: x.std())

risk_out = risk[[i for i in risk.columns if ('Mean' in i)|('Std' in i)]]

risk_out.columns = risk_out.columns.str.split('_', expand=True)
risk_out = risk_out.stack(0).rename_axis(('Model', 'Max_Legs', 'Risk')).reset_index()

risk_out = tabulate(risk_out.values, headers=risk_out.columns, tablefmt='prettytable', floatfmt='.2f')

with open(out_path + '/TotalRisk.tex', 'w') as f:
		f.write(risk_out)

# tex_out = combos[model][leg_str].describe()
# with open(out_path + '/OptCombo' + model + "Legs_" + str(max_legs) + '.tex', 'w') as f:
# 	f.write(tex_out)

plot_params = {'title' : "Option Combination Optimization Plot: " + model, 'file' : out_path + '/Plot_' + model + "Legs_" + str(max_legs) + '.png'}
combos[model][leg_str].plot_profit(plot_params)

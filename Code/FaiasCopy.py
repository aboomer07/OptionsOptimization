################################################################################
# Import Libraries and Scripts
################################################################################

import os
import sys
import numpy as np
import pandas as pd
from arch import arch_model
import yfinance
from scipy.optimize import minimize, fmin
from scipy.stats import skew, kurtosis
from statsmodels.api import tsa, stats
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import calendar
import cvxpy as cp
from tabulate import tabulate
import mosek
from numba import jit, float64, int64

################################################################################
# Import Libraries and Scripts
################################################################################

base_path = '/Users/andrewboomer/Desktop/M2_Courses/Thesis/Options/tradingbot'
data_path = base_path + '/Data'
code_path = base_path + '/OptionsOptimization/Code'
out_path = base_path + '/OptionsOptimization'
report_path = base_path + '/OptionsOptimization/Report'

################################################################################
# Import Data
################################################################################

# cboe = pd.concat([pd.read_csv(os.path.join(data_path, f)) for f in os.listdir(data_path) if 'Underlying' in f])
# cboe = cboe.rename(columns={'underlying_symbol' : 'Ticker', 'quote_date' : 'Date', 'expiration' : 'Expiry', 'strike' : 'Strike', 'trade_volume' : 'Volume'})
# cboe['Date'] = pd.to_datetime(cboe['Date'], format='%Y-%m-%d')
# cboe['Expiry'] = pd.to_datetime(cboe['Expiry'], format='%Y-%m-%d')
# cboe['T'] = (cboe['Expiry'] - cboe['Date']).dt.days

# cboe_trim = cboe[cboe['Volume'] != 0]
# cboe_trim = cboe_trim[cboe_trim['bid_eod'] >= 0.125]
# cboe_trim = cboe_trim[cboe_trim['bid_eod'] <= cboe_trim['ask_eod']]

# def is_third_friday(s):
#     return((s.weekday() == 4)&(15 <= s.day)&(s.day <= 21))

# expiries = cboe_trim['Expiry'].unique()
# expiries = pd.to_datetime(expiries)
# exp_dict = {x : is_third_friday(x) for x in expiries}
# cboe_trim['ThirdFri'] = cboe_trim['Expiry'].map(exp_dict)
# cboe_trim['ThirdFri'] = np.where(cboe_trim.Expiry == '2019-04-18', True, cboe_trim['ThirdFri'])
# cboe_trim = cboe_trim[cboe_trim['ThirdFri'] == True]

# cboe_trim = cboe_trim[cboe_trim['T'].isin([28, 29, 30, 31])]
# cboe_trim = cboe_trim[((cboe_trim['T'].isin([28, 29]))&(cboe_trim['Date'].dt.month == 2))|((cboe_trim['T'] == 30)&(cboe_trim['Date'].dt.month.isin([4, 6, 9, 11])))|((cboe_trim['T'] == 31)&(cboe_trim['Date'].dt.month.isin([1, 3, 5, 7, 8, 10, 12])))]
# cboe_trim['Leap'] = cboe_trim.Date.apply(lambda x: calendar.isleap(x.year))
# cboe_trim = cboe_trim[~((cboe_trim['T'] == 29)&(cboe_trim['Leap'] == False))]
# cboe_trim = cboe_trim[~((cboe_trim['T'] == 28)&(cboe_trim['Leap'] == True))]
# cboe_trim = cboe_trim[cboe_trim.root == 'SPX']
# cboe_trim.to_csv(data_path + '/CBOE_Trimmed.csv', index=False)

cboe = pd.read_csv(data_path + '/CBOE_Trimmed.csv')
cboe['Date'] = pd.to_datetime(cboe['Date'], format='%Y-%m-%d')
cboe['Expiry'] = pd.to_datetime(cboe['Expiry'], format='%Y-%m-%d')

shill = pd.read_excel("http://www.econ.yale.edu/~shiller/data/ie_data.xls", sheet_name='Data', skiprows=7)
shill = shill.drop(shill.index[-1], axis=0)
shill = shill[['Date', 'P']]
shill.Date = pd.to_datetime(shill.Date.astype(str).str.ljust(7, '0'), format='%Y.%m')
shill['P'] = shill.P.astype(float)
shill = shill.rename(columns={'P' : 'Price'})

################################################################################
# Clean Data to Get Log Returns
################################################################################

shill['RR'] = np.log(shill.Price).diff()
returns = shill[(shill['Date'].dt.year >= 1950)]

yT = returns[returns.Date.dt.year<=2018].RR

am = arch_model(yT, vol='Garch', mean='zero')
am_fit = am.fit()
print(am_fit.summary().as_latex())

horizon = returns[returns.Date.dt.year > 2018].shape[0]
n = len(am_fit.resid)

condvol = np.array(list(am_fit.conditional_volatility) + [0]*horizon)
yT = returns.RR.values

omega = am_fit.params['omega']
alpha = am_fit.params['alpha[1]']
beta = am_fit.params['beta[1]']

# set.seed(1230910)
for i in range(horizon):
	condvol[n+i] = np.sqrt(omega+(alpha*(yT[n+i-1]**2))+(beta*(condvol[n+i-1]**2)))

returns.loc[:, 'SDEV'] = condvol
returns['CondVar'] = returns['SDEV']**2
returns['SR'] = returns['RR'] / returns['SDEV']
returns = returns.reset_index().drop('index', axis=1)

################################################################################
# Plot Log Returns and Variance from GARCH
################################################################################

fig, axes = plt.subplots(2, 1)
axes[0].plot(returns.Date, returns.RR, label='Log Returns $y_{t}$')
axes[1].plot(returns.Date, returns.CondVar, label=r'$\hat{\sigma}_{t}^{2}$')
axes[1].axvline(dt.datetime(2019, 1, 1), color='black', linestyle='dotted', label='Forecast Date')
axes[0].legend()
axes[1].legend()
fig.suptitle('Log Returns and Conditional Variance')
plt.savefig(out_path + '/GARCH_TS_plot.png')
plt.show()

################################################################################
# Replicate Table 1 Faias Santa Clara
################################################################################

def get_desc(data):
	data = data.copy(deep=True)
	desc = {}
	desc['Nobs'] = data.shape[0]
	desc['StartYear'] = data.Date.min().year
	desc['EndDate'] = data.Date.max().year
	desc['Skewness'] = {}
	desc['Skewness']['RR'] = skew(data.RR)
	desc['Skewness']['SR'] = skew(data.SR)

	desc['Kurtosis'] = {}
	desc['Kurtosis']['RR'] = kurtosis(data.RR)
	desc['Kurtosis']['SR'] = kurtosis(data.SR)

	desc['Auto'] = {}
	desc['Auto']['RR'] = tsa.acf(data.RR, nlags=5, fft=False)[1]
	desc['Auto']['SR'] = tsa.acf(data.SR, nlags=5, fft=False)[1]

	desc['SqAuto'] = {}
	desc['SqAuto']['RR'] = tsa.acf(data.RR**2, nlags=5, fft=False)[1]
	desc['SqAuto']['SR'] = tsa.acf(data.SR**2, nlags=5, fft=False)[1]

	desc['Llung_Box'] = {}
	desc['Llung_Box']['RR'] = stats.acorr_ljungbox(data.RR, lags=[1], return_df=True).values[0]
	desc['Llung_Box']['SR'] = stats.acorr_ljungbox(data.SR, lags=[1], return_df=True).values[0]

	desc['ARCH'] = {}
	desc['ARCH']['RR'] = stats.het_arch(data.RR, nlags=1)[2:]
	desc['ARCH']['SR'] = stats.het_arch(data.SR, nlags=1)[2:]

	return(desc)

def get_tab_list(descs):
	tab = []
	tab.extend([desc['Nobs'] for desc in descs] * 2)

	tab.extend([round(desc['Skewness']['RR'], 2) for desc in descs])
	tab.extend([round(desc['Skewness']['SR'], 2) for desc in descs])

	tab.extend([round(desc['Kurtosis']['RR'], 2) for desc in descs])
	tab.extend([round(desc['Kurtosis']['SR'], 2) for desc in descs])

	tab.extend([round(desc['Auto']['RR'], 2) for desc in descs])
	tab.extend([round(desc['Auto']['SR'], 2) for desc in descs])

	tab.extend([round(desc['SqAuto']['RR'], 2) for desc in descs])
	tab.extend([round(desc['SqAuto']['SR'], 2) for desc in descs])

	tab.extend([round(desc['Llung_Box']['RR'][0], 2) for desc in descs])
	tab.extend([round(desc['Llung_Box']['SR'][0], 2) for desc in descs])

	tab.extend([round(desc['Llung_Box']['RR'][1], 2) for desc in descs])
	tab.extend([round(desc['Llung_Box']['SR'][1], 2) for desc in descs])

	tab.extend([round(desc['ARCH']['RR'][0], 2) for desc in descs])
	tab.extend([round(desc['ARCH']['SR'][0], 2) for desc in descs])

	tab.extend([round(desc['ARCH']['RR'][1], 2) for desc in descs])
	tab.extend([round(desc['ARCH']['SR'][1], 2) for desc in descs])

	return(tuple(tab))

def get_latex(tab):
	string = """\\begin{tabular}{lccccccc} 
	& \\multicolumn{3}{c} { Log Returns $y_{t}$} & & \\multicolumn{3}{c} { Standardized Returns $\\hat{\epsilon}_{t}$} \\\\
	\\cline { 2 - 4 } \\cline { 6 - 8 } Statistics & $1950-2018$ & $2019-2021$ & $1950-2021$ & & $1950-2018$ & $2019-2021$ & $1950-2021$ \\\\
	\\hline

	No. of obs. & %s & %s & %s & & %s & %s & %s \\\\
	Skewness & $%s$ & $%s$ & $%s$ & & $%s$ & $%s$ & $%s$ \\\\
	Excess kurtosis & $%s$ & $%s$ & $%s$ & & $%s$ & $%s$ & $%s$ \\\\
	$\\rho_{1}(z)$ & $%s$ & $%s$ & $%s$ & & $%s$ & $%s$ & $%s$ \\\\
	$\\rho_{1}\\left(z^{2}\\right)$ & $%s$ & $%s$ & $%s$ & & $%s$ & $%s$ & $%s$ \\\\
	$Q_{1}(z)$ & $%s$ & $%s$ & $%s$ & & $%s$ & $%s$ & $%s$ \\\\
	& {$[%s]$} & {$[%s]$} & {$[%s]$} & & {$[%s]$} & {$[%s]$} & {$[%s]$} \\\\
	ARCH(1) & $%s$ & $%s$ & $%s$ & & $%s$ & $%s$ & $%s$ \\\\
	& {$[%s]$} & {$[%s]$} & {$[%s]$} & & {$[%s]$} & {$[%s]$} & {$[%s]$} \\\\
	\\hline
	\\end{tabular}"""

	string = string%tab
	return(string)

descs = [get_desc(returns[(returns.Date.dt.year <= 2018)])]
descs.append(get_desc(returns[(returns.Date.dt.year >= 2019)]))
descs.append(get_desc(returns))
tab = get_tab_list(descs)
tex = get_latex(tab)

################################################################################
# Summarize Option Data
################################################################################

col_dict = {'underlying_symbol' : 'Asset Symbol', 'quote_date' : 'Quote Time',
'expiration' : 'Contract Expiration Date', 'strike' : 'Contract Strike Price', 
'option_type' : 'Contract Type: Call/Put', 'open' : 'Option Price at Open',
'close' : "Option Price at Close", 'high' : "High Option Price", 
'low' : "Low Option Price", 'Volume' : 'Contract Trading Volume'}

print(tabulate(col_dict.items(), headers=['Col Name', 'Description'], tablefmt='latex'))

count = cboe.groupby(['Date', 'option_type'])['Strike'].count().unstack(1).reset_index()
count['Date'] = pd.to_datetime(count['Date'], format='%Y-%m-%d')

fig, ax = plt.subplots(1)
count.plot('Date', 'C', label='Calls', ax=ax)
count.plot('Date', 'P', label='Puts', ax=ax)
ax.set_ylabel('Count')
plt.legend()
plt.title('Count of Contracts by Date and Type')
plt.savefig(out_path + '/Contract_Count.png')
plt.close()
# plt.show()

################################################################################
# Step 1 and 2 Faias Santa Clara
################################################################################

N = 10000
rem = returns[returns.Date.dt.year >= 2019].shape[0]
SimRet = np.empty(shape=(rem, N))
SR = returns.SR.values
for i in range(returns.shape[0]):
	if returns.loc[returns.index[i]].Date.year >= 2019:
		SimRet[i-n, :] = np.random.choice(SR[0:i-1], N)


adj_ret = np.multiply(condvol[n:].reshape(rem, 1), SimRet)
St = returns[returns.Date >= '2018-12-01'][:-1].Price.values.reshape((rem, 1))
ST = np.multiply(St, np.exp(adj_ret))

################################################################################
# Step 3 Faias Santa Clara
################################################################################

opts = cboe[['Date', 'Expiry', 'Strike', 'option_type', 'open', 'close', 'bid_eod', 'ask_eod', 'Volume']].sort_values(by='Date')
# opts = opts[opts.Volume > 200]
opts = opts.groupby(['Date', 'Expiry', 'Strike', 'option_type'], as_index=False)[['open', 'close', 'bid_eod', 'ask_eod', 'Volume']].mean()
opts['ExpYM'] = opts.Expiry.dt.year.astype(str) + opts.Expiry.dt.month.astype(str).str.rjust(2, '0')
fcst_dates = [str(x.year) + str(x.month).rjust(2, '0') for x in returns[returns.Date.dt.year > 2018].Date]
SR_dict = {fcst_dates[i] : SimRet[i, :] for i in range(rem)}
ST_dict = {fcst_dates[i] : ST[i, :] for i in range(rem)}
opts['SimRet'] = opts['ExpYM'].map(ST_dict)
opts['SimPrices'] = opts['ExpYM'].map(ST_dict)
opts = opts.reset_index().drop('index', axis=1)
opts['Payoff'] = np.where(opts['option_type'] == 'C', opts.SimPrices - opts.Strike, opts.Strike - opts.SimPrices)
opts['Payoff'] = opts['Payoff'].apply(lambda x: np.maximum(x, 0))
returns['YM'] = returns.Date.dt.year.astype(str) + returns.Date.dt.month.astype(str).str.rjust(2, '0')
price_dict = dict(zip(returns.YM, returns.Price))
opts['ActPrice'] = opts['ExpYM'].map(price_dict)
opts = opts.sort_values(by=['option_type', 'Date', 'Strike']).reset_index().drop('index', axis=1)
opts['Moneyness'] = opts['ActPrice'] / opts['Strike']
four_df = pd.concat([opts[opts['option_type'] == 'C'].groupby(['Date', 'Expiry'], as_index=False)['Moneyness'].apply(lambda x: opts.loc[abs(x - 0.95).idxmin(), ['Strike', 'option_type']]).assign(Region='OTM_Call'), opts[opts['option_type'] == 'C'].groupby(['Date', 'Expiry'], as_index=False)['Moneyness'].apply(lambda x: opts.loc[abs(x - 1).idxmin(), ['Strike', 'option_type']]).assign(Region='ATM_Call'), opts[opts['option_type'] == 'P'].groupby(['Date', 'Expiry'], as_index=False)['Moneyness'].apply(lambda x: opts.loc[abs(x - 1).idxmin(), ['Strike', 'option_type']]).assign(Region='ATM_Put'), opts[opts['option_type'] == 'P'].groupby(['Date', 'Expiry'], as_index=False)['Moneyness'].apply(lambda x: opts.loc[abs(x - 1.05).idxmin(), ['Strike', 'option_type']]).assign(Region='OTM_Put')], axis=0, ignore_index=True)
opts = opts.merge(four_df, how='left', on=['Date', 'Expiry', 'Strike', 'option_type'])
# opts = pd.concat([opts.assign(Side='Long'), opts.assign(Side='Short')], axis=0, ignore_index=True)

################################################################################
# Visualize Returns
################################################################################

fig, ax = plt.subplots(1)
sns.histplot(returns[returns.Date.dt.year < 2019].SR, ax=ax)
ax.set_xlabel('$\hat{\epsilon}_{t}^{n}$')
plt.title('Historic Standardized Returns 2019-01-15')
plt.savefig(out_path + "/SR_20190115.png")
plt.close()
# plt.show()

fig, ax = plt.subplots(1)
sns.histplot(returns[returns.Date.dt.year < 2019].RR, ax=ax)
ax.set_xlabel('$\hat{y}_{t}^{n}$')
plt.title('Historic Log Returns 2019-01-15')
plt.savefig(out_path + "/RR_20190115.png")
plt.close()
# plt.show()

fig, ax = plt.subplots(1)
sns.histplot(returns[returns.Date.dt.year < 2019].Price, ax=ax)
# ax.axvline(St[0], label='$S_{t}$', color='black', linestyle='dotted')
ax.set_xlabel('$S_{t}$')
plt.title('Historic Prices 2019-01-15')
ax.legend()
plt.savefig(out_path + "/SimPrice_20190115.png")
plt.close()
# plt.show()

################################################################################
# Step 4 Faias Santa Clara
################################################################################

tbill = pd.read_csv(data_path + '/Tbill.csv')
tbill['DATE'] = pd.to_datetime(tbill['DATE'], format='%Y-%m-%d')
tbill = tbill.rename(columns={'DATE' : 'Date', 'DGS1MO' : 'Rf'})
tbill['Rf'] = tbill['Rf'].apply(lambda x: float(x) if x != '.' else np.nan)
tbill['Rf'] = tbill['Rf'] / 100
opts = opts.merge(tbill, how='left', on='Date')

opts['OptRet'] = (opts['Payoff'] / opts['close']) - 1 - opts['Rf']
# opts['OptRet'] = np.where(opts['Side'] == 'Long', (opts['Payoff'] / opts['close']) - 1 - opts['Rf'], 1 + opts['Rf'] - (opts['Payoff'] / opts['close']))

################################################################################
# Step 5 Faias Santa Clara
################################################################################

@jit(float64(float64[::1], float64[:, ::1], float64, int64), nopython=True, parallel=True)
def scipy_optim(w, As, rf, gamma):
	# As = np.array(optret.tolist()).T
	# mat_var = As@w + rf + 1
	# if gamma > 1:
	gam = 1 - gamma
	utils = np.mean(np.power(np.dot(w, As) + rf + 1, gam)/(gam))
	# else:
		# utils = -np.sum(mat_var)/As.shape[0]

	return(-1*utils)

def jac(w, As, rf, gamma):

	jacob = np.power(np.dot(w, As) + rf + 1, -gamma)
	jacob = np.multiply(jacob, As)
	jacob = np.mean(jacob, axis=1)
	return(jacob)

def optim(optret, rf, gamma, solver):
	As = np.array(optret.tolist()).T
	# rf_arr = (np.ones(As.shape[0])*rf.values[0]).reshape((As.shape[0], 1))
	# As = np.concatenate([As, rf_arr], axis=1)

	w = cp.Variable(As.shape[1])
	# w.value = np.zeros(As.shape[1])
	mat_var = As@w + rf.values[0] + 1

	utils = cp.sum(cp.power(mat_var, (1-gamma))/(1-gamma))/As.shape[0]

	objective = cp.Maximize(utils)
	# constraints = [w >= -1, w <= 1, cp.sum(w) <= 1]
	constraints = []

	prob = cp.Problem(objective)
	result = prob.solve(solver=solver, verbose=True)

	return(w.value)

dates = opts.Date.unique()[:1]
df = opts[(opts.Date.isin(dates))&(opts.Region.notnull())]
# df = opts[(opts.Date.isin(dates))]
# df = df[(df.Moneyness >= 0.995)&(df.Moneyness <= 1.005)]
# df = df[df.index.isin(np.random.choice(df.index, 10))]

df = df.sort_values(by='Date')


weights = []
for date in dates:
	curr = df[df.Date == date]
	weights.append(optim(curr.OptRet, curr.Rf, 10, 'MOSEK'))
	# w = np.random.uniform(0, 1, curr.shape[0])
	# bounds = [(-100, 100) for i in range(len(w))]
	# res = minimize(scipy_optim, method='Powell', x0=w, args=(np.array(curr.OptRet.tolist()), curr.Rf.values[0], 10), bounds=bounds, tol=1e-50, options={"maxiter":2000, 'disp':True})
	# res = fmin(scipy_optim, x0=w, args=(np.array(curr.OptRet.tolist()), curr.Rf.values[0], 10), full_output=True, maxiter=2000)
	# weights.append(res[0])
	# weights.append(res.x)

df['Weights'] = np.concatenate([np.array(weight).reshape(len(weight), 1) for weight in weights])

################################################################################
# Step 6, 7 Faias Santa Clara
################################################################################

df['ActPay'] = np.where(df['option_type'] == 'C', np.maximum(df.ActPrice - df.Strike, 0), np.maximum(df.Strike - df.ActPrice, 0))
df['ActRet'] = (df['ActPay'] / df['close']) - 1 - df['Rf']
# df['ActRet'] = np.where(df['Side'] == 'Long', (df['ActPay'] / df['close']) - 1 - df['Rf'], 1 + df['Rf'] - (df['ActPay'] / df['close']))

################################################################################
# Step 8 Faias Santa Clara
################################################################################

df['TotAct'] = df['ActRet'] * df['Weights']
df['TotSim'] = (df['OptRet'] * df['Weights']).apply(lambda x: np.mean(x))

act = df.groupby(['Date', 'Expiry'])[['TotAct', 'Rf']].apply(lambda x: (x['Rf'].mean() + np.sum(x['TotAct']))).reset_index().rename(columns={0:'rp_t+1'})

################################################################################
# Plot the Weights
################################################################################

sns.lineplot(data=df[df.Date == '2019-01-15'], 
	x='Strike', y='Weights', hue='option_type')
plt.title('Optimization Weights 2019-01-15')
# plt.savefig(out_path + "/Weights_20190115.png", bbox_inches='tight')
# plt.close()
plt.show()


tbill[tbill.Date.dt.year > 2015].plot(x='Date', y='Rf')
plt.legend()
plt.title("1 Month Treasury Bill Return")
plt.savefig(out_path + '/Tbill_TS.png')
plt.show()



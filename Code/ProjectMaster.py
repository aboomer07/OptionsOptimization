################################################################################
# Import Libraries and Scripts
################################################################################

print('#################################')
print('Importing Libraries')
print('#################################')

import os
import sys
import numpy as np
import pandas as pd
from arch import arch_model
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
import matplotlib.ticker as mtick
from matplotlib.font_manager import FontProperties
from itertools import chain

################################################################################
# Define directory paths
################################################################################

base_path = '/Users/andrewboomer/Desktop/M2_Courses/Thesis/Options/tradingbot'
data_path = base_path + '/Data'
code_path = base_path + '/OptionsOptimization/Code'
out_path = base_path + '/OptionsOptimization'
report_path = base_path + '/OptionsOptimization/Report'
tex_out = out_path + '/Report/TexFiles'
graph_out = out_path + '/Report/Graphs'

################################################################################
# Define parameters
################################################################################

global incl_side
global show_plots
global import_clean
global refit
global N
global four
global cut
global covid
global detail_out
global plt_pct_fmt

incl_side = True
show_plots = False
import_clean = True
refit = True
N = 3000
four = True
cut = np.inf
covid = False
detail_out = False
plt_pct_fmt = mtick.PercentFormatter(xmax=1, symbol='')

################################################################################
# Import Data
################################################################################

print('#################################')
print('Importing/Cleaning Data')
print('#################################')

if not import_clean:
	cboe = pd.concat([pd.read_csv(os.path.join(data_path, f)) for f in os.listdir(data_path) if 'Underlying' in f])
	cboe = cboe.rename(columns={'underlying_symbol' : 'Ticker', 'quote_date' : 'Date', 'expiration' : 'Expiry', 'strike' : 'Strike', 'trade_volume' : 'Volume'})
	cboe['Date'] = pd.to_datetime(cboe['Date'], format='%Y-%m-%d')
	cboe['Expiry'] = pd.to_datetime(cboe['Expiry'], format='%Y-%m-%d')
	cboe['T'] = (cboe['Expiry'] - cboe['Date']).dt.days

	cboe_trim = cboe[cboe['Volume'] != 0]
	cboe_trim = cboe_trim[cboe_trim['bid_eod'] >= 0.125]
	cboe_trim = cboe_trim[cboe_trim['bid_eod'] <= cboe_trim['ask_eod']]

	def is_third_friday(s):
	    return((s.weekday() == 4)&(15 <= s.day)&(s.day <= 21))

	expiries = cboe_trim['Expiry'].unique()
	expiries = pd.to_datetime(expiries)
	exp_dict = {x : is_third_friday(x) for x in expiries}
	cboe_trim['ThirdFri'] = cboe_trim['Expiry'].map(exp_dict)
	cboe_trim['ThirdFri'] = np.where(cboe_trim.Expiry == '2019-04-18', True, cboe_trim['ThirdFri'])
	cboe_trim = cboe_trim[cboe_trim['ThirdFri'] == True]

	cboe_trim = cboe_trim[cboe_trim['T'].isin([28, 29, 30, 31])]
	cboe_trim = cboe_trim[((cboe_trim['T'].isin([28, 29]))&(cboe_trim['Date'].dt.month == 2))|((cboe_trim['T'] == 30)&(cboe_trim['Date'].dt.month.isin([4, 6, 9, 11])))|((cboe_trim['T'] == 31)&(cboe_trim['Date'].dt.month.isin([1, 3, 5, 7, 8, 10, 12])))]
	cboe_trim['Leap'] = cboe_trim.Date.apply(lambda x: calendar.isleap(x.year))
	cboe_trim = cboe_trim[~((cboe_trim['T'] == 29)&(cboe_trim['Leap'] == False))]
	cboe_trim = cboe_trim[~((cboe_trim['T'] == 28)&(cboe_trim['Leap'] == True))]
	cboe_trim = cboe_trim[cboe_trim.root == 'SPX']
	cboe_trim.to_csv(data_path + '/CBOE_Trimmed.csv', index=False)

else:
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

print('#################################')
print('Running GARCH Model')
print('#################################')

garch_sum = """\\begin{center}
\\begin{tabular}{lcccc}
	& \\textbf{coef} & \\textbf{std err} & \\textbf{t} & \\textbf{P$>|$t$|$} \\\\
\\midrule
$\\mathbf{\\omega}$ & %s & %s & %s & %s \\\\
$\\mathbf{\\alpha}$ & %s & %s & %s & %s \\\\
$\\mathbf{\\beta}$ & %s & %s & %s & %s \\\\
\\bottomrule
\\end{tabular}
\\end{center}"""

shill['RR'] = np.log(shill.Price).diff()
returns = shill[(shill['Date'].dt.year >= 1950)]

yT = returns[returns.Date.dt.year<=2018].RR

am = arch_model(yT, vol='Garch', mean='zero')
am_fit = am.fit()

params = ["{:.3f}".format(i) for i in am_fit.params]
std_err = ["{:.3f}".format(i) for i in am_fit.std_err]
tvals = ["{:.3f}".format(i) for i in am_fit.tvalues]
pvals = ["{:.3f}".format(i) for i in am_fit.pvalues]
sub_vals = tuple(chain(*zip(params, std_err, tvals, pvals)))
mod_str = garch_sum%sub_vals

with open(tex_out + '/ARCH_Model.tex', "w") as text_file:
    text_file.write(mod_str)

horizon = returns[returns.Date.dt.year > 2018].shape[0]
n = len(am_fit.resid)

condvol = np.array(list(am_fit.conditional_volatility) + [0]*horizon)
yT = returns.RR.values

omega = am_fit.params['omega']
alpha = am_fit.params['alpha[1]']
beta = am_fit.params['beta[1]']

for i in range(horizon):
	if (i != 0)&(refit):
		fit = arch_model(returns[:(n+i)].RR, vol='Garch', mean='zero').fit()
		omega = fit.params['omega']
		alpha = fit.params['alpha[1]']
		beta = fit.params['beta[1]']

	condvol[n+i] = np.sqrt(omega+(alpha*(yT[n+i-1]**2))+(beta*(condvol[n+i-1]**2)))

returns.loc[:, 'SDEV'] = condvol
returns['CondVar'] = returns['SDEV']**2
returns['SR'] = returns['RR'] / returns['SDEV']
returns = returns.reset_index().drop('index', axis=1)

################################################################################
# Plot Log Returns and Variance from GARCH
################################################################################

print('#################################')
print('Plotting GARCH Model')
print('#################################')

fig, axes = plt.subplots(2, 1)
axes[0].plot(returns.Date, returns.RR, label='Log Returns $y_{t}$')
axes[1].plot(returns.Date, returns.CondVar, label=r'$\hat{\sigma}_{t}^{2}$')
axes[1].axvline(dt.datetime(2019, 1, 1), color='black', linestyle='dotted', label='Forecast Date')
axes[1].set_xlabel('Date')
axes[0].set_ylabel('Returns')
axes[1].set_ylabel('Volatility')
axes[0].legend()
axes[1].legend()
plt.tight_layout()

if show_plots:
	plt.show()
else:
	plt.savefig(graph_out + '/GARCH_TS_plot.png')
	plt.close()

################################################################################
# Replicate Table 1 Faias Santa Clara
################################################################################

print('#################################')
print('Describing SP500 Data')
print('#################################')

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
tab1_str = get_latex(tab)

with open(tex_out + '/Table1.tex', "w") as text_file:
    text_file.write(tab1_str)

################################################################################
# Visualize Returns
################################################################################

fig, ax = plt.subplots(1)
sns.histplot(returns[returns.Date.dt.year < 2019].SR, ax=ax)
ax.set_xlabel('$\epsilon_{t}$', fontdict={'size':20})
ax.set_ylabel('Frequency', fontdict={'size':20})
ax.xaxis.set_major_formatter(plt_pct_fmt)
ax.yaxis.set_ticklabels([])
ax.tick_params(axis='x', labelsize=15)
plt.title('Historic Standardized Returns 2019-01-15', fontsize='15')
plt.tight_layout()

if show_plots:
	plt.show()
else:
	plt.savefig(graph_out + "/SR_20190115.png")
	plt.close()

fig, ax = plt.subplots(1)
sns.histplot(returns[returns.Date.dt.year < 2019].RR, ax=ax)
ax.set_xlabel('$y_{t}$', fontdict={'size':20})
ax.set_ylabel('Frequency', fontdict={'size':20})
ax.xaxis.set_major_formatter(plt_pct_fmt)
ax.yaxis.set_ticklabels([])
ax.tick_params(axis='x', labelsize=15)
plt.title('Historic Log Returns 2019-01-15', fontsize='15')
plt.tight_layout()

if show_plots:
	plt.show()
else:
	plt.savefig(graph_out + "/RR_20190115.png")
	plt.close()

################################################################################
# Summarize Option Data
################################################################################

print('#################################')
print('Describing CBOE Data')
print('#################################')

col_dict = {'underlying_symbol' : 'Asset Symbol', 'quote_date' : 'Quote Time',
'expiration' : 'Contract Expiration Date', 'strike' : 'Contract Strike Price', 
'option_type' : 'Contract Type: Call/Put', 'open' : 'Option Price at Open',
'close' : "Option Price at Close", 'high' : "High Option Price", 
'low' : "Low Option Price", 'Volume' : 'Contract Trading Volume'}

cboe_str = tabulate(col_dict.items(), headers=['Variable Name', 'Description'], tablefmt='latex')

with open(tex_out + '/CBOE_Dict.tex', "w") as text_file:
    text_file.write(cboe_str)

count = cboe.groupby(['Date', 'option_type'])['Strike'].count().unstack(1).reset_index()
count['Date'] = pd.to_datetime(count['Date'], format='%Y-%m-%d')

fig, ax = plt.subplots(1)
count.plot('Date', 'C', label='Calls', ax=ax, ylim=[0, 1.1*count['C'].max()])
count.plot('Date', 'P', label='Puts', ax=ax, ylim=[0, 1.1*count['P'].max()])
ax.set_ylabel('Count')
plt.legend()
plt.tight_layout()

if show_plots:
	plt.show()
else:
	plt.savefig(graph_out + '/Contract_Count.png')
	plt.close()

################################################################################
# Step 1 and 2 Faias Santa Clara
################################################################################

print('#################################')
print('Creating Simluated Prices')
print('#################################')

rem = returns[returns.Date.dt.year >= 2019].shape[0]
SimRet = np.empty(shape=(rem - 1, N))
SR = returns.SR.values
for i in range(1, rem):
	SimRet[i - 1, :] = np.random.choice(SR[:(i + n)], N)


adj_ret = np.multiply(condvol[n + 1:].reshape(rem - 1, 1), SimRet)
St = returns[n:-1].Price.values.reshape((rem - 1, 1))
ST = np.multiply(St, np.exp(adj_ret))

################################################################################
# Step 3 Faias Santa Clara
################################################################################

print('#################################')
print('Constructing Options Dataframe')
print('#################################')

opts = cboe[['Date', 'Expiry', 'Strike', 'option_type', 'open', 'close', 'bid_eod', 'ask_eod', 'Volume']].sort_values(by='Date')
opts = opts.groupby(['Date', 'Expiry', 'Strike', 'option_type'], as_index=False)[['open', 'close', 'bid_eod', 'ask_eod', 'Volume']].mean()

opts['YM'] = opts.Date.dt.year.astype(str) + opts.Date.dt.month.astype(str).str.rjust(2, '0')
opts['ExpYM'] = opts.Expiry.dt.year.astype(str) + opts.Expiry.dt.month.astype(str).str.rjust(2, '0')
fcst_dates = [str(x.year) + str(x.month).rjust(2, '0') for x in returns[returns.Date.dt.year > 2018].Date]
SR_dict = {fcst_dates[i] : adj_ret[i, :] for i in range(rem-1)}
ST_dict = {fcst_dates[i] : ST[i, :] for i in range(rem-1)}
opts['SimRet'] = opts['YM'].map(SR_dict)
opts['SimPrices'] = opts['YM'].map(ST_dict)

opts = opts.reset_index().drop('index', axis=1)
opts['Payoff'] = np.where(opts['option_type'] == 'C', opts.SimPrices - opts.Strike, opts.Strike - opts.SimPrices)
opts['Payoff'] = opts['Payoff'].apply(lambda x: np.maximum(x, 0))
returns['YM'] = returns.Date.dt.year.astype(str) + returns.Date.dt.month.astype(str).str.rjust(2, '0')

price_dict = dict(zip(returns.YM, returns.Price))
opts['Price_t'] = opts['YM'].map(price_dict)
opts['Price_t+1'] = opts['ExpYM'].map(price_dict)
opts = opts.sort_values(by=['option_type', 'Date', 'Strike']).reset_index().drop('index', axis=1)
opts['Moneyness'] = (opts['Price_t'] / opts['Strike']) - 1

four_df = pd.concat([
	opts[(opts['option_type'] == 'C')&(opts['Moneyness'] >= -0.01)&(opts['Moneyness'] <= 0.01)].groupby(['Date', 'Expiry'], as_index=False)[['ask_eod', 'bid_eod']].apply(lambda x: opts.loc[abs(x['ask_eod'] - x['bid_eod']).idxmin(), ['Strike', 'option_type']]).assign(Region='ATM Call'),
	opts[(opts['option_type'] == 'P')&(opts['Moneyness'] >= -0.01)&(opts['Moneyness'] <= 0.01)].groupby(['Date', 'Expiry'], as_index=False)[['ask_eod', 'bid_eod']].apply(lambda x: opts.loc[abs(x['ask_eod'] - x['bid_eod']).idxmin(), ['Strike', 'option_type']]).assign(Region='ATM Put'), 
	opts[(opts['option_type'] == 'C')&(opts['Moneyness'] >= -0.05)&(opts['Moneyness'] <= -0.02)].groupby(['Date', 'Expiry'], as_index=False)[['ask_eod', 'bid_eod']].apply(lambda x: opts.loc[abs(x['ask_eod'] - x['bid_eod']).idxmin(), ['Strike', 'option_type']]).assign(Region='OTM Call'), 
	opts[(opts['option_type'] == 'P')&(opts['Moneyness'] >= 0.02)&(opts['Moneyness'] <= 0.05)].groupby(['Date', 'Expiry'], as_index=False)[['ask_eod', 'bid_eod']].apply(lambda x: opts.loc[abs(x['ask_eod'] - x['bid_eod']).idxmin(), ['Strike', 'option_type']]).assign(Region= 'OTM Put')], axis=0, ignore_index=True)

opts = opts.merge(four_df, how='left', on=['Date', 'Expiry', 'Strike', 'option_type'])
opts.Region = opts.Region.fillna('Other')

if incl_side:
	opts = pd.concat([opts.assign(Side='Long'), opts.assign(Side='Short')], axis=0, ignore_index=True)

################################################################################
# Step 4 Faias Santa Clara
################################################################################

print('#################################')
print('Importing and Merging Risk Free Data')
print('#################################')

tbill = pd.read_csv(data_path + '/Tbill.csv')
tbill['DATE'] = pd.to_datetime(tbill.DATE, format='%Y-%m-%d')
tbill = tbill.rename(columns={'DATE' : 'Date', 'DGS1MO' : 'Rf'})
tbill['Rf'] = tbill['Rf'].apply(lambda x: float(x) if x != '.' else np.nan)
tbill['Rf'] = tbill['Rf'] / 100
opts = opts.merge(tbill, how='left', on='Date')

if incl_side:
	opts['OptRet'] = np.where(opts['Side'] == 'Long', 
		(opts['Payoff'] / opts['ask_eod']) - 1 - opts['Rf'], 
		1 + opts['Rf'] - (opts['Payoff'] / opts['bid_eod']))
else:
	opts['OptRet'] = (opts['Payoff'] / opts['close']) - 1 - opts['Rf']

tbill.plot(x='Date', y='Rf', legend=None, label='')
plt.axvspan('2019-01-15', '2021-08-02', color='red', alpha=0.5, label='CBOE Data')
plt.legend(loc='upper center')
plt.gca().set_ylabel('Returns (%)')
plt.gca().yaxis.set_major_formatter(plt_pct_fmt)
plt.tight_layout()

if show_plots:
	plt.show()
else:
	plt.savefig(graph_out + '/Tbill_TS.png')
	plt.close()

################################################################################
# Step 5 Faias Santa Clara
################################################################################

print('#################################')
print('Running Optimization')
print('#################################')

def optim(optret, rf, gamma, solver):
	As = optret.T

	w = cp.Variable(As.shape[1])
	mat_var = As@w + rf + 1

	utils = cp.sum(cp.power(mat_var, (1-gamma))/(1-gamma))/As.shape[0]

	objective = cp.Maximize(utils)
	if incl_side:
		constraints = [w >= 0]
	else:
		constraints = []

	prob = cp.Problem(objective, constraints)
	result = prob.solve(solver=solver, verbose=True)

	return(w.value)

dates = opts.Date.unique()
if four:
	df = opts[(opts.Date.isin(dates))&(opts.Region != 'Other')]
else:
	df = opts[(opts.Date.isin(dates))&(opts.Moneyness >= -0.1)&(opts.Moneyness <= 0.1)]
df = df.sort_values(by='Date')

weights = []
for date in dates:
	str_date = pd.to_datetime(date).strftime(format='%Y-%m-%d')
	curr = df[df.Date == date]
	curr = curr.reset_index()
	curr_ret = np.array(curr.OptRet.to_list()).reshape(curr.shape[0], N)
	curr_rf = curr.Rf.values[0]

	if (covid)&(str_date == '2020-03-17')&(four):
		curr_ret[curr[curr.Region == 'ATM Put'].index.values] = np.zeros((2, N))

	curr_w = optim(curr_ret, curr_rf, 10, 'MOSEK')

	if (not covid)&(cut < np.inf):
		while any(curr_w > cut):
			curr_ret[curr_w > cut] = np.zeros((sum(curr_w > cut), N))
			curr_w = optim(curr_ret, curr_rf, 10, 'MOSEK')

	weights.append(curr_w)

df['Weights'] = np.concatenate([np.array(weight).reshape(len(weight), 1) for weight in weights])

################################################################################
# Step 6, 7 Faias Santa Clara
################################################################################

print('#################################')
print('Calculating Realized Returns')
print('#################################')

df['ActPay'] = np.where(df['option_type'] == 'C', np.maximum(df['Price_t+1'] - df.Strike, 0), np.maximum(df.Strike - df['Price_t+1'], 0))

if incl_side:
	df['ActRet'] = np.where(df['Side'] == 'Long', 
		(df['ActPay'] / df['ask_eod']) - 1 - df['Rf'], 
		1 + df['Rf'] - (df['ActPay'] / df['bid_eod']))
else:
	df['ActRet'] = (df['ActPay'] / df['close']) - 1 - df['Rf']

df2 = df.pivot(index=['Date', 'option_type', 'Strike'], columns='Side', values='Weights').reset_index()
df2['Weights'] = df2.Long - df2.Short
df2['Side'] = np.where(df2.Weights >= 0, 'Long', 'Short')
df2 = df2.merge(df.drop('Weights', axis=1), how='left', on=['Date', 'option_type', 'Strike', 'Side'])
df = df2.copy()
del df2

################################################################################
# Step 8 Faias Santa Clara
################################################################################

print('#################################')
print('Constructing Output')
print('#################################')

df['Weights'] = abs(df.Weights)
df['TotAct'] = df['ActRet'] * df['Weights']
df['TotSim'] = (df['OptRet'] * df['Weights']).apply(lambda x: np.mean(x))

act_og = df.copy()
if incl_side:
	act_og['Weights'] = np.where(act_og['Side'] == 'Short', 
		-1*act_og['Weights'], act_og['Weights'])
act_og = act_og.groupby(['Date', 'Expiry', 'Region'], as_index=False)[['TotAct', 'Rf', 'Weights']].agg({'TotAct' : sum, 'Rf' : np.mean, 'Weights' : sum})
act = act_og.groupby(['Date', 'Expiry'])[['TotAct', 'Rf', 'Weights']].apply(lambda x: [(x['Rf'].mean() + np.sum(x['TotAct'])), x['Weights'].sum()]).reset_index()
act['ExPost'] = act[0].apply(lambda x: x[0])

if four:
	act = act.merge(act_og[act_og.Region == 'ATM Call'][['Date', 'Expiry', 'Weights']], on=['Date', 'Expiry'], how='left').rename(columns={'Weights':'ATM Call Weight'})
	act = act.merge(act_og[act_og.Region == 'OTM Call'][['Date', 'Expiry', 'Weights']], on=['Date', 'Expiry'], how='left').rename(columns={'Weights':'OTM Call Weight'})
	act = act.merge(act_og[act_og.Region == 'ATM Put'][['Date', 'Expiry', 'Weights']], on=['Date', 'Expiry'], how='left').rename(columns={'Weights':'ATM Put Weight'})
	act = act.merge(act_og[act_og.Region == 'OTM Put'][['Date', 'Expiry', 'Weights']], on=['Date', 'Expiry'], how='left').rename(columns={'Weights':'OTM Put Weight'})

act['SumWeight'] = act[0].apply(lambda x: x[1])
act = act.drop(0, axis=1)
act['Rf Weight'] = 1 - act['SumWeight']
act = act.drop('SumWeight', axis=1)

if (detail_out)&(four):
	act_out = act.copy()
	act_out['ExPost'] = 100 * act_out['ExPost']
	act_out['ATM Call Weight'] = 100 * act_out['ATM Call Weight']
	act_out['OTM Call Weight'] = 100 * act_out['OTM Call Weight']
	act_out['ATM Put Weight'] = 100 * act_out['ATM Put Weight']
	act_out['OTM Put Weight'] = 100 * act_out['OTM Put Weight']
	act_out['Rf Weight'] = 100 * act_out['Rf Weight']

	headers = ['Date', 'Expiry', 'ExPost', 'ATM Call', 'OTM Call', 
		'ATM Put', 'OTM Put', 'Risk Free']

	out_str = tabulate(act_out[['Date', 'Expiry', 'ExPost', 'ATM Call Weight', 'OTM Call Weight', 'ATM Put Weight', 'OTM Put Weight', 'Rf Weight']].astype((str, float)).values, floatfmt='.1f', tablefmt='latex', headers=headers)

	name = tex_out + '/Output_4Contracts'
	if covid:
		name += '_Covid'
	if (cut < np.inf)&(not covid):
		name += '_Cutoff'
	name += '.tex'

	with open(name, "w") as text_file:
	    text_file.write(out_str)

################################################################################
# Plot the Weights
################################################################################

if incl_side:
	df['Weights'] = np.where(df['Side'] == 'Short', 
		-1*df['Weights'], df['Weights'])

if four:
	df['Money'] = df['Region'].apply(lambda x: x.split(" ")[0])
	df['Contract'] = df['Region'].apply(lambda x: x.split(" ")[1])
	df['timedates'] = df['Date'].map(lambda x: x.strftime('%Y-%m'))

	fontP = FontProperties()
	fontP.set_size('x-small')
	fig, axes = plt.subplots(ncols=2, nrows=2, sharey=False)

	sns.lineplot(x='Date', y='Weights', hue='Contract', 
		data=df[(df.Money=='ATM')], ax=axes[0][0])
	axes[0][0].set_ylim([-0.3, 0.3])
	leg1 = axes[0][0].legend(loc='upper left', ncol=2,
		prop=fontP)
	leg1.set_title('ATM', prop={'size':9})
	axes[0][0].yaxis.set_major_formatter(plt_pct_fmt)
	axes[0][0].set_ylabel('Weights (%)')

	sns.lineplot(x='Date', y='Weights', hue='Contract', 
		data=df[(df.Money=='OTM')], ax=axes[0][1])
	axes[0][1].set_ylim([-0.3, 0.3])
	leg2 = axes[0][1].legend(loc='upper left', ncol=2,
		prop=fontP)
	leg2.set_title('OTM', prop={'size':9})
	axes[0][1].yaxis.set_major_formatter(plt_pct_fmt)
	axes[0][1].set_ylabel('Weights (%)')

	sns.lineplot(data=df[['Date', 'Weights', 'Money', 'Contract']].pivot(index=['Date', 'Money'], columns='Contract', values='Weights').reset_index().assign(Diff=lambda x: x['Call'] - x['Put']),
		x='Date', y='Diff', hue='Money', ax=axes[1][0])
	axes[1][0].set_ylim([-0.3, 0.3])
	leg3 = axes[1][0].legend(loc='upper left', ncol=2,
		prop=fontP)
	leg3.set_title('Diff C-P', prop={'size':9})
	axes[1][0].yaxis.set_major_formatter(plt_pct_fmt)
	axes[1][0].set_ylabel('Weights (%)')

	sns.lineplot(data=act, x='Date', y='Rf Weight', ax=axes[1][1], label='Risk-free')
	axes[1][1].set_ylim([0.9, 1.3])
	leg4 = axes[1][1].legend(loc='upper left', ncol=2,
		prop=fontP)
	axes[1][1].yaxis.set_major_formatter(plt_pct_fmt)
	axes[1][1].set_ylabel('Weights (%)')

	fig.autofmt_xdate(rotation=45)
	fig.subplots_adjust(wspace=0.5)

	name = graph_out + '/WeightsPlot'
	if covid:
		name += '_Covid'
	if (cut < np.inf)&(not covid):
		name += '_Cutoff'
	name += '.png'

	plt.tight_layout()

	if show_plots:
		plt.show()
	else:
		plt.savefig(name, bbox_inches='tight')
		plt.close()

################################################################################
# Figure 3 Densities of Contract Type Returns
################################################################################

if four:
	sub = df.copy()
	sub['ActualReturns'] = (sub['ActPay'] / sub['close']) - 1
	sub = sub[sub.ActualReturns <= 10]

	fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)

	axes[0][0].hist(sub[sub.Region == 'ATM Call'].ActualReturns)
	axes[0][0].xaxis.set_major_formatter(plt_pct_fmt)
	axes[0][0].text(0.7, .9, 'ATM Call', horizontalalignment='left',
		transform=axes[0][0].transAxes)
	axes[0][0].set_ylabel('Frequency')
	axes[0][0].set_xlim([-1.1, 9])
	axes[0][0].yaxis.set_ticklabels([])

	axes[0][1].hist(sub[sub.Region == 'OTM Call'].ActualReturns)
	axes[0][1].xaxis.set_major_formatter(plt_pct_fmt)
	axes[0][1].text(0.7, .9, 'OTM Call', horizontalalignment='left',
		transform=axes[0][1].transAxes)
	axes[0][1].set_xlim([-1.1, 9])
	axes[0][1].yaxis.set_ticklabels([])

	axes[1][0].hist(sub[sub.Region == 'ATM Put'].ActualReturns)
	axes[1][0].xaxis.set_major_formatter(plt_pct_fmt)
	axes[1][0].text(0.7, .9, 'ATM Put', horizontalalignment='left',
		transform=axes[1][0].transAxes)
	axes[1][0].set_xlabel('Actual Returns (%)')
	axes[1][0].set_ylabel('Frequency')
	axes[1][0].set_xlim([-1.1, 9])
	axes[1][0].yaxis.set_ticklabels([])

	axes[1][1].hist(sub[sub.Region == 'OTM Put'].ActualReturns)
	axes[1][1].xaxis.set_major_formatter(plt_pct_fmt)
	axes[1][1].text(0.7, .9, 'OTM Put', horizontalalignment='left',
		transform=axes[1][1].transAxes)
	axes[1][1].set_xlabel('Actual Returns (%)')
	axes[1][1].set_xlim([-1.1, 9])
	axes[1][1].yaxis.set_ticklabels([])

	plt.tight_layout()

	name = graph_out + '/Fig3_kde.png'

	if show_plots:
		plt.show()
	else:
		plt.savefig(name)
		plt.close()

################################################################################
# Figure 4 Densities of OOPS Returns
################################################################################

fig, ax = plt.subplots(1)
ax.hist(data=act, x='ExPost')
ax.set_xlabel('ExPost Returns (%)')
ax.xaxis.set_major_formatter(plt_pct_fmt)
ax.set_ylabel('Frequency')
ax.yaxis.set_ticklabels([])
plt.tight_layout()

name = graph_out + '/Fig4_OOPSRet'
if covid:
	name += '_Covid'
if (cut < np.inf)&(not covid):
	name += '_Cutoff'
if not four:
	name += "_AllContract"
name += '.png'

if show_plots:
	plt.show()
else:
	plt.savefig(name)
	plt.close()

################################################################################
# Table 3 F/SC Returns Summary Stats
################################################################################

if four:
	tab_df = df[['Date', 'Expiry', 'Strike', 'Region', 'Rf', 'Price_t', 'Price_t+1', 'close', 'option_type', 'ActPay']].reset_index().drop('index', axis=1)
	tab_df['ActRet'] = (tab_df['ActPay'] / tab_df['close']) - 1
	tab_df['S&P 500'] = (tab_df['Price_t+1']/tab_df['Price_t'])-1
	tab_df['Payoff'] = np.where(tab_df['option_type'] == 'C', tab_df['Price_t+1'] - tab_df.Strike, tab_df.Strike - tab_df['Price_t+1'])
	tab_df['Payoff'] = np.maximum(tab_df['Payoff'], 0)

	grp = tab_df.groupby('Region', as_index=False)['ActRet'].agg({'Mean' : np.mean, 'Std' : np.std, 'Min' : min, 'Max' : max, 'Skew' : skew, 'Kurtosis' : kurtosis})
	grp.loc[grp.index[-1]+1, 'Region'] = '1/N rule'
	N_ret = tab_df.groupby('Date', as_index=False).apply(lambda x: (np.sum(x['Payoff'])/np.sum(x['close'])) - 1).rename(columns={None : 'Return'}).Return

	grp.loc[grp.Region == '1/N rule', 'Mean'] = N_ret.mean()
	grp.loc[grp.Region == '1/N rule', 'Std'] = N_ret.std()
	grp.loc[grp.Region == '1/N rule', 'Min'] = N_ret.min()
	grp.loc[grp.Region == '1/N rule', 'Max'] = N_ret.max()
	grp.loc[grp.Region == '1/N rule', 'Skew'] = skew(N_ret)
	grp.loc[grp.Region == '1/N rule', 'Kurtosis'] = kurtosis(N_ret)

	sp = tab_df.groupby('Date', as_index=False)[['Price_t', 'Rf']].first()
	sp['S&P 500'] = (sp['Price_t']/sp['Price_t'].shift())-1
	sp['Risk Free'] = (sp['Rf']/sp['Rf'].shift())-1
	sp_ret = sp['S&P 500'].dropna()
	tbill_ret = sp['Risk Free'].dropna()

	grp.loc[grp.index[-1]+1, 'Region'] = 'S&P 500'
	grp.loc[grp.Region == 'S&P 500', 'Mean'] = sp_ret.mean()
	grp.loc[grp.Region == 'S&P 500', 'Std'] = sp_ret.std()
	grp.loc[grp.Region == 'S&P 500', 'Min'] = sp_ret.min()
	grp.loc[grp.Region == 'S&P 500', 'Max'] = sp_ret.max()
	grp.loc[grp.Region == 'S&P 500', 'Skew'] = skew(sp_ret)
	grp.loc[grp.Region == 'S&P 500', 'Kurtosis'] = kurtosis(sp_ret)

	grp.loc[grp.index[-1]+1, 'Region'] = 'Risk Free'
	grp.loc[grp.Region == 'Risk Free', 'Mean'] = tbill_ret.mean()
	grp.loc[grp.Region == 'Risk Free', 'Std'] = tbill_ret.std()
	grp.loc[grp.Region == 'Risk Free', 'Min'] = tbill_ret.min()
	grp.loc[grp.Region == 'Risk Free', 'Max'] = tbill_ret.max()
	grp.loc[grp.Region == 'Risk Free', 'Skew'] = skew(tbill_ret)
	grp.loc[grp.Region == 'Risk Free', 'Kurtosis'] = kurtosis(tbill_ret)

	grp['Mean'] = 100 * grp['Mean']
	grp['Std'] = 100 * grp['Std']
	grp['Min'] = 100 * grp['Min']
	grp['Max'] = 100 * grp['Max']

	grp['SR'] = grp['Mean']/grp['Std']

	tab3_str = tabulate(grp.set_index('Region'), tablefmt='latex', floatfmt=(".1f", ".1f", ".1f", ".1f",".1f", '.2f', '.2f', '.2f', '.2f'), headers=[''] + grp.columns[1:])

	with open(tex_out + '/Table3.tex', "w") as text_file:
	    text_file.write(tab3_str)

################################################################################
# Table 4 OOPS vs. SP500 Returns Summary
################################################################################

grp4 = grp[grp.Region == 'S&P 500'].reset_index().drop('index', axis=1)
grp4.loc[grp4.index[-1]+1, 'Region'] = 'OOPS'
grp4.loc[grp4.Region == 'OOPS', 'Mean'] = act['ExPost'].mean()
grp4.loc[grp4.Region == 'OOPS', 'Std'] = act['ExPost'].std()
grp4.loc[grp4.Region == 'OOPS', 'Min'] = act['ExPost'].min()
grp4.loc[grp4.Region == 'OOPS', 'Max'] = act['ExPost'].max()
grp4.loc[grp4.Region == 'OOPS', 'Skew'] = skew(act['ExPost'])
grp4.loc[grp4.Region == 'OOPS', 'Kurtosis'] = kurtosis(act['ExPost'])

grp4['Mean'] = np.where(grp4.Region == 'S&P 500',grp4['Mean'],100*grp4['Mean'])
grp4['Std'] = np.where(grp4.Region == 'S&P 500', grp4['Std'], 100 * grp4['Std'])
grp4['Min'] = np.where(grp4.Region == 'S&P 500', grp4['Min'], 100 * grp4['Min'])
grp4['Max'] = np.where(grp4.Region == 'S&P 500', grp4['Max'], 100 * grp4['Max'])

grp4['SR'] = grp4['Mean']/grp4['Std']

grp4['Region'] = np.where(grp4.Region == 'OOPS', 'ExPost', grp4.Region)

tab4_str = tabulate(grp4.set_index('Region'), tablefmt='latex', floatfmt=(".1f", ".1f", ".1f", ".1f",".1f", '.2f', '.2f', '.2f', '.2f'), headers=[''] + grp4.columns[1:])

name = tex_out + '/Table4'
if covid:
	name += '_Covid'
if (cut < np.inf)&(not covid):
	name += '_Cutoff'
if not four:
	name += "_AllContract"
name += '.tex'

with open(name, "w") as text_file:
    text_file.write(tab4_str)

################################################################################
# Table 5 F/SC Mean Weights by Type
################################################################################

if four:
	agg_df = df.groupby('Region', as_index=False)['Weights'].agg({'Mean': np.mean, 'Minimum': min, 'Maximum': max}).set_index('Region')

	agg_df['Mean'] = 100 * agg_df['Mean']
	agg_df['Minimum'] = 100 * agg_df['Minimum']
	agg_df['Maximum'] = 100 * agg_df['Maximum']

	agg_df = agg_df.transpose()
	agg_df.columns.name = None

	tab5_str = tabulate(agg_df, floatfmt=".1f", tablefmt='latex', headers=agg_df.columns)

	name = tex_out + '/Table5'
	if covid:
		name += '_Covid'
	if (cut < np.inf)&(not covid):
		name += '_Cutoff'
	name += '.tex'

	with open(name, "w") as text_file:
	    text_file.write(tab5_str)

################################################################################
# Cumulative Returns
################################################################################

sp_cum = (1 + returns[(returns.Date >= '2019-01-01')&(returns.Date <= '2021-05-01')].RR).cumprod() * 100
rf_cum = (1 + opts.groupby('Date')['Rf'].first().values).cumprod() * 100
oops_cum = (1 + act['ExPost']).cumprod() * 100

cum_df = pd.DataFrame(act['Date'])
cum_df['S&P 500'] = sp_cum.values
cum_df['Risk-free'] = rf_cum
cum_df['ExPost'] = oops_cum
cum_df = cum_df.set_index('Date')

fig, ax = plt.subplots(1)

cum_df['S&P 500'].plot(ylabel='Wealth ($)', ax=ax, ylim=[80, 1.3*cum_df['S&P 500'].max()])
cum_df['Risk-free'].plot(ax=ax, ylim=[80, 1.3*cum_df['Risk-free'].max()])
cum_df['ExPost'].plot(ax=ax, ylim=[80, 1.3*cum_df['ExPost'].max()])
ax.axhline(100, color='black', linestyle='dotted', label='Starting Wealth')
ax.axvline('2020-03-17', color='red', linestyle='dotted', label='COVID')
plt.legend(loc='upper left')
plt.tight_layout()

name = graph_out + '/Cum_Returns'
if covid:
	name += '_Covid'
if (cut < np.inf)&(not covid):
	name += '_Cutoff'
if not four:
	name += "_AllContract"
name += '.png'

if show_plots:
	plt.show()
else:
	plt.savefig(name)
	plt.close()

################################################################################
# COVID Simulated Returns per Contract
################################################################################

if four:
	dat = opts[(opts.Date == '2020-03-17')]
	if incl_side:
		dat = dat[(dat.Side == 'Long')]

	fix, axes = plt.subplots(nrows=2, ncols=2, sharey=True)
	axes[0][0].hist(data=dat[dat.Region == 'ATM Call'], 
		x='OptRet', range=[-2, 12], label='ATM Call')
	axes[0][0].xaxis.set_major_formatter(plt_pct_fmt)
	axes[0][0].legend()

	axes[0][1].hist(data=dat[dat.Region == 'OTM Call'], 
		x='OptRet', range=[-2, 12], label='OTM Call')
	axes[0][1].xaxis.set_major_formatter(plt_pct_fmt)
	axes[0][1].legend()

	axes[1][0].hist(data=dat[dat.Region == 'ATM Put'], 
		x='OptRet', range=[-2, 12], label='ATM Put')
	axes[1][0].xaxis.set_major_formatter(plt_pct_fmt)
	axes[1][0].legend()

	axes[1][1].hist(data=dat[dat.Region == 'OTM Put'], 
		x='OptRet', range=[-2, 12], label='OTM Put')
	axes[1][1].xaxis.set_major_formatter(plt_pct_fmt)
	axes[1][1].legend()
	plt.tight_layout()

	if show_plots:
		plt.show()
	else:
		plt.savefig(graph_out + "/20200317_OptRet.png")
		plt.close()

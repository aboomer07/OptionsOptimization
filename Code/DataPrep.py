################################################################################
# Import Libraries and Scripts
################################################################################

import numpy as np
import pandas as pd
import os
import sys
import datetime as dt
from itertools import product
import random
from Params import get_params, vals

################################################################################
# Define folders
################################################################################

base_path = '/Users/andrewboomer/Desktop/M2_Courses/Thesis/Options/tradingbot'

code_path = base_path + '/OptionsOptimization/Code'

data_path = base_path + '/Data'
rhood_daily = data_path + '/Rhood_Daily_Data.csv'
rhood_daily_stock = data_path + '/Rhood_Daily_Stock_Data.csv'
yahoo_daily = data_path + '/Yahoo_Trim.csv'

################################################################################
#Merge Stock Price and Option Data
################################################################################

def get_data(params):

	stocks = params['stocks']
	dates = params['dates']

	df = pd.read_csv(eval(vals['source'] + '_daily'))
	df[vals['dt_col']] = pd.to_datetime(df[vals['dt_col']], format=vals['dt_form'])
	df[vals['exp_col']] = pd.to_datetime(df[vals['exp_col']], format='%Y-%m-%d')
	df = df[df[vals['tick_col']].isin(stocks)]
	df = df[df[vals['exp_col']].isin(dates)]

	if vals['source'] == 'rhood':
		stock = pd.read_csv(rhood_daily_stock)[['begins_at', 'symbol', 'open_price', 'close_price', 'volume']].rename({'open_price':'Stock_Open', 'close_price':'Stock_Close','volume':'Stock_Volume'}, axis=1)
		df = df.merge(stock, how='left', on=['begins_at', 'symbol'])

	df['T'] = (df[vals['exp_col']] - df[vals['dt_col']])/dt.timedelta(days=365)

	if vals['source'] == 'yahoo':
		df = df[~df['Vol'].isna()]
		df = df[df['Vol'] != 0]
		df = df.drop(['Last_Trade_Date', 'IV', 'Open_Int'], axis=1)
		df = df[~df.duplicated()]

	df = df.sort_values(by=[vals['type_col'], vals['exp_col'], 'T', vals['strike_col']])
	df = df.reset_index().drop(['index'], axis=1)

	return(df)

def fit_optim_test(data, fit_size, min_dt):

	Ts = list(data['T'].unique())
	combs = [(i[0], i[1]) for i in product(Ts, Ts) if i[1] - i[0] >= min_dt]
	random.shuffle(combs)
	optim_test = combs.pop()
	optim_T = optim_test[0]
	test_T = optim_test[1]

	fit_maturs = [i for i in Ts if (i != optim_T)&(i != test_T)]
	fit_T = np.random.choice(fit_maturs, fit_size)

	fit_data = data[data['T'].isin(fit_T)]
	optim_data = data[data['T'] == optim_T]
	test_data = data[data['T'] == test_T]

	optim_data['Unique'] = optim_data[[vals['strike_col'], vals['type_col']]].apply(lambda x: str(x[vals['strike_col']]) + "_" + x[vals['type_col']], axis=1)
	test_data['Unique'] = test_data[[vals['strike_col'], vals['type_col']]].apply(lambda x: str(x[vals['strike_col']]) + "_" + x[vals['type_col']], axis=1)

	optim_data = optim_data[optim_data['Unique'].isin(test_data['Unique'].unique())]

	optim_data = optim_data.drop('Unique', axis=1)
	test_data = test_data.drop('Unique', axis=1)

	fit_data = fit_data.reset_index().drop('index', axis=1)
	optim_data = optim_data.reset_index().drop('index', axis=1)
	test_data = test_data.reset_index().drop('index', axis=1)

	return(fit_data, optim_data, test_data)
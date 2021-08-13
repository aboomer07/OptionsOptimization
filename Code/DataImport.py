import pandas as pd
import os
# from yahoo_fin import options
# from yahoo_fin import stock_info as si
import robin_stocks as rhood
from robin_stocks import authentication as auth
from robin_stocks import options
import pandas_datareader.data as web
import datetime
import time

data_path = "/Users/andrewboomer/Desktop/M2_Courses/Thesis/Options/tradingbot/Data"

top = ['SPY', 'PLTR', 'XLF', 'BAC', 'EEM', 'XLE', 'VALE', 'EFA', 'FCX', 'PBR', 'X', 'GGB', 'ARKK', 'AAPL', 'TSLA', 'AMZN', 'MSFT', 'GOOGL']

top = ['SPY']
###############################################################################
# Connecting to Robinhood;
###############################################################################

connect = input("Do you want to connect to robinhood account? Yes or No: ")

if connect == 'Yes':
	pwd = input("Please Enter RobinHood password to connect")
	uname = input('Please Enter RobinHood username to connect')

	cnxn = auth.login(username=uname, password=pwd)

###############################################################################
# Importing Sample Option Data
###############################################################################

data_file = data_path + "/Yahoo_Daily_Data.csv"

num_steps = 24 * 21
sleep_time = 3600

for step in range(num_steps):

	print("Running Step #" + str(step + 1))

	data = {}
	for tick in top:
		print('Running Ticker ' + tick)
		data[tick] = web.YahooOptions(tick).get_all_data()

	data = pd.concat(data)
	data = data.reset_index().rename({'level_0' : 'Ticker'}, axis=1)

	if os.path.isfile(data_file):
		data.to_csv(data_file, mode='a', header=False, index=False)
	else:
		data.to_csv(data_file, mode='w', header=True, index=False)

	time.sleep(sleep_time)

###############################################################################
# Robinhood Option Historicals
###############################################################################

opts = {}
for tick in top:
	opts[tick] = options.find_tradable_options(tick)

# data_file = data_path + '/Rhood_Daily_Data.csv'
data_file = data_path + '/Rhood_SPY_Data.csv'

for tick in top:

	print("Running Symbol " + tick)

	# curr_opts = opts[tick]
	curr_opts = opts[~opts['Unique'].isin(test['Unique'])]

	for index, entry in curr_opts.iterrows():

		try:
			if [tick, entry['expiration_date'], float(entry['strike_price'])] in left_list:
				continue
		except:
			pass

		calls = options.get_option_historicals(tick, entry['expiration_date'], entry['strike_price'], 'call', interval='day', span='5year')
		puts = options.get_option_historicals(tick, entry['expiration_date'], entry['strike_price'], 'put', interval='day', span='5year')

		calls = pd.DataFrame.from_dict(calls)
		puts = pd.DataFrame.from_dict(puts)

		calls['Direction'] = 'call'
		puts['Direction'] = 'put'

		data = pd.concat([calls, puts], axis=0)

		data['expiration_date'] = entry['expiration_date']
		data['strike_price'] = entry['strike_price']
		data['tradability'] = entry['tradability']
		data[data.filter(like='price').columns] = data[data.filter(like='price').columns].astype(float)

		if os.path.isfile(data_file):
			data.to_csv(data_file, mode='a', header=False, index=False)
		else:
			data.to_csv(data_file, mode='w', header=True, index=False)

test = pd.read_csv(data_file)

###############################################################################
# Robinhood Stock Historicals
###############################################################################

data_file = data_path + "/Rhood_Hourly_Stock_Data.csv"
hourly_stocks = rhood.stocks.get_stock_historicals(top, interval='hour', span='3month')
hourly = pd.DataFrame.from_dict(hourly_stocks)
hourly[hourly.filter(like='price').columns] = hourly[hourly.filter(like='price').columns].astype(float)
hourly.to_csv(data_file, header=True, index=False)

data_file = data_path + "/Rhood_Daily_Stock_Data.csv"
daily_stocks = rhood.stocks.get_stock_historicals(top, interval='day', span='5year')
daily = pd.DataFrame.from_dict(daily_stocks)
daily[daily.filter(like='price').columns] = daily[daily.filter(like='price').columns].astype(float)
daily.to_csv(data_file, header=True, index=False)


###############################################################################
# Importing Libraries
###############################################################################
import pandas as pd
import numpy as np
import robin_stocks as rhood
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as si
import datetime
from itertools import combinations, groupby, permutations
from random import choice
from yahoo_fin import options
from yahoo_fin import stock_info as si

###############################################################################
# Finding the list of tradable options on robinhood (takes forever)
###############################################################################

stocks = pd.read_csv(os.path.abspath("..") + "/Data/nasdaqlisted.txt", sep='|')
stocks1 = pd.read_csv(os.path.abspath("..") + "/Data/otherlisted.txt", sep='|')

other_list = list(stocks1['NASDAQ Symbol'].unique())
lst = list(stocks['Symbol'].unique())
tot_list = sorted([i for i in set(other_list + lst) if i is not np.nan])

tot_list[tot_list.index(valid_symbs[-1]):]
valid_symbs = []
for val in tot_list:
	try:
		symb_check = rhood.options.find_tradable_options(val)
		if symb_check[0] is not None:
			valid_symbs.append(val)
	except:
		pass

# Write data to file
with open(os.path.abspath("..") + "/Data/option_ticks.csv",'w') as f:
	for r in valid_symbs:
	    f.write(r + "\n")
f.close()

###############################################################################
# Gathering Options Data
###############################################################################

data = rhood.options.find_tradable_options('AAPL')
df = pd.DataFrame(data)

expiries = sorted(df['expiration_date'].unique())
strikes = sorted(df['strike_price'].unique())

test1 = rhood.options.get_option_historicals('AAPL',
                                  expiries[0],
                                  strikes[0],
                                  'call',
                                  interval='10minute',
                                  span='week',
                                  bounds='regular')
test_df = pd.DataFrame(test1)


test2 = rhood.options.find_options_by_expiration('AAPL', expiry)

df = pd.DataFrame.from_dict(test2)
df['strike_price'] = df['strike_price'].astype(np.float64)
df2 = pd.DataFrame.from_dict(test1)

valid_symbs = pd.read_csv(os.path.abspath("..") + "/Data/option_ticks.csv",
	header=None)
valid_symbs = [i[0] for i in valid_symbs.values]

top = rhood.markets.get_top_100()
top_ticks = [i['symbol'] for i in top if i['symbol'] in valid_symbs]

expiries = ['2021-01-29']
data_dicts = []
for tick in ['AMC']:
	for expiry in expiries:
		try:
			curr_data = rhood.options.find_options_by_expiration(tick, expiry)
			for i in range(len(curr_data)):
				curr_data[i]['Accessed_DT'] = str(datetime.datetime.now())
				curr_data[i]['StockPrice'] = float(rhood.stocks.get_stock_quote_by_symbol(tick)['last_trade_price'])
			data_dicts.append(curr_data)
		except:
			pass

df = pd.concat([pd.DataFrame.from_dict(i) for i in data_dicts])

df.to_csv(os.path.abspath("..") + "/Data/test_option_data.csv", index=False)


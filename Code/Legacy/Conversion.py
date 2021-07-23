conversion_df = df[['chain_symbol', 'created_at', 'type', 'expiration_date', 'strike_price', 'ask_price', 'ask_size', 'bid_price', 'bid_size', 'last_trade_price', 'volume', 'delta', 'gamma', 'implied_volatility', 'rho', 'theta', 'vega', 'StockPrice']]
conversion_df[['strike_price', 'ask_price', 'ask_size', 'bid_price', 'bid_size', 'last_trade_price', 'volume', 'delta', 'gamma', 'implied_volatility', 'rho', 'theta', 'vega', 'StockPrice']] = conversion_df[['strike_price', 'ask_price', 'ask_size', 'bid_price', 'bid_size', 'last_trade_price', 'volume', 'delta', 'gamma', 'implied_volatility', 'rho', 'theta', 'vega', 'StockPrice']].astype(float)
conversion_df = conversion_df.pivot(index=['chain_symbol', 'created_at', 'expiration_date', 'strike_price', 'StockPrice'], columns='type', values=['ask_price', 'ask_size', 'bid_price', 'bid_size', 'last_trade_price', 'volume', 'delta', 'gamma', 'implied_volatility', 'rho', 'theta', 'vega']).reset_index()
conversion_df['Conversion'] = conversion_df[('last_trade_price', 'call')] - conversion_df[('last_trade_price', 'put')] + conversion_df['strike_price'] - conversion_df['StockPrice']
conversion_df['ConversionPct'] = (conversion_df[('last_trade_price', 'call')] - conversion_df[('last_trade_price', 'put')] + conversion_df['strike_price'] - conversion_df['StockPrice']) / conversion_df['StockPrice']
conversion_df = conversion_df[(conversion_df[('volume', 'call')]>0)&(conversion_df[('volume', 'put')]>0)]

df = pd.concat(options.get_options_chain('amc', date='01/29/2021')).reset_index().drop('level_1', axis=1).rename({'level_0' : 'type'}, axis=1)
df = df.pivot(index=['Strike'], columns='type', values=['Last Price', 'Bid', 'Ask', 'Change', '% Change', 'Volume', 'Open Interest', 'Implied Volatility', 'Last Trade Date']).reset_index()
df['StockPrice'] = si.get_live_price('amc')
df['Conversion'] = df[('Bid', 'calls')] - df[('Ask', 'puts')] + df['Strike'] - df['StockPrice']
 
# scrape the options data for each Dow ticker

stock_data = {}
opt_data = {}
expiry = 'January 29, 2021'
for ticker in valid_symbs:
	try:
		first_expiry = options.get_expiration_dates(ticker)[0]
	except:
		continue

	if first_expiry != expiry:
		continue

    opt_data[ticker] = pd.concat(options.get_options_chain(ticker, date=expiry)).reset_index()
    stock_data[ticker] = si.get_live_price(ticker)
    print(ticker)

df = pd.concat(top_data).rename({'level_0' : 'type'}, axis=1).drop(['level_1', 'Contract Name'], axis=1).reset_index().rename({'level_0' : 'ticker'}, axis=1).drop('level_1', axis=1)
df = df.replace("-", "0")
df = df.replace('%')
df[['Bid', 'Volume']] = df[['Bid', 'Volume']].astype(float)
df = df.pivot(index=['ticker', 'Strike'], columns='type', values=['Last Price', 'Bid', 'Ask', 'Change', '% Change', 'Volume', 'Open Interest', 'Implied Volatility', 'Last Trade Date']).reset_index()
df['StockPrice'] = df['ticker'].map(stock_data)
df = df.dropna(axis=0)
df = df[(df[('Volume', 'calls')]>2)&(df[('Volume', 'puts')]>2)]
df['Conversion'] = df[('Bid', 'calls')] - df[('Ask', 'puts')] + df['Strike'] - df['StockPrice']
df['ConversionPct'] = (df[('Bid', 'calls')] - df[('Ask', 'puts')] + df['Strike'] - df['StockPrice']) / df['StockPrice']
df = df[df['ConversionPct'] > 0]
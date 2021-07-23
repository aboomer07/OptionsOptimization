import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class Option:
    def __init__(self, type_, K, price, side, Expiry):
        self.type = type_
        self.K = K
        self.price = price
        self.side = side
        self.Expiry = Expiry
        # self.T = self.Expiry - datetime.datetime.today()
    
    def __repr__(self):
        side = 'Long' if self.side == 'Long' else 'Short'
        return f'Option(type={self.type}, K=${self.K}, price=${self.price}, side={side})'

class OptionStrat:
	def __init__(self, res, params):
		self.base_funcs = {
				'sc' : lambda P, K, X: min(K - X, 0) + P,
				'lc' : lambda P, K, X: max(X - K, 0) - P,
				'sp' : lambda P, K, X: min(X - K, 0) + P,
				'lp' : lambda P, K, X: max(K - X, 0) - P}
		self.side = {'sc' : 'Short', 'sp' : 'Short',
			'lc' : 'Long', 'lp' : 'Long'}
		self.prod = {'sc' : 'Call', 'sp' : 'Put',
			'lc' : 'Call', 'lp' : 'Put'}

		res = res[abs(res['Results']) > 0]
		res = res.loc[res.index.repeat(abs(res.Results))]

		self.Legs = list(res.apply(lambda x: ['s', 'l'][x.Results > 0] + x.Type[0], axis=1).values)
		self.Price = res[self.stock_col].min()
		self.Expiry = res.Expiry.min()
		self.T = res['T'].min()

		self.x_rng = np.linspace(0, self.Price * 2, 200)

		self.K = list(res.Strike.values)
		self.P = list(res.Last.values)

		self.instruments = []

		self.y_rng = np.zeros_like(self.x_rng)

		self.Funcs = [self.base_funcs[t] for t in self.Legs]

		for i in range(len(self.Funcs)):

			prod = self.prod[self.Legs[i]]
			side = self.side[self.Legs[i]]
			o = Option(prod, self.K[i], self.P[i], side, self.Expiry)
			self.instruments.append(o)

			self.y_rng += np.array([self.Funcs[i](self.P[i], self.K[i], x) for x in self.x_rng])

	def get_combo(self):
		combo_dict = {'Legs' : self.Legs, self.stock_col : self.Price, 
		self.exp_col : self.Expiry, 'Contracts' : self.instruments, 
		'Strikes' : self.K, 'Last' : self.P, 'T' : self.T}
		return(combo_dict)

	def plot_profit(self, plot_params, plot=False):
		fig, ax = plt.subplots(1)
		ax.plot(self.x_rng, self.y_rng)
		ax.hlines(xmin=min(self.x_rng), xmax=max(self.x_rng), y=0, 
			linestyles='dashed', color='black')

		ax.fill_between(self.x_rng, self.y_rng, 
			where=(self.y_rng >= 0), facecolor='g', alpha=0.4)
		ax.fill_between(self.x_rng, self.y_rng,
			where=(self.y_rng < 0), facecolor='r', alpha=0.4)

		ax.vlines(ymin=min(self.y_rng), ymax=max(self.y_rng), 
			x=self.Price, linestyles='dashed', color='green')
		ax.text(0.9 * self.Price, (min(self.y_rng) + max(self.y_rng))/2, 
			'Current Price', rotation=90)

		ax.set_xlabel("<---- Price at Expiration ---->")
		ax.set_ylabel("Profit/Loss")
		plt.title(plot_params['title'])

		if plot_params['file'] is not None:
			plt.savefig(plot_params['file'])

		plt.close()

		if plot:
			plt.show()

	def describe(self):
		max_profit  = self.y_rng.max()
		max_loss = -1*self.y_rng.min()
		string = f"Max Profit: \\${round(max_profit, 2)}" + "\n\\\\"
		string += f"Max loss: \\${round(max_loss, 2)}" + "\n\\\\"
		c = 0
		for o in self.instruments:
			string += str(o) + '\n\\\\'
			# print(o)
			if o.type == 'Call' and o.side=='Long':
				c += o.price
			elif o.type == 'Call' and o.side=='Short':
				c -= o.price
			elif o.type =='Put' and o.side=='Long':
				c += o.price
			elif o.type =='Put' and o.side=='Short':
				c -= o.price

		string += f"Cost of entering position \\${round(c, 2)}"
		print(string)
		return(string)

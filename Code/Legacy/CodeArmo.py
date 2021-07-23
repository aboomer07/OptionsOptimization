import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

################################################################################
# Basic Object Class Definitions for Options and Option Combs
################################################################################

class Option:
    def __init__(self, type_, K, price, side):
        self.type = type_
        self.K = K
        self.price = price
        self.side = side
    
    def __repr__(self):
        side = 'long' if self.side == 1 else 'short'
        return f'Option(type={self.type}, K={self.K}, price={self.price}, side={side})'

class OptionStrat:
    def __init__(self, name, S0, params=None):
        self.name = name
        self.S0 = S0
        if params:
            self.STs = np.arange(params.get('start', 0), 
                params.get('stop', S0*2), params.get('by', 1))
        else:
            self.STs = np.arange(0, S0*2, 1)
        self.payoffs = np.zeros_like(self.STs)
        self.instruments = [] 
           
    def long_call(self, K, C, Q=1):
        payoffs =  np.array([max(s-K,0) - C for s in self.STs])*Q
        self.payoffs = self.payoffs + payoffs
        self._add_to_self('call', K, C, 1, Q)
    
    def short_call(self, K, C, Q=1):
        payoffs =  np.array([max(s-K,0) * -1 + C for s in self.STs])*Q
        self.payoffs = self.payoffs + payoffs
        self._add_to_self('call', K, C, -1, Q)
    
    def long_put(self, K, P, Q=1):
        payoffs = np.array([max(K-s,0) - P for s in self.STs])*Q
        self.payoffs = self.payoffs + payoffs
        self._add_to_self('put', K, P, 1, Q)
      
    def short_put(self, K, P, Q=1):
        payoffs = np.array([max(K-s,0)*-1 + P for s in self.STs])*Q
        self.payoffs = self.payoffs + payoffs
        self._add_to_self('put', K, P, -1, Q)
        
    def _add_to_self(self, type_, K, price, side, Q):
        o = Option(type_, K, price, side)
        for _ in range(Q):
            self.instruments.append(o)
        
    def plot(self, **params):
        plt.plot(self.STs, self.payoffs, **params)
        plt.title(f"Payoff Diagram for {self.name}")
        plt.fill_between(self.STs, self.payoffs,
                         where=(self.payoffs >= 0), facecolor='g', alpha=0.4)
        plt.fill_between(self.STs, self.payoffs,
                         where=(self.payoffs < 0), facecolor='r', alpha=0.4)
        
        plt.xlabel(r'$S_T$')
        plt.ylabel('Profit in $')
        plt.show()
        
    def describe(self):
        max_profit  = self.payoffs.max()
        max_loss = self.payoffs.min()
        print(f"Max Profit: ${round(max_profit,3)}")
        print(f"Max loss: ${round(max_loss,3)}")
        c = 0
        for o in self.instruments:
            print(o)
            if o.type == 'call' and o.side==1:
                c += o.price
            elif o.type == 'call' and o.side == -1:
                c -= o.price
            elif o.type =='put' and o.side == 1:
                c += o.price
            elif o.type =='put' and o.side == -1:
                c -+ o.price
        
        print(f"Cost of entering position ${c}")

obj = OptionStrat('Iron Condor', 100, {'start': 50, 'stop':150,'by':1})
obj.long_call(120, 2, 1)        
obj.short_call(110, 4, 1)
obj.short_put(90, 4, 1)  
obj.long_put(80, 2, 1)    
obj.plot(color='black', linewidth=2) 
obj.describe()

################################################################################
# Binomial Pricing Model
################################################################################

############params###############3

N = 4
S0  = 100
T = 0.5
sigma = 0.4
dt = T/N
K =105
r = 0.05
u = np.exp( sigma * np.sqrt(dt) )
d =  np.exp( -sigma * np.sqrt(dt) )
p = ( np.exp(r*dt) - d) / (u -d)


######showing terminal stock prices for 4 step model################

for k in reversed(range(N+1)):
    ST = S0 * u**k * d ** (N-k)
    print(round(ST,2), round(max(ST-K,0),2))


#176.07 71.07
#132.69 27.69
#100.0 0
#75.36 0
#56.8 0


############showing node probabilities

def combos(n, i):
    return math.factorial(n) / (math.factorial(n-i)*math.factorial(i))
    
for k in reversed(range(N+1)):
    p_star = combos(N, k)*p**k *(1-p)**(N-k)
    print(round(p_star,2))
    
#0.06
#0.24
#0.37
#0.26
#0.07


######valuing the call from example#######################

C=0   
for k in reversed(range(N+1)):
    p_star = combos(N, k)*p**k *(1-p)**(N-k)
    ST = S0 * u**k * d ** (N-k)
    C += max(ST-K,0)*p_star
    
print(np.exp(-r*T)*C)

#10.60594883990603

################################################################################
# Black-Scholes Pricing Model
################################################################################

class BsOption:
    def __init__(self, S, K, T, r, sigma, q=0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r 
        self.sigma = sigma
        self.q = q
        
    
    @staticmethod
    def N(x):
        return norm.cdf(x)
    
    @property
    def params(self):
        return {'S': self.S, 
                'K': self.K, 
                'T': self.T, 
                'r':self.r,
                'q':self.q,
                'sigma':self.sigma}
    
    def d1(self):
        return (np.log(self.S/self.K) + (self.r -self.q + self.sigma**2/2)*self.T) / (self.sigma*np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma*np.sqrt(self.T)
    
    def _call_value(self):
        return self.S*np.exp(-self.q*self.T)*self.N(self.d1()) - \
                    self.K*np.exp(-self.r*self.T) * self.N(self.d2())
                    
    def _put_value(self):
        return self.K*np.exp(-self.r*self.T) * self.N(-self.d2()) -\
                self.S*np.exp(-self.q*self.T)*self.N(-self.d1())
    
    def price(self, type_ = 'C'):
        if type_ == 'C':
            return self._call_value()
        if type_ == 'P':
            return self._put_value() 
        if type_ == 'B':
            return  {'call': self._call_value(), 'put': self._put_value()}
        else:
            raise ValueError('Unrecognized type')

        
if __name__ == '__main__':
    K = 100
    r = 0.1
    T = 1
    sigma = 0.3
    S = 100
    print(BsOption(S, K, T, r, sigma).price('B')) 

################################################################################
# Calculating Greeks
################################################################################

def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + sigma**2/2)*T) /\
                     sigma*np.sqrt(T)

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma* np.sqrt(T)

def delta_call(S, K, T, r, sigma):
    N = norm.cdf
    return N(d1(S, K, T, r, sigma))
    
def delta_fdm_call(S, K, T, r, sigma, ds = 1e-5, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_CALL(S+ds, K, T, r, sigma) -BS_CALL(S-ds, K, T, r, sigma))/\
                        (2*ds)
    elif method == 'forward':
        return (BS_CALL(S+ds, K, T, r, sigma) - BS_CALL(S, K, T, r, sigma))/ds
    elif method == 'backward':
        return (BS_CALL(S, K, T, r, sigma) - BS_CALL(S-ds, K, T, r, sigma))/ds
    
    
def delta_put(S, K, T, r, sigma):
    N = norm.cdf
    return - N(-d1(S, K, T, r, sigma))

def delta_fdm_put(S, K, T, r, sigma, ds = 1e-5, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_PUT(S+ds, K, T, r, sigma) -BS_PUT(S-ds, K, T, r, sigma))/\
                        (2*ds)
    elif method == 'forward':
        return (BS_PUT(S+ds, K, T, r, sigma) - BS_PUT(S, K, T, r, sigma))/ds
    elif method == 'backward':
        return (BS_PUT(S, K, T, r, sigma) - BS_PUT(S-ds, K, T, r, sigma))/ds



S = 100
K = 100
T = 1
r = 0.00
sigma = 0.25

prices = np.arange(1, 250, 1)

deltas_c = delta_call(prices, K, T, r, sigma)
deltas_p = delta_put(prices, K, T, r, sigma)
deltas_back_c = delta_fdm_call(prices, K, T,r, sigma, ds = 0.01,method='backward')
deltas_forward_p = delta_fdm_put(prices, K, T,r, sigma, ds = 0.01,method='forward')

plt.plot(prices, deltas_c, label='Delta Call')
plt.plot(prices, deltas_p, label='Delta Put')
plt.xlabel('$S_0$')
plt.ylabel('Delta')
plt.title('Stock Price Effect on Delta for Calls/Puts' )
plt.axvline(K, color='black', linestyle='dashed', linewidth=2,label="Strike")
plt.legend()
plt.show()

def gamma(S, K, T, r, sigma):
    N_prime = norm.pdf
    return N_prime(d1(S,K, T, r, sigma))/(S*sigma*np.sqrt(T))


def gamma_fdm(S, K, T, r, sigma , ds = 1e-5, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_CALL(S+ds , K, T, r, sigma) -2*BS_CALL(S, K, T, r, sigma) + 
                    BS_CALL(S-ds , K, T, r, sigma) )/ (ds)**2
    elif method == 'forward':
        return (BS_CALL(S+2*ds, K, T, r, sigma) - 2*BS_CALL(S+ds, K, T, r, sigma)+
                   BS_CALL(S, K, T, r, sigma) )/ (ds**2)
    elif method == 'backward':
        return (BS_CALL(S, K, T, r, sigma) - 2* BS_CALL(S-ds, K, T, r, sigma)
                + BS_CALL(S-2*ds, K, T, r, sigma)) /  (ds**2)  


gammas = gamma(prices, K, T, r, sigma)
gamma_forward = gamma_fdm(prices, K, T, r, sigma, ds =0.01,method='forward')

plt.plot(prices, gammas)
plt.plot(prices, gamma_forward)
plt.title('Gamma by changing $S_0$')
plt.xlabel('$S_0$')
plt.ylabel('Gamma')

def vega(S, K, T, r, sigma):
    N_prime = norm.pdf
    return S*np.sqrt(T)*N_prime(d1(S,K,T,r,sigma)) 

def vega_fdm(S, K, T, r, sigma, dv=1e-4, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_CALL(S, K, T, r, sigma+dv) -BS_CALL(S, K, T, r, sigma-dv))/\
                        (2*dv)
    elif method == 'forward':
        return (BS_CALL(S, K, T, r, sigma+dv) - BS_CALL(S, K, T, r, sigma))/dv
    elif method == 'backward':
        return (BS_CALL(S, K, T, r, sigma) - BS_CALL(S, K, T, r, sigma-dv))/dv

def theta_call(S, K, T, r, sigma):
    p1 = - S*N_prime(d1(S, K, T, r, sigma))*sigma / (2 * np.sqrt(T))
    p2 = r*K*np.exp(-r*T)*N(d2(S, K, T, r, sigma)) 
    return p1 - p2

def theta_put(S, K, T, r, sigma):
    p1 = - S*N_prime(d1(S, K, T, r, sigma))*sigma / (2 * np.sqrt(T))
    p2 = r*K*np.exp(-r*T)*N(-d2(S, K, T, r, sigma)) 
    return p1 + p2

def theta_call_fdm(S, K, T, r, sigma, dt, method='central'):
    method = method.lower() 
    if method =='central':
        return -(BS_CALL(S, K, T+dt, r, sigma) -BS_CALL(S, K, T-dt, r, sigma))/\
                        (2*dt)
    elif method == 'forward':
        return -(BS_CALL(S, K, T+dt, r, sigma) - BS_CALL(S, K, T, r, sigma))/dt
    elif method == 'backward':
        return -(BS_CALL(S, K, T, r, sigma) - BS_CALL(S, K, T-dt, r, sigma))/dt
    
def theta_put_fdm(S, K, T, r, sigma, dt, method='central'):
    method = method.lower() 
    if method =='central':
        return -(BS_PUT(S, K, T+dt, r, sigma) -BS_PUT(S, K, T-dt, r, sigma))/\
                        (2*dt)
    elif method == 'forward':
        return -(BS_PUT(S, K, T+dt, r, sigma) - BS_PUT(S, K, T, r, sigma))/dt
    elif method == 'backward':
        return -(BS_PUT(S, K, T, r, sigma) - BS_PUT(S, K, T-dt, r, sigma))/dt


theta_call(100,100,1,0.05,0.2, 0.1,0.05)
Ts = [1,0.75,0.5,0.25,0.1,0.05]
for t in Ts:
    plt.plot(theta_call(prices, K, t, r, sigma), label=f'T = {t}')

plt.legend()
plt.title('Theta of a call')
plt.xlabel('$S_0$')
plt.ylabel('Theta')

def rho_call(S, K, T, r, sigma):
    return K*T*np.exp(-r*T)*N(d2(S, K, T, r, sigma))

def rho_put(S, K, T, r, sigma):
    return -K*T*np.exp(-r*T)*N(-d2(S, K, T, r, sigma))


def rho_call_fdm(S, K, T, r, sigma, dr, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_CALL(S, K, T, r+dr, sigma) -BS_CALL(S, K, T, r-dr, sigma))/\
                        (2*dr)
    elif method == 'forward':
        return (BS_CALL(S, K, T, r+dr, sigma) - BS_CALL(S, K, T, r, sigma))/dr
    elif method == 'backward':
        return (BS_CALL(S, K, T, r, sigma) - BS_CALL(S, K, T, r-dr, sigma))/dr
  
def rho_put_fdm(S, K, T, r, sigma, dr, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_PUT(S, K, T, r+dr, sigma) - BS_PUT(S, K, T, r-dr, sigma))/\
                        (2*dr)
    elif method == 'forward':
        return (BS_PUT(S, K, T, r+dr, sigma) - BS_PUT(S, K, T, r, sigma))/dr
    elif method == 'backward':
        return (BS_PUT(S, K, T, r, sigma) - BS_PUT(S, K, T, r-dr, sigma))/dr

################################################################################
# Calculating Implied Volatility
################################################################################

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar   
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
N = norm.cdf

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)    
    

def implied_vol(opt_value, S, K, T, r, type_='call'):
    
    def call_obj(sigma):
        return abs(BS_CALL(S, K, T, r, sigma) - opt_value)
    
    def put_obj(sigma):
        return abs(BS_PUT(S, K, T, r, sigma) - opt_value)
    
    if type_ == 'call':
        res = minimize_scalar(call_obj, bounds=(0.01, 6), method='bounded')
        return res.x
    elif type_ == 'put':
        res = minimize_scalar(put_obj, bounds=(0.01, 6), method='bounded')
        return res.x
    else:
        raise ValueError("type_ must be 'put' or 'call'")

################################################################################
# Merton Jump Diffusion
################################################################################

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

def merton_jump_paths(S, T, r, sigma, lam, m, v, steps, Npaths):
    size = (steps, Npaths)
    dt = T/steps 
    poi_rv = np.multiply(np.random.poisson( lam*dt, size=size),
                         np.random.normal(m,v, size=size)).cumsum(axis=0)
    geo = np.cumsum(((r -  sigma**2/2 -lam*(m  + v**2*0.5))*dt +\
                              sigma*np.sqrt(dt) * \
                              np.random.normal(size=size)), axis=0)
    
    return np.exp(geo+poi_rv)*S


S = 100 # current stock price
T = 1 # time to maturity
r = 0.02 # risk free rate
m = 0 # meean of jump size
v = 0.3 # standard deviation of jump
lam = 1 # intensity of jump i.e. number of jumps per annum
steps = 10000 # time steps
Npaths = 1 # number of paths to simulate
sigma = 0.02 # annaul standard deviation , for weiner process

j = merton_jump_paths(S, T, r, sigma, lam, m, v, steps, Npaths)

plt.plot(j)
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Jump Diffusion Process')
    
def merton_jump_call(S, K, T, r, sigma, m , v, lam):
    p = 0
    for k in range(40):
        r_k = r - lam*(m-1) + (k*np.log(m) ) / T
        sigma_k = np.sqrt( sigma**2 + (k* v** 2) / T)
        k_fact = np.math.factorial(k)
        p += (np.exp(-m*lam*T) * (m*lam*T)**k / (k_fact)) * BS_CALL(S, K, T, r_k, sigma_k)
    
    return p 

def merton_jump_put(S, K, T, r, sigma, m , v, lam):
    p = 0 # price of option
    for k in range(40):
        r_k = r - lam*(m-1) + (k*np.log(m) ) / T
        sigma_k = np.sqrt( sigma**2 + (k* v** 2) / T)
        k_fact = np.math.factorial(k) # 
        p += (np.exp(-m*lam*T) * (m*lam*T)**k / (k_fact)) \
                    * BS_PUT(S, K, T, r_k, sigma_k)
    return p 

S = 100 # current stock price
T = 1 # time to maturity
r = 0.02 # risk free rate
m = 0 # meean of jump size
v = 0.3 # standard deviation of jump
lam = 1 # intensity of jump i.e. number of jumps per annum
steps = 255 # time steps
Npaths = 200000 # number of paths to simulate
sigma = 0.2 # annaul standard deviation , for weiner process
K = 100
np.random.seed(3)
j = merton_jump_paths(S, T, r, sigma, lam, m, v, steps, Npaths) #generate jump diffusion paths

mcprice = np.maximum(j[-1]-K,0).mean() * np.exp(-r*T) # calculate value of call

cf_price =  merton_jump_call(S, K, T, r, sigma, np.exp(m+v**2*0.5), v, lam)

print('Merton Price =', cf_price)
print('Monte Carlo Merton Price =', mcprice)
print('Black Scholes Price =', BS_CALL(S,K,T,r, sigma))

#Merton Price = 14.500570058304778
#Monte Carlo Merton Price = 14.597509592911369
#Black Scholes Price = 8.916037278572539

S = 100
strikes = np.arange(50, 150, 1)
r = 0.02
m = 1 
v = 0.3 
lam =1 
sigma = 0.02 
T= 1

bs_prices = BS_CALL(S, strikes, T, r, sigma)
mjd_prices = merton_jump_call(S, strikes, T, r, sigma, m, v, lam)
merton_ivs = [implied_vol(c, S, k, T, r) for c, k in zip(mjd_prices, strikes)]
bs_ivs = [implied_vol(c, S, k, T, r) for c, k in zip(bs_prices, strikes)]

plt.plot(strikes, merton_ivs, label='IV Smile')
plt.plot(strikes, bs_ivs, label='BS_IV')
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.axvline(S, color='black', linestyle='dashed', linewidth=2,label="Spot")
plt.title('MJD Volatility Smile')
plt.legend()
plt.show()

import pandas as pd
import time
from scipy.optimize import minimize

df = pd.read_csv('https://raw.githubusercontent.com/codearmo/data/master/calls_calib_example.csv')

print(df.head(10))

def optimal_params(x, mkt_prices, strikes):
    candidate_prices = merton_jump_call(S, strikes, T, r,
                                        sigma=x[0], m= x[1] ,
                                        v=x[2],lam= x[3])
    return np.linalg.norm(mkt_prices - candidate_prices, 2)


T = df['T'].values[0]
S = df.F.values[0]
r = 0 
x0 = [0.15, 1, 0.1, 1] # initial guess for algorithm
bounds = ((0.01, np.inf) , (0.01, 2), (1e-5, np.inf) , (0, 5)) #bounds as described above
strikes = df.Strike.values
prices = df.Midpoint.values

res = minimize(optimal_params, method='SLSQP', x0=x0, args=(prices, strikes),
                  bounds = bounds, tol=1e-20, 
                  options={"maxiter":1000})
sigt = res.x[0]
mt = res.x[1]
vt = res.x[2]
lamt = res.x[3]

print('Calibrated Volatlity = ', sigt)
print('Calibrated Jump Mean = ', mt)
print('Calibrated Jump Std = ', vt)
print('Calibrated intensity = ', lamt)

#Calibrated Volatlity =  0.06489478237064618
#Calibrated Jump Mean =  0.8789051095314648
#Calibrated Jump Std =  0.1542041201811455
#Calibrated intensity =  0.9722952134238365

df['least_sq_V'] = merton_jump_call(S, df.Strike, df['T'], 0 ,sigt, mt, vt, lamt)

plt.scatter(df.Strike, df.Midpoint, label= 'Observed Prices')
plt.plot(df.Strike, df.least_sq_V, color='black',label= 'Fitted Prices')
plt.legend()
plt.xlabel('Strike')
plt.ylabel('Value in $')
plt.title('Merton Model Optimal Params')

################################################################################
# Heston Model
################################################################################

def generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi, 
    steps, Npaths, return_vol=False):
    dt = T/steps
    size = (Npaths, steps)
    prices = np.zeros(size)
    sigs = np.zeros(size)
    S_t = S
    v_t = v_0
    for t in range(steps):
        WT = np.random.multivariate_normal(np.array([0,0]), 
            cov = np.array([[1, rho], [rho,1]]), size=paths) * np.sqrt(dt) 
        
        S_t = S_t*(np.exp((r-0.5*v_t)*dt + np.sqrt(v_t)*WT[:,0])) 
        v_t = np.abs(v_t + kappa*(theta-v_t)*dt + xi*np.sqrt(v_t)*WT[:,1])
        prices[:, t] = S_t
        sigs[:, t] = v_t
    
    if return_vol:
        return(prices, sigs)
    
    return(prices)

kappa = 4
theta = 0.02
v_0 = 0.02
xi = 0.9
r = 0.02
S = 100
paths = 50000
steps = 2000
T = 1

prices_pos = generate_heston_paths(S, T, r, kappa, theta, v_0, rho=0.6, xi=xi, steps=steps, Npaths=paths, return_vol=False)[:,-1]  
prices_neg  = generate_heston_paths(S, T, r, kappa, theta, v_0, rho=-0.6, xi=xi, steps=steps, Npaths=paths, return_vol=False)[:,-1]       
gbm_bench = S*np.exp(np.random.normal((r - v_0/2)*T, np.sqrt(theta)*np.sqrt(T), size=paths))

import seaborn as sns

fig, ax = plt.subplots()

ax = sns.kdeplot(data=prices_pos, label=r"$\rho = 0.9$", ax=ax)
ax = sns.kdeplot(data=prices_neg, label=r"$\rho= -0.9$ ", ax=ax)
ax = sns.kdeplot(data=gbm_bench, label="GBM", ax=ax)

ax.set_title(r'Tail Density by Varying $\rho$')
plt.axis([40, 180, 0, 0.055])
plt.xlabel('$S_T$')
plt.ylabel('Density')

strikes = np.arange(50, 150,1)

puts = [] 

for K in strikes:
    P = np.mean(np.maximum(K-prices_neg, 0)) * np.exp(-r*T)
    puts.append(P)


ivs = [implied_vol(P, S, K, T, r, type_ = 'put') for P, K in zip(puts, strikes)]

plt.plot(strikes, ivs, label='Heston')
plt.plot(strikes, merton_ivs, label='Merton')
plt.ylabel('Implied Volatility')
plt.xlabel('Strike')
plt.axvline(S, color='black',linestyle='--',
            label='Spot Price')
plt.title('Implied Volatility Smile from Heston Model')
plt.legend()
plt.show()

kappa = 3
theta = 0.04
v_0 =  0.04
xi = 0.6
r = 0.05
S = 100
paths = 3
steps = 10000
T = 1
rho = -0.8
prices, sigs = generate_heston_paths(S, T, r, kappa, theta,
                                    v_0, rho, xi, steps, paths,
                                    return_vol=True)        
    
plt.figure(figsize=(7,6))
plt.plot(prices.T)
plt.title('Heston Price Paths Simulation')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.show()

plt.figure(figsize=(7,6))
plt.plot(np.sqrt(sigs).T)
plt.axhline(np.sqrt(theta), color='black', label=r'$\sqrt{\theta}$')
plt.title('Heston Stochastic Vol Simulation')
plt.xlabel('Time Steps')
plt.ylabel('Volatility')
plt.legend(fontsize=15)
plt.show()


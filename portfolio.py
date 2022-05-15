# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize as sco
import scipy.interpolate as sli 

# %%
#bank: ANZ.AX
#energy&resource: BHP.AX
#Technology: WTC.AX
p_data = pd.read_csv('portfolio3.csv', index_col = 0)
#calculate daily return & delete fist day NA
p_data['ANZ_Returns'] = p_data['ANZ.AX']/p_data['ANZ.AX'].shift(1) - 1
p_data['BHP_Returns'] = p_data['BHP.AX']/p_data['BHP.AX'].shift(1) - 1
p_data['WTC_Returns'] = p_data['WTC.AX']/p_data['WTC.AX'].shift(1) - 1
p_data = p_data.dropna()

#copy the return data into a new dataframe
rts = pd.DataFrame(p_data, columns= ['ANZ_Returns','BHP_Returns','WTC_Returns']).copy()

#get num of stockse
symbols = rts.columns
num = len(symbols)

# %%
#weight simulation
#run 10000 times
rts_y = [] #simulated portfolio's annual return
vol_y = [] #simulated portfolio's standard deviation(volatility)
i = 10000
rts_mean_annual = rts.mean() * 252
rts_cov_annual = rts.cov() * 252

#simulation
for portfolio in range(i):
    weights = np.random.random(num) #generate random weights
    weights /= np.sum(weights) #standarlized weight to make sure we have a weight total of 1 (no leverage is used)
    rts_y.append(np.sum(rts_mean_annual * weights)) #calculate portfolio's annual return
    vol_y.append(np.sqrt(np.dot(weights.T, np.dot(rts_cov_annual, weights)))) #calculate portfolio's co-variance
#store simulated results into np.array format
rts_y = np.array(rts_y)
vol_y = np.array(vol_y)


# %%
#Draw feasible region of the portfolio
plt.figure(figsize = (16,10))
plt.grid(True)
plt.scatter(vol_y, rts_y, c = rts_y/vol_y, marker = 'o')
plt.xlabel('STD')
plt.ylabel('Return')


# %%
#define some functions to help with effecient frontier
def portfolio_stats(weights, rts):
    weights = np.array(weights)
    rts_y = np.sum(rts_mean_annual * weights) #return of a specific portfolio 
    vol_y = np.sqrt(np.dot(weights.T, np.dot(rts_cov_annual, weights))) #volatility of a specific portfolio (standard deviation)
    return np.array([rts_y, vol_y, rts_y/vol_y]) #calculate the portfolio returns over its volatility

def portfolio_min_ratio(weight, rts):
    return -portfolio_stats(weight, rts)[2] #define a function to find the max rts/vol ratio (minimum in negative)

def portfolio_min_vol(weights, rts):
    return portfolio_stats(weights,rts)[1] #define a funciton to find return the portfolios standard deviation

def portfolio_min_var(weight, rts): 
    return portfolio_stats(weight, rts)[1] ** 2 # define a function to ruturn the co-variance

# %%
#find out the maximum rts/vol ratio y = min(x0*r0 + x1*r1 + x2*r2) 1-x0-x1-x2 = 0 for 0<=x<=1
cons = ({'type':'eq','fun':lambda x:1 - np.sum(x)}) #set up constraint (total weigth = 1)
w_range = tuple((0,1) for x in range (num)) #set up weights ranges for each stock (0~1)
min_ratio = sco.minimize(lambda x:portfolio_min_ratio(x,rts), num * [1.0/num], method = 'SLSQP', bounds = w_range, constraints = cons) 
min_ratio_weights = min_ratio['x'].round(2) 


# %%
#find out minimized volatility
rts_min=min(rts_y)
rts_max=max(rts_y)
vol_min=min(vol_y)
index_min_vol=np.argmin(vol_y)
rts_start=rts_y[index_min_vol]

t_rts=np.linspace(rts_start, rts_max, 100)
t_vols=[]

for t_rt in t_rts:
    cons = ({'type':'eq','fun':lambda x: portfolio_stats(x,rts)[0]-t_rt},
           {'type':'eq','fun':lambda x: 1-np.sum(x)})
    res = sco.minimize(lambda x: portfolio_min_vol(x,rts), num*[1.0/num], method='SLSQP', bounds=w_range, constraints=cons)
    t_vols.append(res['fun'])
tvols=np.array(t_vols) #target volatility
    


# %%
#draw the effecient frontier
plt.figure(figsize=(16,10))
plt.grid(True)
plt.scatter(t_vols, t_rts, c = t_rts/t_vols, marker = 'x')
plt.xlabel('volatility')
plt.ylabel('return')
plt.title('Efficient Frontier')

# %%
#CML
#1 year Treasurey yield is 1.99% ATM

#numerical approximation via interpolation of frontier and its first derivative
index = np.argmin(t_vols)
e_rts = t_rts[index:]
e_vols = t_vols[index:]
tck = sli.splrep(e_vols, e_rts)

def f(x):
    return sli.splev(x, tck, der = 0) #f(x) itsself

def df(x):
    return sli.splev(x, tck, der = 1) #first derivative of f(x)

def equations(p, rf = 0.0199): #parameter p refers to risk-free rate, maximun sharp ratio, tangency portfolio variance 
    eq1 = rf - p[0]                    
    eq2 = rf + p[1]*p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3  #should be all zeros
#find the CML
opt_p = sco.fsolve(equations, [0.0199, 0.5, 0.25], xtol=1e-05) #xtol for acceptable error range


# %%
#combine the feasible area, effecient frontier and CML
plt.figure(figsize = (12,8))
plt.grid(True)
plt.axhline(0, color = 'k', ls = '--', lw =2.0)
plt.axvline(0, color = 'k', ls = '--', lw =2.0)
plt.xlabel('volatility')
plt.ylabel('return')
plt.scatter(vol_y, rts_y, c = rts_y/vol_y, marker = 'o')
plt.plot(e_vols, e_rts, 'b', lw = 2.0)
plt.plot(opt_p[2], f(opt_p[2]), 'r*')
cml_v = np.linspace(0.0, max(vol_y))
plt.plot(cml_v, opt_p[0] + opt_p[1] * cml_v, lw = 1.5)


# %%
#allocation of each assets

target_r = f(opt_p[2])
cons = ({'type':'eq', 'fun':lambda x: portfolio_stats(x,rts)[0]-target_r},
        {'type':'eq', 'fun':lambda x: 1 - np.sum(x)})
w_range = tuple((0,1) for x in range(num))
solution = sco.minimize(lambda x: portfolio_min_vol(x,rts), num*[1.0/num], method='SLSQP', bounds=w_range, constraints=cons)

optimal_w = []
optimal_w.append(solution.x.round(2))

print('The weight of each asset class is\n', optimal_w)



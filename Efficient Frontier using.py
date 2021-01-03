# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:14:06 2020

@author: Meekey
"""

import quandl
import cvxopt
import pandas as pd
import numpy as np
from bokeh.palettes import Set1
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import Range1d, HoverTool, CrosshairTool, NumeralTickFormatter
from bokeh.layouts import row, gridplot
from bokeh.models.callbacks import CustomJS
from bokeh.transform import dodge
from bokeh.io import push_notebook, output_notebook
from datetime import datetime, timedelta
from math import sqrt, log, pi
output_notebook()


import cvxopt
from gurobipy import *
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.io import output_notebook
from bokeh.models import Range1d, HoverTool, NumeralTickFormatter
output_notebook()
import pandas as pd
import yfinance as yf
import datetime as dt
import bokeh.palettes
from bokeh.io import output_file, show, save





stk_price = pd.DataFrame()
for ticker in stocks:    
        stk_price[ticker] = yf.download(ticker, start, end)["Adj Close"]
        
stk_price.isna().any()


stk_price = stk_price.dropna()
            
stk_price.to_csv("temp.csv")

cl_price = pd.read_csv("temp.csv")
cl_price.Date = pd.to_datetime(cl_price.Date)
cl_price = cl_price.set_index('Date').resample('M').mean()

market_price = cl_price[stocks[0]]
cl_price = cl_price.drop([stocks[0]], axis = 1)

market_ret = market_price.pct_change().dropna()
cum_market_ret = (market_ret +1).cumprod()

cl_price_ret = cl_price.pct_change()
cl_price_ret = cl_price_ret.dropna()
cum_stock_ret = (cl_price_ret + 1).cumprod()


matrix_cov_stock_ret = (cl_price_ret.cov()) *12
matrix_cov_stock_ret

from bokeh.palettes import all_palettes
from bokeh.palettes import Category20c
#CAGR
t = len(cum_stock_ret)/12
stk_annual_growth = (cum_stock_ret.tail(1))**(1/t) - 1
mkt_annual_growth = (cum_market_ret[-1])**(1/t) - 1
stk_annual_growth['market'] = mkt_annual_growth

bar_colors = list(Category20c[len(stk_annual_growth.columns)+1])
stk_annual_growth.T.plot.bar(title = 'CAGR', color = bar_colors.pop())

avg_market_ret = market_ret.iloc[-3:].mean()

import seaborn as sns
from scipy import stats

#Calculate Stock Beta
stk_beta ={}
stk_ex_ret ={}
rf = 0.065/12
def calc_stock_beta(mkt_ret, stock_ret, rf):
    for ticker in stock_ret:
        stock_beta, stk_alpha = stats.linregress(mkt_ret, 
                               stock_ret[ticker])[0:2]
        stk_beta[ticker] = stock_beta
        

calc_stock_beta(market_ret, cl_price_ret, rf)
market_ret[-1]


#calculate annual stk exp return
rf = 0.065
annual_expected_stock_ret = {}
for ticker in stk_beta:
    annual_expected_stock_ret[ticker] = rf + stk_beta[ticker] * (mkt_annual_growth - rf)

annual_expected_stock_ret = pd.core.series.Series(annual_expected_stock_ret)

syms = matrix_cov_stock_ret.columns
pd.set_option('display.max_rows', None)
ret_st = 50
ret_end = 200
step = 5
risk = []

df=pd.DataFrame()
for mean_ret in range(ret_st,ret_end,step):
    mean_ret  = mean_ret/1000
    m = Model()
    #ret = ret/1000
    #create one variable for each stock
    portvars = [m.addVar(name = symb, lb = 0.0) for symb in syms]
    portvars = pd.Series(portvars, index = syms)
    portfolio = pd.DataFrame({'Variable' : portvars})
    m.update()
    p_total = portvars.sum()
    #p_return = stats['Mean return'].dot(portvars)
    p_return = annual_expected_stock_ret.dot(portvars)
    p_risk = matrix_cov_stock_ret.dot(portvars).dot(portvars)
    #p_risk = np.dot(weights.T,np.dot(matrix_cov_stock_ret, weights))
    m.setObjective(p_risk, GRB.MINIMIZE)
    #summ of weights
    m.addConstr(p_total==1)
    fixreturn = m.addConstr(p_return==mean_ret)
    m.update()
    #select simplex algorithm
    m.setParam('Method', 1)
    m.optimize()
    #portfolio['2.5% return'] = portvars.apply(lambda x:x.getAttr('x'))
    wts = portvars.apply(lambda x:x.getAttr('x'))
    m.remove(fixreturn)
    #portfolio_var = np.dot(wts,np.dot(matrix_cov_stock_ret, wts.T))
    portfolio_var = np.matmul(wts.T,np.matmul(matrix_cov_stock_ret, wts))
    portfolio_std = np.sqrt(portfolio_var)
    #portfolio_std
    risk.append(portfolio_std)
    df[mean_ret] = wts
    
 
x = np.array(risk)
#y = np.array(ret)
y = np.array(range(ret_st,ret_end,step))/1000
efficient_frontier_table = pd.DataFrame()
efficient_frontier_table['risk'] = x
efficient_frontier_table['ret'] = y

import matplotlib.pyplot as plt
plt.figure(figsize=(16, 5))
plt.scatter(x, y, c = (y-rf)/x)
plt.xlabel('volatility')
plt.ylabel('return')
plt.colorbar(label = 'Sharpe Ratio')



min(risk)
ret4_min_risk = efficient_frontier_table['ret'][efficient_frontier_table['risk']==min(risk)]
#ret4_min_risk = efficient_frontier_table['ret'][efficient_frontier_table['risk']>cal_wts4_exp_ret]
wts4_min_risk = df[ret4_min_risk]
#annual growth

bar_colors = list(Category20c[len(wts4_min_risk.T.columns)+1])
wts4_min_risk.plot.bar(title = 'ret:{},risk:{}'.format(ret4_min_risk,round(min(risk),2)), color = bar_colors.pop())


i=0
annualized_return=0
weighted_expected_stock_return ={}
for x, y in expected_stock_ret.items():
    weighted_expected_stock_return[x] = np.sum(y * wts4_min_risk.iloc[i:i+1])
    wt_exp_port_monthly_ret = annualized_return + (np.sum(y * wts4_min_risk.iloc[i:i+1]))
    i += 1
    #temp[x] = np.sum(y * weights[i])
    
wt_exp_port_monthly_ret


i=0
wt_portfolio_inv = {}
for ticker in cl_price.columns:
    wt_portfolio_inv[ticker] = cl_price[ticker]*wts4_min_risk.iloc[i:i+1]
    i += 1

risk[int(len(risk)/2)]

portfolio_var = np.dot(weights.T,np.dot(matrix_cov_stock_ret, weights))
portfolio_std = np.sqrt(portfolio_var)
portfolio_std

sharpe_ratio = (wt_exp_port_monthly_ret - rf)/portfolio_std
sharpe_ratio

weighted_price = np.matmul(cl_price,wts4_min_risk)


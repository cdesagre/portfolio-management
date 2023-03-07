# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:06:27 2023

@author: CDES1
"""

import numpy as np
import pandas as pd

def compute_portfolio_rets(weights, rets):
    portfolio_rets = weights @ rets
    df = pd.DataFrame(portfolio_rets, 
                      columns=['PortfolioRets'])
    return df

def annualize_rets(rets, frequency):
    AnnR = np.prod(1+rets)**(frequency/rets.shape[0]) - 1 
    return AnnR

def compute_portfolio_volatility(weights, rets, freq=12):
    cov = rets.cov() 
    portfolio_var = weights.T @ cov @ weights 
    portfolio_vol = np.sqrt(portfolio_var * freq)
    return portfolio_vol 
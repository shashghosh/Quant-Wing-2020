# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 23:38:42 2020

@author: Shaswat
Binomial call price of option is calculated below using the same logic as pseudocode in the question paper
"""
import math


def binomial_call(A, K, t, r, vol, n):
    """
    A - stock price
    K - strike price
    t - time until expiry 
    r - risk-free rate
    vol - volatility
    n - number of steps in the model
    
    """
    timestep = t/n
    disc_f = math.exp(-r*timestep)
    temp1 = math.exp((r+vol*vol)*timestep)
    temp2 = 0.5*(disc_f+temp1)
    u = temp2 + math.sqrt(temp2*temp2-1)
    d = 1/u
    p = (math.exp(r*timestep) - d) / (u - d)
    c = {}
    for m in range(0, n+1):
            c[(n, m)] = max(A * (u ** (2*m - n)) - K, 0)
    for k in range(n-1, -1, -1):
        for m in range(0,k+1):
            c[(k, m)] = disc_f * (p * c[(k+1, m+1)] + (1-p) * c[(k+1, m)])
    price = c[0,0]
    return price

binomial_call(100,105,1,0.06,0.1,1000)

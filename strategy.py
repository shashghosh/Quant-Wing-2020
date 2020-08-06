# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 01:39:16 2020

@author: Shaswat
"""
"""
Simple MACD strategy which uses consecutive fibonacci numbers as parameters for signal lines and MACD.
Alpha, CAGR, Sharpe ratio, Win% and total number of trades are the performance measures used
Performs best on an intermediate timeframe of 6 months to 2 years.
Performs poorly when markets trade flat
Ignore the Copy Warning
"""
import pandas as pd
import datetime as dt
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
n = 0
wins = 0
losses = 0
trades = 0

delta = 365 #number of days lookback
fibo = 5#first fibonacci number for the signal line in MACD
start = dt.datetime.today() - dt.timedelta(delta) 
end = dt.datetime.today()

tickers = "^NSEI" #Nifty 50 index ticker

ohlcv = yf.download(tickers,start,end)

def MACD(DF,a,b,c): #fn to calculate MACD
    df = DF.copy()
    df["MAfast"] = df["Adj Close"].ewm(span=a,min_periods=a).mean()
    df["MAslow"] = df["Adj Close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MAfast"]-df["MAslow"]
    df["Signal"] = df["MACD"].ewm(span=c,min_periods=c).mean()
    df.dropna(inplace=True)
    return df


def strategy(DF,a,b,c):
    df = DF.copy()
    df = MACD(df,a,b,c)
    
    df["Position"] = None
    for row in range(len(df)):
        if (df["MACD"].iloc[row]>df["Signal"].iloc[row]):
            df["Position"].iloc[row] = 1
        elif (df["MACD"].iloc[row] < df["Signal"].iloc[row]):
            df["Position"].iloc[row] = -1

    df["Market_Return"] = np.log(df["Adj Close"]/df["Adj Close"].shift(1))
    df["Strategy_ret"] = df["Market_Return"] * df["Position"]
    return df

def CAGR(DF):
    df = DF.copy()
    
    df["cum_ret"] = 1+(df["Strategy_ret"]).cumsum()
    n = len(df)/252
    CAGR = (df["cum_ret"].tolist()[-1])**(1/n) - 1
    return CAGR

def volatility(DF):
    df = DF.copy()
    vola = df["Strategy_ret"].std() * np.sqrt(252)
    return vola


def sharpe(DF,rf):
    df = DF.copy()
    ratio = (CAGR(df) - rf)/volatility(df)
    return ratio

def Fibonacci(n): 
    if n==1: 
        return 0 
    elif n==2: 
        return 1
    else: 
        return Fibonacci(n-1)+Fibonacci(n-2)


fig, ax = plt.subplots()    

finaldf = pd.DataFrame

finaldf = strategy(ohlcv,Fibonacci(fibo+1),Fibonacci(fibo+2),Fibonacci(fibo)) 
plt.plot((finaldf["Strategy_ret"]).cumsum())
plt.plot((finaldf["Market_Return"]).cumsum())   

plt.title("Market Return vs Strategy Return")
plt.ylabel("Profit %")
plt.xlabel("Time")
ax.legend(["Strategy Returns","Market Return"])
 
for i in range(len(finaldf)):
    if(finaldf["Strategy_ret"][i]<0):
        n= n+1
    losses = n 
for i in range(1,len(finaldf)):
    if(finaldf["Position"][i]!=finaldf["Position"][i-1] ):
        trades = trades + 1
    
wins = len(finaldf) - losses 
    
alpha = (1+finaldf["Strategy_ret"]).cumsum()[-1] - (1+finaldf["Market_Return"]).cumsum()[-1]
print("Alpha = ", round(alpha*100,2),"%")
print("CAGR = ", round(CAGR(finaldf)*100,2), "%")
print("Sharpe = ", round(sharpe(finaldf, 0.06),2))
print("Wins % = ", wins/(losses+wins) * 100)    
print("Total Trades = ", trades, " in ",delta,"days" )

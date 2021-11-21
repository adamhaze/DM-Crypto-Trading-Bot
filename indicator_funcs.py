import pandas as pd
import numpy as np

##RSI Formula given data:
#using 14 periods (time indicators)
#calc avg gain:
#calc avg loss:
# |close - open| / open = //gain or loss over period

def RSI(array): #pass in date, and dataframe (array)
    #use pandas to interate through index: each has 
    avg_gain = 0
    avg_loss = 0
    rsi_arr = []
    x = 0
    
    ##range(0,13) close = 3 open = 0
    for x in range(0,14):
        Close = array.iloc[x,3]
        Open = array.iloc[x,0]
        if Open > Close:
            gain = (Open - Close) / Open
            avg_gain = avg_gain + gain
            
        elif Close > Open:
            loss = (Close - Open) / Open
            avg_loss = avg_loss + loss

        else: continue

    avg_gain = avg_gain / len(array)
    avg_loss = avg_loss / len(array)

    if avg_loss == 0.0 or avg_gain == 0.0:
        return [0.0]

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1+rs))
    rsi_arr.append(rsi)
    
    ##greater than 14 entries:
    
    for x in range(14,len(array)):
        Close = array.iloc[x,3]
        Open = array.iloc[x,0]
        if Open > Close:
            gain = (Open - Close) / Open
            avg_gain = ((avg_gain + gain) * 13) / 14
            avg_loss = ((avg_loss + 0) * 13) / 14
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1+rs))
            rsi_arr.append(rsi)
            
        elif Close > Open:
            loss = (Close - Open) / Open
            avg_gain = ((avg_gain + 0) * 13) / 14
            avg_loss = ((avg_loss + loss) * 13) / 14
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1+rs))
            rsi_arr.append(rsi)
            
        else: continue

    
    #rsi will be last value (maybe get an array of last rsi values?)
    return rsi_arr


#moving average
def MA(array):
    
    total = 0
    moving_avg = 0
    for x in range(len(array)):
        Close = array.iloc[x,3]
        total = total + Close
    
    moving_avg = total / (len(array))
    return moving_avg

#moving average convergence divergence
#Moving average convergence divergence (MACD) is a trend-following momentum indicator 
#that shows the relationship between two moving averages of a security’s price. 
#The MACD is calculated by subtracting the 26-period exponential moving average (EMA) 
#from the 12-period EMA
def MACD(array):
    MACD = 0
    SMA = MA(array)
    smooth_factor = 2 / ((len(array))+1)
    EMA_t = 0
    EMA_y = 0
    short_EMA_arr = []
    long_EMA_arr = []
    
    #short term EMA = 12 periods
    for x in range(14,26):
        price = array.iloc[x,0]
        EMA_t = (price * (smooth_factor)) + (EMA_y * (1-smooth_factor))
        short_EMA_arr.append(EMA_t)
        EMA_y = EMA_t
    
    
    #long term EMA = 26 periods
    EMA_y = 0
    for x in range(0,26):
        price = array.iloc[x,0]
        EMA_t = (price * (smooth_factor)) + (EMA_y * (1-smooth_factor))
        long_EMA_arr.append(EMA_t)
        EMA_y = EMA_t
    
    
    
    return MACD
#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[16]:


df = [pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\RELIANCE.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\SBIN.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\SHREECEM.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\stock_metadata.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\SUNPHARMA.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\TATAMOTORS.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\TATASTEEL.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\TCS.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\TECHM.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\TITAN.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\ULTRACEMCO.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\UPL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\VEDL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\WIPRO.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\ZEEL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\ADANIPORTS.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\ASIANPAINT.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\AXISBANK.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\BAJAJ-AUTO.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\BAJAJFINSV.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\BAJFINANCE.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\BHARTIARTL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\BPCL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\BRITANNIA.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\CIPLA.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\COALINDIA.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\DRREDDY.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\EICHERMOT.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\GAIL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\GRASIM.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\HCLTECH.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\HDFC.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\HDFCBANK.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\HEROMOTOCO.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\HINDALCO.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\HINDUNILVR.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\ICICIBANK.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\INDUSINDBK.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\INFRATEL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\INFY.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\IOC.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\ITC.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\JSWSTEEL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\KOTAKBANK.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\LT.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\MARUTI.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\MM.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\NESTLEIND.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\NIFTY50_all.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\NTPC.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\ONGC.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\POWERGRID.csv")]
df


# In[17]:


combined_dataset = pd.concat(df, ignore_index = True)
combined_dataset


# In[34]:


combined_dataset.to_csv('combined_dataset.csv', index = False)


# In[35]:


industry_ranking = combined_dataset.groupby('Series').agg({'Volume': 'mean', 'Trades': 'mean'}).sort_values(by='Volume', ascending=False).head(5)
industry_ranking.plot(kind='bar', y=['Volume', 'Trades'], title='Top 5 Series by Mean Volume and Trades') 


# In[22]:


since_inception_ranking = combined_dataset.groupby('Symbol').agg({'Trades': ['mean', 'sum']}).sort_values(by=('Trades', 'sum'), ascending=False).head(5)
since_inception_ranking.plot(kind='bar', y=('Trades', 'sum'), 
                             title='Top 5 Tickers by Sum of Trades Since Inception')


# In[23]:


combined_dataset['Date'] = pd.to_datetime(combined_dataset['Date'])
price_change = combined_dataset.groupby('Symbol')['Close'].agg(['first', 'last'])
price_change['delta_percentage'] = ((price_change['last'] - price_change['first']) / price_change['first']) * 100
best_stocks = price_change.sort_values(by='delta_percentage', ascending=False).head(5)
best_stocks.plot(kind='bar', y='delta_percentage', title='Top 5 Stocks by Delta Price % Since Inception')


# In[9]:


worst_stocks = price_change.sort_values(by='delta_percentage').head(5)
worst_stocks.plot(kind='bar', y='delta_percentage', title='Top 5 Stocks by Delta Price % Since Inception')


# In[24]:


std_deviation = combined_dataset.groupby('Symbol')['Close'].std()
mean_prices = combined_dataset.groupby('Symbol')['Close'].mean()
coefficient_of_variation = std_deviation / mean_prices
coefficient_of_variation.plot(kind='bar', title='Coefficient of Variation for Each Stock')


# In[46]:


import pandas as pd
import numpy as np

def calculate_daily_returns(df):
    df['Daily_Return'] = df['Close'].pct_change()
    return df

def calculate_volatility(df):
    daily_returns = df['Daily_Return'].dropna()
    volatility = np.std(daily_returns) * np.sqrt(252)  # Assuming 252 trading days in a year
    return volatility

def calculate_sharpe_ratio(df):
    daily_returns = df['Daily_Return'].dropna()
    avg_daily_return = daily_returns.mean()
    volatility = np.std(daily_returns) * np.sqrt(252)
    risk_free_rate = 0.01  # Adjust as needed
    sharpe_ratio = (avg_daily_return - risk_free_rate) / volatility
    return sharpe_ratio

# Load multiple datasets
df = [pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\RELIANCE.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\SBIN.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\SHREECEM.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\stock_metadata.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\SUNPHARMA.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\TATAMOTORS.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\TATASTEEL.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\TCS.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\TECHM.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\TITAN.csv"), pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\ULTRACEMCO.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\UPL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\VEDL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\WIPRO.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\ZEEL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\ADANIPORTS.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\ASIANPAINT.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\AXISBANK.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\BAJAJ-AUTO.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\BAJAJFINSV.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\BAJFINANCE.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\BHARTIARTL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\BPCL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\BRITANNIA.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\CIPLA.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\COALINDIA.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\DRREDDY.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\EICHERMOT.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\GAIL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\GRASIM.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\HCLTECH.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\HDFC.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\HDFCBANK.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\HEROMOTOCO.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\HINDALCO.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\HINDUNILVR.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\ICICIBANK.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\INDUSINDBK.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\INFRATEL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\INFY.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\IOC.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\ITC.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\JSWSTEEL.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\KOTAKBANK.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\LT.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\MARUTI.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\MM.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\NESTLEIND.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\NIFTY50_all.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\NTPC.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\ONGC.csv"),pd.read_csv(r"C:\Users\BOOMI\Downloads\archive (18)\POWERGRID.csv")]


# Calculate and print risk and Sharpe ratio for each dataset
for i, dataset in enumerate(df, start=1):
    try:
        dataset = calculate_daily_returns(dataset)
        risk = calculate_volatility(dataset)
        sharpe_ratio = calculate_sharpe_ratio(dataset)

        print(f"\nDataset {i}:")
        print(f"Risk Ratio: {risk}")
        print(f"Sharpe Ratio: {sharpe_ratio}")
    except KeyError:
        print(f"\nDataset {i} Not fit")


# In[ ]:





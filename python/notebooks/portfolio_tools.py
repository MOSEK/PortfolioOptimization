import os
import re
import time 
import glob
    
import numpy as np
import pandas as pd
    
URL = r"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={token}&outputsize=full&datatype=csv"
FOLDER = "stock_data"
FNAME = "daily_adjusted_{ticker}.csv"


def data_download(list_tickers, token):
    import requests
    for ticker in list_tickers:
        url = URL.format(ticker=ticker, token=token)
        result = requests.get(url, allow_redirects=True)
        folder = FOLDER.format(ticker=ticker)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        fname = FNAME.format(ticker=ticker)
        open(os.path.join(folder, fname), "wb").write(result.content)
        print(f"Downloaded {fname}.")
        # Limit is 5 downloads per minute.
        time.sleep(12)   
    
    
def alpha_numerator(Z, S):
    s = 0
    T = Z.shape[1]
    for k in range(T):
        z = Z[:, k][:, np.newaxis]
        X = z @ z.T - S
        s += np.trace(X @ X)
    s /= (T**2)
    return s
    
    
def mean_shrinkage_JS(m_estim, S_estim, return_array):
    # James--Stein estimator
    T = return_array.shape[0]
    N = m_estim.shape[0]
    m = m_estim[:, np.newaxis]
    o = np.ones(N)[:, np.newaxis]
    iS = np.linalg.inv(S_estim)
    b = (o.T @ m / N) * o
    N_eff = np.trace(S_estim) / np.max(np.linalg.eigvalsh(S_estim))
    alpha_num = max(N_eff - 3, 0)  # Negative value would not give improvement.
    alpha_den = T * (m - b).T @ iS @ (m - b) 
    alpha = alpha_num / alpha_den
    m_shrunk = b + max(1 - alpha, 0) * (m - b)
    m_shrunk = m_shrunk[:, 0]
    
    return m_shrunk


def cov_shrinkage_LW(m_estim, S_estim, return_array):
    # Ledoit--Wolf shrinkage
    N = S_estim.shape[0]
    s2_avg = np.trace(S_estim) / N 
    B = s2_avg * np.eye(N)
    Z = return_array.T - m_estim[:, np.newaxis]
    alpha_num = alpha_numerator(Z, S_estim) 
    alpha_den = np.trace((S_estim - B) @ (S_estim - B))
    alpha = alpha_num / alpha_den
    S_shrunk = (1 - alpha) * S_estim + alpha * B
    
    return S_shrunk
    
    
def compute_inputs(
        list_df_prices, 
        sample_period='W', 
        investment_horizon=1, 
        show_histograms=False, 
        shrinkage=False, 
        security_num=None,
        return_log=False
    ):
    map_period = {
        'W': 52
    }
    
    # We can generate return distribution based on multiple periods of price data
    if not isinstance(list_df_prices, list):
        list_df_prices = [list_df_prices]
        
    df_weekly_log_returns = pd.DataFrame()
    for df_prices in list_df_prices:
        # PREPROC: Remove factors
        if security_num is not None: 
            df_prices = df_prices.iloc[:, 0:security_num]

        # 1. Compute weekly logarithmic return
        df_weekly_prices = df_prices.resample(sample_period).last()
        df_weekly_log_returns_part = np.log(df_weekly_prices) - np.log(df_weekly_prices.shift(1))
        df_weekly_log_returns_part = df_weekly_log_returns_part.dropna(how='all')
        df_weekly_log_returns_part = df_weekly_log_returns_part.fillna(0)
        
        df_weekly_log_returns = pd.concat([df_weekly_log_returns, df_weekly_log_returns_part], ignore_index=True)

    if show_histograms:
        df_weekly_log_returns.hist(bins=50)
    
    # 2. Compute the distribution of weekly logarithmic return
    return_array = df_weekly_log_returns.to_numpy()
    T = return_array.shape[0]
    m_weekly_log = np.mean(return_array, axis=0)
    S_weekly_log = np.cov(return_array.transpose())
    
    # Apply shrinkage if needed
    if shrinkage:
        m_weekly_log = mean_shrinkage_JS(m_weekly_log, S_weekly_log, return_array)
        S_weekly_log = cov_shrinkage_LW(m_weekly_log, S_weekly_log, return_array)
    
    # 3. Project the distribution to the investment horizon
    scale_factor = investment_horizon * map_period[sample_period]
    m_log = scale_factor * m_weekly_log
    S_log = scale_factor * S_weekly_log
    
    if return_log:
        return m_log, S_log
    
    # 4. Compute the distribution of yearly linear return
    p_0 = np.ones(len(m_log))  # We use a dummy price here to see the method in two steps. It will be canceled out later. 
    m_P = p_0 * np.exp(m_log + 1/2*np.diag(S_log))
    S_P = np.outer(m_P, m_P) * (np.exp(S_log) - 1)
    
    m = 1 / p_0 * m_P - 1
    S = 1 / np.outer(p_0, p_0) * S_P
        
    return m, S
    

class DataReader(object):
    def __init__(self, folder_path, symbol_list=None):
        self.folder_path = folder_path
        self.name_format = r"daily_adjusted_*.csv"
        self.symbol_list = symbol_list if symbol_list is not None else []
        self.df_prices = None
        self.df_volumes = None

    def read_data(self, read_volume=False):
        # Get list of files from path, named as name_format 
        list_files = glob.glob(os.path.join(self.folder_path, self.name_format))
        file_names = "\n".join(list_files)
        print("Found data files: \n{}\n".format(file_names))

        # Keep only ones in symbol list (if given)
        if self.symbol_list:
            list_to_read = [
                os.path.join(self.folder_path, self.name_format.replace("*", symbol))
                for symbol in self.symbol_list
            ]
            list_missing = [fname for fname in list_to_read if fname not in list_files]
            if list_missing: 
            	raise Exception(f"Files are missing: {list_missing}")
            file_names = "\n".join(list_to_read)
            print("Using data files: \n{}\n".format(file_names))
        else:
            list_to_read = list_files
            print("Using all data files.")
            
        # Collect data from the files into a Dataframe
        dict_prices = {}
        dict_volumes = {}
        for file_name in list_to_read: 
            m = re.search(self.name_format.replace("*", "(.+)"), file_name)
            
            # Get symbol name
            symbol = m.group(1)
        
            # Read data file
            df_data = pd.read_csv(file_name)

            # Set timestamp as index 
            df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
            df_data = df_data.set_index('timestamp')
            df_data.index.name = "date"

            # Obtain adjusted close price data 
            dict_prices[symbol] = df_data['adjusted_close']
            
            # Obtain volumes data
            if read_volume:
                dict_volumes[symbol] = df_data['volume']

        self.df_prices = pd.concat(dict_prices.values(), axis=1, keys=dict_prices.keys()).sort_index()
        if read_volume:
            self.df_volumes = pd.concat(dict_volumes.values(), axis=1, keys=dict_volumes.keys()).sort_index()
        
    def get_period(self, start_date, end_date):         
        start_idx = self.df_prices.index.get_indexer([pd.to_datetime(start_date)], method='nearest')[0]
        end_idx = self.df_prices.index.get_indexer([pd.to_datetime(end_date)], method='nearest')[0]
        df_prices = self.df_prices.iloc[start_idx:(end_idx + 1)].copy()
        if self.df_volumes is not None:
            df_volumes = self.df_volumes.iloc[start_idx:(end_idx + 1)].copy()
        else:
            df_volumes = pd.DataFrame()
        return df_prices, df_volumes

import marimo

__generated_with = "0.17.7"
app = marimo.App(width = "medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![MOSEK ApS](https://www.mosek.com/static/images/branding/webgraphmoseklogocolor.png )
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports and configuration
    """)
    return


@app.cell
def _():
    import sys, os, re, glob
    import datetime as dt
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Options
    np.set_printoptions(precision=5, linewidth=120, suppress=True)
    pd.set_option("display.max_rows", None)
    plt.rcParams["figure.figsize"] = [12, 8]

    # Diagnostic
    print(f"Python: {sys.version}")
    print(f"marimo: {mo.__version__}, matplotlib: {matplotlib.__version__}, pandas: {pd.__version__}, numpy: {np.__version__}")

    return glob, matplotlib, np, os, pd, plt, re, sys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prepare input data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we compute the optimization input variables, the vector $\mu$ of expected returns, and the covariance matrix $\Sigma$, from raw data. The data consists of daily stock prices of $8$ stocks from the US market. The data processing routines introduced in this notebook are used routinely in the remaining portfolio optimization notebooks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download data
    """)
    return


@app.cell
def _(DataReader):
    # Data format:
    #
    # This notebook loads data from the folder "stock_data", containing files with names
    # "TICKER.csv", where TICKER is the symbol of a stock. Each csv file contains (at least) columns
    # "date", "price", "volume".
    #
    # The DataReader class (see Appendix) will load data into two dataframes
    # df_prices and df_volumes, which are then used in the notebook.
    #
    # To use your own data prepare your own files and modify the configuration below. You can also
    # modify the DataReader to consume a different format or plug in your df_prices, df_volumes directly.

    list_stocks = ["PM", "LMT", "MCD", "MMM", "AAPL", "MSFT", "TXN", "CSCO"]
    list_factors = []
    list_tickers = list_stocks + list_factors
    investment_start = "2016-03-18"
    investment_end = "2021-03-18"

    dr = DataReader(
        folder_path="stock_data", symbol_list=list_tickers
    )
    dr.read_data()
    df_prices, _ = dr.get_period(
        start_date=investment_start, end_date=investment_end
    )
    return (df_prices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute yearly return statistics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we use the loaded daily price data to compute the corresponding yearly mean return and covariance matrix. 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Compute weekly logarithmic return
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First we convert the daily prices to weekly prices.
    """)
    return


@app.cell
def _(df_prices):
    df_weekly_prices = df_prices.resample('W').last()
    return (df_weekly_prices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Convert the weekly prices to weekly logarithmic return.
    """)
    return


@app.cell
def _(df_weekly_prices, np):
    df_weekly_log_returns = np.log(df_weekly_prices) - np.log(df_weekly_prices.shift(1))
    df_weekly_log_returns = df_weekly_log_returns.dropna(how='all')
    df_weekly_log_returns = df_weekly_log_returns.fillna(0)
    return (df_weekly_log_returns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see based on the histograms that the distribution of weekly logarithmic return is approximately normal.
    """)
    return


@app.cell
def _(df_weekly_log_returns, plt):
    df_weekly_log_returns.hist(bins=50)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Compute the distribution of weekly logarithmic return
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Assuming that the distribution is normal, we estimate the mean and covariance of the weekly logarithmic return.
    """)
    return


@app.cell
def _(df_weekly_log_returns, np):
    return_array = df_weekly_log_returns.to_numpy()
    T = return_array.shape[0]
    m_weekly_log = np.mean(return_array, axis=0)
    S_weekly_log = np.cov(return_array.transpose())
    return S_weekly_log, m_weekly_log


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Project the distribution to the investment horizon
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we project the logarithmic return statistics to the investment horizon of 1 year.
    """)
    return


@app.cell
def _(S_weekly_log, m_weekly_log):
    m_log = 52 * m_weekly_log
    S_log = 52 * S_weekly_log
    return S_log, m_log


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. Compute the distribution of yearly linear return
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We recover the distribution of prices from the distribution of logarithmic returns.
    """)
    return


@app.cell
def _(S_log, df_weekly_prices, m_log, np):
    p_0 = df_weekly_prices.iloc[0].to_numpy()
    m_P = p_0 * np.exp(m_log + 1/2*np.diag(S_log))
    S_P = np.outer(m_P, m_P) * (np.exp(S_log) - 1)
    return S_P, m_P, p_0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally we convert the distribution of prices to the distribution of yearly linear returns.
    """)
    return


@app.cell
def _(S_P, m_P, np, p_0):
    m = 1 / p_0 * m_P - 1
    S = 1 / np.outer(p_0, p_0) * S_P
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Appendix
    Data preparation tools used in this notebook can be reached from the functions defined below.
    """)
    return


@app.cell
def _(glob, os, pd, re):
    class DataReader(object):
        def __init__(self, folder_path, symbol_list=None):
            self.folder_path = folder_path
            self.name_format = r"*.csv"
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
                m = re.search(self.name_format.replace("*", "(.+)"), os.path.basename(file_name))

                # Get symbol name
                symbol = m.group(1)

                # Read data file
                df_data = pd.read_csv(file_name)

                # Set timestamp as index 
                df_data['date'] = pd.to_datetime(df_data['date'])
                df_data = df_data.set_index('date')
                df_data.index.name = "date"

                # Obtain adjusted close price data 
                dict_prices[symbol] = df_data['price']

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
    return (DataReader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. The **MOSEK** logo and name are trademarks of <a href="http://mosek.com">Mosek ApS</a>. The code is provided as-is. Compatibility with future release of **MOSEK** or the `Fusion API` are not guaranteed. For more information contact our [support](mailto:support@mosek.com).
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


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

    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix, Var, SolutionStatus
    import mosek.fusion.pythonic    # Requires MOSEK >= 10.2

    # Options
    np.set_printoptions(precision=5, linewidth=120, suppress=True)
    pd.set_option("display.max_rows", None)
    plt.rcParams["figure.figsize"] = [12, 8]

    # Diagnostic
    print(f"Python: {sys.version}")
    print(f"marimo: {mo.__version__}, matplotlib: {matplotlib.__version__}, pandas: {pd.__version__}, numpy: {np.__version__}, mosek: {Model.getVersion()}")

    return (
        Domain,
        Expr,
        Model,
        ObjectiveSense,
        SolutionStatus,
        glob,
        matplotlib,
        np,
        os,
        pd,
        plt,
        re,
        sys,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prepare input data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we load the raw data that will be used to compute the yearly return observation series used for the optimization. The data consists of daily stock prices of $8$ stocks from the US market, and SPY as the benchmark.
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
    list_factors = ["SPY"]
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
    return (df_prices)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Run the optimization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the optimization model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below we implement the optimization model in Fusion API. We create it inside a function so we can call it later.
    """)
    return


@app.cell
def _(
    Domain,
    Expr,
    Model,
    ObjectiveSense,
    SolutionStatus,
    df_prices,
    np,
    pd,
    sys,
):
    def absval(M, x, z):
        M.constraint(z >= x)
        M.constraint(z >= -x)

    def norm1(M, x, t):
        z = M.variable(x.getSize(), Domain.greaterThan(0.0))
        absval(M, x, z)
        M.constraint(Expr.sum(z) == t)

    def MinTrackingError(N, R, r_bm, x0, lambda_1, lambda_2, beta=1.5):

        with Model("Case study") as M:
            # Settings
            M.setLogHandler(sys.stdout)

            # Variables 
            # The variable x is the fraction of holdings in each security. 
            # It is restricted to be positive, which imposes the constraint of no short-selling.   
            x = M.variable("x", N, Domain.greaterThan(0.0))
            xt = x - x0

            # The variable t models the OLS objective function term (tracking error).
            t = M.variable("t", 1, Domain.unbounded())
            # The variables u and v model the regularization terms (transaction cost penalties).
            u = M.variable("u", 1, Domain.unbounded())
            v = M.variable("v", N, Domain.unbounded())   

            # Budget constraint
            M.constraint('budget', Expr.sum(x) == 1.0)

            # Objective 
            penalty_lin = lambda_1 * u
            penalty_32 = lambda_2 * Expr.sum(v)
            M.objective('obj', ObjectiveSense.Minimize, t + penalty_lin + penalty_32)

            # Constraints for the penalties
            norm1(M, xt, u)
            M.constraint('market_impact', Expr.hstack(v, Expr.constTerm(N, 1.0), xt), Domain.inPPowerCone(1.0 / beta))

            # Constraint for the tracking error 
            residual = R.T @ x - r_bm
            M.constraint('tracking_error', Expr.vstack(t, 0.5, residual), Domain.inRotatedQCone())

            # Create DataFrame to store the results. Last security name (the SPY) is removed.
            columns = ["track_err", "lin_tcost", "mkt_tcost"] + df_prices.columns[:N].tolist()
            df_result = pd.DataFrame()

            # Solve optimization
            M.solve()

            # Check if the solution is an optimal point
            solsta = M.getPrimalSolutionStatus()
            if (solsta != SolutionStatus.Optimal):
                # See https://docs.mosek.com/latest/pythonfusion/accessing-solution.html about handling solution statuses.
                raise Exception("Unexpected solution status!") 

            # Save results
            tracking_error = t.level()[0]
            linear_tcost = u.level()[0]
            market_impact_tcost = np.sum(v.level())
            row = pd.Series([tracking_error, linear_tcost, market_impact_tcost] + list(x.level()), 
                            index=columns)
            df_result = pd.concat([df_result, pd.DataFrame([row])], ignore_index=True)

            return df_result
    return (MinTrackingError,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute optimization input variables
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we use the loaded daily price data to compute the corresponding yearly mean return and covariance matrix for logarithmic returns, and compute linear return observations from that. The benchmark will be the last data series, corresponding to SPY.
    """)
    return


@app.cell
def _(compute_inputs, df_prices, np):
    # Number of securities and observations
    N = df_prices.shape[1] - 1
    T = 1000

    # Mean and covariance of historical yearly log-returns.  
    m_log, S_log = compute_inputs(df_prices, return_log=True)

    # Generate logarithmic return observations assuming normal distribution
    scenarios_log = np.random.default_rng().multivariate_normal(m_log, S_log, T)

    # Convert logarithmic return observations to linear return observations 
    scenarios_lin = np.exp(scenarios_log) - 1
    return N, T, scenarios_lin


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We center and normalize the data matrices.
    """)
    return


@app.cell
def _(N, T, np, scenarios_lin):
    # Center the return data
    centered_return = scenarios_lin - scenarios_lin.mean(axis=0)

    # Security return scenarios
    security_return = scenarios_lin[:, :N] / np.sqrt(T - 1)

    # Benchmark return scenarios
    benchmark_return = scenarios_lin[:, -1] / np.sqrt(T - 1)
    return benchmark_return, security_return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Call the optimizer function
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We run the optimization for the given penalty coefficients, and initial portfolio.
    """)
    return


@app.cell
def _(MinTrackingError, N, benchmark_return, np, security_return):
    lambda_1 = 0.0001 
    lambda_2 = 0.0001 
    x0 = np.ones(N) / N

    df_result = MinTrackingError(N, security_return.T, benchmark_return, x0, lambda_1, lambda_2)
    return (df_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check the results
    """)
    return


@app.cell
def _(df_result):
    df_result
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

            self.df_prices = pd.concat(dict_prices.values(), axis=1, keys=dict_prices.keys(), sort=True).sort_index()
            if read_volume:
                self.df_volumes = pd.concat(dict_volumes.values(), axis=1, keys=dict_volumes.keys(), sort=True).sort_index()

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


@app.cell
def _(cov_shrinkage_LW, mean_shrinkage_JS, np, pd):
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
    return (compute_inputs,)


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

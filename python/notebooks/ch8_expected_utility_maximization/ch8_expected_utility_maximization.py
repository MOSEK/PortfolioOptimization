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
        LinearSegmentedColormap,
        Model,
        ObjectiveSense,
        SolutionStatus,
        Var,
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
    Here we load the raw data that will be used to compute the optimization input variables, the vector $\mu$ of expected returns and the covariance matrix $\Sigma$. The data consists of daily stock prices of $8$ stocks from the US market. We split the data into three parts, each having different market regime. The return of each regime will be modeled using a Gaussian distribution.
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

    # Get first regime (low volatility)
    df_prices_1, _ = dr.get_period(start_date="2016-03-18", end_date="2018-01-31")

    # Get second regime (high volatility: crash)
    df_prices_2a, _ = dr.get_period(start_date="2018-02-01", end_date="2018-12-31")
    df_prices_2b, _ = dr.get_period(start_date="2020-02-01", end_date="2020-03-23")

    # Get third regime (high volatility: recovery)
    df_prices_3a, _ = dr.get_period(start_date="2019-01-01", end_date="2020-01-31")
    df_prices_3b, _ = dr.get_period(start_date="2020-03-24", end_date="2021-03-18")

    return df_prices_1, df_prices_2a, df_prices_2b, df_prices_3a, df_prices_3b


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
    Var,
    list_stocks,
    np,
    pd,
):
    # log(sum_i(p_i * exp(x_i))) <= t
    def logsumexp(M, x, p, t):
        n = int(x.getSize())
        u = M.variable(n)
        M.constraint(Expr.hstack(u, Expr.constTerm(n, 1.0), x - Var.repeat(t, n)), Domain.inPExpCone())
        M.constraint(u.T @ p <= 1.0)

    def ExpectedUtility(N, m, G, pi, deltas):

        with Model("Case study") as M:
            # Settings
            #M.setLogHandler(sys.stdout)

            assert len(m) == len(G), "Wrong number of components!"
            K = len(m)

            # Variables 
            # The variable x is the fraction of holdings relative to the initial capital.
            # It is constrained to take only positive values. 
            x = M.variable("x", N, Domain.greaterThan(0.0))

            # Budget constraint
            M.constraint('budget', Expr.sum(x) == 1.0)

            # Auxiliary variables.
            z = M.variable("z", 1, Domain.unbounded())
            q = M.variable("q", K, Domain.unbounded())

            # Constraint modeling log-sum-exp
            delta = M.parameter()
            mmat = np.vstack(m)
            expconex = delta * (-mmat @ x + delta * q)
            logsumexp(M, expconex, pi, z)

            # Constraints modeling quadratic terms
            for i in range(K):
                M.constraint(f'quad_{i}', Expr.vstack(q[i], 1.0, G[i].T @ x), Domain.inRotatedQCone())

            # Objective
            M.objective('obj', ObjectiveSense.Minimize, z)

            # Create DataFrame to store the results.
            columns = ["delta", "inv_delta", "obj"] + list_stocks
            df_result = pd.DataFrame()
            for d in deltas:
                # Update parameter
                delta.setValue(d);

                # Solve optimization
                M.solve()

                # Check if the solution is an optimal point
                solsta = M.getPrimalSolutionStatus()
                if (solsta != SolutionStatus.Optimal):
                    # See https://docs.mosek.com/latest/pythonfusion/accessing-solution.html about handling solution statuses.
                    raise Exception("Unexpected solution status!")

                # Save results
                row = pd.Series([d, 1/d, M.primalObjValue()] + list(x.level()), 
                                index=columns)

                df_result = pd.concat([df_result, pd.DataFrame([row])], ignore_index=True)

            return df_result
    return (ExpectedUtility,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute optimization input variables
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we use the loaded daily price data to compute the corresponding yearly mean return and covariance matrix.
    """)
    return


@app.cell
def _(
    compute_inputs,
    df_prices_1,
    df_prices_2a,
    df_prices_2b,
    df_prices_3a,
    df_prices_3b,
):
    # Regime probabilities
    pi1 = 0.6
    pi2 = 0.1
    pi3 = 0.3

    # Get optimization parameters
    m1, S1 = compute_inputs([df_prices_1])
    m2, S2 = compute_inputs([df_prices_2a, df_prices_2b])
    m3, S3 = compute_inputs([df_prices_3a, df_prices_3b])

    # Number of securities
    N = m1.shape[0]
    return N, S1, S2, S3, m1, m2, m3, pi1, pi2, pi3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we compute the matrix $G$ such that $\Sigma=GG^\mathsf{T}$, this is the input of the conic form of the optimization problem. Here we use Cholesky factorization.
    """)
    return


@app.cell
def _(S1, S2, S3, np):
    G1 = np.linalg.cholesky(S1)
    G2 = np.linalg.cholesky(S2)
    G3 = np.linalg.cholesky(S3)
    return G1, G2, G3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Call the optimizer function
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We run the optimization for a range of risk aversion parameter values: $\delta = 10^{-1},\dots,10^{2}$. We compute the efficient frontier this way both with and without using shrinkage estimation.
    """)
    return


@app.cell
def _(ExpectedUtility, G1, G2, G3, N, m1, m2, m3, np, pi1, pi2, pi3):
    # Compute efficient frontier with and without shrinkage
    deltas = np.logspace(start=-0.5, stop=3, num=30)[::-1]
    df_result = ExpectedUtility(N, [m1, m2, m3], [G1, G2, G3], [pi1, pi2, pi3], deltas)
    return (df_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check the results.
    """)
    return


@app.cell
def _(df_result):
    df_result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize the results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot the portfolio composition.
    """)
    return


@app.cell
def _(df_result, np):
    # Round small values to 0 to make plotting work
    mask = np.absolute(df_result) < 1e-7
    mask.iloc[:, :-8] = False
    df_result[mask] = 0
    return


@app.cell
def _(LinearSegmentedColormap, df_result, plt):
    my_cmap = LinearSegmentedColormap.from_list("non-extreme gray", ["#111111", "#eeeeee"], N=256, gamma=1.0)
    ax = df_result.set_index('inv_delta').iloc[:, 2:].plot.area(colormap=my_cmap, xlabel='risk seeking', ylabel="x", logx=True) 
    ax.grid(which='both', axis='x', linestyle=':', color='k', linewidth=1)
    plt.show()
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

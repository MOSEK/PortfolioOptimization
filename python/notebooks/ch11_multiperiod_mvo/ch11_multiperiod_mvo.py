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
    import mosek.fusion.pythonic   # Requires MOSEK >= 10.2

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
    Here we load the raw data that will be used to compute the optimization input variables, the vector $\mu$ of expected returns and the covariance matrix $\Sigma_t$ for all periods $t = 1, \dots, T$. The data consists of daily stock prices of $8$ stocks from the US market.
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
    dr.read_data(read_volume=True)
    df_prices, df_volumes = dr.get_period(
        start_date=investment_start, end_date=investment_end
    )
    return (df_prices,df_volumes)


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
    We will solve the following multiperiod optimization problem:

    $$
        \begin{array}{lrcl}
        \text{maximize}     & \sum_{t=1}^T\mu_t^\mathsf{T}\mathbf{x}_t - \delta_t \mathbf{x}_t^\mathsf{T}\Sigma_t\mathbf{x}_t - \left(\sum_{i=1}^N a_{t,i}|x_{t,i}-x_{t-1,i}| + \tilde{b}_{t,i}|x_{t,i}-x_{t-1,i}|^{3/2}\right)      &          &\\
        \text{subject to}   & \mathbf{1}^\mathsf{T}\mathbf{x}_t              & =        & 1,\\
                            & \mathbf{x}_t                                   & \geq     & 0.\\
        \end{array}
    $$

    The first term is the portfolio return in period $i$, the second term is the portfolio risk in period $i$, and the third term is a transaction cost term for period $i$. The $a_{t,i}$ are the coefficients of the linear cost term, and the $\tilde{b}_{t,i}$ are the coefficients of the market impact cost term: $\tilde{b}_{t,i} = b_{t,i}\sigma_{t,i}/\left(\frac{q_{t,i}}{V_t}\right)^{1/2}$, where $b_{t,i} = 1$, $\sigma_{t,i}$ is the volatility of security $i$ in period $t$, and $\frac{q_{t,i}}{V_t}$ is the portfolio value normalized dollar volume of security $i$ in period $t$. The total objective is the sum of these terms for all periods.

    Then we rewrite the above problem into conic form, and implement it in Fusion API:

    $$
        \begin{array}{lrcl}
        \text{maximize}     & \sum_{t=1}^T\mu_t^\mathsf{T}\mathbf{x}_t - \delta_t s_{t} - \left(\sum_{i=1}^N a_{t,i}v_{t,i} + \tilde{b}_{t,i}w_{t,i}\right)      &          &\\
        \text{subject to}   & (s_{t}, 0.5, \mathbf{G}_{t}^\mathsf{T}\mathbf{x}_t) & \in      & Q_\mathrm{r}^{N+2},\quad t = 1,\dots,T\\
                            & |x_{t}-x_{t-1}|                            & \leq     & v_{t},\quad t = 1,\dots,T\\
                            & (w_{t,i}, 1, x_{t,i}-x_{t-1,i})                & \in      & \mathcal{P}_3^{2/3,1/3},\quad t = 1,\dots,T,\ i = 1,\dots,N\\
                            & \mathbf{1}^\mathsf{T}\mathbf{x}_t              & =        & 1,\\
                            & \mathbf{x}_t                                   & \geq     & 0.\\
        \end{array}
    $$

    We create it inside a function so we can call it later.
    """)
    return


@app.cell
def _(Domain, Expr, Model, ObjectiveSense, SolutionStatus, np, sys):
    def absval(M, x, t):
        M.constraint(t + x >= 0)
        M.constraint(t - x >= 0)

    def norm1(M, x, t):
        z = M.variable(x.getSize(), Domain.greaterThan(0.0))
        absval(M, x, z)
        M.constraint(Expr.sum(z) == t)

    def multiperiod_mvo(N, T, m, G, x_0, delta, a, b):

        with Model("multiperiod") as M:
            # Settings
            M.setLogHandler(sys.stdout)

            # Variable
            x = M.variable("x", [N, T], Domain.greaterThan(0.0))
            s = M.variable("s", T)
            v = M.variable("v", [N, T])
            w = M.variable("w", [N, T])

            # Constraint
            M.constraint("budget", Expr.sum(x, 0) == np.ones(T))

            # Objective
            M.objective("obj", ObjectiveSense.Maximize, 
                Expr.add([
                    x[:, t].T @ m[t] - delta[t] * s[t] - v[:, t].T @ a[:, t] - w[:, t].T @ b[:, t]
                    for t in range(T)
                ])
            )

            # Objective cones
            for t in range(T):
                xt = x[:, t]
                xtprev = x_0 if t == 0 else x[:, t - 1]
                xtdiff = xt - xtprev
                M.constraint(f'risk_{t}', Expr.flatten(Expr.vstack(s[t], 0.5, G[t].T @ xt)), Domain.inRotatedQCone())
                absval(M, xtdiff, v[:, t])
                M.constraint(f'market_impact_{t}', Expr.hstack(w[:, t], Expr.constTerm(N, 1.0), xtdiff), Domain.inPPowerCone(2 / 3))

            # Solve the problem
            M.solve()

            # Check if the solution is an optimal point
            solsta = M.getPrimalSolutionStatus()
            if (solsta != SolutionStatus.Optimal):
                # See https://docs.mosek.com/latest/pythonfusion/accessing-solution.html about handling solution statuses.
                raise Exception("Unexpected solution status!")

            # Get the solution values
            x_value = x.level().reshape(N, T)

            return x_value
    return (multiperiod_mvo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute optimization input variables
    """)
    return


@app.cell
def _(df_prices, np):
    # Number of securities
    N = df_prices.shape[1]

    # Number of periods
    T = 10

    # Initial weights
    x_0 = np.array([1] * N) / N
    portfolio_value = 10**8
    return N, T, portfolio_value, x_0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we use the loaded daily price data to compute an estimate of the yearly mean return and covariance matrix for each trading period. These are "dummy" estimates, created from one sample mean and sample covariance based on the data.
    """)
    return


@app.cell
def _(T, compute_inputs, df_prices, np):
    def symmat(m):
        return (m + m.T) / 2

    def makepsd(m):
        mineig = np.min(np.linalg.eigvals(m))
        if mineig < 0:
            m = m - (mineig - 0.0001) * np.identity(m.shape[0])
        return m

    mu, Sigma = compute_inputs(df_prices)
    m = [mu + np.random.normal(0, mu/10) for i in range(T)]
    S = [makepsd(Sigma + symmat(np.random.normal(0, Sigma/10))) for i in range(T)]
    return S, m


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we compute the matrix $G$ such that $\Sigma=GG^\mathsf{T}$ for all periods. This is the input of the conic form of the optimization problem. Here we use Cholesky factorization.
    """)
    return


@app.cell
def _(S, np):
    G = [np.linalg.cholesky(s) for s in S]
    return (G,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also compute the average daily volume and daily volatility (std. dev.) for all periods. These are also dummy values.
    """)
    return


@app.cell
def _(T, df_prices, df_volumes, np):
    df_lin_returns = df_prices.pct_change(fill_method=None)
    volatility = df_lin_returns.std()
    volume = (df_volumes * df_prices).mean()
    vty = [abs(volatility + np.random.normal(0, volatility/10)) for i in range(T)]
    vol = [abs(volume + np.random.normal(0, volume/10)) for i in range(T)]
    return vol, vty


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we specify the transaction cost parameters for each period.
    """)
    return


@app.cell
def _(N, T, np, portfolio_value, vol, vty):
    # Transaction cost
    a = 0.05 * np.ones((N, T))

    # Market impact
    beta = 3 / 2
    b = 1
    rel_volume = [v / portfolio_value for v in vol] # Relative volume (the variable x is also portfolio relative).
    impact_coef = np.vstack([(b * v / r**(beta - 1)).to_numpy() for v, r in zip(vty, rel_volume)]).T

    # Holding cost
    s = 0.01
    return a, impact_coef


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Call the optimizer function
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We run the optimization with the risk aversion parameter $\delta = 1$ for each period.
    """)
    return


@app.cell
def _(G, N, T, a, impact_coef, m, multiperiod_mvo, np, x_0):
    delta = np.array([10] * T)
    x = multiperiod_mvo(N, T, m, G, x_0, delta, a, impact_coef)
    return (x,)


@app.cell
def _(x):
    x
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

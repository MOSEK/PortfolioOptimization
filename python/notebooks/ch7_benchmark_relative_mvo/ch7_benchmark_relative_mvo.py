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
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
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
        glob,
        matplotlib,
        np,
        os,
        pd,
        plt,
        re,
        sm,
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
    Here we load the raw data that will be used to compute the optimization input variables, the vector $\mu$ of expected returns and the covariance matrix $\Sigma$. The data consists of daily stock prices of $8$ stocks from the US market.
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
    return (df_prices)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also create the benchmark. It is the so called $\frac{1}{n}$ portfolio.
    """)
    return


@app.cell
def _(df_prices):
    # Create benchmark
    df_prices['bm'] = df_prices.iloc[:-2, 0:8].mean(axis=1)
    return


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

    The parameters:
    - `a`: The vector of alphas. (Used instead of mean returns.)
    - `B`: The vector of betas ().
    - `xbm`: The benchmark portfolio.
    - `uh`/`ub`: Upper bound on active holdings and active beta.
    - `lh`/`lb`: Lower bound on active holdings and active beta.
    """)
    return


@app.cell
def _(Domain, Expr, Model, ObjectiveSense, SolutionStatus, df_prices, np, pd):
    def EfficientFrontier(N, a, B, G, xbm, deltas, uh, ub, lh, lb):

        with Model("Case study") as M:
            # Settings
            #M.setLogHandler(sys.stdout)

            # Variables 
            # The variable x is the fraction of holdings in each security. 
            # It is restricted to be positive, which imposes the constraint of no short-selling.   
            x = M.variable("x", N, Domain.greaterThan(0.0))

            # Active holdings
            xa = x - xbm

            # The variable s models the portfolio variance term in the objective.
            s = M.variable("s", 1, Domain.unbounded())

            # Budget constraint
            M.constraint('budget_x', Expr.sum(x) == 1.0)

            # Constraint on active holdings
            M.constraint('bound-h', xa, Domain.inRange(lh, uh))

            # Constraint on portfolio active beta 
            port_act_beta = x.T @ B - 1
            M.constraint('bound-b', port_act_beta, Domain.inRange(lb, ub))

            # Conic constraint for the portfolio variance
            M.constraint('risk', Expr.vstack(s, 1, G.T @ xa), Domain.inRotatedQCone())

            # Objective (quadratic utility version)
            delta = M.parameter()
            M.objective('obj', ObjectiveSense.Maximize, x.T @ a - delta * s)

            # Create DataFrame to store the results. Last security name (benchmark) is removed.
            columns = ["delta", "obj", "return", "risk"] + df_prices.columns[:-1].tolist()
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
                portfolio_return = a @ x.level()
                portfolio_risk = np.sqrt(2 * s.level()[0])
                row = pd.Series([d, M.primalObjValue(), portfolio_return, portfolio_risk] + list(x.level()), 
                                index=columns)

                df_result = pd.concat([df_result, pd.DataFrame([row])], ignore_index=True)

            return df_result
    return (EfficientFrontier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the factor model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we define a function that computes the factor model

    $$
    R_t = \alpha + \beta R_{F,t} + \varepsilon_t.
    $$

    It can handle any number of factors, and returns estimates $\beta$, $\Sigma_F$, and $\Sigma_\theta$. The factors are assumed to be at the last coordinates of the data.

    The input of the function is the expected return and covariance of yearly logarithmic returns. The reason is that it is easier to generate logarithmic return scenarios from normal distribution instead of generating linear return scenarios from lognormal distribution.
    """)
    return


@app.cell
def _(N, np, sm):
    def factor_model(m_log, S_log, factor_num):
        """
        It is assumed that the last factor_num coordinates correspond to the factors.
        """
        if factor_num < 1: 
            raise Exception("Does not make sense to compute a factor model without factors!")

        # Generate logarithmic return scenarios
        scenarios_log = np.random.default_rng().multivariate_normal(m_log, S_log, 100000)

        # Convert logarithmic return scenarios to linear return scenarios 
        scenarios_lin = np.exp(scenarios_log) - 1

        # Do linear regression 
        params = []
        resid = []
        X = scenarios_lin[:, -factor_num:]
        X = sm.add_constant(X, prepend=False)

        for k in range(N):
            y = scenarios_lin[:, k]
            model = sm.OLS(y, X, hasconst=True).fit()
            resid.append(model.resid)
            params.append(model.params)
        resid = np.array(resid)
        params = np.array(params)

        # Get parameter estimates
        a = params[:, 1]
        B = params[:, 0:factor_num]
        S_F = np.atleast_2d(np.cov(X[:, 0:factor_num].T))
        S_theta = np.cov(resid)
        S_theta = np.diag(np.diag(S_theta))

        return a, B, S_F, S_theta
    return (factor_model,)


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
def _(compute_inputs, df_prices):
    # Number of factors
    fnum = 1

    # Number of securities (We subtract fnum to account for factors at the end of the price data)
    N = df_prices.shape[1] - fnum

    # Get optimization parameters
    m, S = compute_inputs(df_prices, security_num=N)
    return N, S, fnum


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we compute the matrix $G$ such that $\Sigma=GG^\mathsf{T}$, this is the input of the conic form of the optimization problem. Here we use Cholesky factorization.
    """)
    return


@app.cell
def _(S, np):
    G = np.linalg.cholesky(S)
    return (G,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we compute the estimates $\alpha$ and $\beta$ using the factor model. First we compute logarithmic return statistics and use them to compute the factor exposures and covariances.
    """)
    return


@app.cell
def _(compute_inputs, df_prices, factor_model, fnum):
    m_log, S_log = compute_inputs(df_prices, return_log=True)
    a, B, _, _ = factor_model(m_log, S_log, fnum)
    return B, a


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Call the optimizer function
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We run the optimization for a range of risk aversion parameter values: $\delta = 10^{-0.5},\dots,10^{2}$.
    """)
    return


@app.cell
def _(B, EfficientFrontier, G, N, a, np):
    # Parameters
    xbm = np.ones(N) / N
    uh = np.ones(N) * 0.5
    lh = -np.ones(N) * 0.5
    ub = 0.5
    lb = -0.5

    deltas = np.logspace(start=-0.5, stop=2, num=20)[::-1]

    df_result = EfficientFrontier(N, a, B, G, xbm, deltas, uh, ub, lh, lb)
    mask = df_result < 0
    mask.iloc[:, :2] = False
    df_result[mask] = 0
    return (df_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize the results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot the efficient frontier.
    """)
    return


@app.cell
def _(df_result, plt):
    # Efficient frontier
    ax = df_result.plot(x="risk", y="return", style="-o", xlabel="portfolio tracking error", ylabel="portfolio alpha", grid=True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot the portfolio composition.
    """)
    return


@app.cell
def _(LinearSegmentedColormap, df_result, plt):
    my_cmap = LinearSegmentedColormap.from_list("non-extreme gray",
                                                ["#111111", "#eeeeee"],
                                                N=256, gamma=1.0)

    ax1 = (df_result.set_index('risk')
                     .iloc[:, 3:]
                     .plot.area(logx=False,
                                colormap=my_cmap,
                                xlabel='portfolio risk (std. dev.)',
                                ylabel="x"))

    # Move legend to bottom right
    ax1.legend(loc='lower right', bbox_to_anchor=(1, 0))

    ax1.grid(which='both', axis='x', linestyle=':', color='k', linewidth=1)
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

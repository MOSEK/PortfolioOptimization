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
    We will solve a fund of funds problem. Suppose there are fund managers optimizing their funds wrt. specific benchmarks, and there is an overall benchmark, against which all funds are optimized together. This can be modeled as the following optimization problem:

    $$
        \begin{array}{lrcl}
        \text{minimize}     & (\mathbf{x}_\mathrm{o}-\mathbf{x}_{\mathrm{bm},\mathrm{o}})^\mathsf{T}\Sigma(\mathbf{x}_{\mathrm{o}}-\mathbf{x}_{\mathrm{bm},\mathrm{o}})  &          &\\
          \text{subject to} & \mathbf{x}_\mathrm{o}                          & =        & \sum_i f_i \mathbf{x}_i\\
                            & (\mathbf{x}_i-\mathbf{x}_{\mathrm{bm},i})^\mathsf{T}\Sigma(\mathbf{x}_{i}-\mathbf{x}_{\mathrm{bm},i})  & \leq     &\sigma_{\mathrm{max},i}^2\quad i=1,\dots,K\\
                            & \alpha_i^\mathsf{T}\mathbf{x}_i                & \geq     & \alpha_{\mathrm{min},i}\quad i=1,\dots,K\\
                            & \mathbf{1}^\mathsf{T}\mathbf{x}_i              & =        & 1,\quad i=1,\dots,K\\
                            & \mathbf{x}_i                                   & \geq     & 0,\quad i=1,\dots,K\\
        \end{array}
    $$

    The objective is the squared tracking error of the overall fund. $\mathbf{x}_\mathrm{o}$ is the overall fund portfolio, and $\mathrm{x}_i$ is the portfolio of fund $i$. Likewise, $\mathbf{x}_{\mathrm{bm},\mathrm{o}}$ is the the overall benchmark, and $\mathbf{x}_{\mathrm{bm},i}$ is the benchmark of fund $i$. $f_i$ is the weight of fund $i$, and has to satisfy $f_i\geq 0$, $\sum_if_i=1$. $\sigma_{\mathrm{max},i}^2$ is the squared tracking error upper bound for fund $i$, and $\alpha_{\mathrm{min},i}$ is the portfolio alpha lower bound for fund $i$.

    Then we rewrite the above problem into conic form, and implement it in Fusion API:

    $$
        \begin{array}{lrcl}
        \text{minimize}     & t_\mathrm{o}                                   &          &\\
        \text{subject to}   & (t_\mathrm{o}, 0.5, \mathbf{G}^\mathrm{T}(\mathbf{x}_{\mathrm{o}}-\mathbf{x}_{\mathrm{bm},\mathrm{o}}))  & \in     &Q_\mathrm{r}^{N+2}\\
                            & \mathbf{x}_\mathrm{o}                          & =        & \sum_i f_i \mathbf{x}_i\\
                            & (\sigma_{\mathrm{max},i}^2, 0.5, \mathbf{G}^\mathrm{T}(\mathbf{x}_i-\mathbf{x}_{\mathrm{bm},i}))  & \in     &Q_\mathrm{r}^{N+2},\quad i=1,\dots,K\\
                            & \alpha_i^\mathsf{T}\mathbf{x}_i                & \geq     & \alpha_{\mathrm{min},i}\quad i=1,\dots,K\\
                            & \mathbf{1}^\mathsf{T}\mathbf{x}_i              & =        & 1,\quad i=1,\dots,K\\
                            & \mathbf{x}_i                                   & \geq     & 0,\quad i=1,\dots,K\\
        \end{array}
    $$

    We create it inside a function so we can call it later.

    Below we implement the optimization model in Fusion API. We create it inside a function so we can call it later.

    The parameters:
    - `a`: The vectors of alphas for each fund.
    - `ao`: The vector of alphas for the overall fund.
    - `a_min`: The minimum required portfolio alpha for each fund.
    - `s2_max`: The maximum tracking error for each fund.
    - `f`: The weigth of each fund portfolio in the overall portfolio.
    - `xobm`: The overall benchmark portfolio.
    - `XFbm`: The benchmark portfolio for each fund.
    """)
    return


@app.cell
def _(Domain, Expr, Model, ObjectiveSense, SolutionStatus, df_prices, np, pd):
    def EfficientFrontier(N, K, a, ao, a_min, s2_max, G, f, xobm, Xfbm):

        with Model("Case study") as M:
            # Settings
            #M.setLogHandler(sys.stdout)

            # Variables 
            # The variable x is the fraction of holdings in each security. 
            # It is restricted to be positive, which imposes the constraint of no short-selling. 
            xo = M.variable("xo", N, Domain.greaterThan(0.0))
            Xf = M.variable("Xf", [N, K], Domain.greaterThan(0.0))

            # Active holdings
            xoa = xo - xobm
            Xfa = Xf - Xfbm

            # The variable teo models the overall tracking error in the objective.
            te2o = M.variable("teo", 1, Domain.unbounded())

            # Relate overall portfolio to fund portfolios
            M.constraint("combine", xo == Xf @ f)

            # Budget constraint for each fund
            M.constraint('budget_f', Expr.sum(Xf, 0) == np.ones(K))

            # Conic constraint for the fund sq. tracking errors
            sigma2 = M.parameter()
            for i in range(K):
                M.constraint(f'fund_te2_{i}', 
                             Expr.flatten(Expr.vstack(sigma2, 0.5, G.T @ Xfa[:, i])),
                             Domain.inRotatedQCone())

            # Conic constraint for the overall sq. tracking error
            M.constraint('overall_te2', Expr.vstack(te2o, 0.5, G.T @ xoa), Domain.inRotatedQCone())

            # Alpha constraint for each fund.
            for i in range(K):
                M.constraint(f'fund_alpha_{i}', Xf[:, i].T @ a[:, i]>= a_min[i])

            # Objective
            M.objective('obj', ObjectiveSense.Minimize, te2o)

            # Create DataFrame to store the results. Last security name (benchmark) is removed.
            columns = ["s2", "obj", "return", "te_o", "te_1", "te_2"] + df_prices.columns[:-1].tolist()
            df_result = pd.DataFrame()
            for s2 in s2_max:
                # Update parameter
                sigma2.setValue(s2) 

                # Solve optimization
                M.solve()
                # Check if the solution is an optimal point
                solsta = M.getPrimalSolutionStatus()
                if (solsta != SolutionStatus.Optimal):
                    # See https://docs.mosek.com/latest/pythonfusion/accessing-solution.html about handling solution statuses.
                    raise Exception("Unexpected solution status!")

                # Save results
                portfolio_return = ao @ xo.level()
                overall_te2 = te2o.level()[0]
                r1 = G.T @ (Xf.level().reshape(N, K)[:, 0] - Xfbm[:, 0])
                fund_te2_1 = np.dot(r1, r1)
                r2 = G.T @ (Xf.level().reshape(N, K)[:, 1] - Xfbm[:, 1])
                fund_te2_2 = np.dot(r2, r2)
                row = pd.Series([s2, M.primalObjValue(), portfolio_return, overall_te2, fund_te2_1, fund_te2_2] + list(xo.level()), 
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
    # Number of securities (We subtract fnum to account for factors at the end of the price data)
    N = 8
    K = 2

    # Get optimization parameters
    m, S = compute_inputs(df_prices, security_num=N)
    return K, N, S


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also create three benchmarks, one for each fund, and an overall benchmark.
    """)
    return


@app.cell
def _(S, df_prices, np):
    # Create benchmarks
    # - Benchmark for fund 1 
    w1 = np.diag(S)
    w1 = w1 / sum(w1)
    bm_1 = df_prices.iloc[:-2, 0:8].dot(w1)

    # - Benchmark for fund 2
    w2 = np.diag(S)**2
    w2 = w2 / sum(w2)
    bm_2 = df_prices.iloc[:-2, 0:8].dot(w2)

    # - Overall benchmark
    wo = (1.0 / np.diag(S))
    wo = wo / sum(wo)
    bm_o = df_prices.iloc[:-2, 0:8].dot(wo)
    return bm_1, bm_2, bm_o, w1, w2, wo


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
    Now we compute the estimates $\alpha$ and $\beta$ using the factor model, for each benchmark. First we compute logarithmic return statistics and use them to compute the factor exposures and covariances.
    """)
    return


@app.cell
def _(bm_1, bm_2, bm_o, compute_inputs, df_prices, factor_model, np):
    df_prices['bm'] = bm_1
    m_log, S_log = compute_inputs(df_prices, return_log=True)
    a_1, _, _, _ = factor_model(m_log, S_log, 1)

    df_prices['bm'] = bm_2
    m_log, S_log = compute_inputs(df_prices, return_log=True)
    a_2, _, _, _ = factor_model(m_log, S_log, 1)

    df_prices['bm'] = bm_o
    m_log, S_log = compute_inputs(df_prices, return_log=True)
    a_3, _, _, _ = factor_model(m_log, S_log, 1)

    a = np.vstack([a_1, a_2]).T
    ao = a_3
    return a, ao


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Also define the benchmark weights for the funds, and the overall benchmark.
    """)
    return


@app.cell
def _(np, w1, w2, wo):
    Xfbm = np.vstack([w1, w2]).T
    xobm = wo

    # Fund weights
    f = [0.5, 0.5]

    # Alpha lower bounds
    a_min = [0.05, 0.05]
    return Xfbm, a_min, f, xobm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Call the optimizer function
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We run the optimization for a range of tracking error limits.
    """)
    return


@app.cell
def _(EfficientFrontier, G, K, N, Xfbm, a, a_min, ao, f, np, xobm):
    # Tracking error upper bounds (in percent)
    s2_max = np.linspace(start=1, stop=0.1, num=10) / 100

    df_result = EfficientFrontier(N, K, a, ao, a_min, s2_max, G, f, xobm, Xfbm)
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
    Plot the squared tracking errors of the funds in function of the tracking error limit.
    """)
    return


@app.cell
def _(df_result):
    df_result
    return


@app.cell
def _(df_result, plt):
    # Efficient frontier
    ax = df_result.plot(x="s2", y="te_o", style="-o", xlabel="risk limit", ylabel="portfolio tracking error", grid=True)
    df_result.plot(ax=ax, x="s2", y="te_1", style="-o", xlabel="risk limit", ylabel="portfolio tracking error", grid=True) 
    df_result.plot(ax=ax, x="s2", y="te_2", style="-o", xlabel="risk limit", ylabel="portfolio tracking error", grid=True)
    ax.legend(["overall portfolio", "fund 1", "fund_2"]);
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

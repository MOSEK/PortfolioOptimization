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
    import statsmodels.api as sm
    import scipy.stats as stats
    from scipy.optimize import brentq
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from mosek.fusion import  Model, Domain, Expr, ObjectiveSense, Matrix, Var, SolutionStatus
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
        LinearSegmentedColormap,
        Model,
        ObjectiveSense,
        SolutionStatus,
        brentq,
        glob,
        matplotlib,
        np,
        os,
        pd,
        plt,
        re,
        sm,
        stats,
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
    list_factors = ["SPY", "IWM"]
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
def _(Domain, Expr, Model, ObjectiveSense, SolutionStatus, df_prices, np, pd):
    # |x| <= t
    def absval(M, x, t):
        M.constraint(t + x >= 0)
        M.constraint(t - x >= 0)


    def sqrtm_symm(m):
        e, v = np.linalg.eigh(m)
        sqrt_e = np.sqrt(e)
        sqrt_m = np.dot(v, np.dot(np.diag(sqrt_e), v.T))
        return sqrt_m


    def EfficientFrontier(N, mu0, gamma, beta0, Gmx, rho, diag_S_theta_upper, Q0, Nmx, zeta, deltas):

        with Model("Case study") as M:
            # Settings
            #M.setLogHandler(sys.stdout)

            # Get number of factors
            K = Q0.shape[0]

            # Variables 
            # The variable x is the fraction of holdings in each security. 
            # It is restricted to be positive, which imposes the constraint of no short-selling.   
            x = M.variable("x", N, Domain.greaterThan(0.0))
            z = M.variable("z", N, Domain.greaterThan(0.0))

            # Constrain absolute value
            absval(M, x, z)

            # The variable t1 and t2 models the factor and specific portfolio variance terms.
            t1 = M.variable("t1", 1, Domain.greaterThan(0.0))
            t2 = M.variable("t2", 1, Domain.greaterThan(0.0))

            # The variables tau, s, u help modeling the factor risk.
            tau = M.variable("tau", 1, Domain.greaterThan(0.0))
            s = M.variable("s", 1, Domain.greaterThan(0.0))
            u = M.variable("u", K, Domain.greaterThan(0.0))

            # Budget constraint
            M.constraint('budget', Expr.sum(x), Domain.equalsTo(1.0))

            # Objective (variance minimization)
            delta = M.parameter()
            wc_return = x.T @ mu0 - z.T @ gamma
            M.objective('obj', ObjectiveSense.Maximize, wc_return - delta * (t1 + t2))

            # Risk constraint (specific)
            M.constraint('spec-risk', Expr.vstack(t2, 0.5, Expr.mulElm(np.sqrt(diag_S_theta_upper), x)), Domain.inRotatedQCone())

            # Risk constraint (factor)
            siG = sqrtm_symm(np.linalg.inv(Gmx))            
            H = siG @ (Q0 + zeta * Nmx) @ siG 
            lam, V = np.linalg.eigh(H)
            w = (V.T @ sqrtm_symm(H) @ sqrtm_symm(Gmx) @ beta0.T) @ x
            M.constraint('fact-risk-1', t1 >= tau + Expr.sum(u))
            M.constraint('fact-risk-2', s <= 1.0 / lam[-1])
            M.constraint('fact-risk-3', Expr.vstack(s, 0.5 * tau, z.T @ rho), Domain.inRotatedQCone())
            col1 = Expr.constTerm(K, 1.0) - Expr.mulElm(Expr.repeat(s, K, 0), lam)
            M.constraint('fact-risk-4', Expr.hstack(col1, 0.5 * u, w), Domain.inRotatedQCone())

            # Create DataFrame to store the results. Last security names (the factors) are removed.
            columns = ["delta", "obj", "return", "risk", "zdiff"] + df_prices.columns[:-K].tolist()
            df_result = pd.DataFrame()
            for d in deltas:
                # Update parameter
                delta.setValue(d)

                # Solve optimization
                M.solve()

                # Check if the solution is an optimal point
                solsta = M.getPrimalSolutionStatus()
                if (solsta != SolutionStatus.Optimal):
                    # See https://docs.mosek.com/latest/pythonfusion/accessing-solution.html about handling solution statuses.
                    raise Exception("Unexpected solution status!")

                # Save results
                portfolio_return = mu0 @ x.level() - gamma @ z.level()
                portfolio_risk = np.sqrt((t1.level() + t2.level())[0])
                zdiff = np.sum(np.abs(x.level()) - z.level())
                row = pd.Series([d, M.primalObjValue(), portfolio_return, portfolio_risk, zdiff] + list(x.level()), 
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
    We create a function to make scenarios. The input is the expected return and covariance of yearly logarithmic returns. The reason for this is that it is easier to generate logarithmic return scenarios from normal distribution than generating linear return scenarios from lognormal distribution.
    """)
    return


@app.cell
def _(np):
    def scenarios(m_log, S_log, factor_num):
        """
        It is assumed that the last factor_num coordinates correspond to the factors.
        """
        if factor_num < 1: 
            raise Exception("Does not make sense to compute a factor model without factors!")

        # Generate logarithmic return scenarios
        scenarios_log = np.random.default_rng().multivariate_normal(m_log, S_log, 100000)

        # Convert logarithmic return scenarios to linear return scenarios 
        scenarios = np.exp(scenarios_log) - 1

        R = scenarios[:, :-factor_num]
        F = scenarios[:, -factor_num:]

        return R, F
    return (scenarios,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we define a function that computes the factor model

    $$
    R_t = \mu + \beta F_{t} + \theta_t.
    $$

    The function can handle any number of factors, and returns estimates $\mu$, $\beta$, $\Sigma_F$, $\Sigma_\theta$, and the factor return matrix. The factors are assumed to be at the last coordinates of the data.
    """)
    return


@app.cell
def _(N, np, sm):
    def factor_model(R, F):
        """
        It is assumed that the last factor_num coordinates correspond to the factors.
        """
        factor_num = F.shape[1]

        # Do linear regression 
        params = []
        resid = []
        X = F
        X = sm.add_constant(X, prepend=True)

        for k in range(N):
            y = R[:, k]
            model = sm.OLS(y, X, hasconst=True).fit()        
            resid.append(model.resid)        
            params.append(model.params)
        resid = np.array(resid)
        params = np.array(params)


        # Get parameter estimates
        mu = params[:, 0]
        B = params[:, 1:]

        S_F = np.atleast_2d(np.cov(X[:, 1:].T))  # MLE computed from data
        S_theta = np.cov(resid, ddof=factor_num + 1)   
        diag_S_theta = np.diag(S_theta)

        return mu, B, S_F, diag_S_theta, X
    return (factor_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we define functions that will compute the parametrization for the uncertainty sets of the factor model parameters.
    """)
    return


@app.cell
def _(brentq, np, stats):
    def unc_mu(mu_est, diag_S_theta_est, A, omega):
        K = A.shape[1] - 1
        T = A.shape[0]
        iAA = np.linalg.inv(A.T @ A)
        c = stats.f.ppf(omega, K + 1, T - K - 1)

        # Parametrization
        mu0 = mu_est
        gamma = np.sqrt(diag_S_theta_est) * np.sqrt((K + 1) * iAA[0, 0] * c)

        return mu0, gamma

    def unc_beta(beta_est, diag_S_theta_est, A, omega):
        K = A.shape[1] - 1
        T = A.shape[0]
        F = A[:,1:].T
        F1 = F @ np.ones((T, 1))
        c = stats.f.ppf(omega, K + 1, T - K - 1)

        # Parametrization
        beta0 = beta_est
        Q = np.array([[0, 1, 0], [0, 0, 1]])
        iAA = np.linalg.inv(A.T @ A)
        Gmx = F @ F.T - F1 @ F1.T / T
        rho = np.sqrt(diag_S_theta_est) * np.sqrt((K + 1) * c)

        return beta0, Gmx, rho

    def unc_d(diag_S_theta_est, percent):    
        # Here we just add a percentage to the estimated error variance, to get an upper bound estimate. 
        diag_S_theta_upper = diag_S_theta_est * (1.0 + percent)
        return diag_S_theta_upper

    def unc_q(S_F, A, omega):
        T = A.shape[0]


        def fun(eta, T, omega):
            return stats.gamma.cdf(1 + eta, (T + 1) / 2, scale=2 / (T - 1)) - \
                   stats.gamma.cdf(1 - eta, (T + 1) / 2, scale=2 / (T - 1)) - \
                   omega


        eta = brentq(fun, 0, 1, args=(T, omega))

        # Parametrization
        Q0 = S_F
        Nmx = S_F

        zeta = eta / (1 - eta)
        return Q0, Nmx, zeta
    return unc_beta, unc_d, unc_mu, unc_q


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
def _(df_prices, list_factors):
    # Number of factors
    fnum = len(list_factors)

    # Number of securities (We subtract fnum to account for factors at the end of the price data)
    N = df_prices.shape[1] - fnum
    return N, fnum


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we compute the same using the factor model. First we compute logarithmic return statistics and use them to compute the factor exposures and covariances.
    """)
    return


@app.cell
def _(compute_inputs, df_prices, factor_model, fnum, scenarios):
    m_log, S_log = compute_inputs(df_prices, return_log=True)
    R, F = scenarios(m_log, S_log, fnum)

    # Center factors, so we have the same model as in the article (Goldfarb--Iyengar 2003). 
    F -= F.mean(axis=0)

    # Compute factor model
    mu, B, S_F, diag_S_theta, X = factor_model(R, F)
    return B, S_F, X, diag_S_theta, mu


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We compute the parameters of the uncertainty sets.
    """)
    return


@app.cell
def _(B, N, S_F, X, diag_S_theta, mu, np, unc_beta, unc_d, unc_mu, unc_q):
    # Uncertainty set parameters
    omega = 0.95
    percent = 0.2
    mu0, gamma = unc_mu(mu, diag_S_theta, X, omega)
    beta0, Gmx, rho = unc_beta(B, diag_S_theta, X, omega)
    diag_S_theta_upper = unc_d(diag_S_theta, percent)
    Q0, Nmx, zeta = unc_q(S_F, X, omega)

    # To get back the non_robust case, we have to zero the bounds
    gamma_z = np.zeros(N)
    rho_z = np.zeros(N)
    zeta_z = 0.0
    return (
        Gmx,
        Nmx,
        Q0,
        beta0,
        diag_S_theta_upper,
        gamma,
        gamma_z,
        mu0,
        rho,
        rho_z,
        zeta,
        zeta_z,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Call the optimizer function
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We run the optimization for a range of risk aversion parameter values: $\delta = 10^{-1},\dots,10^{2}$. We compute the efficient frontier this way both with and without using factor model.
    """)
    return


@app.cell
def _(
    EfficientFrontier,
    Gmx,
    N,
    Nmx,
    Q0,
    beta0,
    diag_S_theta_upper,
    gamma,
    gamma_z,
    mu0,
    np,
    rho,
    rho_z,
    zeta,
    zeta_z,
):
    # Compute efficient frontier with and without factor model
    deltas = np.logspace(start=-1, stop=2, num=20)[::-1] / 2
    df_result_orig = EfficientFrontier(N, mu0, gamma_z, beta0, Gmx, rho_z, diag_S_theta_upper, Q0, Nmx, zeta_z, deltas)
    df_result_robust = EfficientFrontier(N, mu0, gamma, beta0, Gmx, rho, diag_S_theta_upper, Q0, Nmx, zeta, deltas)
    df_result_orig
    return df_result_orig, df_result_robust


@app.cell
def _(df_result_orig, df_result_robust):
    # Set small negatives to zero to make plotting work
    mask = df_result_orig < 0
    mask.iloc[:, :-8] = False
    df_result_orig[mask] = 0

    # Set small negatives to zero to make plotting work
    mask = df_result_robust < 0
    mask.iloc[:, :-8] = False
    df_result_robust[mask] = 0
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
    Plot the efficient frontier for both cases.
    """)
    return


@app.cell
def _(df_result_orig, df_result_robust, plt):
    ax = df_result_robust.plot(x="risk", y="return", style="-o", xlabel="portfolio risk (std. dev.)", ylabel="portfolio return", grid=True)
    df_result_orig.plot(ax=ax, x="risk", y="return", style="-o", xlabel="portfolio risk (std. dev.)", ylabel="portfolio return", grid=True)   
    ax.legend(["robust return", "return"]);
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot the portfolio composition for both cases.
    """)
    return


@app.cell
def _(LinearSegmentedColormap, df_result_orig, df_result_robust, plt):
    # Plot portfolio composition
    my_cmap = LinearSegmentedColormap.from_list("non-extreme gray", ["#111111", "#eeeeee"], N=256, gamma=1.0)
    ax1 = df_result_robust.set_index('risk').iloc[:, 4:].clip(0, None).plot.area(colormap=my_cmap, xlabel='portfolio risk (std. dev.)', ylabel="x")
    ax1.grid(which='both', axis='x', linestyle=':', color='k', linewidth=1)
    ax1.legend(loc='lower right')
    ax2 = df_result_orig.set_index('risk').iloc[:, 4:].clip(0, None).plot.area(colormap=my_cmap, xlabel='portfolio risk (std. dev.)', ylabel="x") 
    ax2.grid(which='both', axis='x', linestyle=':', color='k', linewidth=1)
    ax2.legend(loc='lower right')
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

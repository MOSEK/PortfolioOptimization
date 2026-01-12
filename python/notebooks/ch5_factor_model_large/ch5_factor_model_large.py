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
    import sys
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
        matplotlib,
        np,
        pd,
        plt,
        sys,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Define the optimization model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We create a function to randomize factor models, i. e., large random covariance matrices with only a few significant eigenvalues.
    """)
    return


@app.cell
def _(np):
    def random_factor_model(N, K, T):
        # Generate K + N zero mean factors, with block covariance: 
        # - K x K weighted diagonal block for the factors, 
        # - N x N white noise (uncorrelated to the factors)
        S_F = np.diag(np.sqrt(range(1, K + 1)))
        Cov = np.block([
            [S_F,              np.zeros((K, N))],
            [np.zeros((N, K)), np.eye(N)]
        ])
        Y = np.random.default_rng(seed=1).multivariate_normal(np.zeros(K + N), Cov, T).T
        Z_F = Y[:K, :]

        # Generate random factor model parameters
        B = np.random.default_rng(seed=2).normal(size=(N, K))
        a = np.random.default_rng(seed=3).normal(loc=1, size=(N, 1))
        e = Y[K:, :]

        # Generate N time-series from the factors
        Z = a + B @ Z_F + e

        # Residual covariance
        S_theta = np.cov(e)
        diag_S_theta = np.diag(S_theta)

        # Optimization parameters
        m = np.mean(Z, axis=1)
        S = np.cov(Z)
        #print(np.linalg.eigvalsh(np.corrcoef(Z))[-20:])

        return m, S, B, S_F, diag_S_theta
    return (random_factor_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we define the optimization model in MOSEK Fusion.
    """)
    return


@app.cell
def _(Domain, Expr, Model, ObjectiveSense, SolutionStatus):
    # Solve optimization
    def Markowitz(N, m, G, gamma2):
        with Model("markowitz") as M:
            # Settings
            #M.setLogHandler(sys.stdout) 

            # Decision variable (fraction of holdings in each security)
            # The variable x is restricted to be positive, which imposes the constraint of no short-selling.   
            x = M.variable("x", N, Domain.greaterThan(0.0))

            # Budget constraint
            M.constraint('budget', Expr.sum(x) == 1.0)

            # Objective 
            M.objective('obj', ObjectiveSense.Maximize, x.T @ m)

            # Imposes a bound on the risk
            if isinstance(G, tuple):
                G_factor = G[0]
                g_specific = G[1]

                factor_risk = G_factor.T @ x 
                specific_risk = Expr.mulElm(g_specific, x)
                total_risk = Expr.vstack(factor_risk, specific_risk)

                M.constraint('risk', Expr.vstack(gamma2**0.5, total_risk), Domain.inQCone())
            else:
                M.constraint('risk', Expr.vstack(gamma2**0.5, G.T @ x), Domain.inQCone())

            # Solve optimization
            M.solve()

            # Check if the solution is an optimal point
            solsta = M.getPrimalSolutionStatus()
            if (solsta != SolutionStatus.Optimal):
                # See https://docs.mosek.com/latest/pythonfusion/accessing-solution.html about handling solution statuses.
                raise Exception("Unexpected solution status!") 

            returns = M.primalObjValue()
            portfolio = x.level()
            time = M.getSolverDoubleInfo("optimizerTime")

            return returns, time
    return (Markowitz,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Run the optimization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the parameters
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The parameters are the number of factors $K$ and the risk limit $\gamma^2$.
    """)
    return


@app.cell
def _():
    # Risk limit
    gamma2 = 0.1

    # Number of factors
    K = 10
    return K, gamma2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solve the optimization problem
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we generate random factor structured covariance matrices of different sizes, and solve the portfolio optimization both when we utilize the factor structure and when we do Cholesky factorization on it instead.
    """)
    return


@app.cell
def _(K, Markowitz, gamma2, np, random_factor_model):
    # Generate runtime data 
    # NOTE: This can have a long runtime, depending on the range given for n below!
    list_runtimes_orig = []
    list_runtimes_factor = []
    for n in range(5, 13):
        N = 2**n
        T = 10000
        m, S, B, S_F, diag_S_theta = random_factor_model(N, K, T)

        F = np.linalg.cholesky(S_F)
        G_factor = B @ F
        g_specific = np.sqrt(diag_S_theta)

        G_orig = np.linalg.cholesky(S)

        optimum_orig, runtime_orig = Markowitz(N, m, G_orig, gamma2)
        optimum_factor, runtime_factor = Markowitz(N, m, (G_factor, g_specific), gamma2)
        list_runtimes_orig.append((N, runtime_orig))
        list_runtimes_factor.append((N, runtime_factor))

    tup_N_orig, tup_time_orig = list(zip(*list_runtimes_orig))
    tup_N_factor, tup_time_factor = list(zip(*list_runtimes_factor))
    return tup_N_factor, tup_N_orig, tup_time_factor, tup_time_orig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plot results
    """)
    return


@app.cell
def _(plt, tup_N_factor, tup_N_orig, tup_time_factor, tup_time_orig):
    # Runtime plot
    plt.plot(tup_N_orig, tup_time_orig, "-o")
    plt.plot(tup_N_factor, tup_time_factor, "-o")
    plt.xlabel("N")
    plt.ylabel("runtime (s)")
    ax = plt.gca()
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid()
    legend = ["Cholesky", "factor model"]
    plt.legend(legend)
    return


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

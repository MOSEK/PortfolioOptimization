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
        matplotlib,
        np,
        pd,
        plt,
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
    In this example, the input data is given. It consists of the vector $\mu$ of expected returns, and the covariance matrix $\Sigma$.
    """)
    return


@app.cell
def _(np):
    # Linear return statistics on the investment horizon
    mu = np.array([0.07197349, 0.15518171, 0.17535435, 0.0898094 , 0.42895777, 0.39291844, 0.32170722, 0.18378628])
    Sigma = np.array([
            [0.09460323, 0.03735969, 0.03488376, 0.03483838, 0.05420885, 0.03682539, 0.03209623, 0.03271886],
            [0.03735969, 0.07746293, 0.03868215, 0.03670678, 0.03816653, 0.03634422, 0.0356449 , 0.03422235],
            [0.03488376, 0.03868215, 0.06241065, 0.03364444, 0.03949475, 0.03690811, 0.03383847, 0.02433733],
            [0.03483838, 0.03670678, 0.03364444, 0.06824955, 0.04017978, 0.03348263, 0.04360484, 0.03713009],
            [0.05420885, 0.03816653, 0.03949475, 0.04017978, 0.17243352, 0.07886889, 0.06999607, 0.05010711],
            [0.03682539, 0.03634422, 0.03690811, 0.03348263, 0.07886889, 0.09093307, 0.05364518, 0.04489357],
            [0.03209623, 0.0356449 , 0.03383847, 0.04360484, 0.06999607, 0.05364518, 0.09649728, 0.04419974],
            [0.03271886, 0.03422235, 0.02433733, 0.03713009, 0.05010711, 0.04489357, 0.04419974, 0.08159633]
          ])
    return Sigma, mu


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Define the optimization model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The optimization problem we would like to solve is

    $$
    \begin{array}{lrcl}
    \text{maximize}   & \boldsymbol{\mu}^\mathsf{T}\mathbf{x}                    &        &\\
    \text{subject to} & \left(\gamma^2, \frac{1}{2}, \mathbf{G}^\mathsf{T}\mathbf{x}\right)
                      & \in    & \mathcal{Q}_\mathrm{r}^{N+2},\\
                      & \mathbf{1}^\mathsf{T}\mathbf{x}                          & =      & 1,\\
                      & \mathbf{x}                                               & \geq   & 0.
    \end{array}
    $$


    Here we define this model in MOSEK Fusion.
    """)
    return


@app.cell
def _(Domain, Expr, Model, ObjectiveSense, SolutionStatus, sys):
    # Define function solving the optimization model
    def Markowitz(N, m, G, gamma2):
        with Model("markowitz") as M:
            # Settings
            M.setLogHandler(sys.stdout) 

            # Decision variable (fraction of holdings in each security)
            # The variable x is restricted to be positive, which imposes the constraint of no short-selling.   
            x = M.variable("x", N, Domain.greaterThan(0.0)) 

            # Budget constraint
            M.constraint('budget', Expr.sum(x) == 1)

            # Objective 
            M.objective('obj', ObjectiveSense.Maximize, x.T @ m)

            # Imposes a bound on the risk
            M.constraint('risk', Expr.vstack(gamma2, 0.5, G.T @ x), Domain.inRotatedQCone())

            # Solve optimization
            M.solve()

            # Check if the solution is an optimal point
            solsta = M.getPrimalSolutionStatus()
            if (solsta != SolutionStatus.Optimal):
                # See https://docs.mosek.com/latest/pythonfusion/accessing-solution.html about handling solution statuses.
                raise Exception("Unexpected solution status!") 

            returns = M.primalObjValue()
            portfolio = x.level()

        return returns, portfolio
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
    The problem parameters are the number of securities $N$ and the risk limit $\gamma^2$.
    """)
    return


@app.cell
def _(mu):
    N = mu.shape[0]  # Number of securities
    gamma2 = 0.05   # Risk limit (variance)
    return N, gamma2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Factorize the covariance matrix
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we factorize $\Sigma$ because the model is defined in conic form, and it expects a matrix $G$ such that $\Sigma = GG^\mathsf{T}$.
    """)
    return


@app.cell
def _(Sigma, np):
    G = np.linalg.cholesky(Sigma)  # Cholesky factor of S to use in conic risk constraint
    return (G,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solve the optimization problem
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we call the function that defines the Fusion model and runs the optimization.
    """)
    return


@app.cell
def _(G, Markowitz, N, gamma2, mu, np):
    # Run optimization 
    f, x = Markowitz(N, mu, G, gamma2)
    print("========================\n")
    print("RESULTS:")
    print(f"Optimal expected portfolio return: {f*100:.4f}%")
    print(f"Optimal portfolio weights: {x}")
    print(f"Sum of weights: {np.sum(x)}")
    return (x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test result
    """)
    return


@app.cell
def _(np, x):
    expected_x = np.array([0., 0.09126, 0.26911, 0., 0.02531, 0.32162, 0.17652, 0.11618])
    diff = np.sum(np.abs(expected_x - x))
    assert diff < 1e-4, f"Resulting portfolio does not match expected one. Difference is {diff}"
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

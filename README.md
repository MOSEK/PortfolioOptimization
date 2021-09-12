# MOSEK Portfolio Optimization Cookbook

The [MOSEK Portfolio Optimization Cookbook](https://docs.mosek.com/portfolio-cookbook/index.html) book provides an introduction to the topic of portfolio optimization and discusses several branches of practical interest from this broad subject. You can read it here:

* [HTML](https://docs.mosek.com/portfolio-cookbook/index.html)
* [PDF (A4)](https://docs.mosek.com/MOSEKPortfolioCookbook-a4paper.pdf), [PDF (letter)](https://docs.mosek.com/MOSEKPortfolioCookbook-letter.pdf)

It is illustrated with complete code examples using MOSEK which can be found in this repository.

# Python notebooks

Notebook | Type | Keywords | Links
--- | --- | --- | ---
[Mean-variance optimization](./python/notebooks/./ch2_mean-variance_optimization.ipynb) | CQO | Markowitz, efficient frontier, conic model, risk, return | [CoLab](https://colab.research.google.com/github/MOSEK/PortfolioOptimization/blob/master/python/notebooks-colab/./ch2_mean-variance_optimization.ipynb), [Cookbook](https://docs.mosek.com/portfolio-cookbook/markowitz.html)
[Shrinkage](./python/notebooks/./ch4_shrinkage.ipynb) | CQO | shrinkage, Ledoit-Wolf, James-Stein | [CoLab](https://colab.research.google.com/github/MOSEK/PortfolioOptimization/blob/master/python/notebooks-colab/./ch4_shrinkage.ipynb), [Cookbook](https://docs.mosek.com/portfolio-cookbook/estimationerror.html)
[Single factor model](./python/notebooks/./ch5_factor_model_small.ipynb) | CQO | factor model | [CoLab](https://colab.research.google.com/github/MOSEK/PortfolioOptimization/blob/master/python/notebooks-colab/./ch5_factor_model_small.ipynb), [Cookbook](https://docs.mosek.com/portfolio-cookbook/factormodels.html)
[Large scale factor model](./python/notebooks/./ch5_factor_model_large.ipynb) | CQO | factor model | [CoLab](https://colab.research.google.com/github/MOSEK/PortfolioOptimization/blob/master/python/notebooks-colab/./ch5_factor_model_large.ipynb), [Cookbook](https://docs.mosek.com/portfolio-cookbook/factormodels.html)
[Market impact costs](./python/notebooks/./ch6_market_impact.ipynb) | CQO, POW | market impact, power law | [CoLab](https://colab.research.google.com/github/MOSEK/PortfolioOptimization/blob/master/python/notebooks-colab/./ch6_market_impact.ipynb), [Cookbook](https://docs.mosek.com/portfolio-cookbook/transaction.html#market-impact-costs)
[Transaction costs](./python/notebooks/./ch6_transaction_cost.ipynb) | CQO, mixed-int | leverage, buy-in threshold, fixed costs | [CoLab](https://colab.research.google.com/github/MOSEK/PortfolioOptimization/blob/master/python/notebooks-colab/./ch6_transaction_cost.ipynb), [Cookbook](https://docs.mosek.com/portfolio-cookbook/transaction.html#transaction-cost-models)
[Benchmark relative optimization](./python/notebooks/./ch7_benchmark_relative_mvo.ipynb) | CQO | active return, error tracking | [CoLab](https://colab.research.google.com/github/MOSEK/PortfolioOptimization/blob/master/python/notebooks-colab/./ch7_benchmark_relative_mvo.ipynb), [Cookbook](https://docs.mosek.com/portfolio-cookbook/benchmarkrel.html)

With a Google account you can launch the notebook directly in Google CoLab by following the CoLab link. 
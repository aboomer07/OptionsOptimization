# Portfolio Optimization with Options

This project was for my second year thesis during my masters program at the Toulouse School of Economics.

## Overview
This project was a replcation and extension of the paper "Optimal Option Portfolio Strategies: Deepening the Puzzle of Index Option Mispricing" found [here](https://run.unl.pt/bitstream/10362/67617/1/OOPS_2017.pdf). I extend their paper by considering a GARCH volatility model, applying the model to more recent data including the Covid pandemic, and discussing extensions for practical application in an automated trading program. I implemented the rolling GARCH model using the Statsmodels python package, and built the optimization framework using CVXPY. I analyzed how pricing and liquidity dynamics in the option markets affected the performance of the optimization model, as well as the impact of the Covid pandemic and potential small sample issues.

## Data Sources
The S&P500 price data was downloaded from the Case Shiller index. The risk free asset was modeled with the four week treasury bill from the FRED database. The S&P500 options data was provided from the Chicago Board of Exchange (CBOE).

## Tools
The paper is built using Latex. The coding is done in Python using the CVXPY, Numpy, Pandas, Seaborn, and Statsmodels packages. The optimization was done with the MOSEK solver under an academic license.

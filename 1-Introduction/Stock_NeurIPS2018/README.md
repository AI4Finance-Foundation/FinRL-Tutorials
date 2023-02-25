# Contents:

The three tutorials in this directory show a workflow of applying ML/RL in algorithmic trading. They are the reproduction and improvement of the process in this [paper](https://arxiv.org/abs/1811.07522).

# Usage

## Part 1. Data

First, run the notebook *Stock_NeurIPS2018_1_Data.ipynb*. It gives a demonstration of how to download and preprocess stock OHLCV data into the form that we could use in the further steps.

It will generate two csv files: *train.csv*, *trade.csv*. Here in this directory, we've provide the two sample files to you.

## Part 2. Train

After data files are prepared, run the second notebook *Stock_NeurIPS2018_1_Train.ipynb*. It shows how to use FinRL to construct the data into a proper OpenAI gym-style envrionment, and then train different DRL algorithms on it.

It will generate the trained RL model .zip files. Here in this directory, we've provide a sample A2C model .zip file. Feel free to use it directly on next part.

## Part 3. Backtest

Now we have the RL agents. The last step is backtesting. Run the third notebook *Stock_NeurIPS2018_1_Backtest.ipynb* It backtest the agents, and compare their performances with two baselines: Mean Variance Optimization and the DJIA index. At the end, it will plot the figure that records the change of asset total value during the backtest process.
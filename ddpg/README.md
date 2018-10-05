# Reinforcement Learning for Finance
We apply reinforcement learning for stock trading. 


## Fetch Data Example
```
import utils 
# fetch symbols from yahoo finance
symbols = utils.get_sap_symbols('sap500')
# fetch Open value from 01/04/2015 to 01/04/2016
input_data = utils.get_data_list_key(symbols, '2015-04-01', '2016-04-01', 'Open')
```

We have two models:
## Exit Rule
When is optimal to sell out stocks is challenging task. I implemented the following alogrithm to determine if selling out stocks is more profitable than holding stocks. A learning is based on based on [DQN](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html). To give stability, I introduced [Double Q-Learning](https://www.aaai.org/Conferences/AAAI/2016/Papers/12vanHasselt12389.pdf).
The figure below is a learned result from 01/04/2015 to 01/04/2016. The blue points inidiate points where it's better to sell stocks if you have.

![exit](https://github.com/jjakimoto/DQN/blob/master/assets/exit_result.jpg)

Edit DQNConfig of config.py from
```
self.save_path = '/path/to/your/save/path/model.ckpt'
```
to your save path. Then, you can try
```
python main.py "dqn"
```


## Optimal Portfolio
Constructing optimal portfolio that makes profits safely is important for fund management. I implemented an algorithm to prdocue portfolios that makes profits. A learning algorighm is based on [DDPG](https://arxiv.org/pdf/1509.02971v5.pdf).
The figure below is a trading result.
After learned with data from 01/04/2012 to 01/04/2013, trade on data from 01/04/2013 to 01/04/2014.

![trade](https://github.com/jjakimoto/DQN/blob/master/assets/trade_result.jpg)

Edit DDPGConfig of config.py from
```
self.save_path = '/path/to/your/save/path/model.ckpt'
```
to your save path. Then, you can try
```
python main.py "ddpg"
```

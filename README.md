# Practical Deep Reinforcement Learning Approach for Stock Trading

## FinRL library
The codes here are incooperated into the [FinRL library](https://github.com/AI4Finance-LLC/FinRL-Library)

## Prerequisites
Python 3.6+ envrionment

## Step 1: Install OpenAI Baselines System Packages [OpenAI Instruction](https://github.com/openai/baselines)
### Ubuntu
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```
### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```


## Step 2: Create and Activate Virtual Environment
Clone the repository to folder /DQN-DDPG_Stock_Trading:
```bash
git clone https://github.com/hust512/DQN-DDPG_Stock_Trading.git
cd DQN-DDPG_Stock_Trading
```
Under folder /DQN-DDPG_Stock_Trading, create a virtual environment
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages.
Create a virtualenv called venv under folder /DQN-DDPG_Stock_Trading/venv
```bash
virtualenv -p python3 venv
```
To activate a virtualenv:
```
source venv/bin/activate
```

## Step 3: Install openAI gym environment under this virtual environment: venv
#### Tensorflow versions
The master branch supports Tensorflow from version 1.4 to 1.14. For Tensorflow 2.0 support, please use tf2 branch. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
for more details.
- Install gym and tensorflow packages:
    ```bash
    pip install gym
    pip install gym[atari] 
    pip install tensorflow==1.14
    ```
- Other packages that might be missing:
    ```bash
    pip install filelock
    pip install matplotlib
    pip install pandas
    ```
## Step 4: Download and Install Official Baseline Package
- Clone the baseline repository to folder DQN-DDPG_Stock_Trading/baselines:
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```

- Install baselines package
    ```bash
    pip install -e .
    ```

## Step 5: Testing the installation
Run all unit tests in baselines:
```
pip install pytest
pytest
```
A result like '94 passed, 49 skipped, 72 warnings in 355.29s' is expected. Check the OpenAI baselines [Issues](https://github.com/openai/baselines/issues) or stackoverflow if fixes on failed tests are needed.

## Step 6: Test OpenAI Atari Pong game
### If this works then it's ready to implement the stock trading application
```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=1e4 --load_path=~/models/pong_20M_ppo2 --play
```
A mean reward per episode around 20 is expected.

## Step 7: Register Stock Trading Environment under gym

Register the RLStock-v0 environment in folder /DQN-DDPG_Stock_Trading/venv:
From
```bash
DQN-DDPG_Stock_Trading/gym/envs/__init__.py
```
Copy following:
```bash
register(
    id='RLStock-v0',
    entry_point='gym.envs.rlstock:StockEnv',
)
register(
    id='RLTestStock-v0',
    entry_point='gym.envs.rlstock:StockTestEnv',
)
```
into the venv gym environment:
```bash
/DQN-DDPG_Stock_Trading/venv/lib/python3.6/site-packages/gym/envs/__init__.py
```
## Step 8: Build Stock Trading Environment under gym

- Copy folder
```bash
DQN_Stock_Trading/gym/envs/rlstock
```
into the venv gym environment folder:
```bash
/DQN-DDPG_Stock_Trading/venv/lib/python3.6/site-packages/gym/envs
```

- Open
```bash
/DQN-DDPG_Stock_Trading/venv/lib/python3.6/site-packages/gym/envs/rlstock/rlstock_env.py 
/DQN-DDPG_Stock_Trading/venv/lib/python3.6/site-packages/gym/envs/rlstock/rlstock_testenv.py
```
change the import data path in these two files (cd into the rlstock folder and pwd to check the folder path).
### Baseline
Replace 
```bash
/DQN-DDPG_Stock_Trading/baselines/baselines/run.py
```
with
```bash
/DQN-DDPG_Stock_Trading/run.py
```

## Step 9: Training model and Testing

### Pre-steps:
Go to folder 
```
/DQN-DDPG_Stock_Trading/
```
Activate the virtual environment 
```
source venv/bin/activate
```
Go to the baseline folder
```
/DQN-DDPG_Stock_Trading/baselines
```
### Train
To train the model, run this
```bash
python -m baselines.run --alg=ddpg --env=RLStock-v0 --network=mlp --num_timesteps=1e4
```
### Trade
To see the testing/trading result, run this
```bash
python -m baselines.run --alg=ddpg --env=RLStock-v0 --network=mlp --num_timesteps=2e4 --play
```

The result images are under folder /DQN-DDPG_Stock_Trading/baselines.

(You can tune the hyperparameter num_timesteps to better train the model, note that if this number is too high, then you will face an overfitting problem, if it's too low, then you will face an underfitting problem.)

Compare to our result:

<img src=result_trading.png width="500">


### Some Other Commands May Need:
```bash
pip3 install opencv-python
pip3 install lockfile
pip3 install -U numpy
pip3 install mujoco-py==0.5.7
```

#### Please cite the following paper
Xiong, Z., Liu, X.Y., Zhong, S., Yang, H. and Walid, A., 2018. Practical deep reinforcement learning approach for stock trading, NeurIPS 2018 AI in Finance Workshop.

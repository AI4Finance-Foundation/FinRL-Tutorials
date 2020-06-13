# Practical Deep Reinforcement Learning Approach for Stock Trading


## Prerequisites
Python 3.6 envrionment

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
Clone this repo and cd into it:
```bash
git clone https://github.com/hust512/DQN-DDPG_Stock_Trading.git
cd DQN-DDPG_Stock_Trading
```
Under this folder DQN-DDPG_Stock_Trading, create a virtual environment
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages.
To create a virtualenv called venv with python3, one runs
```bash
virtualenv -p python3 venv
```
To activate a virtualenv:
```
source venv/bin/activate
```
Your terminate bash will become something like this:
```
(venv) bruceyang-MBP:DQN-DDPG_Stock_Trading bruce$
```
There will be a folder named venv under DQN-DDPG_Stock_Trading

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
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```

- Install baselines package
    ```bash
    pip install -e .
    ```

## Step 5: Testing the installation
All unit tests in baselines can be run using pytest runner:
```
pip install pytest
pytest
```
All unit tests have to get passed, in the end you will see something like: 94 passed, 49 skipped, 72 warnings in 355.29s. If there are any errors or failed tests, you have to debug it, check the openai baselines [Issues](https://github.com/openai/baselines/issues) or stackoverflow to make sure all unit tests passed in the end.

Some failed tests will not affect our stock trading application (for example, a ssl verification error we had), you can proceed to see if it runs or not.

## Step 6: Test-run OpenAI Atari Pong game
### If this works for you then you are ready to implement the stock trading application
Set num_timesteps to 1e4 for test-run purpose
```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=1e4 --save_path=~/models/pong_20M_ppo2
```
This should get to the mean reward per episode about 20. To load and visualize the model, we'll do the following - load the model, train it for 0 steps, and then visualize:
```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=0 --load_path=~/models/pong_20M_ppo2 --play
```
Now, you have successfully used the OpenAI baseline PPO algorithm to play the Atari Pong game.

## Step 7: Register Stock Trading Environment under gym

Find your gym package under environment folder, in my computer (or an EC2 instance) it is under
```bash
/Users/bruceyang/Documents/GitHub/DQN-DDPG_Stock_Trading/venv/lib/python3.6/site-packages/gym/
```
If the virtual environment doesn't work for you, then you have to install everything into your local, then the gym package will be installed under anaconda3:
```bash
/Users/bruceyang/anaconda3/lib/python3.6/site-packages/gym/
```

Register the RLStock-v0 environment into your venv gym environment:
Check this file from our repository
```bash
DQN-DDPG_Stock_Trading/gym/envs/__init__.py
```
Copy this part:
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
into your venv gym environment:
```bash
/DQN-DDPG_Stock_Trading/venv/lib/python3.6/site-packages/gym/envs/__init__.py
```
## Step 8: Build Stock Trading Environment under gym

- Add the folder from our repository 
```bash
DQN_Stock_Trading/gym/envs/rlstock of our repository
```
into your venv gym environment folder:
```bash
/DQN-DDPG_Stock_Trading/venv/lib/python3.6/site-packages/gym/envs
```

- Open
```bash
gym/envs/rlstock/rlstock_env.py and gym/envs/rlstock/rlstock_testenv.py
```
change the data path which is hardcoded.

### Baseline
- Open your baselines folder cloned before, find
```bash
baselines/baselines/run.py
```

- Replace it with
```bash
DQN_Stock_Trading/run.py in this reposotory
```

## Step 9: Training model and Testing

If you only want to train the model run this
```bash
python -m baselines.run --alg=ddpg --env=RLStock-v0 --network=mlp --num_timesteps=1e4
```

If you also want to see the testing/trading result
```bash
python -m baselines.run --alg=ddpg --env=RLStock-v0 --network=mlp --num_timesteps=2e4 --play
```

Your result image is in the baseline folder.

You can tune the hyperparameter num_timesteps to better train the model, note that if this number is too high, then you will face an overfitting problem, if it's too low, then you will face an underfitting problem.

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

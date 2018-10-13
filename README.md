# Practical Deep Reinforcement Learning Approach for Stock Trading


## Prerequisites 
Python 3.6 envrionment 

### CMake, OpenMPI
Installation of system packages CMake, OpenMPI on Mac 
```bash
brew install cmake openmpi
```
    
### Activate your envrionment using using conda or Anaconda
```bash
source activate myenv
```

### Install gym under this environment
```bash
pip install gym
```

## Download Official Baseline Package
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```
- Install tendorflow
    ```bash
    pip install Tensorflow
    ```
    should be sufficient. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
    for more details. 

- Install baselines package
    ```bash
    pip install -e .
    ```

### Testing the installation
All unit tests in baselines can be run using pytest runner:
```
pip install pytest
pytest
```

## Replace files with files in this repository and change file address

### gym
Find your gym package under environment folder, in my computer it is under
```bash
/Users/xiongzhuoran/anaconda3/envs/venv/lib/python3.6/site-packages/gym/
```
- Replece the file 
```bash
gym\envs\__init__.py
``` 
with file from this repository
```bash
DQN_Stock_Trading/gym/envs/__init__.py 
```

- Add folder in this repository to gym\envs in your computer
```bash
DQN_Stock_Trading/gym/envs/zxstock of this repository 
```

- Open 
```bash
gym/envs/zxstock/zxstock_env.py and gym/envs/zxstock/zxstock_testenv.py
```
change the address at line 9 and line 10 into where you want to save the image

### Baseline
- Open your baselines folder cloned before, find 
```bash
baselines/baselines/run.py
```

- Replace it with 
```bash
DQN_Stock_Trading/baselines/baselines/run.py in this reposotory
```

## Training model and Testing
If you only want to train the model run this
```bash
python -m baselines.run --alg=ddpg --env=ZXStock-v0 --network=mlp --num_timesteps=1e4
```

If you also want to see the testing result
```bash
python -m baselines.run --alg=ddpg --env=ZXStock-v0 --network=mlp --num_timesteps=1e4 --play
```



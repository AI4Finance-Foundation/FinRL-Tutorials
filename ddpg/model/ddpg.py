import tensorflow as tf
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D 
from keras.layers.core import Flatten, Lambda
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers.core import Dense
from keras.engine.topology import Merge
from keras.layers.advanced_activations import PReLU
from keras.layers import SpatialDropout2D
from keras.layers import Dropout
from keras import backend as K
import numpy as np
import pandas as pd
import time
# local library
from memory import SequentialMemory

class DDPG(object):
    """Deep Deterministic Poilicy Gradient
    
    Basend on DDPG and Multiscale CNN, seek out 
    optimal strategy for stock trading.
    
    Available function
    - build_model: build network based on tensorflow and keras
    - train: given DateFrame stock data, train network
    - predict_action: givne DataFrame stock data, return optimal protfolio
    """
    
    def __init__(self, config):
        """initialized approximate value function
        
        config should have the following attributes
        
        Args:
            device: the device to use computation, e.g. '/gpu:0'
            gamma(float): the decay rate for value at RL
            history_length(int): input_length for each scale at CNN
            n_feature(int): the number of type of input 
                (e.g. the number of company to use at stock trading)
            trade_stock_idx(int): trading stock index
            gam (float): discount factor
            n_history(int): the nubmer of history that will be used as input
            n_smooth, n_down(int): the number of smoothed and down sampling input at CNN
            k_w(int): the size of filter at CNN
            n_hidden(int): the size of fully connected layer
            n_batch(int): the size of mini batch
            n_epochs(int): the training epoch for each time
            update_rate (0, 1): parameter for soft update
            learning_rate(float): learning rate for SGD
            memory_length(int): the length of Replay Memory
            n_memory(int): the number of different Replay Memories
            alpha, beta: [0, 1] parameters for Prioritized Replay Memories
            action_scale(float): the scale of initialized ation
        """
        self.device = config.device
        self.save_path = config.save_path
        self.is_load = config.is_load
        self.gamma = config.gamma
        self.history_length = config.history_length
        self.n_stock = config.n_stock
        self.n_smooth = config.n_smooth
        self.n_down = config.n_down
        self.n_batch = config.n_batch
        self.n_epoch = config.n_epoch
        self.update_rate = config.update_rate
        self.alpha = config.alpha
        self.beta = config.beta
        self.lr = config.learning_rate
        self.memory_length = config.memory_length
        self.n_memory = config.n_memory
        self.noise_scale = config.noise_scale
        self.model_config = config.model_config
        # the length of the data as input
        self.n_history = max(self.n_smooth + self.history_length, (self.n_down + 1) * self.history_length)
        print ("building model....")
        # have compatibility with new tensorflow
        tf.python.control_flow_ops = tf
        # avoid creating _LEARNING_PHASE outside the network
        K.clear_session()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        K.set_session(self.sess)
        with self.sess.as_default():
            with tf.device(self.device):
                self.build_model()
        print('finished building model!')
    
    def train(self, input_data):
        self.max_action = 100
        """training DDPG, where action is confined to integer space
        
        Args:
            data (DataFrame): stock price for self.n_feature companies
        """
        stock_data = input_data.values
        date = input_data.index
        T = len(stock_data)
        
        # frequency for output
        print_freq = int(T / 10)
        if print_freq == 0:
            print_freq = 1
            
        print ("training....")
        st = time.time()
        # prioritizomg parameter
        db = (1 - self.beta) / 1000
        
        # result for return value
        values = []
        date_label = []
        value = 0
        values.append(value)
        date_label.append(date[0])
        # keep half an year data 
        t0 = self.n_history + self.n_batch
        self.initialize_memory(stock_data[:t0])
        plot_freq = 10
        save_freq = 100000
        count = 0
        for t in range(t0, T - 1):
            self.update_memory(stock_data[t], stock_data[t+1])
            reward = self.take_action(stock_data[t], stock_data[t+1])
            value += reward
            date_label.append(date[t+1])
            values.append(value)
            count += 1
            for epoch in range(self.n_epoch):    
                # select transition from pool
                self.update_weight()
                # update prioritizing paramter untill it goes over 1
                # self.beta  += db
                if self.beta >= 1.0:
                    self.beta = 1.0
                 
            if t % print_freq == 0:
                print ("time:",  date[t + 1])
                action = self.predict_action(stock_data[t+1])
                print("portfolio:", action)
                print("reward:", reward)
                print("value:", value)
                print ("elapsed time", time.time() - st)
                print("********************************************************************")
                
            if count % plot_freq == 0:
                result = pd.DataFrame(values, index=pd.DatetimeIndex(date_label))
                result.to_csv("training_result.csv")
                
            if count % save_freq == 0:
                save_path = self.saver.save(self.sess, self.save_path)
                print("Model saved in file: %s" % self.save_path)

        save_path = self.saver.save(self.sess, self.save_path)
        print("Model saved in file: %s" % self.save_path)
        print ("finished training")
           
        return pd.DataFrame(values, index=pd.DatetimeIndex(date_label))
    
    def norm_action(self, action):
        max_action = np.max(np.abs(action))
        if max_action > 1:
            return action / max_action
        else:
            return action
    
    def predict_action(self, state):
        """Preduct Optimal Portfolio
        
        Args:
            state(float): stock data with size: [self.n_stock, ]
        Retrun:
            np.array with size: [self.n_stock, ]
        """
        pred_state = self.memory[0].sample_state_uniform(self.n_batch, self.n_history)
        new_state = pred_state[-1]
        new_state = np.concatenate((new_state[1:], [state]), axis=0)
        pred_state = np.concatenate((pred_state[:-1], [new_state]), axis=0)
        action = self.actor_output.eval(
            session=self.sess,
            feed_dict={self.state: pred_state, K.learning_phase(): 0})[-1]
        # action = self.norm_action(action)
        return action
    
    def update_weight(self):
        # pararel memory update
        idx = np.random.randint(0, self.n_memory)
        experiences, weights = self.memory[idx].sample(self.n_batch, self.n_history, self.alpha, self.beta)
        self.sess.run(self.critic_optim, 
                      feed_dict={self.state: experiences.state0,
                                 self.state_target: experiences.state1,
                                 self.reward: experiences.reward,
                                 self.action: experiences.action,
                                 self.weights: weights,
                                 self.learning_rate: self.lr,
                                 K.learning_phase(): 1})  
        self.sess.run(self.actor_optim,
                      feed_dict={self.state: experiences.state0,
                                 self.learning_rate: self.lr,
                                 K.learning_phase(): 1})  
                
        error = self.sess.run(self.error,
                              feed_dict={self.state: experiences.state0,
                                         self.state_target: experiences.state1,
                                         self.reward: experiences.reward,
                                         self.action: experiences.action,
                                         K.learning_phase(): 0})
        self.memory[idx].update_priority(error)
                    
        # softupdate for critic network
        old_weights = self.critic_target.get_weights()
        new_weights = self.critic.get_weights()
        weights = [self.update_rate * new_w + (1 - self.update_rate) * old_w
                   for new_w, old_w in zip(new_weights, old_weights)]
        self.critic_target.set_weights(weights)
        
    def initialize_memory(self, stocks):
        self.memory = []
        for i in range(self.n_memory):
            self.memory.append(SequentialMemory(self.memory_length))
        for t in range(len(stocks) - 1):
            for idx_memory in range(self.n_memory):
                action = np.random.normal(0, self.noise_scale, self.n_stock)
                action = self.norm_action(action)
                reward = np.sum((stocks[t + 1] - stocks[t]) * action)
                self.memory[idx_memory].append(stocks[t], action, reward)
        
    def update_memory(self, state, state_forward):
        # update memory without updating weight
        for i in range(self.n_memory):
            self.memory[i].observations.append(state)
            self.memory[i].priority.append(1.0)
        # to stabilize batch normalization, use other samples for prediction
        pred_state = self.memory[0].sample_state_uniform(self.n_batch, self.n_history)
        # off policy action and update portfolio
        actor_action = self.actor_output.eval(session=self.sess,
                                      feed_dict={self.state: pred_state,
                                                          K.learning_phase(): 0})[-1]
        action_scale = np.mean(np.abs(actor_action))
        # action_off = np.round(actor_value_off + np.random.normal(0, noise_scale, self.n_stock))
        for i in range(self.n_memory):
            action_off = actor_action + np.random.normal(0, action_scale * self.noise_scale, self.n_stock)
            action_off = self.norm_action(action_off)
            # action_off = actor_value_off
            reward_off = reward = np.sum((state_forward - state) * action_off)
            self.memory[i].rewards.append(reward_off)
            self.memory[i].actions.append(action_off)
       
    def take_action(self, state, state_forward):
        # to stabilize batch normalization, use other samples for prediction
        pred_state = self.memory[0].sample_state_uniform(self.n_batch, self.n_history)
        # off policy action and update portfolio
        action = self.actor_output.eval(session=self.sess,
                                      feed_dict={self.state: pred_state,
                                                          K.learning_phase(): 0})[-1]
        reward = np.sum((state_forward - state) * action)
        return reward
    
    
    def build_model(self):
        """Build all of the network and optimizations
        
        just for conveninece of trainig, seprate placehoder for train and target network
        critic network input: [raw_data, smoothed, downsampled, action]
        actor network input: [raw_data, smoothed, downsampled]
        """
        self.critic = self.build_critic()
        self.critic_target = self.build_critic()
        # actor network input should be [raw_data, smoothed, downsampled]
        self.actor = self.build_actor()
        # transform input into the several scales and smoothing
        self.state =  tf.placeholder(tf.float32, [None, self.n_history, self.n_stock], name='state')
        self.state_target = tf.placeholder(tf.float32, [None, self.n_history, self.n_stock], name='state_target')
        # reshape to convolutional input
        state_ = tf.reshape(self.state, [-1, self.n_history, self.n_stock, 1])
        state_target_ = tf.reshape(self.state_target, [-1, self.n_history, self.n_stock, 1])
        raw, smoothed, down = self.transform_input(state_)
        raw_target, smoothed_target, down_target = self.transform_input(state_target_)
        
        # build graph for citic training
        self.action = tf.placeholder(tf.float32, [None, self.n_stock])
        input_q = [raw,] +  smoothed + down + [self.action,]
        self.Q = tf.squeeze(self.critic(input_q))
        # target network
        # for double q-learning we use actor network not for target network
        self.actor_target_output = self.actor([raw_target,] +  smoothed_target + down_target)
        input_q_target = [raw_target,] +  smoothed_target + down_target + [self.actor_target_output,]
        Q_target = tf.squeeze(self.critic_target(input_q_target))
        self.reward = tf.placeholder(tf.float32, [None], name='reward')
        target = self.reward  + self.gamma * Q_target
        self.target_value = self.reward  + self.gamma * Q_target
        # optimization
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        # get rid of bias of prioritized
        self.weights = tf.placeholder(tf.float32, shape=[None], name="weights")
        self.loss = tf.reduce_mean(self.weights * tf.square(target - self.Q), name='loss')
        # TD-error for priority
        self.error = tf.abs(target - self.Q)
        self.critic_optim = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.loss, var_list=self.critic.trainable_weights)
        
        # build graph for actor training
        self.actor_output = self.actor([raw,] +  smoothed + down)
        input_q_actor = [raw,] +  smoothed + down + [self.actor_output,]
        self.Q_actor = tf.squeeze(self.critic(input_q_actor))
        # optimization
        self.actor_optim = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(-self.Q_actor, var_list=self.actor.trainable_weights)
        
        self.saver = tf.train.Saver()
        is_initialize = True
        if self.is_load:
            if self.load(self.save_path):
                print('succeded to load')
                is_initialize = False
            else:
                print('failed to load')
        
        # initialize network
        if is_initialize:
            tf.global_variables_initializer().run(session=self.sess)
            weights = self.critic.get_weights()
            self.critic_target.set_weights(weights)
        
    def build_critic(self):
        """Build critic network
        
        recieve convereted tensor: raw_data, smooted_data, and downsampled_data
        """
        # lower layer
        lower_model = [self.build_network(self.model_config['critic_lower'], input_shape=(self.history_length, self.n_stock, 1)) 
                       for _ in range(1  + self.n_smooth + self.n_down)]
        merged = Merge(lower_model, mode='concat')
        # upper layer
        upper_model = self.build_network(self.model_config['critic_upper'],  model=merged)
        # action layer
        action = self.build_network(self.model_config['critic_action'], input_shape=(self.n_stock,), is_conv=False)
        # output layer
        merged = Merge([upper_model, action], mode='mul')
        model = Sequential()
        model.add(merged)
        model.add(Dense(1))
        return model
    
    def build_actor(self):
        """Build actor network
        
        recieve convereted tensor: raw_data, smooted_data, and downsampled_data
        """
        # lower layer
        lower_model = [self.build_network(self.model_config['actor_lower'], input_shape=(self.history_length, self.n_stock, 1)) 
                       for _ in range(1  + self.n_smooth + self.n_down)]
        merged = Merge(lower_model, mode='concat')
        # upper layer
        model = self.build_network(self.model_config['actor_upper'],  model=merged)
        return model
    
    def build_network(self, conf, model=None, input_shape=None, is_conv=True):
        """Build network"""
        _model = model
        model = Sequential()
        if _model is None:
            model.add(Lambda(lambda x: x,  input_shape=input_shape))
        else:
            model.add(_model)
            
        for x in conf:
            if x['is_drop']:
                model.add(Dropout(x['drop_rate']))
            if x['type'] is 'full':
                if is_conv:
                    model.add(Flatten())
                    is_conv = False
                model.add(Dense(x['n_feature']))
            elif x['type'] is 'conv':
                model.add(Convolution2D(nb_filter=x['n_feature'], 
                                        nb_row=x['kw'], 
                                        nb_col=1, 
                                        border_mode='same'))  
                is_conv=True
            if x['is_batch']:
                if x['type'] is 'full':
                    model.add(BatchNormalization(mode=1, axis=-1))
                if x['type'] is 'conv':
                    model.add(BatchNormalization(mode=2, axis=-1))
            model.add(x['activation'])
        return model
    
    
    def transform_input(self, input):
        """Transform data into the Multi Scaled one
        
        Args:
            input: tensor with shape: [None, self.n_history, self.n_stock]
        Return:
            list of the same shape tensors, [None, self.length_history, self.n_stock]
        """
        # the last data is the newest information
        raw = input[:, self.n_history - self.history_length:, :, :]
        # smooth data
        smoothed = []
        for n_sm in range(2, self.n_smooth + 2):
            smoothed.append(
                tf.reduce_mean(tf.pack([input[:, self.n_history - st - self.history_length:self.n_history - st, :, :]
                                        for st in range(n_sm)]),0))
        # downsample data
        down = []
        for n_dw in range(2, self.n_down + 2):
            sampled_ = tf.pack([input[:, idx, :, :] 
                                for idx in range(self.n_history-n_dw*self.history_length, self.n_history, n_dw)])
            down.append(tf.transpose(sampled_, [1, 0, 2, 3]))
        return raw, smoothed, down
    
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        try:
            self.saver.restore(self.sess, self.save_path)
            return True
        except:
            return False
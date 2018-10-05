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
from keras.layers import Dropout, Reshape
from keras import backend as K
import numpy as np
import pandas as pd
import time
# local library
from memory import SequentialMemory

class DQN(object):
    """Deep Q-Learning Networ
    
    Basend on DQN and Multiscale CNN, find the optimal time to 
    exit from a stock market.
    
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
        """
        self.device = config.device
        self.save_path = config.save_path
        self.is_load = config.is_load
        self.gamma = config.gamma
        self.history_length = config.history_length
        self.n_stock = config.n_stock
        self.n_feature = config.n_feature
        self.n_smooth = config.n_smooth
        self.n_down = config.n_down
        self.k_w = config.k_w
        self.n_hidden = config.n_hidden
        self.n_batch = config.n_batch
        self.n_epochs = config.n_epochs
        self.update_rate = config.update_rate
        self.alpha = config.alpha
        self.beta = config.beta
        self.lr = config.learning_rate
        self.memory_length = config.memory_length
        self.n_memory = config.n_memory
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
    
    def train(self, input_data, noise_scale=0.1):
        """training DQN, which has two actions: 0-exit, 1-stay
        
        Args:
            data (DataFrame): stock price for self.n_feature companies
        """
        stock_data = input_data.values
        date = input_data.index
        T = len(stock_data)
        self.noise_scale = noise_scale
        
        # frequency for output
        print_freq = int(T / 100)
        if print_freq == 0:
            print_freq = 1
        print ("training....")
        st = time.time()
        #  udpate rate for prioritizing parameter
        db = (1 - self.beta) / 1000
        
        # result for return value
        values = [[] for _ in range(self.n_stock)]
        date_label = [[] for _ in range(self.n_stock)]
        date_use = []
        stock_use = []
        # will not train until getting enough data
        t0 = self.n_history + self.n_batch
        self.initialize_memory(stock_data[:t0], scale=noise_scale)
        save_data_freq = 10
        save_weight_freq = 10
        count = 0
        input_data.to_csv("stock_price.csv")
        for t in range(t0, T):
            stock_use.append(stock_data[t])
            date_use.append(date[t])
            action = self.predict_action(stock_data[t])
            for i in range(self.n_stock):
                if action[i] == 0:
                    date_label[i].append(date[t])
                    values[i].append(stock_data[t][i])
            self.update_memory(stock_data[t])
            count += 1
            for epoch in range(self.n_epochs):    
                # select transition from pool
                self.update_weight()
                # update prioritizing paramter untill it goes over 1
            self.beta  += db
            if self.beta >= 1.0:
                self.beta = 1.0
            idx = np.random.randint(0, self.n_memory)
            
            experiences, weights = self.memory[idx].sample(self.n_batch, self.n_history, self.alpha, self.beta)
            max_idx = self.get_max_idx(experiences.state1)
            target_value = self.sess.run(self.target_value,
                                     feed_dict={self.state_target: experiences.state1,
                                 self.reward: experiences.reward,
                                               self.max_idx_target: max_idx})
            
            if t % print_freq == 0:
                print ("time:",  date[t])
                error = self.sess.run(self.error,
                              feed_dict={self.state: experiences.state0,
                                         self.target: target_value,
                                         self.reward: experiences.reward,
                                         K.learning_phase(): 0})
                print("error:", np.mean(error))
                action = self.predict_action(stock_data[t])
                print("portfolio:", action)
                print ("elapsed time", time.time() - st)
                print("********************************************************************")
                
            if count % save_data_freq == 0:
                for i in range(self.n_stock):
                    result = pd.DataFrame(values[i], index=pd.DatetimeIndex(date_label[i]))
                    result.to_csv("exit_result_{}.csv".format(i))
                data_use = pd.DataFrame(stock_use, index=pd.DatetimeIndex(date_use))
                data_use.to_csv("stock_price.csv")
                
            if count % save_weight_freq == 0:
                save_path = self.saver.save(self.sess, self.save_path)
                print("Model saved in file: %s" % self.save_path)

        save_path = self.saver.save(self.sess, self.save_path)
        print("Model saved in file: %s" % self.save_path)
        print ("finished training")
        
        return [pd.DataFrame(values[i], index=pd.DatetimeIndex(date_label[i])) for i in range(self.n_stock)]
    
    def predict_action(self, state):
        """Preduct Optimal strategy
        
        Args:
            state(float): stock data with size: [self.n_stock, ]
        Retrun:
            integer: 0-exit, 1-stay
        """
        pred_state = self.memory[0].sample_state_uniform(self.n_batch, self.n_history)
        new_state = pred_state[-1]
        new_state = np.concatenate((new_state[1:], [state]), axis=0)
        pred_state = np.concatenate((pred_state[:-1], [new_state]), axis=0)
        action = self.max_action.eval(
            session=self.sess,
            feed_dict={self.state: pred_state, K.learning_phase(): 0})[-1]
        return action
    
    def update_weight(self):
        """Update networks' parameters and memories"""
        idx = np.random.randint(0, self.n_memory)
        experiences, weights = self.memory[idx].sample(self.n_batch, self.n_history, self.alpha, self.beta)
        max_idx = self.get_max_idx(experiences.state1)
        # get target value for optimization
        target_value = self.sess.run(self.target_value,
                                     feed_dict={self.state_target: experiences.state1,
                                 self.reward: experiences.reward,
                                               self.max_idx_target: max_idx})
        # optimize network
        self.sess.run(self.critic_optim, 
                      feed_dict={self.state: experiences.state0,
                                 self.target: target_value,
                                 self.weights: weights,
                                 self.learning_rate: self.lr,
                                 K.learning_phase(): 1})  
        # compute errors to determine prioritizing ratio
        error = self.sess.run(self.error,
                              feed_dict={self.state: experiences.state0,
                                         self.target: target_value,
                                         self.reward: experiences.reward,
                                         K.learning_phase(): 0})
        self.memory[idx].update_priority(error)
        # softupdate for critic network
        old_weights = self.critic_target.get_weights()
        new_weights = self.critic.get_weights()
        weights = [self.update_rate * new_w + (1 - self.update_rate) * old_w
                   for new_w, old_w in zip(new_weights, old_weights)]
        self.critic_target.set_weights(weights)
        
    def initialize_memory(self, stocks, scale=10):
        self.memory = []
        for i in range(self.n_memory):
            self.memory.append(SequentialMemory(self.memory_length))
        for t in range(len(stocks)):
            for idx_memory in range(self.n_memory):
                action = None
                reward = np.concatenate((np.reshape(stocks[t], (self.n_stock, 1)), np.zeros((self.n_stock, 1))), axis=-1)
                self.memory[idx_memory].append(stocks[t], action, reward)
        
    def update_memory(self, state):
        """Update memory without updating weight"""
        for i in range(self.n_memory):
            self.memory[i].observations.append(state)
            self.memory[i].priority.append(1.0)
        # to stabilize batch normalization, use other samples for prediction
        pred_state = self.memory[0].sample_state_uniform(self.n_batch, self.n_history)
        for i in range(self.n_memory):
            action_off = None
            reward_off = np.concatenate((np.reshape(state, (self.n_stock, 1)), np.zeros((self.n_stock, 1))), axis=-1)
            self.memory[i].rewards.append(reward_off)
            self.memory[i].actions.append(action_off)
    
    def get_max_idx(self, state):
        max_action = self.sess.run(self.max_action_target, feed_dict={self.state_target: state})
        shape = max_action.shape
        max_idx = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                max_idx.append([i, j, max_action[i][j]])
        return np.array(max_idx, dtype=int)
    
    
    def build_model(self):
        """Build all of the network and optimizations
        
        just for conveninece of trainig, seprate placehoder for train and target network
        critic network input: [raw_data, smoothed, downsampled]
        """
        self.critic = self.build_critic()
        self.critic_target = self.build_critic()
        # transform input into the several scales and smoothing
        self.state =  tf.placeholder(tf.float32, [None, self.n_history, self.n_stock], name='state')
        self.state_target = tf.placeholder(tf.float32, [None, self.n_history, self.n_stock], name='state_target')
        # reshape to convolutional input
        state_ = tf.reshape(self.state, [-1, self.n_history, self.n_stock, 1])
        state_target_ = tf.reshape(self.state_target, [-1, self.n_history, self.n_stock, 1])
        raw, smoothed, down = self.transform_input(state_)
        raw_target, smoothed_target, down_target = self.transform_input(state_target_)
        
        # build graph for citic training
        input_q = [raw,] +  smoothed + down
        self.Q = self.critic(input_q)
        self.max_action = tf.argmax(self.Q, dimension=2)
        # target network
        input_q_target = [raw_target,] +  smoothed_target + down_target
        Q_target = self.critic_target(input_q_target)
        self.reward = tf.placeholder(tf.float32, [None, self.n_stock, 2], name='reward')
        double_Q = self.critic(input_q_target)
        self.max_action_target = tf.argmax(double_Q, 2)
        self.max_idx_target = tf.placeholder(tf.int32, [None, 3], "double_idx")
        Q_max = tf.gather_nd(Q_target, self.max_idx_target)
        Q_max = tf.reshape(Q_max, [-1, self.n_stock, 1])
        Q_value = tf.concat(2, (tf.zeros_like(Q_max), Q_max))
        self.target_value = self.reward  + self.gamma * Q_value
        self.target_value = tf.cast(self.target_value, tf.float32)
        self.target = tf.placeholder(tf.float32, [None, self.n_stock, 2], name="target_value")
        # optimization
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        # get rid of bias of prioritized
        self.weights = tf.placeholder(tf.float32, shape=[None], name="weights")
        self.loss = tf.reduce_mean(self.weights * tf.reduce_sum(tf.square(self.target - self.Q), [1, 2]), name='loss')
        # TD-error for priority
        self.error = tf.reduce_sum(tf.abs(self.target - self.Q), [1, 2])
        self.critic_optim = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.loss, var_list=self.critic.trainable_weights)
        
        self.saver = tf.train.Saver()
        is_initialize = True
        if self.is_load:
            if self.load(self.save_path):
                print('succeded to load')
                is_initialize = False
            else:
                print('failed to load')
        
        # initialize network
        tf.initialize_all_variables().run(session=self.sess)
        weights = self.critic.get_weights()
        self.critic_target.set_weights(weights)
        
    def build_critic(self):
        """Build critic network
        
        recieve transformed tensor: raw_data, smooted_data, and downsampled_data
        """
        nf = self.n_feature
        # layer1
        # smoothed input
        sm_model = [Sequential() for _ in range(self.n_smooth)]
        for m in sm_model:
            m.add(Lambda(lambda x: x,  input_shape=(self.history_length, self.n_stock, 1)))
            m.add(Convolution2D(nb_filter=nf, nb_row=self.k_w, nb_col=1, border_mode='same'))
            m.add(BatchNormalization(mode=2, axis=-1))
            m.add(PReLU())
        # down sampled input
        dw_model = [Sequential() for _ in range(self.n_down)]
        for m in dw_model:
            m.add(Lambda(lambda x: x,  input_shape=(self.history_length, self.n_stock, 1)))
            m.add(Convolution2D(nb_filter=nf, nb_row=self.k_w, nb_col=1, border_mode='same'))
            m.add(BatchNormalization(mode=2, axis=-1))
            m.add(PReLU())
        # raw input
        state = Sequential()
        nf = self.n_feature
        state.add(Lambda(lambda x: x,  input_shape=(self.history_length, self.n_stock, 1)))
        state.add(Convolution2D(nb_filter=nf, nb_row=self.k_w, nb_col=1, border_mode='same'))
        state.add(BatchNormalization(mode=2, axis=-1))
        state.add(PReLU())
        merged = Merge([state,] + sm_model + dw_model, mode='concat', concat_axis=-1)
        # layer2
        nf = nf * 2
        model = Sequential()
        model.add(merged)
        model.add(Convolution2D(nb_filter=nf, nb_row=self.k_w, nb_col=1, border_mode='same'))
        model.add(BatchNormalization(mode=2, axis=-1))
        model.add(PReLU())
        model.add(Flatten())
        # layer3
        model.add(Dense(self.n_hidden))
        model.add(BatchNormalization(mode=1, axis=-1))
        model.add(PReLU())
        # layer4
        model.add(Dense(int(np.sqrt(self.n_hidden))))
        model.add(PReLU())
        # output
        model.add(Dense(2 * self.n_stock))
        model.add(Reshape((self.n_stock, 2)))
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
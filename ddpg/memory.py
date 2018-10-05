from collections import deque, namedtuple
import numpy as np
from numpy.random import choice

Experiecne = namedtuple('Experience', 'state0,  action, reward, state1')

class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        # self.data = [None for _ in range(maxlen)]
        self.data = []
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx< 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]
    
    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item
            self.data[:-1] = self.data[1:]
        else:
            # This should never happen
            raise RuntimeError()
        self.data.append(v)
        
class SequentialMemory(object):
    def __init__(self, limit=1000):
        self.limit = limit
        self.priority = []
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.observations = RingBuffer(limit)
        self.batch_idx = None

        
    def sample(self, batch_size, window_length, alpha=1.0, beta=1.0, epsilon=0.05):
        # udpate priority when sampling
        if len(self.priority) > self.limit:
            self.priority = self.priority[-self.limit:]
        # draw random indexes such that is bigger than window_length to enough length data
        index_space = np.arange(window_length, self.nb_entries)
        # prioritized sample
        p = np.array(self.priority)[window_length:]
        p_tilde = p + np.ones(self.nb_entries - window_length) * np.mean(p) * epsilon
        p_tilde[-1] = np.mean(p)
        p_tilde = p_tilde ** alpha
        p_tilde = p_tilde / np.sum(p_tilde)
        batch_idx = choice(index_space, p=p_tilde, size=batch_size - 1)
        # take the newest data
        batch_idx = np.concatenate((batch_idx, [self.nb_entries - 1]))
        assert len(batch_idx) == batch_size
        # keep batch_idx to update pritority
        self.batch_idx = batch_idx
        
        # weights to modify biased update
        weights = 1. / (p_tilde**beta)
        weights = weights / np.max(weights)
        ret_w = weights[batch_idx - window_length]
        
        # create experiences
        state0 = np.array([[self.observations[i] for i in range(idx - window_length,idx)] for idx in batch_idx])
        action = np.array([self.actions[idx - 1] for idx in batch_idx])
        reward = np.array([self.rewards[idx - 1] for idx in batch_idx])
        state1 = np.array([[self.observations[i] for i in range(idx - window_length + 1,idx + 1)] for idx in batch_idx])
        return Experiecne(state0, action, reward, state1), ret_w
    
    def sample_state(self, batch_size, window_length, alpha=0.5, epsilon=0.05):
        # udpate priority when sampling
        if len(self.priority) > self.limit:
            self.priority = self.priority[-self.limit:]
        # draw random indexes such that is bigger than window_length to enough length data
        index_space = np.arange(window_length, self.nb_entries)
        # prioritized sample
        p = np.array(self.priority)[window_length:]
        p_tilde = p + np.ones(self.nb_entries - window_length) * np.mean(p) * epsilon
        p_tilde[-1] = np.mean(p)
        p_tilde = p_tilde ** alpha
        p_tilde = p_tilde / np.sum(p_tilde)
        batch_idx = choice(index_space, p=p_tilde, size=batch_size - 1)
        # take the newest data
        batch_idx = np.concatenate((batch_idx, [self.nb_entries]))
        assert len(batch_idx) == batch_size
        
        # create experiences
        state = np.array([[self.observations[i] for i in range(idx - window_length,idx)] for idx in batch_idx])
        return state
    
    def sample_state_uniform(self, batch_size, window_length):
        # draw random indexes such that is bigger than window_length to enough length data
        batch_idx = np.random.random_integers(window_length, self.nb_entries - 1, size=batch_size - 1)
        # take the newest data
        batch_idx = np.concatenate((batch_idx, [self.nb_entries]))
        assert len(batch_idx) == batch_size
        
        # create experiences
        state = np.array([[self.observations[i] for i in range(idx - window_length, idx)] for idx in batch_idx])
        return state
    
    def update_priority(self,error):
        for idx, i in enumerate(self.batch_idx):
            self.priority[i] = error[idx]
    
    
    def append(self, observation, action, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        # initialize new sample with 1
        self.priority.append(1.0)
    
    @property
    def nb_entries(self):
        return  len(self.observations)
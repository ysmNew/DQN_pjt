import collections
import random
import numpy as np
import torch


class ReplayBuffer(): # 보관은 array로 하고 꺼낼때 tensor로 꺼내기
    def __init__(self,buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        h_lst, a_lst, r_lst, h_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            h, a, r, h_prime, done_mask = transition
            h_lst.append(h)
            a_lst.append([a])
            r_lst.append([r])
            h_prime_lst.append(h_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(np.array(h_lst), dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), \
               torch.tensor(np.array(h_prime_lst), dtype=torch.float), torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

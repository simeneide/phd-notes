from pyro.nn import PyroSample, PyroModule
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample, pyro_method
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pyro.distributions as dist
import copy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import models
import agents

def collect_simulated_data(sim, policy_epsilon=0.5, **kwargs):
    randomagent = agents.RandomSystem(num_items=kwargs['num_items'],
                                      maxlen_slate=kwargs['maxlen_slate'])

    def policy_epsilon_greedy(policy_epsilon=policy_epsilon, *args, **kwargs):
        if random.random() < policy_epsilon:
            return sim.env.recommend(*args, **kwargs)
        else:
            return randomagent.recommend(*args, **kwargs)

    # Build training data:
    logging.info(
        f"Simulating data: num users: {kwargs['num_users']}, optimal games: {policy_epsilon}*100%"
    )
    sim.generate_dataset(policy_epsilon_greedy)

    itemattr = {'category': sim.env.item_group.numpy()}
    dataloaders = sim.build_dataloaders(**kwargs)

    ind2val = {
        'itemId': {key: str(key)
                   for key in range(kwargs['num_items'])},
        'userId':
        {key: str(key)
         for key in sim.data['userId'].numpy()}
    }

    return ind2val, itemattr, dataloaders, sim


#################
### SIMULATOR-DATASET
#################


class Simulator(Dataset):
    def __init__(self, **kwargs):
        defaults = {
            'num_users' : None,
            'num_items' : None,
            'batch_size' : 128,
            'env' : None,
            'maxlen_time' : None,
            'maxlen_slate' : None,
            'device' : "cpu",
            }
        # Register all input vars in module:
        for key, val in defaults.items():
            setattr(self,key, kwargs.get(key, val))

        with torch.no_grad():
            self.pos = 0  # counter when adding

            self.data = {
                'userId':
                torch.arange(self.num_users),
                'action':
                torch.zeros((self.num_users, self.maxlen_time,
                             self.maxlen_slate)).long().to(self.device),
                'click':
                torch.zeros(
                    (self.num_users, self.maxlen_time)).long().to(self.device),
                'click_idx':
                torch.zeros(
                    (self.num_users, self.maxlen_time)).long().to(self.device)
            }

    ## DATASET FUNCTIONS
    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        return {key: val[idx, ] for key, val in self.data.items()}

    def build_dataloaders(self, batch_size=512, num_testusers =100, t_testsplit=10, **kwargs):
        torch.manual_seed(0)
        perm_user = torch.randperm(self.num_users)
        valid_user_idx = perm_user[:num_testusers]
        train_user_idx = perm_user[num_testusers:]
        self.data['mask_train'] = torch.ones_like(self.data['click'])
        self.data['mask_train'][valid_user_idx, t_testsplit:] = 0

        subsets = {'train': self, 'valid': torch.utils.data.Subset(self, valid_user_idx)}
        dataloaders = {
            phase: DataLoader(ds, batch_size=batch_size, shuffle=True)
            for phase, ds in subsets.items()
        }
        for key, dl in dataloaders.items():
            logging.info(
                f"In {key}: num_users: {len(dl.dataset)}, num_batches: {len(dl)}"
            )

        return dataloaders

    ## SIMULATOR FUNCTIONS
    def reset(self, userIds = None, t_start = 0):
        if userIds is None:
            userIds = self.data['userId'][:self.batch_size]

        self.batch_user = userIds
        self.t= t_start
        return self.return_data()

    def return_data(self, t=None):
        if t is None:
            t = self.t

        dat = {
            key: val[self.batch_user, :t]
            for key, val in self.data.items() if key != "userId"
        }
        dat['userId'] = self.data['userId'][self.batch_user,]
        return dat

    def step(self, action):
        """Takes an action from policy, evaluates and returns click/reward"""
        done = False
        
        action = action.long().to(self.device)
        # Concat the no click option as an alternative:
        noclick = torch.ones_like(self.batch_user).unsqueeze(1).long().to(self.device)
        self.data['action'][self.batch_user, self.t] = torch.cat( (noclick, action), dim=1)
        action = self.data['action'][self.batch_user, self.t]

        # Let environment sample click or no click:
        self.data['click_idx'][self.batch_user, self.t] = self.env.simulate(
            batch=self.return_data(t=self.t + 1)).to(self.device)

        self.data['click'][self.batch_user, self.t] = self.data[
            'action'][self.batch_user, self.t].gather(
                -1, self.data['click_idx'][self.batch_user, self.t].unsqueeze(
                    -1)).squeeze()
        reward = (self.data['click'][self.batch_user, self.t] >= 2).float().mean()

        self.t += 1

        if self.t == self.maxlen_time:
            done = True

        return self.return_data(), reward, done

    def play_game(self, agent_function, userIds = None, t_end = None):
        """ Play a full game with a given environment function and a given agent function"""
        if t_end is None:
            t_end = self.maxlen_time
        # Sample U_b users with no interactions:
        batch = self.reset(userIds=userIds)

        reward = torch.zeros((t_end, ))
        for t in range(t_end):
            # Let agent recommend:
            action = agent_function(batch=batch, max_rec=self.maxlen_slate -
                                    1).to(self.device)
            #print(action)
            # Let environment generate a click and return an updated user history
            batch, reward[t], done = self.step(action)
            if done is True:
                return reward
        return reward  # return avg cum reward

    def generate_dataset(self, agent_function, t_end=None):
        for userIds in chunker(self.data['userId'], size=self.batch_size):
            self.play_game(agent_function, t_end = t_end, userIds = userIds)


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
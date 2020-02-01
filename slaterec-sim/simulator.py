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
#import model_pyro

#################
### DATASET
#################

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import random

class SequentialData(Dataset):
    def __init__(self, capacity, maxlen_time, maxlen_slate, device="cpu"):
        with torch.no_grad():
            self.capacity = capacity
            self.maxlen_time = maxlen_time
            self.maxlen_slate = maxlen_slate
            self.device = device

            self.cur_size = 0  # how big is dataset now
            self.pos = 0  # counter when adding

            self.data = {
                'userId' : torch.arange(self.capacity),
                'action':
                torch.zeros((self.capacity, maxlen_time,
                             maxlen_slate)).long().to(self.device),
                'click':
                torch.zeros((self.capacity, maxlen_time)).long().to(self.device),
                'click_idx':
                torch.zeros((self.capacity, maxlen_time)).long().to(self.device)
            }

    def __len__(self):
        return min(self.cur_size, self.capacity)

    def __getitem__(self, idx):
        return {key: val[idx, ] for key, val in self.data.items()}

    def push(self, ep_data):
        """ Saves a episode of batch of users to the dataset """
        with torch.no_grad():
            bs = len(ep_data['click'])
            start = self.pos
            end = (self.pos + bs)

            # If at end of batch, clip first steps of episode:
            if end >= self.capacity:
                avail = self.capacity - start
                for key, val in ep_data.items():
                    ep_data[key] = ep_data[key][-avail]
                end = self.capacity

            for key, val in ep_data.items():
                self.data[key][start:end, ] = ep_data[key].float().to(
                    self.device)

            self.cur_size += bs
            self.pos = (self.pos + bs) % self.capacity

    def build_dataloaders(self, batch_size=512, split_trainvalid=0.95):
        torch.manual_seed(0)
        tl = int(len(self) * split_trainvalid)

        subsets = torch.utils.data.random_split(dataset=self,
                                                lengths=[tl,
                                                         len(self) - tl])

        subsets = {'train': subsets[0], 'valid': subsets[1]}
        dataloaders = {
            phase: DataLoader(ds, batch_size=batch_size, shuffle=True)
            for phase, ds in subsets.items()
        }
        for key, dl in dataloaders.items():
            logging.info(
                f"In {key}: num_users: {len(dl.dataset)}, num_batches: {len(dl)}"
            )

        return dataloaders

class RandomSystem:
    def __init__(self, num_items, maxlen_slate, batch_user, *args, **kwargs):
        super().__init__()
        self.num_items = num_items
        self.maxlen_slate = maxlen_slate
        self.batch_user = batch_user

    def forward(self, *args, **kwargs):
        return None

    def recommend(self, *args, **kwargs):
        action = 2 + torch.cat([
            torch.randperm(self.num_items - 2).unsqueeze(0)
            for _ in range(self.batch_user)
        ])
        action = action[:, :(self.maxlen_slate - 1)]
        return action


def collect_simulated_data(env, sim, policy_epsilon=0.5, **kwargs):
    randomagent = RandomSystem(num_items=kwargs['num_items'],
                            maxlen_slate=kwargs['maxlen_slate'],
                            batch_user=kwargs['batch_user'])

    def policy_epsilon_greedy(policy_epsilon=policy_epsilon, *args, **kwargs):
        if random.random()<policy_epsilon:
            return env.recommend(*args, **kwargs)
        else:
            return randomagent.recommend(*args, **kwargs)

    num_games = int(kwargs['num_users']/kwargs['batch_user'])
    # Build training data:
    logging.info(f"Simulating data: num users: {kwargs['num_users']}, optimal games: {policy_epsilon}*100%")
    reward_opt = sim.play_many_games(policy_epsilon_greedy,
                        num_games=num_games).mean()

    itemattr = {'category': env.item_group.numpy()}
    dataloaders = sim.dataset.build_dataloaders(
        batch_size=kwargs['batch_size'])

    ind2val = {
        'itemId': {key: str(key)
                   for key in range(kwargs['num_items'])},
        'userId' : {key : str(key) for key in sim.dataset.data['userId'].numpy()}
    }


    return ind2val, itemattr, dataloaders, sim


import copy

class Simulator:
    def __init__(
        self,
        num_items,
        num_users,
        batch_user,
        maxlen_time,
        maxlen_slate,
        env=None,
        device="cpu",
        # DATA HANDLING
        **kwargs):

        self.device = device
        self.maxlen_time = maxlen_time
        self.num_items = num_items
        self.num_users = num_users
        self.batch_user = batch_user
        self.maxlen_slate = maxlen_slate
        self.env = env
        self.dataset = SequentialData(capacity=num_users,
                                      maxlen_time=maxlen_time,
                                      maxlen_slate=maxlen_slate)
        self.user_iter = 0

    ## GAME FUNCTIONS ###
    def reset(self):
        """Reset game"""
        self.t = 0
        user_endidx = self.user_iter+self.batch_user
        if user_endidx > self.num_users:
            logging.info("All users interacted with!")
            return None
        self.episode_data = {
            'action':
            torch.zeros(
                (self.batch_user, self.maxlen_time, self.maxlen_slate)).to(self.device).long(),
            'click':
            torch.zeros((self.batch_user, self.maxlen_time)).to(self.device).long(),
            'click_idx':
            torch.zeros((self.batch_user, self.maxlen_time)).to(self.device).long(),
            'userId' : torch.arange(start=self.user_iter, end = user_endidx)
        }

        self.user_iter += self.batch_user
        return self.return_data()

    def return_data(self, t=None):
        if t is None:
            t = self.t
        dat = {key: val[:, :t] for key, val in self.episode_data.items() if key != "userId"}
        dat['userId'] = self.episode_data['userId']
        return dat

    def step(self, action):
        """ Takes an action from policy, evaluates and returns click/reward"""
        done = False
        # Concat the no click option as an alternative
        action = action.long().to(self.device)
        self.episode_data['action'][:, self.t] = torch.cat(
            (torch.ones(self.batch_user, 1).to(self.device).long(), action), dim=1)
        action = self.episode_data['action'][:, self.t]

        # Let environment sample click or no click:
        self.episode_data['click_idx'][:, self.t] = self.env.simulate(
            batch=self.return_data(t=self.t + 1) ).to(self.device)
        self.episode_data['click'][:, self.t] = self.episode_data[
            'action'][:, self.t].gather(
                -1, self.episode_data['click_idx'][:, self.t].unsqueeze(
                    -1)).squeeze()
        reward = (self.episode_data['click'][:, self.t] >= 2).float().mean()

        self.t += 1

        if self.t == self.maxlen_time:
            self.dataset.push(copy.deepcopy(self.episode_data))
            done = True

        return self.return_data(), reward, done

    def play_game(self, agent_rec_func):
        """ Play a full game with a given environment function and a given agent function"""

        # Sample U_b users with no interactions:
        batch = self.reset()

        reward = torch.zeros((self.maxlen_time, ))
        for t in range(self.maxlen_time):
            # Let agent recommend:
            action = agent_rec_func(batch=batch, max_rec=self.maxlen_slate - 1).to(self.device)
            #print(action)
            # Let environment generate a click and return an updated user history
            batch, reward[t], done = self.step(action)
            if done is True:
                return reward
        return reward  # return avg cum reward

    def play_many_games(self, agent_rec_func, num_games=1):
        reward = torch.zeros((num_games))
        for t in range(num_games):
            reward[t] = self.play_game(agent_rec_func).mean()

        return reward


### WRAPPER FOR MODELS:
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

class PyroRecommender(PyroModule):
    """ A Pyro Recommender Module. Have some common functionalities that are shared among all recommenders"""
    def __init__(self, **kwargs):
        super().__init__()
        # Register all input vars in module:
        for key, val in kwargs.items():
            setattr(self,key,val)
        self.init_set_of_real_parameters()

    @pyro_method
    def simulate(self, batch):
        return pyro.condition(
            lambda batch: self.forward(batch, mode="simulate"),
            data=self.get_real_par(batch))(batch)

    @pyro_method
    def likelihood(self, batch, par = None): 
        if par is None:
            par = self.get_real_par(batch)

        return pyro.condition(
            lambda batch: self.forward(batch, mode="likelihood"),
            data=par)(batch)

    @pyro_method
    def predict_cond(self, batch, par=None, **kwargs): 
        if par is "real":
            par = self.get_real_par(batch)
        else:
            par = par(batch)
        return pyro.condition(
            lambda batch: self.predict(batch, **kwargs),
            data=par)(batch)

    def recommend(self, batch, max_rec=1, chunksize=3, t=-1, par="real", **kwargs):
        """
        Compute predict & rank on a batch in chunks (for memory)
        """

        click_seq = batch['click']
        topk = torch.zeros((len(click_seq), max_rec), device=self.device)

        i = 0
        for click_chunck, userId in zip(
            chunker(click_seq, chunksize),
            chunker(batch['userId'], chunksize)):
            pred, ht = self.predict_cond(
                batch={'click' :click_chunck, 'userId' : userId}, t=t, par=par)
            topk_chunk = 3 + pred[:, 3:].argsort(dim=1,
                                                 descending=True)[:, :max_rec]
            topk[i:(i + len(pred))] = topk_chunk
            i += len(pred)
        return topk

    def visualize_item_space(self):
        V = self.par_real['item_model.itemvec.weight'].cpu()
        sns.scatterplot(V[:, 0].cpu(),
                        V[:, 1].cpu(),
                        hue=self.item_group.cpu())
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.show()


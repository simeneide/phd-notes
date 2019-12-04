from pyro.nn import PyroSample, PyroModule
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample
import matplotlib.pyplot as plt
import pyro.distributions as dist
##################
### SIMULATOR
##################

class Simulator(nn.Module):
    def __init__(self, item_dim, num_items, batch_user, max_time, max_rho, hidden_dim = None):
        super(Simulator, self).__init__()
        with torch.no_grad():
            self.item_dim = item_dim
            self.max_time = max_time
            self.hidden_dim = item_dim if hidden_dim == None else hidden_dim
            self.num_items = num_items
            self.batch_user = batch_user
            self.max_rho = max_rho

            # item model
            self.itemvec = nn.Embedding(num_embeddings=num_items, embedding_dim = item_dim)
            
            self.item_group = torch.zeros((num_items,)).long()
            self.item_group[5:] = 1

            self.groupvec = torch.randn((2, item_dim))

            V = self.groupvec[self.item_group] + 0.2*torch.randn((num_items,item_dim))
            
            #V = torch.nn.functional.normalize(V, p=2, dim = 1)
            #V[1,:] = 0 # place no click in centre
            self.itemvec.weight = nn.Parameter(V)

            # user model
            self.gru = nn.GRU(
                input_size = self.item_dim, 
                hidden_size = self.hidden_dim, 
                bias = False,
                num_layers = 1,
                batch_first=True)

        for key, par in self.named_parameters():
            par.requires_grad = False

    def reset(self):
        with torch.no_grad():
            # Set initial vars:
            self.zt = torch.zeros(self.batch_user, 1, self.item_dim)
            self.hidden_state = None
            self.t = -1
            return {
                't' : self.t,
                'click' : torch.zeros((self.batch_user,1),).long(),
                'reward' : torch.zeros((self.batch_user,))
                }

    def step(self, action, render=False):
        with torch.no_grad():
            # User samples click given action:
            action_vec = self.itemvec(action)
            score = (self.zt * action_vec).sum(-1)

            #score += (action == 1).float()*1
            click_idx = dist.Categorical(logits=score).sample()
            click = action.gather(-1, click_idx.unsqueeze(-1)).squeeze()

            # Update user state for next time step:
            click_vec = self.itemvec(click.unsqueeze(-1))
            self.zt, self.hidden_state = self.gru(click_vec, self.hidden_state)
            #self.hidden_state = self.zt
            #self.zt = 0.1*(self.hidden_state + click_vec)
            

            if render:
                plt.scatter(self.zt[0,:,0], self.zt[0,:,1],color = "green")
                plt.scatter(self.itemvec.weight[:,0], self.itemvec.weight[:,1])
            
            # Return info:
            self.t += 1
            done =  self.t >= (self.max_time-1)
            
            return {
                't' : self.t,
                'action' : action,
                'click_idx' : click_idx,
                'click' : click.long(),
                'reward' : int((click>1).float().sum()),
                'done' : done,
                'score' : score
            }

    def play_game(self, policy, dataset=None, render=False):
        with torch.no_grad():
            episode_data = {
                'action' : torch.zeros((self.batch_user, self.max_time, self.max_rho)).long(),
                'click' : torch.zeros((self.batch_user, self.max_time)).long(),
                'click_idx' : torch.zeros((self.batch_user, self.max_time)).long(),
                'score' : torch.zeros((self.batch_user, self.max_time, self.max_rho))
                }

            dat = self.reset()
            reward = 0
            for t in range(self.max_time):
                if t == 0:
                    user_history = episode_data['click'][:,:1]
                else:
                    user_history = episode_data['click'][:,:t]

                # Build recommendations from policy (and no click)
                rec = policy(user_history)[:,:(self.max_rho-1)]
                noclick = torch.ones((self.batch_user,1)).long()
                action = torch.cat((noclick,rec), dim = 1)
                
                dat = self.step(action, render=render)
                reward += dat['reward']/self.batch_user/self.max_time

                for key, val in episode_data.items():
                    episode_data[key][:,t] = dat.get(key)
                if dat['done']:
                    break

            if dataset is not None:
                dataset.push(episode_data)
            return reward
#################
### DATASET
#################

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
class SequentialData(Dataset):
    def __init__(self, capacity, max_time, max_rho, device = "cpu"):
        with torch.no_grad():
            self.capacity = capacity
            self.max_time = max_time
            self.max_rho = max_rho
            self.device = device

            self.cur_size = 0 # how big is dataset now
            self.pos = 0 # counter when adding

            self.data = {
                'action' : torch.zeros((self.capacity,    max_time, max_rho)).long().to(self.device),
                'click' : torch.zeros((self.capacity,     max_time)).long().to(self.device),
                'click_idx' : torch.zeros((self.capacity, max_time)).long().to(self.device),
                'score' : torch.zeros((self.capacity,    max_time, max_rho)).to(self.device)
            }

    def __len__(self):
        return min(self.cur_size, self.capacity)

    def __getitem__(self, idx):
        return {key : val[idx,] for key, val in self.data.items()}
    def push(self, ep_data):
        """ Saves a episode of batch of users to the dataset """
        with torch.no_grad():
            bs = len(ep_data['click'])
            start = self.pos
            end = (self.pos+bs)

            # If at end of batch, clip first steps of episode:
            if end >= self.capacity:
                avail = self.capacity-start
                for key, val in ep_data.items():
                    ep_data[key] = ep_data[key][-avail]
                end = self.capacity
            
            for key, val in self.data.items():
                self.data[key][start:end,] = ep_data[key].float().to(self.device)
            
            
            self.cur_size += bs
            self.pos = (self.pos + bs) % self.capacity

    def build_dataloader(self, batch_size = 16):
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True
            )
        return dataloader

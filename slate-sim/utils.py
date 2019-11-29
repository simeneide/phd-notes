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
    def __init__(self, item_dim, num_items, batch_user, max_time, hidden_dim = None):
        super(Simulator, self).__init__()
        with torch.no_grad():
            self.item_dim = item_dim
            self.max_time = max_time
            self.hidden_dim = item_dim if hidden_dim == None else hidden_dim
            self.num_items = num_items
            self.batch_users = batch_user

            # item model
            self.itemvec = nn.Embedding(num_embeddings=num_items, embedding_dim = item_dim)

            V =  torch.randn((num_items,item_dim))
            V = torch.nn.functional.normalize(V, p=2, dim = 1)
            V[1,:] = 0 # place no click in centre
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
            self.zt = torch.zeros(self.batch_users, 1, self.item_dim)
            self.hidden_state = None
            self.t = -1
            return {
                't' : self.t, 
                'click' : torch.zeros((self.batch_users,1),).long(),
                'reward' : torch.zeros((self.batch_users,))
                }

    def step(self, action, render=False):
        with torch.no_grad():
            # User samples click given action:
            action_vec = self.itemvec(action)
            score = (self.zt * action_vec).sum(-1)
            click_idx = dist.Categorical(logits=score).sample()
            click = action.gather(-1, click_idx.unsqueeze(-1)).squeeze()

            # Update user state for next time step:
            click_vec = self.itemvec(click.unsqueeze(-1))
            #self.zt, self.hidden_state = self.gru(click_vec, self.hidden_state)
            self.hidden_state = self.zt
            self.zt = self.hidden_state*0.5 + click_vec*0.5
            

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
                'click_idx' : torch.zeros((self.capacity, max_time)).long().to(self.device)
            }

    def __len__(self):
        return min(self.cur_size, self.capacity)

    def __getitem__(self, idx):
        return {key : val[idx,] for key, val in self.data.items()}
    def push(self, data):
        """ Saves a episode of batch of users to the dataset """
        with torch.no_grad():
            bs = len(data['click'])
            
            for key, val in self.data.items():
                self.data[key][self.pos:(self.pos+bs),] = data[key].float().to(self.device)
            
            
            self.cur_size += bs
            self.pos = (self.pos + bs) % self.capacity

    def build_dataloader(self, batch_size = 16):
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True
            )
        return dataloader

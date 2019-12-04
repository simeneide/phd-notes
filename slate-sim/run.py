#%%
import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample
import matplotlib.pyplot as plt
import pyro.distributions as dist
import utils

from pyro import optim
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import Predictive
from pyro.nn import PyroSample, PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, init_to_feasible

# DATA GENERATION
item_dim = 2
num_items = 10
max_time = 35
max_rho = 3
batch_user = 50
num_users = 10000
item_group = torch.zeros((num_items,)).long()
item_group[5:] = 1
# Rules
# item 0 is PADDING
# item 1 is no Click
#%%
env = utils.Simulator(
    item_dim=item_dim, 
    num_items =num_items, 
    batch_user = batch_user,
    max_rho = max_rho,
    max_time=max_time)

dataset = utils.SequentialData(capacity = num_users, max_time = max_time, max_rho = max_rho)
def set_noninform_prior(mod, scale = 1.0):
    for key, par in list(mod.named_parameters()):
        setattr(mod, key, PyroSample(dist.Normal(torch.zeros_like(par), scale*torch.ones_like(par)).independent() ))


#%%
# #%%
from pyro import poutine
pyro.enable_validation(True)
class GRU_MODEL(PyroModule):
    def __init__(self, item_dim, num_items, num_slates, item_group=torch.ones((num_items,)).long(), hidden_dim =None):
        super().__init__()
        self.item_dim = item_dim
        self.hidden_dim = item_dim if hidden_dim == None else hidden_dim
        self.num_items = num_items
        self.num_slates = num_slates
        self.item_group = item_group
        self.init_guide()

        #### item model

        # Group-vec:
        self.num_group = len(item_group.unique())
        self.groupvec = PyroModule[nn.Embedding](num_embeddings=self.num_group, embedding_dim = self.item_dim)
        self.groupvec.weight = PyroSample(dist.Normal(
            torch.zeros_like(self.groupvec.weight), 
            torch.ones_like(self.groupvec.weight)
            ).to_event(2))
        self.groupscale = PyroModule[nn.Embedding](num_embeddings=self.num_group, embedding_dim = self.item_dim)
        self.groupscale.weight = PyroSample(dist.Uniform(
            0.001*torch.ones_like(self.groupscale.weight), 
            torch.ones_like(self.groupscale.weight)
            ).to_event(2))
        
        # Item vec based on group vec hier prior:
        self.itemvec = PyroModule[nn.Embedding](num_embeddings=num_items, embedding_dim = item_dim)
        self.itemvec.weight = PyroSample( lambda x: dist.Normal(
            self.groupvec(self.item_group),
            self.groupscale(self.item_group)).to_event(2)
        )

        # user model
        self.gru = PyroModule[nn.GRU](
            input_size = self.item_dim,
            hidden_size = self.hidden_dim,
            bias = False,
            num_layers = 1,
            batch_first=True)

        set_noninform_prior(self.gru)

    def model(self, batch):
        user_click_ids = batch['click'][:,:-1]
        action_ids = batch['action'][:,1:]
        target_idx = batch['click_idx'][:,1:]

        ### SCORE CALC
        # Sample itemvecs:
        itemvec = self.itemvec(torch.arange(self.num_items))
        # User side:
        click_vecs = itemvec[user_click_ids]
        zt, ht = self.gru(click_vecs)
        action_vecs = itemvec[action_ids]

        scores = (zt.unsqueeze(2) * action_vecs).sum(-1)

        ### Flatten and compute likelihood:
        slatelen = scores.size()[-1]
        scores_flat = scores.reshape(-1, slatelen)
        target_idx_flat = target_idx.reshape(-1)

        with pyro.plate("data", size = self.num_slates, subsample = target_idx_flat):
            obsdistr = dist.Categorical(logits=scores_flat)
            pyro.sample("obs", obsdistr, obs=target_idx_flat)

        return scores
        
    def init_guide(self):
        self.guide = AutoDiagonalNormal(self.model) # , init_loc_fn = init_to_feasible
    def init_opt(self,lr = 0.01):
        adam = optim.Adam({"lr": lr})
        self.svi = SVI(self.model, self.guide, adam, loss=Trace_ELBO())

    def sample_guide_trace(self, batch=None):
        return poutine.trace(self.guide).get_trace(batch)

    def recommend(self, click_seq, t=-1):
        with poutine.replay(trace=self.sample_guide_trace()):
            itemvec = self.itemvec(torch.arange(self.num_items))
            user_click_vec = itemvec[click_seq]
            zt_all, _ = self.gru(user_click_vec)
            zt = zt_all[:,t,]

            score = (zt.unsqueeze(1) * itemvec.unsqueeze(0)).sum(-1)
            sort_score, sort_item =  score[:,2:].sort(dim=-1, descending=True)
            sort_item +=2
            return sort_item
            
class RandomPolicy:
    def __init__(self, num_items, max_rho, batch_user):
        super().__init__()
        self.num_items = num_items
        self.max_rho = max_rho
        self.batch_user = batch_user
    
    def recommend(self, *args, **kwargs):
        action = 2+torch.cat([torch.randperm(self.num_items-2).unsqueeze(0) for _ in range(batch_user)])
        #action[:,0] = 1
        return action

random_policy = RandomPolicy(num_items = num_items, max_rho= max_rho, batch_user = batch_user)


net = GRU_MODEL(item_dim=item_dim, num_items =num_items, num_slates = 1, item_group=item_group)

# Init guide (dataset is empty):
pyro.clear_param_store()
net.init_opt(lr = 0.01)
net.guide(dataset.data)
net.sample_guide_trace()
#%% Visuailze game
env.play_game(policy=net.recommend,render=True)
#%%
# Data collect step:
for n in range(1):
    with torch.no_grad():
        print(f"--- n = {n} ---")
        num_episodes = 100
        reward = torch.zeros((num_episodes,))
        for ep in range(num_episodes):
            reward[ep] = env.play_game(policy=net.recommend, dataset=dataset, render=False)
        print(f"Interacted with users for {num_episodes} episodes. Average reward: {reward.mean():.3f}")

    # Train step
    net.num_slates = len(dataset)*max_time
    dataloader = dataset.build_dataloader(batch_size=128)
    num_epochs = 100

    loss = torch.zeros((num_epochs,))

    for ep in range(num_epochs):
        for i, batch in enumerate(dataloader):
            # calculate the loss and take a gradient step
            loss[ep] += net.svi.step(batch)
        
    with torch.no_grad():
        predictive = Predictive(net.model, guide=net.guide, num_samples=80, return_sites=("itemvec.weight","_RETURN",))
        samples = predictive(batch)
        score_hat = samples['_RETURN']
        mae_score = (batch['score'][:,1:] - score_hat).abs().mean() / batch['score'][:,1:].abs().mean()
        acc = (score_hat.float().mean(0).argmax(-1) == batch['click_idx'][:,1:]).float().mean()
        print(f"epoch {ep}: loss {loss[ep]:.1f}, accuracy {acc:.2f}, mae scores {mae_score:.2f}")

# %% PREDICTION
predictive = Predictive(net.model, guide=net.guide, num_samples=200, return_sites=('groupscale.weight','groupvec.weight',"itemvec.weight","_RETURN",))
samples = predictive(batch)

V_hat = samples['itemvec.weight'].detach().squeeze()
V_mean = V_hat.mean(0)

for i in range(10):
    plt.scatter(V_hat[:,i,0], V_hat[:,i,1])
#%%
Vg_hat = samples['groupvec.weight'].detach().squeeze()

for i in range(2):
    plt.scatter(Vg_hat[:,i,0], Vg_hat[:,i,1])
#%%
sigma_g_hat = samples['groupscale.weight'].detach().squeeze()

for i in range(2):
    plt.scatter(sigma_g_hat[:,i,0], sigma_g_hat[:,i,1])

#%%
V_true = env.itemvec.weight.detach()
#V_true = torch.nn.functional.normalize(V_true, p=2, dim =-1)
plt.scatter(V_mean[:,0], V_mean[:,1], color = "red")
plt.scatter(V_true[:,0],V_true[:,1], color = "green")
#%%

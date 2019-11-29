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
from pyro.infer import Predictive
from pyro.nn import PyroSample, PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, init_to_feasible

# DATA GENERATION
item_dim = 2
num_items = 10
num_users = 4
batch_size = 90
max_time = 13
max_rho = 10
batch_user = 50
# Rules
# item 0 is PADDING
# item 1 is no Click
#%%


env = utils.Simulator(item_dim=item_dim, num_items =num_items, batch_user = batch_user, max_time=max_time)

dataset = utils.SequentialData(capacity = 10000, max_time = max_time, max_rho = max_rho)
#%%
#%%
class RandomPolicy:
    def __init__(self, num_items, max_rho, batch_user):
        super().__init__()
        self.num_items = num_items
        self.max_rho = max_rho
        self.batch_user = batch_user
    
    def recommend(self, *args, **kwargs):
        action = torch.cat([torch.randperm(self.num_items)[:max_rho].unsqueeze(0) for _ in range(batch_user)])
        action[:,0] = 1
        return action
policy = RandomPolicy(num_items = num_items, max_rho= max_rho, batch_user = batch_user)

#%% COLLECT DATA

num_episodes = 1

for ep in range(num_episodes):
    episode_data = {
    'action' : torch.zeros((batch_user, max_time, max_rho)),
    'click' : torch.zeros((batch_user, max_time)),
    'click_idx' : torch.zeros((batch_user, max_time)),
    }

    dat = env.reset()
    while True:
        action = policy.recommend(dat)
        dat = env.step(action, render=True)
        for key, val in episode_data.items():
            episode_data[key][:,dat['t']] = dat.get(key)
        if dat['done']:
            break
    dataset.push(episode_data)

len(dataset)

#%% TRAINING
dataloader = dataset.build_dataloader(batch_size=128)

# %%

class GRU_MODEL(PyroModule):
    def __init__(self, item_dim, num_items, num_slates, hidden_dim =None):
        super().__init__()
        self.item_dim = item_dim
        self.hidden_dim = item_dim if hidden_dim == None else hidden_dim
        self.num_items = num_items
        self.num_slates = num_slates

        # item model
        self.itemvec = PyroModule[nn.Embedding](num_embeddings=num_items, embedding_dim = item_dim)
        self.itemvec.weight = PyroSample(
            dist.Normal(
                torch.zeros_like(self.itemvec.weight), 
                torch.ones_like(self.itemvec.weight)
                ).to_event(2))

        # user model
        self.gru = PyroModule[nn.GRU](
            input_size = self.item_dim, 
            hidden_size = self.hidden_dim, 
            bias = False,
            num_layers = 1,
            batch_first=True)

        for key, par in list(self.gru.named_parameters()):
            setattr(self.gru, key, PyroSample(dist.Normal(torch.zeros_like(par), 0.01*torch.ones_like(par)).independent() ))
            

    def forward(self, batch):
        user_click_ids = batch['click'][:,:-1]
        action_ids = batch['action'][:,1:]
        

        # Sample itemvecs:
        itemvec = self.itemvec(torch.arange(self.num_items))
        #itemvec[1,:] = 0

        # User side:
        click_vecs = itemvec[user_click_ids]
        click_vecs.size()

        zt, ht = self.gru(click_vecs)
        zt.size()


        action_vecs = itemvec[action_ids]
        action_vecs.size()

        scores = (zt.unsqueeze(2) * action_vecs).sum(-1)
        return scores

    def model(self, batch):
        target_idx = batch['click_idx'][:,1:]
        scores = self.forward(batch)
        slatelen = scores.size()[-1]

        scores_flat = scores.reshape(-1, slatelen)
        scores_flat.size()
        target_idx_flat = target_idx.reshape(-1)

        target_idx_flat.size()

        with pyro.plate("data", size = self.num_slates, subsample = target_idx_flat):
            obsdistr = dist.Categorical(logits=scores_flat)
            pyro.sample("obs", obsdistr, obs=target_idx_flat)

        return scores

net = GRU_MODEL(item_dim=item_dim, num_items =num_items, num_slates = len(dataset)*max_rho)

# %%
pyro.enable_validation(True)
guide = AutoDiagonalNormal(net.model) # , init_loc_fn=init_to_feasible

#%%
pyro.clear_param_store()
adam = optim.Adam({"lr": 0.01})
svi = SVI(net.model, guide, adam, loss=Trace_ELBO())
num_epochs = 200

loss = torch.zeros((num_epochs,))

for ep in range(num_epochs):
    for i, batch in enumerate(dataloader):
        # calculate the loss and take a gradient step
        loss[ep] += svi.step(batch)
    
    if (ep %10) == 0:
        with torch.no_grad():
            predictive = Predictive(net.model, guide=guide, num_samples=80, return_sites=("itemvec.weight","_RETURN",))
            samples = predictive(batch)
            acc = (samples['_RETURN'].float().mean(0).argmax(-1) == batch['click_idx'][:,1:]).float().mean()
            print(f"epoch {ep}: loss {loss[ep]:.1f}, accuracy {acc:.2f}")


plt.plot(loss)
plt.yscale("log")
# %% QUANTILES OF PARAMETERS
#for key, val in guide.quantiles([0.25,0.5,0.75]).items():
#    print(f"{key}: \t {val[0].detach().numpy()}")

# %% PREDICTION

predictive = Predictive(net.model, guide=guide, num_samples=200, return_sites=("itemvec.weight","_RETURN",))
samples = predictive(batch)

V_hat = samples['itemvec.weight'].detach().squeeze()
V_mean = V_hat.mean(0)

for i in range(10):
    plt.scatter(V_hat[:,i,0], V_hat[:,i,1])

#%%
V_true = env.itemvec.weight.detach()
#V_true = torch.nn.functional.normalize(V_true, p=2, dim =-1)
plt.scatter(V_mean[:,0], V_mean[:,1], color = "red")
plt.scatter(V_true[:,0],V_true[:,1], color = "green")
#%%

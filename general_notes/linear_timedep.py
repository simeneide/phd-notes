#%%
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro import optim
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroSample, PyroModule, pyro_method
from pyro.infer.autoguide import AutoDiagonalNormal, init_to_sample
from pyro import poutine

#%% Some very simple data where each time step just increases by 1.
num_cl = 5
x = torch.cat((
    torch.arange(0,num_cl).unsqueeze(0),
    (torch.arange(0,num_cl).unsqueeze(0)+1) %5
))
x

# %%
class Model(PyroModule):
    def __init__(self, num_cl):
        super().__init__()
        self.num_cl = num_cl
        self.trans_score = PyroModule[nn.Embedding](num_embeddings=self.num_cl, embedding_dim = self.num_cl)
        self.trans_score.weight = PyroSample(
            dist.Uniform(
                -2.0*torch.ones_like(self.trans_score.weight), 
                2*torch.ones_like(self.trans_score.weight)
                ).to_event(2))
    
    @pyro_method
    def forward_normal(self, x):
        state_prev = x[:,:-1]
        state_next = x[:,1:]
        logits = self.trans_score(state_prev)
        for u in pyro.plate("data_loop", len(x)):
            for t in pyro.plate(f"time_loop_{u}", state_prev.size()[1]):
                obs = pyro.sample(
                    f"obs_{u}_{t}", 
                    dist.Categorical(logits=logits[u,t,:]), 
                    obs=state_next[u,t])
        return obs
    @pyro_method
    def forward_block_two_grads(self, x):
        state_prev = x[:,:-1]
        state_next = x[:,1:]
        time_mask = (state_prev != 1).float()
        scores = self.trans_score(state_prev)
        expscore = scores.exp()
        expscore[0,0,1] = 0#expscore[0,0,1]*0
        probs = expscore / expscore.sum(1, keepdim=True)
        #scores.size(), state_prev.size()
        
        
        #%%
        slatelen = scores.size()[-1]
        scores_flat = scores.reshape(-1, slatelen)
        target_idx_flat = state_next.reshape(-1)
        time_mask_flat = time_mask.reshape(-1)

        with pyro.plate("data_loop", size = len(target_idx_flat)):
            distr = dist.Categorical(logits=scores_flat).mask(time_mask_flat)
            obs = pyro.sample("obs", distr, obs=target_idx_flat)
        return obs

#%%
def train(model):
    pyro.clear_param_store()
    
    model(x)

    guide = AutoDiagonalNormal(model)
    guide(x)

    adam = optim.Adam({"lr": 0.1})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    for _ in range(1000):
        svi.step(x)

    import matplotlib.pyplot as plt
    med = guide.median()['trans_score.weight'].detach()
    plt.imshow(med)
    return med


#%% Normal forward with all gradients available to optimizer
model = Model(num_cl)
train(model.forward_normal)

#%% Forward when some gradients are blocked
model = Model(num_cl)
train(model.forward_block_two_grads)

# %%

#%%
import torch
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pyro
import logging
import simulator
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
import pyro
import pyro.distributions as dist
import models
import numpy as np
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')
import prepare
from prepare import SequentialDataset
import pickle
#%%
param = utils.load_param()
if param['device'] == "cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
# Overwrite param with whatever is in kwargs:
try:
    for key, val in kwargs.items():
        logging.info(f"Overwriting parameter {key} to {val}.")
        param[key] = val
except:
    logging.info("Did no overwrite of default param.")

if param.get('real_data'):
    logging.info("Loading real data")


    ind2val, itemattr, dataloaders = prepare.load_dataloaders(
            data_dir="data_real",
            data_type="lake-noclickrate-0.2",
            batch_size=1024,
            split_trainvalid=0.95,
            num_workers=0,
            override_candidate_sampler="actual")

    param['num_items'] = len(ind2val['itemId'])
    param['num_groups'] = len(np.unique(itemattr['category']))
    param['num_users'], param['maxlen_time'], param['maxlen_slate'] = dataloaders['train'].dataset.dataset.data['action'].size()
    dataloaders['train'].dataset.dataset.data['userId'] = torch.arange(0, param['num_users'])
else:
    #%% Place all items in a group:
    item_group = 1 + (torch.arange(param['num_items']) //
                (param['num_items'] / (param['num_groups']-1))).long()
    item_group[:3] = 0 # first three items are special group
    itemattr = {'category': item_group.cpu().numpy()}

    # %% TRAIN: MODEL+CALLBACKS+TRAINER
    pyro.clear_param_store()
    env = models.AR_Model(**param, item_group=torch.tensor(itemattr['category']))
    sim = simulator.Simulator(**param, env=env)
    ind2val, itemattr, dataloaders, sim = simulator.collect_simulated_data(
        sim, policy_epsilon=0.5, **param)

#%%
pyro.clear_param_store()
import pyrotrainer
dummybatch = next(iter(dataloaders['train']))
dummybatch['phase_mask'] = dummybatch['mask_train']
dummybatch = {key: val.long().to(param.get("device")) for key, val in dummybatch.items()}

if param['model_type'] == "rnn":
    model = models.RNN_Model(**param, item_group=torch.tensor(itemattr['category']).long())
elif param['model_type'] == "ar1":
    model = models.AR_Model(**param, item_group=torch.tensor(itemattr['category']).long())


#%%
from pyro.infer.mcmc import NUTS, MCMC
hmc_kernel = NUTS(model, jit_compile=True)
init_par = {key: val for key, val in model.par_real.items() if key not in ['softmax_mult',"h0"]}
init_par['h0-batch'] = model.par_real['h0']
mcmc = MCMC(hmc_kernel, num_samples=10000, warmup_steps=100, initial_params=init_par)
#%%
all_data = dataloaders['train'].dataset.data
all_data['phase_mask'] = all_data['mask_train']
#%% TRAIN
mcmc.run(all_data)
# %% SAVE TO FILE

par = mcmc.get_samples()
par = {key : val.detach().cpu() for key, val in par.items()}

with open("mcmc-cuda2.parameter", "wb") as file:
    pickle.dump(par, file=file)
#%% ANALYSIS
with open("mcmc.parameter", "rb") as file:
    par = pickle.load(file=file)
# %%
list(par.keys())

# %% Plot gamma over iteration steps:
plt.plot(par['gamma'])
#%%
plt.scatter(par['bias_noclick'][:,4], par['gamma'])
#%%
t0 = 5000
V = par['item_model.itemvec.weight']
for i in range(param['num_items']):
    plt.scatter(V[t0:,i,0], V[t0:,i,1])

#%% Variance of V samples
logging.info(f"mean( std(v_j)): {torch.std(V, dim = 0).mean()}")
logging.info(f"mean( std(h0^u)): {torch.std(par['h0-batch'], dim = 0).mean()}")
#plt.scatter(par['h0-batch'].mean(0))
# %%
t0 = 9000
Vg = par['item_model.groupvec.weight']

Vg.size()

for i in range(param['num_groups']):
    plt.scatter(Vg[t0:,i,0], Vg[t0:,i,1])
#%%
d_i = 0
d_j = 0
i = 2
j = 3
t0 = 100
plt.scatter(Vg[t0:, i, d_i], Vg[t0:,j, d_j])

#np.corcoef(Vg[:,i,d_])
# %%
h0 = par['h0-batch']

t0 = 0
for t0 in [0,2500, 5000, 7500]:
    for i in range(3):
        plt.scatter(h0[t0:(t0+1000),i,0], h0[t0:(t0+1000),i,1])
    plt.show()

# %%
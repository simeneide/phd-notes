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

logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')

param = utils.load_param()
item_group = (torch.arange(param['num_items']) //
              (param['num_items'] / param['num_groups'])).long()
itemattr = {'category': item_group.numpy()}
# %% TRAIN: MODEL+CALLBACKS+TRAINER
pyro.clear_param_store()
env = models.Model(**param, item_group=torch.tensor(itemattr['category']))
sim = simulator.Simulator(**param, env=env)
ind2val, itemattr, dataloaders, sim = simulator.collect_simulated_data(
    sim, policy_epsilon=0.5, **param)

#%%
pyro.clear_param_store()
import pyrotrainer
dummybatch = next(iter(dataloaders['train']))
model = models.Model(**param, item_group=torch.tensor(itemattr['category']))
guide = models.MeanFieldGuide(model=env, batch=dummybatch, **param)
#%%
from pyro import poutine
tr = poutine.trace(
    model).get_trace(batch=dummybatch)

guide_tr = poutine.trace(
    guide).get_trace(batch=dummybatch)  
for node, obj in tr.iter_stochastic_nodes():
    if model.par_real.get(node) is None:
        print(node, "\t", obj['value'].size())

#%%
"""
import torch.distributions.constraints as constraints
pyro.clear_param_store()
for key, val in model.par_real.items():
    pyro.param(f"{key}-mean", val)
    pyro.param(f"{key}-scale", 0.00001+torch.zeros_like(val), constraint=constraints.interval(0,0.1))
    print(key)
"""
#%%
trainer = pyrotrainer.RecTrainer(model=model,
                                 guide=guide,
                                 max_epoch=1000,
                                 name=param['name'],
                                 param=param,
                                 patience=param['patience'],
                                 learning_rate=param['learning_rate'])
#%%
trainer.fit(dataloaders)


# %%

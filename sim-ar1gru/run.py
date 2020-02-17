#%%
import torch
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pyro
import logging
import simulator
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer, logging
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
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
#env.visualize_item_space()

#%%
sim = simulator.Simulator(**param, env=env)
#%%
ind2val, itemattr, dataloaders, sim = simulator.collect_simulated_data(
    sim, policy_epsilon=0.5, **param)

#%%
import pyrotrainer
dummybatch = next(iter(dataloaders['train']))
model = models.Model(**param, item_group=torch.tensor(itemattr['category']))
guide = models.MeanFieldGuide(model=env, batch=dummybatch, **param)
#%%
self = model
batch = dummybatch

#%%

trainer = pyrotrainer.RecTrainer(model=model,
                                 guide=guide,
                                 max_epoch=5000,
                                 name=param['name'],
                                 param=param,
                                 patience=param['patience'],
                                 learning_rate=param['learning_rate'])
trainer.fit(dataloaders)
# %%

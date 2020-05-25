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

# %%
param = utils.load_param()
    
# Overwrite param with whatever is in kwargs:
try:
    for key, va.l in kwargs.items():
        logging.info(f"Overwriting parameter {key} to {val}.")
        param[key] = val
except:
    logging.info("Did no overwrite of default param.")

if param['device'] == "cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
if param.get('real_data'):
    logging.info("Loading real data")

    ind2val, itemattr, dataloaders = prepare.load_dataloaders(
            data_dir="data_real",
            data_type="lake-noclickrate-0.2",
            batch_size=param['batch_size'],
            split_trainvalid=param['split_trainvalid'],
            num_workers=0,
            override_candidate_sampler="actual",
            t_testsplit = param['t_testsplit'])

    param['num_items'] = len(ind2val['itemId'])
    param['num_groups'] = len(np.unique(itemattr['category']))
    param['num_users'], param['maxlen_time'], _ = dataloaders['train'].dataset.data['action'].size()
    param['num_users'] = param['num_users']+1
    param['num_displayTypes'] = 3
else:
    #%% Place all items in a group:
    item_group = 1 + (torch.arange(param['num_items']) //
                (param['num_items'] / (param['num_groups']-1))).long()
    item_group[:3] = 0 # first three items are special group
    itemattr = {'category': item_group.cpu().numpy()}

    # %% TRAIN: MODEL+CALLBACKS+TRAINER
    pyro.clear_param_store()
    env = models.PyroRecommender(**param, item_group=torch.tensor(itemattr['category']))
    env.init_set_of_real_parameters()
    sim = simulator.Simulator(**param, env=env)
    ind2val, itemattr, dataloaders, sim = simulator.collect_simulated_data(
        sim, policy_epsilon=param['collect_data_randompct'], **param)

#%%
pyro.clear_param_store()
pyro.get_param_store().load("checkpoints/Irene-Folger, user_model:gru, item_model:pretrained, dist:dot, user_init:False, guide_maxscale:1.0, hidden_dim:100.pyro")

#%%
torch.manual_seed(param['train_seed'])
import pyrotrainer
dummybatch = next(iter(dataloaders['train']))
dummybatch['phase_mask'] = dummybatch['mask_train']
dummybatch = {key: val.long().to(param.get("device")) for key, val in dummybatch.items()}

model = models.PyroRecommender(**param, item_group=torch.tensor(itemattr['category']).long())
guide = models.MeanFieldGuide(model=model, batch=dummybatch, **param)


trainer = pyrotrainer.PyroTrainer(
    model, 
    guide, 
    dataloaders, 
    **param)

#%% ADD EMBEDDINGS
trainer.step=2
emb = pyrotrainer.VisualizeEmbeddings(ind2val=ind2val)
emb(trainer)
# %%
V = pyro.param("item_model.itemvec.weight-mean").detach().cpu()#.numpy()

V.min(), V.max()
V[:10,:10].numpy().round(1)

#%%
Vg = pyro.param("item_model.groupvec.weight-mean").detach().cpu().numpy()
#%%
N  = 10000
h0 = pyro.param("h0-mean").detach().cpu()[:N]  
plt.scatter(h0[:N,0], h0[:N,1], alpha = 0.1)

# %%
idx = 30
dataloaders['train'].dataset.data
smallbatch = {
    'click' : dummybatch['click'][idx,:10].unsqueeze(0).to(model.device), 
    'userId' : dummybatch['userId'][idx].unsqueeze(0).to(model.device),
    'displayType' : dummybatch['displayType'][idx,:10].unsqueeze(0).to(model.device)}
recs = model.recommend(smallbatch, par=guide, num_rec=5)
#plt.imshow(plot_itemidx_array(recs[:num_time]).permute(1,2,0))
import FINNPlot

views = smallbatch['click'].flatten()
num_recs=5
num_time=7
M = torch.zeros(num_recs+1, num_time)
M[0,:] = views[:num_time]
for t_rec in range(num_time):
    M[1:,t_rec] = model.recommend(smallbatch, par=lambda *args, **kwargs: guide(temp=0.0, *args, **kwargs), num_rec=num_recs, t_rec=t_rec)

def plot_itemidx_array(arr,nrow=None):
    if nrow is None:
        nrow = arr.size()[1]
    finnkoder = [ind2val['itemId'][r.item()] for r in arr.flatten()]
    return FINNPlot.add_image_line(finnkoder, nrow=nrow)

plt.figure(figsize=(30,30))
plt.imshow(plot_itemidx_array(M).permute(1,2,0))

#%%
t_rec = 10
for t_rec in range(1,10):
    score, h = model.predict_cond(smallbatch, par=lambda *args, **kwargs: guide(temp=0.0, *args, **kwargs),t_rec=t_rec)
    print(t_rec, score.mean(), float(h[:,t_rec,:].abs().mean()), float(h[:,t_rec,:].abs().max()))

    #_ = plt.hist(score.detach().cpu(),bins=100)
    plt.hist(h[:,t_rec,].detach().cpu().flatten(), bins=30)

#plt.xlim(-3,0)
#%%
#%%
score.max()
# %%
plt.hist(V.flatten())
# %%


idx, cnts = dataloaders['train'].dataset.data['click'].unique(return_counts=True)
most_clicked_items = cnts.argsort()[-20:]
plt.imshow(plot_itemidx_array(most_clicked_items,nrow=10).permute(1,2,0))

# %%

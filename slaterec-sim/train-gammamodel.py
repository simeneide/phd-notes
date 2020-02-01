#%%
import torch
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pyro
import logging
import model_gamma
import simulator

logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')

param = utils.load_param()
item_group = (torch.arange(param['num_items']) //
                    (param['num_items'] / param['num_groups'])).long()
itemattr = {'category': item_group.numpy()}

#%%
#self = model_gamma.Model(**param, item_group = torch.tensor(itemattr['category']))

#%%

pyro.clear_param_store()


# VISUALIZE A BATCH OF PATHS
def visualize_batch(batch):
    sim.dataset.build_dataloaders()
    batch = next(iter(dataloaders['train']))
    res = env.likelihood(batch)
    res['score']
    V = res['V']
    zt = res['zt']
    sns.scatterplot(V[:, 0].cpu(),
                    V[:, 1].cpu())

    for k in range(len(zt)):
        sns.lineplot(zt[k,:,0], zt[k,:,1])
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

#%% TRAIN
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer, logging
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from torch.nn import functional as F
import pyro
import pyro.distributions as dist
class PyroOptWrap(pyro.infer.SVI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def state_dict(self,):
        return {}

    def load_state_dict(self, *args, **kwargs):
        logging.info("Loading empty state dict")
        return {}

class PyroCoolSystem(pl.LightningModule):
    def __init__(self, param, model, guide):
        super(PyroCoolSystem, self).__init__()
        
        self.model = model
        self.guide = guide

    def forward(self, batch):
        self.model.likelihood(batch)
        return yhat

    def training_step(self, batch, batch_idx):
        
        loss = self.svi.step(batch)
        loss = torch.tensor(loss).requires_grad_(True)
        tensorboard_logs = {}
        tensorboard_logs['running/loss'] =  loss
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self.svi.evaluate_loss(batch)
        loss = torch.tensor(loss).requires_grad_(True)
        res = self.model(batch, mode = "likelihood")

        output = self.calc_scorediff(batch)
        output['loss'] = loss
        tensorboard_logs = {}
        for key,val in output.items():
            tensorboard_logs[f"outputs/{key}"] = val.mean()

        tensorboard_logs['val_loss'] = loss
        output['log'] = tensorboard_logs
        return output

    def training_end(self, outputs):
        avg_loss = outputs['loss']
        tensorboard_logs = {'train/loss': avg_loss}
        return {'loss': avg_loss, 'log': tensorboard_logs}

    def validation_end(self, outputs):
        # OPTIONAL
        
        tensorboard_logs = {}
        keys = outputs[0]['log'].keys()
        for key in keys:
            avg = torch.stack([x['log'][key] for x in outputs]).mean()
            tensorboard_logs[f'{key}'] = avg
        
        #logging.info(outputs[0]['log'])
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs['valid/loss'] =  avg_loss
        for samplename, vals in system.guide.get_parameters().items():
            for p, val in vals.items():
                tensorboard_logs[f"param/{samplename}-{p}-l1"] = val.abs().mean()


        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        self.svi = PyroOptWrap(model=self.model,
                guide=self.guide,
                optim=pyro.optim.Adam({"lr": 1e-2}),
                loss=pyro.infer.Trace_ELBO())

        return [self.svi]
        
    @pl.data_loader
    def train_dataloader(self):
        return dataloaders['train']

    @pl.data_loader
    def val_dataloader(self):
        return dataloaders['valid']

    def optimizer_step(self, *args, **kwargs):
        pass
    def backward(self, *args, **kwargs):
        pass
    def calc_scorediff(self, batch):
        
        res_hat = pyro.condition(
                    lambda batch: self.model.forward(batch),
                    data=system.guide(batch))(batch)

        res = self.model.likelihood(batch)
        
        # general mean values:
        stats = {f"outputs/{key}-L1" : val.abs().mean() for key,val in res_hat.items()}
        
        score2prob = lambda s: (s.exp()/s.exp().sum(2,keepdims=True))
        stats['outputs/score-mae'] = (score2prob(res['score'])-score2prob(res_hat['score'])).abs().mean()
        return stats
# %% TRAIN: MODEL+CALLBACKS+TRAINER
#for num_users in [100,1000]:
pyro.clear_param_store()
#param['num_users'] = num_users

# Generate data
env = model_gamma.Model(**param, item_group = torch.tensor(itemattr['category']))
train_sim = simulator.Simulator(**param, env = env)
ind2val, itemattr, dataloaders, sim = simulator.collect_simulated_data(env, train_sim , policy_epsilon=0.5, **param)
#%%
dummybatch = next(iter(dataloaders['train']))
model = model_gamma.Model(**param, item_group = torch.tensor(itemattr['category']))
guide = model_gamma.MeanFieldGuide(model = env, batch = dummybatch, **param)

#%% EXPERIMENT PARAMETERS
print(param)
#%%
from pytorch_lightning import Trainer
import pytorch_lightning
system = PyroCoolSystem(param=param, model=model, guide=guide)
early_stopping = pytorch_lightning.callbacks.EarlyStopping(monitor="loss", patience=10)
from pytorch_lightning.logging import TensorBoardLogger
# lightning uses tensorboard by default
tb_logger = TensorBoardLogger("test_gamma/", name=f"gammamodel_num_user={param.get('num_users')}-num_time={param.get('maxlen_time')}-num_items={param.get('num_items')}")
# most basic trainer, uses good defaults
trainer = Trainer(early_stop_callback=early_stopping,logger=tb_logger)
trainer.fit(system)
#%%

# %% PLOT OF H0 parameters of users
h0 = system.guide.get_parameters()['h0-batch']['mean'].detach()
fig = plt.figure()
plt.scatter(h0[:,0], h0[:,1])
tb_logger.experiment.add_figure('h0', fig, 0)
#%%

# %% PLOT OF item vector parameters
h0 = system.guide.get_parameters()['item_model.itemvec.weight']['mean'].detach()
fig = plt.figure()
plt.scatter(h0[:,0], h0[:,1])
tb_logger.experiment.add_figure('V', fig, 0)

# %% # EVALUATE PERFORMANCE IN SIMULATOR wrt REWARD:
sim = simulator.Simulator(**param, env = env)
g = lambda *args, **kwargs: system.guide(temp = 0.0, *args, **kwargs)
rec = lambda *args, **kwargs: system.model.recommend(par=g, *args, **kwargs)
reward_est = sim.play_many_games(rec, 20).mean()
tb_logger.log_metrics({"reward" : reward_est}, step=1)

# %% OPTIMAL
sim = simulator.Simulator(**param, env = env)
rec = lambda *args, **kwargs: system.model.recommend(par="real", *args, **kwargs)
sim.play_many_games(rec, 20).mean()

#%% # RANDOM
kwargs = param
randomagent = simulator.RandomSystem(num_items=kwargs['num_items'],
                        maxlen_slate=kwargs['maxlen_slate'],
                        batch_user=kwargs['batch_user'])
sim = simulator.Simulator(**param, env = env)
#rec = lambda *args, **kwargs: system.model.recommend(par="real", *args, **kwargs)
sim.play_many_games(randomagent.recommend, 20).mean()

# %%

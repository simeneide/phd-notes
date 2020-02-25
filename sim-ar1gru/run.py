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

def main(**kwargs):
    param = utils.load_param()

    # Overwrite param with whatever is in kwargs:
    try:
        for key, val in kwargs.items():
            logging.info(f"Overwriting parameter {key} to {val}.")
            param[key] = val
    except:
        logging.info("Did no overwrite of default param.")

    #%% Place all items in a group:
    item_group = 1 + (torch.arange(param['num_items']) //
                (param['num_items'] / (param['num_groups']-1))).long()
    item_group[:3] = 0 # first three items are special group
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

    #%% START WITH TRUE PARAMETERS IF THIS IS TRUE:
    if param.get("start_true"):
        logging.info(f"Starting in true mean parameters...:")
        pyro.clear_param_store()
        for key, val in model.par_real.items():
            pyro.param(f"{key}-mean", val)
            print(key)

    #%%
    test_sim = simulator.Simulator(**param, env=env)

    trainer = pyrotrainer.RecTrainer(model=model,
                                    guide=guide,
                                    dataloaders = dataloaders,
                                    max_epoch=1000,
                                    name=param['name'],
                                    param=param,
                                    patience=param['patience'],
                                    learning_rate=param['learning_rate'],
                                    sim=test_sim)


    #%%
    trainer.fit()

if __name__ == "__main__":
    main()

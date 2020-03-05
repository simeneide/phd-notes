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
#%%
def main(**kwargs):
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
        
    guide = models.MeanFieldGuide(model=model, batch=dummybatch, **param)

    #%% START WITH TRUE PARAMETERS IF THIS IS TRUE:
    if param.get("start_true"):
        logging.info(f"Starting in true mean parameters...:")
        pyro.clear_param_store()
        for key, val in model.par_real.items():
            pyro.param(f"{key}-mean", val)
            print(key)

    #%%
    if not param['real_data']:
        test_sim = simulator.Simulator(**param, env=env)
    else:
        test_sim=None

    trainer = pyrotrainer.RecTrainer(model=model,
                                    guide=guide,
                                    dataloaders = dataloaders,
                                    max_epoch=1000,
                                    name=param['name'],
                                    param=param,
                                    patience=param['patience'],
                                    learning_rate=param['learning_rate'],
                                    sim=test_sim,
                                    device = param.get("device"))


    #%%
    trainer.fit()

if __name__ == "__main__":
    main()

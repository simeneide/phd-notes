#%%
import setGPU
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
        
    # Overwrite param with whatever is in kwargs:
    try:
        for key, val in kwargs.items():
            logging.info(f"Overwriting parameter {key} to {val}.")
            param[key] = val
    except:
        logging.info("ERROR: Did no overwrite of default param.")

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

    else:
        sim_param = utils.load_sim_param()
        #%% Place all items in a group:
        item_group = 1 + (torch.arange(sim_param['num_items']) //
                    (sim_param['num_items'] / (sim_param['num_groups']-1))).long()
        item_group[:3] = 0 # first three items are special group
        itemattr = {'category': item_group.cpu().numpy()}

        # %% TRAIN: MODEL+CALLBACKS+TRAINER
        pyro.clear_param_store()
        env = models.PyroRecommender(**sim_param, item_group=torch.tensor(itemattr['category']))
        env.init_set_of_real_parameters()
        sim = simulator.Simulator(**sim_param, env=env)
        ind2val, itemattr, dataloaders, sim = simulator.collect_simulated_data(
            sim, policy_epsilon=sim_param['collect_data_randompct'], **sim_param)

    param['num_items'] = len(ind2val['itemId'])
    param['num_groups'] = len(np.unique(itemattr['category']))
    param['num_users'], param['maxlen_time'], _ = dataloaders['train'].dataset.data['action'].size()
    param['num_users'] = param['num_users']+1
    #param['num_displayTypes'] = 3
    # Move data to device
    #for key, val in dataloaders['train'].dataset.data.items():
    #    dataloaders['train'].dataset.data[key] = val.to(param['device'])
    #%%
    pyro.clear_param_store()
    pyro.validation_enabled(True)
    torch.manual_seed(param['train_seed'])
    import pyrotrainer
    dummybatch = next(iter(dataloaders['train']))
    dummybatch['phase_mask'] = dummybatch['mask_train']
    dummybatch = {key: val.long().to(param.get("device")) for key, val in dummybatch.items()}

    model = models.PyroRecommender(**param, item_group=torch.tensor(itemattr['category']).long())
    guide = models.MeanFieldGuide(model=model, batch=dummybatch, **param)

    #%% START WITH TRUE PARAMETERS IF THIS IS TRUE:
    if param.get("start_true"):
        logging.info(f"Starting in true mean parameters...:")
        pyro.clear_param_store()
        for key, val in env.par_real.items():
            pyro.param(f"{key}-mean", val)
            pyro.param(f"{key}-scale", torch.zeros_like(val)+ 1e-5)
            print(key)

    #%% Define callbacks:

    # Common callbacks:
    optim = pyrotrainer.SviStep(model=model, guide=guide, **param)

    step_callbacks = [optim, pyrotrainer.calc_batch_stats]

    phase_end_callbacks = [
        pyrotrainer.report_phase_end, 
        pyrotrainer.ReportPyroParameters(), 
        pyrotrainer.EarlyStoppingAndCheckpoint(stopping_criteria=param['stopping_criteria'], patience=param['patience'], name =param['name'])
        ]

    after_training_callbacks = []

    if param['real_data']:
        plot_finn_ads = pyrotrainer.PlotFinnAdsRecommended(ind2val, epoch_interval=3)
        phase_end_callbacks.append(plot_finn_ads)
        after_training_callbacks.append(pyrotrainer.VisualizeEmbeddings())
    else:        
        test_sim = simulator.Simulator(**param, env=env)
        step_callbacks.append(pyrotrainer.Simulator_batch_stats(test_sim))
        after_training_callbacks.append(pyrotrainer.VisualizeEmbeddings(sim=test_sim))
        after_training_callbacks.append(pyrotrainer.RewardComputation(param, test_sim))
    
    after_training_callbacks.append(pyrotrainer.ReportHparam(param))
    #%%
    trainer = pyrotrainer.PyroTrainer(
        model, 
        guide, 
        dataloaders, 
        before_training_callbacks = [pyrotrainer.checksum_data],
        after_training_callbacks = after_training_callbacks,
        step_callbacks = step_callbacks, 
        phase_end_callbacks = phase_end_callbacks,
        max_epoch=param['max_epochs'],
        **param)
    return param, ind2val, trainer

if __name__ == "__main__":
    param, ind2val, trainer = main()
    trainer.fit()


# %%

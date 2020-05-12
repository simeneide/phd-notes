#%%
import run
import utils
from joblib import Parallel, delayed
import names
default_param = utils.load_param()
"""
jobs = []
for dist in ['l2']:
    for maxlen_slate in [4]:

        optimal_par = {
            'model_type' : 'ar1',
            'start_true' : True,
            'learning_rate' : 0,
            'dist' : dist,
            'maxlen_slate' : maxlen_slate,
            'name' : f'optimal-{dist}-slatelen_{maxlen_slate}'
        }
        jobs.append(delayed(run.main)(**optimal_par))


        random_par = {
            'model_type' : 'ar1',
            'start_true' : False,
            'learning_rate' : 0,
            'dist' : dist,
            'maxlen_slate' : maxlen_slate,
            'name' : f'random-{dist}-slatelen_{maxlen_slate}'
        }
        jobs.append(delayed(run.main)(**random_par))

        for train_seed in range(1):
            for model_type in ['ar1', 'adalinear']: #  #,,, 'rnn'
                for guide_userinit in [True, False]: # , False  False
                    for guide_maxscale in [0.1, 0.2]:
                        update_pars = {
                            'model_type' : model_type,
                            'guide_userinit' : guide_userinit,
                            'train_seed' : train_seed,
                            'dist' : dist,
                            'maxlen_slate' : maxlen_slate,
                            'guide_maxscale' : guide_maxscale
                            }
                        update_pars['name'] = f"{names.get_full_name().replace(' ','-')}, COLLECT_RANDOM, " + ", ".join([f"{key}:{val}" for key, val in update_pars.items()])
                        jobs.append(delayed(run.main)(**update_pars))
"""
# %%

#%%
parameter_sets = {
    'model_type' : ["linear","gru","markov"],
    'user_init' : [True, False],
    'item_dim' : [5, 10]
}

# %%
from itertools import product
configs = [dict(zip(parameter_sets, v)) for v in product(*parameter_sets.values())]

jobs = []
for update_par in configs:
    update_par['name'] = f"{names.get_full_name().replace(' ','-')}, " + ", ".join([f"{key}:{val}" for key, val in update_par.items()])
    jobs.append(delayed(run.main)(**update_par))

# %%
Parallel(n_jobs=6)(jobs)

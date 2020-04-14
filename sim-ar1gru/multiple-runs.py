#%%
import run
import utils
from joblib import Parallel, delayed
import names
default_param = utils.load_param()
jobs = []
for dist in ['l2','dot']:
    optimal_par = {
        'model_type' : 'ar1',
        'start_true' : True,
        'learning_rate' : 0,
        'dist' : dist,
        'name' : f'optimal-{dist}'
    }
    #jobs.append(delayed(run.main)(**optimal_par))


    random_par = {
        'model_type' : 'ar1',
        'start_true' : False,
        'learning_rate' : 0,
        'dist' : dist,
        'name' : f'random-{dist}'
    }
    #jobs.append(delayed(run.main)(**random_par))

    # %% 
    for train_seed in range(10,20):
        for model_type in ['ar1', 'rnn']: #  #,
            for guide_userinit in [True, False]: # , False  False
                for maxlen_time in [20]: 
                    update_pars = {
                        'model_type' : model_type,
                        'guide_userinit' : guide_userinit,
                        'maxlen_time' : maxlen_time,
                        'train_seed' : train_seed,
                        'dist' : dist
                        }
                    update_pars['name'] = f"{names.get_full_name().replace(' ','-')}_" + ", ".join([f"{key}:{val}" for key, val in update_pars.items()])
                    jobs.append(delayed(run.main)(**update_pars))

# %%
Parallel(n_jobs=6)(jobs)

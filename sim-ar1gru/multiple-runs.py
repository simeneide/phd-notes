#%%
import run
import utils
from joblib import Parallel, delayed
import names
default_param = utils.load_param()
jobs = []
for dist in ['l2']:
    for maxlen_slate in [2, 5, 10,15, 20]:

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

        # %% 
        for train_seed in range(4):
            for model_type in ['ar1']: #  #,,, 'rnn'
                for guide_userinit in [True, False]: # , False  False
                        update_pars = {
                            'model_type' : model_type,
                            'guide_userinit' : guide_userinit,
                            'train_seed' : train_seed,
                            'dist' : dist,
                            'maxlen_slate' : maxlen_slate,
                            }
                        update_pars['name'] = f"{names.get_full_name().replace(' ','-')}_" + ", ".join([f"{key}:{val}" for key, val in update_pars.items()])
                        jobs.append(delayed(run.main)(**update_pars))

# %%
Parallel(n_jobs=8)(jobs)

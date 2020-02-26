#%%
import run
import utils
from joblib import Parallel, delayed
import names
default_param = utils.load_param()
#%%
#run.main(lr=0.1)

# %%
# %%
jobs = []
for num_users in [0.2e3, 0.5e3, 1e3, 10e3]:
    for lr in [0.01, 0.005]:
        for priorscale in [0.5, 1.0]:
            update_pars = {
                'num_users' : int(num_users),
                'lr' : lr,
                'prior_userinit_scale' : priorscale,
                'prior_groupscale_scale' : priorscale,
                'prior_groupvec_scale' : priorscale
                }
            update_pars['name'] = f"{names.get_full_name().replace(' ','-')}_" + ", ".join([f"{key}:{val}" for key, val in update_pars.items()])
            jobs.append(delayed(run.main)(**update_pars))

# %%
Parallel(n_jobs=9)(jobs)

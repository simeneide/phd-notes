#%%
import run
import utils
from joblib import Parallel, delayed
default_param = utils.load_param()
#%%
#run.main(lr=0.1)

# %%

# %%
jobs = []
for num_users in [1e3,5e3,10e3]:
    update_pars = {
        'num_users' : int(num_users)
        }
    update_pars['name'] = str(update_pars)
    jobs.append(delayed(run.main)(**update_pars))

# %%
Parallel(n_jobs=4)(jobs)

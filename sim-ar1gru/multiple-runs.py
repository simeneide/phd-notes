#%%
import train
import utils
from joblib import Parallel, delayed
import names

param = utils.load_param()

jobs = []
if param['real_data'] is False:
    optimal_par = {
        'model_type' : 'linear',
        'start_true' : True,
        'learning_rate' : 0,
        'name' : f'optimal'
    }
    jobs.append(delayed(train.main)(**optimal_par))


    random_par = {
        'model_type' : 'linear',
        'start_true' : False,
        'learning_rate' : 0,
        'name' : f'random'
    }
    jobs.append(delayed(train.main)(**random_par))

# %%

#%%
parameter_sets = {
    'user_model' : ["linear","gru","markov"],
    'user_init' : [True, False],
    'item_dim' : [4]
}

# %%
from itertools import product
configs = [dict(zip(parameter_sets, v)) for v in product(*parameter_sets.values())]

for update_par in configs:
    update_par['name'] = f"{names.get_full_name().replace(' ','-')}, " + ", ".join([f"{key}:{val}" for key, val in update_par.items()])
    jobs.append(delayed(train.main)(**update_par))

# %%
Parallel(n_jobs=6)(jobs)

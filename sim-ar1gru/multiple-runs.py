#%%
import run
import utils
from joblib import Parallel, delayed
import names


jobs = []

optimal_par = {
    'model_type' : 'linear',
    'start_true' : True,
    'learning_rate' : 0,
    'name' : f'optimal'
}
jobs.append(delayed(run.main)(**optimal_par))


random_par = {
    'model_type' : 'linear',
    'start_true' : False,
    'learning_rate' : 0,
    'name' : f'random'
}
jobs.append(delayed(run.main)(**random_par))

# %%

#%%
parameter_sets = {
    'model_type' : ["linear","gru","markov"],
    'user_init' : [True, False],
    'item_dim' : [5]
}

# %%
from itertools import product
configs = [dict(zip(parameter_sets, v)) for v in product(*parameter_sets.values())]

for update_par in configs:
    update_par['name'] = f"{names.get_full_name().replace(' ','-')}, " + ", ".join([f"{key}:{val}" for key, val in update_par.items()])
    jobs.append(delayed(run.main)(**update_par))

# %%
Parallel(n_jobs=6)(jobs)

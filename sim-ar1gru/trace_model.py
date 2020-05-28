#%%
from train import *
def dict2device(d, device="cpu"): 
    return {key: val.to(device) for key, val in d.items()}

param, ind2val, trainer = main(device="cpu")
model = trainer.model
guide = trainer.guide

dummybatch = next(iter(trainer.dataloaders['train']))
dummybatch['phase_mask'] = dummybatch['mask_train']
dummybatch = {key: val.long().to(param.get("device")) for key, val in dummybatch.items()}
batch = dict2device(next(iter(models.dict_chunker(dummybatch, 1))), device="cpu")

# %%
param['name'] = "Asia-Madina, user_model:gru, item_model:pretrained, dist:l2, user_init:False, guide_maxscale:1.0, hidden_dim:100, clip_norm:10, num_particles:2, init_dim:2, h0_negative:true"
pyro.get_param_store().load(
    f"checkpoints/{param['name']}.pyro", 
    map_location=param['device']
    )

model.prepare_model_for_production()

import PYTORCHHelper
with torch.no_grad():
    sampled_par = dict2device(guide(batch, temp=1.0), "cpu")

    with torch.no_grad(), pyro.condition(data=sampled_par):
        item2ind = {item:idx for idx, item in ind2val["itemId"].items()}
        PYTORCHHelper.save_pt_model(model=model, dummy_tensor=batch['click'], it2ind_lookup=item2ind)


# %%

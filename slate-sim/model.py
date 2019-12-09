import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro import optim
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroSample, PyroModule
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro import poutine

def set_noninform_prior(mod, scale = 1.0):
    for key, par in list(mod.named_parameters()):
        setattr(mod, key, PyroSample(dist.Normal(torch.zeros_like(par), scale*torch.ones_like(par)).independent() ))

class GruModel(PyroModule):
    def __init__(self, item_dim, num_items, num_slates, item_group=None, hidden_dim =None, device = "cpu"):
        super().__init__()
        self.item_dim = item_dim
        self.hidden_dim = item_dim if hidden_dim == None else hidden_dim
        self.num_items = num_items
        self.num_slates = num_slates
        self.item_group = item_group
        self.device = device

        #### item model

        # Group-vec:
        if item_group is None:
            item_group = torch.zeros((self.num_items,)).long()
        self.num_group = len(item_group.unique())
        self.groupvec = PyroModule[nn.Embedding](num_embeddings=self.num_group, embedding_dim = self.item_dim)
        self.groupvec.weight = PyroSample(dist.Normal(
            torch.zeros_like(self.groupvec.weight), 
            torch.ones_like(self.groupvec.weight)
            ).to_event(2))
        self.groupscale = PyroModule[nn.Embedding](num_embeddings=self.num_group, embedding_dim = self.item_dim)
        self.groupscale.weight = PyroSample(dist.Uniform(
            0.01*torch.ones_like(self.groupscale.weight), 
            torch.ones_like(self.groupscale.weight)
            ).to_event(2))
        
        # Item vec based on group vec hier prior:
        self.itemvec = PyroModule[nn.Embedding](num_embeddings=num_items, embedding_dim = item_dim)
        self.itemvec.weight = PyroSample( lambda x: dist.Normal(
            self.groupvec(self.item_group),
            self.groupscale(self.item_group)).to_event(2)
        )

        # user model
        self.gru = PyroModule[nn.GRU](
            input_size = self.item_dim,
            hidden_size = self.hidden_dim,
            bias = False,
            num_layers = 1,
            batch_first=True)
        self.gru.flatten_parameters()
        set_noninform_prior(self.gru)
        
        # Initial user state
        self.z0 = PyroSample(
            dist.Normal(torch.zeros((self.item_dim,)), torch.ones((self.item_dim,))).to_event(1)
            )

    def model(self, batch, forecast=False):
        user_click_ids = batch['click'][:,:-1]
        # initialize user click with PAD item 0:
        #init_user_click = (torch.ones((len(user_click_ids),1))*0).long()
        #user_click_ids = torch.cat((init_user_click, user_click_ids), dim=1)

        action_ids = batch['candidates']#[:,1:]
        target_idx = batch['click_idx']#[:,1:]
        time_mask = (batch['click']!=0).float()

        ### SCORE CALC
        # Sample itemvecs:
        itemvec = self.itemvec(torch.arange(self.num_items))
        # User side:
        click_vecs = itemvec[user_click_ids]
        # Add initial vector parameter (before any clicks):
        
        z0 = self.z0
        z0_expanded = z0.expand(len(click_vecs),1,len(z0))
        click_vecs = torch.cat((z0_expanded, click_vecs), dim = 1)

        zt, ht = self.gru(click_vecs)
        action_vecs = itemvec[action_ids]

        scores = (zt.unsqueeze(2) * action_vecs).sum(-1)
        if forecast:
            return scores

        ### LIKELIHOOD COMPUTATION:
        # mask scores through neg values for padded candidates:
        masked_scores = scores + (-1000)*(action_ids==0)

        ### Flatten and compute likelihood:
        slatelen = masked_scores.size()[-1]
        scores_flat = masked_scores.reshape(-1, slatelen)
        target_idx_flat = target_idx.reshape(-1)
        time_mask_flat = time_mask.reshape(-1)
        
        with pyro.plate("data", size = self.num_slates, subsample = target_idx_flat):
            obsdistr = dist.Categorical(logits=scores_flat).mask(time_mask_flat)
            pyro.sample("obs", obsdistr, obs=target_idx_flat)

        return scores
        
    def init_opt(self, guide, lr = 0.005, num_particles=1):
        adam = optim.Adam({"lr": lr})
        self.svi = SVI(self.model, guide, adam, loss=Trace_ELBO(num_particles))

    def predict(self, click_seq, t=-1):
        """
        Computes scores for each user in batch at time step t.
        NB: Not scalable for large batch.
        
        #user_click = batch['user_click'][:2]
        """
        with torch.no_grad():
            itemvec = self.itemvec(torch.arange(self.num_items))
            user_click_vec = itemvec[click_seq]
            zt_all, _ = self.gru(user_click_vec)
            zt = zt_all[:,t,]

            score = (zt.unsqueeze(1) * itemvec.unsqueeze(0)).sum(-1)
            return score

    def recommend(self, click_seq, k = 100, chunksize = 3,  t=-1, trace = None):
        """
        Compute predict & rank on a batch in chunks (for memory)
        """

        with torch.no_grad():
            topk = torch.zeros((len(click_seq), k), device = self.device)

            i = 0
            for click_chunck in chunker(click_seq, chunksize):
                pred = self.predict(click_chunck, t = t)
                topk_chunk = 3+pred[:,3:].argsort(dim=1, descending=True)[:,:k]
                topk[i:(i+len(pred))] = topk_chunk
                i += len(pred)
            return topk

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

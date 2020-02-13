#%%
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro import optim
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroSample, PyroModule, pyro_method
from pyro.infer.autoguide import AutoDiagonalNormal, init_to_sample, AutoDelta
from pyro import poutine
import torch.distributions.constraints as constraints
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')
import warnings
import simulator
def set_noninform_prior(mod, scale=1.0):
    for key, par in list(mod.named_parameters()):
        setattr(
            mod, key,
            PyroSample(
                dist.Normal(torch.zeros_like(par),
                            scale * torch.ones_like(par)).independent()))

class ItemHier(PyroModule):
    def __init__(self,
                 item_dim,
                 num_items,
                 item_group=None,
                 hidden_dim=None,
                 device="cpu", **kwargs):
        super().__init__()
        self.item_dim = item_dim
        self.hidden_dim = item_dim if hidden_dim == None else hidden_dim
        self.num_items = num_items
        self.device = device

        # Group-vec:
        self.item_group = item_group
        if self.item_group is None:
            self.item_group = torch.zeros((self.num_items, )).long()
        self.num_group = len(self.item_group.unique())
        self.groupvec = PyroModule[nn.Embedding](num_embeddings=self.num_group,
                                                 embedding_dim=self.item_dim)
        self.groupvec.weight = PyroSample(
            dist.Normal(torch.zeros_like(self.groupvec.weight),
                        torch.ones_like(self.groupvec.weight)).to_event(2))
        self.groupscale = PyroModule[nn.Embedding](
            num_embeddings=self.num_group, embedding_dim=self.item_dim)
        self.groupscale.weight = PyroSample(
            dist.Normal(0.0 * torch.ones_like(self.groupscale.weight), 0.2 *
                        torch.ones_like(self.groupscale.weight)).to_event(2))

        # Item vec based on group vec hier prior:
        self.itemvec = PyroModule[nn.Embedding](num_embeddings=num_items,
                                                embedding_dim=item_dim)
        self.itemvec.weight = PyroSample(lambda x: dist.Normal(
            self.groupvec(self.item_group),
            self.groupscale(self.item_group).clamp(0.001, 0.99)).to_event(2))

    def forward(self, idx=None):
        if idx is None:
            idx = torch.arange(self.num_items).to(self.device)
        return torch.tanh(self.itemvec(idx))

class Model(simulator.PyroRecommender):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set priors on parameters:
        self.item_model = ItemHier(**kwargs)
        self.gamma = PyroSample( dist.Normal(torch.tensor(0.5),torch.tensor(0.5)) )
        self.softmax_mult = PyroSample( prior = dist.Normal(torch.tensor(1.0), torch.tensor(1.0)))
        self.bias_noclick = PyroSample( 
            prior = dist.Normal(
                torch.zeros((self.maxlen_slate +1,)),
                torch.ones( (self.maxlen_slate +1,))
            ))

    def init_set_of_real_parameters(self, seed = 1):
        torch.manual_seed(seed)
        par_real = {}

        par_real['softmax_mult'] =  torch.tensor(self.softmax_mult).float()
        par_real['gamma'] = torch.tensor(0.99)
        par_real['bias_noclick'] = self.bias_noclick * torch.ones( (self.maxlen_slate +1,))

        groupvec = generate_random_points_on_d_surface(
            d=self.item_dim, 
            num_points=self.num_groups, 
            radius=1,
            max_angle=3.14)
            
        par_real['item_model.groupvec.weight'] = groupvec

        # Get item vector placement locally inside the group cluster
        V_loc = generate_random_points_on_d_surface(
            d=self.item_dim, 
            num_points=self.num_items,
            radius=1,
            max_angle=3.14)

        V = (groupvec[self.item_group] + 0.1 * V_loc)

        # Special items 0 and 1 is in middle:
        V[:2, :] = 0

        par_real['item_model.itemvec.weight'] = V


        # Set user initial conditions to specific groups:
        self.user_init = (torch.arange(self.num_users) //
                        (self.num_users / self.num_groups)).long()

        par_real['h0'] = groupvec[self.user_init]
        #tr = poutine.trace(
        #    self).get_trace(batch=batch)
        #for node, obj in tr.iter_stochastic_nodes():
        #    if par_real.get(node) is None:
        #        print(node, "\t", obj['value'].size())


        self.par_real = par_real

    def get_real_par(self, batch):

        """ Function that outputs a set of real parameters, including setting h0-batch which is batch specific"""
        out = self.par_real
        out['h0-batch'] = self.par_real["h0"][batch['userId']]
        return out

    def forward(self, batch, mode = "likelihood", t=None):
        click_seq = batch['click']
        userIds = batch['userId']

        batch_size, t_maxclick = batch['click'].size()
        # sample item dynamics
        itemvec = self.item_model()
        click_vecs = itemvec[click_seq]

        # Sample user dynamic parameters:
        gamma = self.gamma
        softmax_mult = self.softmax_mult
        bias_noclick = self.bias_noclick

        with pyro.plate("data", size = self.num_users, subsample = userIds):
            ## USER PROFILE
            # container for all hidden states:
            H = torch.zeros( (batch_size, t_maxclick+1, self.hidden_dim))

            # Sample initial hidden state of users:
            h0 = pyro.sample("h0-batch", dist.Normal(
                torch.zeros((batch_size, self.hidden_dim)), torch.ones((batch_size, self.hidden_dim))
                ).to_event(1)
                )

            H_list = [h0]
            for t in range(t_maxclick):
                h_new = gamma * H_list[-1] + (1-gamma)*click_vecs[:,t]
                H_list.append(h_new)

            H = torch.cat( [h.unsqueeze(1) for h in H_list], dim =1)
            Z = H # linear from hidden to Z_t

            if mode =="predict":
                "NB: DOES NOT WORK FOR LARGE BATCHES"
                zt_last = Z[:, t, ]
                score = (zt_last.unsqueeze(1) * itemvec.unsqueeze(0)).sum(-1)
                return score, H

            target_idx = batch['click_idx']
            time_mask = (batch['click'] != 0)*(batch['click'] != 2).float()
            lengths = (batch['action'] != 0).long().sum(-1)
            action_vecs = itemvec[batch['action']]

            # Compute scores for all actions by recommender
            scores = (Z[:,:t_maxclick].unsqueeze(2) * action_vecs).sum(-1)

            scores = scores*softmax_mult

            # Add a constant based on inscreen length to the no click option:
            batch_bias = bias_noclick[lengths]
            batch_bias = (batch['action'][:, :, 0] == 1).float() * batch_bias
            scores[:,:,0] += batch_bias
            # PAD MASK: mask scores through neg values for padded candidates:
            scores[batch['action'] == 0] = -100

            # SIMULATE
            # If we want to simulate, then t_maxclick = t_maxclick+1
            if mode == "simulate":
                # Check that there are not click in last time step:
                if bool((time_mask[:,-1] != 0 ).any() ):
                    warnings.warn("Trying to sample from model, but there are observed clicks in last timestep.")
                
                gen_click_idx = dist.Categorical(logits=scores[:, -1, :]).sample()
                return gen_click_idx
            
            if mode == "likelihood":
                obsdistr = dist.Categorical(logits=scores).mask(time_mask).to_event(1)
                pyro.sample("obs", obsdistr, obs=target_idx)
                
            return {'score' : scores, 'zt' : Z, 'V' : itemvec}

    #@torch.no_grad()
    @pyro_method
    def predict(self, batch, t=-1):
        """
        Computes scores for each user in batch at time step t.
        NB: Not scalable for large batch.
        
        batch['click_seq'] : tensor.Long(), size = [batch, timesteps]
        """
        return self.forward(batch, t=t, mode = "predict")

#%% MODEL GAMMA WITH POSITIVE POLAR COORD

def generate_random_points_on_d_surface(d, num_points, radius, max_angle=3.14, last_angle_factor=2):

    r = torch.ones((num_points,1))*radius
    angles = torch.rand((num_points, d-1))*max_angle
    angles[:,-1] = angles[:,-1]*last_angle_factor

    cosvec = torch.cos(angles)
    cosvec = torch.cat([cosvec, torch.ones((num_points,1))], dim=1)

    sinvec = torch.sin(angles)
    sinvec = torch.cat([torch.ones((num_points,1)), sinvec], dim=1)
    sincum = sinvec.cumprod(1)

    X = cosvec * sincum*r
    return X

def visualize_3d_scatter(dat):
    import plotly
    import plotly.graph_objs as go
    plotly.offline.init_notebook_mode()

    # Configure the trace.
    trace = go.Scatter3d(
        x=dat[:,0],  # <-- Put your data instead
        y=dat[:,1],  # <-- Put your data instead
        z=dat[:,2],  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 2,
            'opacity': 0.8,
        }
    )

    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    data = [trace]

    plot_figure = go.Figure(data=data, layout=layout)

    # Render the plot.
    plotly.offline.iplot(plot_figure)

class MeanFieldGuide:
    def __init__(self, model, batch, item_dim, num_users, hidden_dim, **kwargs):
        self.item_dim = item_dim
        self.num_users = num_users
        self.hidden_dim = hidden_dim
        self.maxscale = 0.1

        self.model_trace = poutine.trace(model).get_trace(batch)
    def __call__(self, batch=None, temp = 1.0):
        posterior = {}
        for node, obj in self.model_trace.iter_stochastic_nodes():
            par = obj['value']    
            if node == 'h0-batch':
                mean = pyro.param(f"{node}-mean",
                                    init_tensor = 0.01*torch.zeros((self.num_users, self.hidden_dim)))
                scale = pyro.param(f"{node}-scale",
                                    init_tensor=0.001 +
                                    0.05 * 0.01*torch.ones((self.num_users, self.hidden_dim)),
                                    constraint=constraints.interval(0,self.maxscale))

                with pyro.plate("data", size = self.num_users, subsample = batch['userId']):
                    posterior[node] = pyro.sample(
                        node,
                        dist.Normal(mean[batch['userId']], temp*scale[batch['userId']]).to_event(1))

            else:
                mean = pyro.param(f"{node}-mean",
                                    init_tensor= 0.05*par.detach().clone())
                scale = pyro.param(f"{node}-scale",
                                    init_tensor=0.05 +
                                    0.01 * par.detach().clone().abs(),
                                    constraint=constraints.interval(0,self.maxscale))
                posterior[node] = pyro.sample(
                    node,
                    dist.Normal(mean, temp*scale).independent())
        return posterior

    def median(self):
        posterior = {}

        for node, obj in self.model_trace.iter_stochastic_nodes():
            par = obj['value']
            mean = pyro.param(f"{node}-mean", init_tensor=par.detach().clone())
            posterior[node] = mean

        posterior['user_noise'] = 0
        return posterior

    def get_parameters(self):
        varpar = {}

        for node, obj in self.model_trace.iter_stochastic_nodes():
            par = obj['value']
            varpar[node] = {}
            varpar[node]['mean'] = pyro.param(f"{node}-mean",
                                              init_tensor=0.1 *
                                              torch.randn_like(par))
            varpar[node]['scale'] = pyro.param(f"{node}-scale",
                                               init_tensor=0.01 *
                                               par.detach().clone().abs(),
                                               constraint=constraints.positive)
        return varpar
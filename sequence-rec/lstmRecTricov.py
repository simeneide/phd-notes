import pyro
import torch.nn as nn
import torch
import torch.distributions.constraints as constraints
import pyro.distributions as dist
from pyro import plate
from pyro.distributions import Categorical


class LSTMrec(nn.Module):
    def __init__(self, num_items, emb_dim, batch_size, device):
        super(LSTMrec, self).__init__()
        self.batch_size = batch_size
        self.num_items = num_items
        self.emb_dim = emb_dim
        
        ### CUDA
        self.device = device
        
        # PRIORS
        self.tri0 = torch.cholesky(torch.diag(torch.ones(self.emb_dim).to(self.device)*5))
        
        # ITEM EMB
        self.V = nn.Embedding(embedding_dim= self.emb_dim,
                                           num_embeddings = self.num_items)
        
        ### LSTM
        self.linear = nn.Linear(1,1) #nn.ModuleList(nn.Parameter(torch.zeros((1,))))
        self.lstm = nn.LSTM(batch_first=True,
                                input_size=self.emb_dim,
                                hidden_size=self.emb_dim)
        
        # Need to permute before and after LSTM because LSTM's batch_first is not supported...
        self.permute_for_lstm = lambda x: x.permute(1,0,2)
        self.permute_back_lstm = lambda x: x.permute(1,0,2)
        
        if device.type == "cuda":
            self.cuda()    
    def forward(self, seq):
        """
        seq: a torch array with dimension (batch_size, sequence length).
        Needs to be long() and on right device. 
        
        Outputs log probabilities of all items. (todo: Let user specify which items to calc)
        """
        x_vecs = self.V(seq)
        z, _ = self.lstm(x_vecs)
        
        
        lprob = self.linear.bias + z.matmul(self.V.weight.t())
        return lprob
    
    def model(self, seq):
        mu0 = torch.zeros(self.emb_dim).to(self.device)
        tri0 = self.tri0 # create this when initializing. (takes 4ms each time!)

        muV = pyro.sample("muV", dist.MultivariateNormal(loc = mu0, scale_tril=tri0))

        with plate("item_loop", self.num_items):
            V = pyro.sample(f"V", dist.MultivariateNormal(muV, scale_tril=tri0))

        # LIFT MODULE:
        prior = {'linear.bias' : dist.Normal(0,1),
                'V.weight' : Deterministic_distr(V)}
        lifted_module = pyro.random_module("net", self, prior= prior)
        

        lifted_reg_model = lifted_module()
        lifted_reg_model.lstm.flatten_parameters()

        with pyro.plate("data", len(seq), subsample_size = self.batch_size) as ind:
            batch_seq = seq[ind,]
            x = batch_seq[:,:-1]
            y = batch_seq[:,1:]
            batch_mask = (y!=0).float()

            lprobs = lifted_reg_model(x)
            data = pyro.sample("obs_x", 
                               dist.Categorical(logits=lprobs).mask(batch_mask).to_event(2), 
                               obs = y)
        return lifted_reg_model
    
    def guide(self, x):
                           
        bias_loc = pyro.param("bias_loc", torch.tensor(0.0).to(self.device))
        bias_scale = pyro.param("bias_scale", torch.tensor(2.0).to(self.device), constraint = constraints.positive)
        bias = dist.Normal(loc = bias_loc, scale = bias_scale)

        
        # GLOBAL V:
        muV_mean = pyro.param("muV_mean", torch.rand(self.emb_dim).to(self.device))
        global_triV = pyro.param("triangVpar", 
              torch.diag(0.5*torch.ones(self.emb_dim)).to(self.device), 
              constraint = constraints.lower_cholesky)
        muV = pyro.sample("muV", dist.MultivariateNormal(muV_mean, scale_tril=global_triV))
        
        ### ITEMS RVs ###
        # Each item has a factor it multiplies L with on diagonal. 
        # Inspiration: covar = L D L^t = (L D^0.5) (L D^0.5)ˆt
        # item_var is the D**2
        item_var = pyro.param("item_var_factor", (torch.rand(self.num_items)+0.5).to(self.device),
                              constraint = constraints.positive)
        D_V = (
            torch.diag(
                torch.ones(self.emb_dim)
                .to(self.device))
            .unsqueeze(0)
            .repeat(len(item_var),1,1)
            *item_var.unsqueeze(1)
            .unsqueeze(1)
        )
        triV = global_triV.matmul(D_V)

        V = pyro.param("Vpar", torch.rand(self.num_items, self.emb_dim).to(self.device)-0.5)

        ## item latent vectors
        with plate("item_loop", self.num_items) as i: # 
            V = pyro.sample(f"V", dist.MultivariateNormal(V[i,], scale_tril=triV[i,]))
            
        posterior = {'linear.bias' : bias,
                'V.weight' : Deterministic_distr(V)}
        
        lifted_module = pyro.random_module("net", self, prior = posterior)
        
        lifted_reg_model = lifted_module()
        lifted_reg_model.lstm.flatten_parameters()
        return lifted_reg_model

    
class Deterministic_distr(pyro.distributions.Distribution):
    def __init__(self, V):
        self.V = V
    def sample(self):
        return self.V
    def log_prob(self,x):
        return torch.zeros(x.size()[0])
    
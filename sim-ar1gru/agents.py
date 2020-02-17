from pyro.nn import PyroSample, PyroModule
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample, pyro_method
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pyro.distributions as dist
import copy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random


class RandomSystem:
    def __init__(self, num_items, maxlen_slate , *args, **kwargs):
        super().__init__()
        self.num_items = num_items
        self.maxlen_slate = maxlen_slate

    def forward(self, *args, **kwargs):
        return None

    def recommend(self, batch, *args, **kwargs):
        batch_size = len(batch['click'])
        action = 2 + torch.cat([
            torch.randperm(self.num_items - 2).unsqueeze(0)
            for _ in range(batch_size)
        ])
        action = action[:, :(self.maxlen_slate - 1)]
        return action


### WRAPPER FOR MODELS:
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

class PyroRecommender(PyroModule):
    """ A Pyro Recommender Module. Have some common functionalities that are shared among all recommenders"""
    def __init__(self, **kwargs):
        super().__init__()
        # Register all input vars in module:
        for key, val in kwargs.items():
            setattr(self,key,val)
        self.init_set_of_real_parameters()
        self.trainmode=True # defaults to training mode.
    @pyro_method
    def simulate(self, batch):
        return pyro.condition(
            lambda batch: self.forward(batch, mode="simulate"),
            data=self.get_real_par(batch))(batch)

    @pyro_method
    def likelihood(self, batch, par = None): 
        if par is None:
            par = self.get_real_par(batch)

        return pyro.condition(
            lambda batch: self.forward(batch, mode="likelihood"),
            data=par)(batch)

    @pyro_method
    def predict_cond(self, batch, par=None, **kwargs): 
        if par is "real":
            par = self.get_real_par(batch)
        else:
            par = par(batch)
        return pyro.condition(
            lambda batch: self.predict(batch, **kwargs),
            data=par)(batch)

    def recommend(self, batch, max_rec=1, chunksize=3, t=-1, par="real", **kwargs):
        """
        Compute predict & rank on a batch in chunks (for memory)
        """

        click_seq = batch['click']
        topk = torch.zeros((len(click_seq), max_rec), device=self.device)

        i = 0
        for click_chunck, userId in zip(
            chunker(click_seq, chunksize),
            chunker(batch['userId'], chunksize)):
            pred, ht = self.predict_cond(
                batch={'click' :click_chunck, 'userId' : userId}, t=t, par=par)
            topk_chunk = 3 + pred[:, 3:].argsort(dim=1,
                                                 descending=True)[:, :max_rec]
            topk[i:(i + len(pred))] = topk_chunk
            i += len(pred)
        return topk

    def visualize_item_space(self):
        if self.item_dim ==2:
            V = self.par_real['item_model.itemvec.weight'].cpu()
            sns.scatterplot(V[:, 0].cpu(),
                            V[:, 1].cpu(),
                            hue=self.item_group.cpu())
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.show()
        if self.item_dim == 3:
            visualize_3d_scatter(self.par_real['item_model.itemvec.weight'])

    def train(self):
        self.trainmode=True

    def eval(self):
        self.trainmode=False


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

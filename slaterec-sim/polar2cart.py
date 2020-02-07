#%%
import torch
import matplotlib.pyplot as plt

# %% FROM ANGLES TO CARTESIAN
def generate_random_points_on_d_surface(d, num_points, radius):
    pi=torch.tensor(3.14)

    r = torch.ones((num_points,1))*radius
    angles = torch.rand((num_points, d-1))*pi
    angles[:,-1] = angles[:,-1]*2
    cosvec = torch.cos(angles)
    cosvec = torch.cat([cosvec, torch.ones((num_points,1))], dim=1)

    sinvec = torch.sin(angles)
    sinvec = torch.cat([torch.ones((num_points,1)), sinvec], dim=1)
    sincum = sinvec.cumprod(1)

    X = cosvec * sincum*r
    return X

# %%
# Import dependencies
if __name__ == "__main__":
    import plotly
    import plotly.graph_objs as go


    #%% Generate some data in balls centered around balls in a ball.
    num_groups = 100
    num_items = 100
    groupvec = generate_random_points_on_d_surface(d=3, num_points=num_groups, radius = 1)

    itemvecs = []
    for g in range(num_groups):
        loc = generate_random_points_on_d_surface(d=3, num_points=int(num_items/num_groups), radius=0.2)
        V_tmp = loc + groupvec[g,:]
        itemvecs.append(V_tmp)

    dat = torch.cat(itemvecs, dim = 0)

    dat = V
    #%% PLOT IN PLOTLY
    # Configure Plotly to be rendered inline in the notebook.

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

# %%

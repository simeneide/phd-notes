# %% [markdown]
# # Hand-in for Sequential Monte Carlo Course
#  Written by Simen Eide (simeneide@gmail.com)

#%% [markdown]
####################
# # H.1 Importance Sampling Theory
####################
# %% [markdown]
# # H.1 (a)
# We need to show that the estimator is unbiased i.e. $E_q(\hat{Z}) = Z$.
#
# $$E_q(\hat{Z}) = \frac{1}{N} \sum_{i=1}^N E_q( \frac{\hat{\pi}(x^i)}{q(x^i)} )$$
# $$ =  \frac{1}{N} \sum_{i=1}^N \int( \frac{\hat{\pi}(x)}{q(x)} ) q(x) dx$$
# $$ =  \int \hat{\pi}(x)dx = Z$$
#
#
# %% [markdown]
# # H.1 (b)
# Assume that $\lambda^{-1}$ is the variance.
#
# If we compute the ratio,
#
# $$ \frac{\pi(x)^2}{q(x)} \propto \frac{e^{-x^2} }{  \sqrt{\lambda} e^{-\lambda x^2 / 2}} $$
# $$ = \sqrt{\lambda} e^{-x^2(1-\lambda/2)}$$
#
# So the integral
# $ \int \sqrt{\lambda} e^{-x^2(1-\lambda/2)} dx$
# will be finite when $1-\lambda/2 > 0$.
# This is when $0 < \lambda < 2$.
#
# If $\lambda$ is too big then the tails of the proposal distribution will not cover the target distribution, and it will not effectiively have support over the target distribution.
# %% [markdown]
####################
# # H.2 Particle filter for a linear Gaussian state-space model
####################
# %% [markdown]
# # H.2 (a)
#
# $$ X_t | (X_{t-1} = x_{t-1}) \sim N(X_t ; 0.9 x_{t-1}, 0.5)$$
# $$ Y_t | (X_t = x_t) \sim N(Y_t; 1.3 x_t, 0.1) $$
# $$ X_0 \sim N(0,1)$$

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sorcery import dict_of
import pandas as pd
from scipy.stats import invgamma
from tqdm import tqdm
#%%
np.random.seed(1)
T = 2000
x_true = np.zeros(T)
y_true = np.zeros(T)
x_true[0] = np.random.randn()

for t in range(1, T):
    x_true[t] = 0.9 * x_true[t - 1] + 0.5 * np.random.randn()
    y_true[t] = 1.3 * x_true[t] + 0.1 * np.random.randn()

plt.plot(y_true)

# %% [markdown]
# # H.2 (b)
#

# %%
A = 0.9
Q = 0.5
C = 1.3
R = 0.1
p0 = 1.0

x_kalman = np.zeros(T)
Ptt = np.zeros(T)
Ptt[0] = p0
for t in range(1, T):
    p_t2tmin1 = A * Ptt[t - 1] * A + Q
    K_t = p_t2tmin1 * C * (C * p_t2tmin1 * C + R)**(-1)
    Ptt[t] = p_t2tmin1 - K_t * C * p_t2tmin1
    x_kalman[t] = A * x_kalman[t - 1] + K_t * (y_true[t] -
                                               C * A * x_kalman[t - 1])
plt.plot(x_true, x_kalman, "o")
print(f"Mean absolute error of kalman filter: {np.mean(np.abs(x_true-x_kalman)):.3f}")
# %% [markdown]
# No, the particle filter is an approximate solution to x, whereas the kalman filter is an analytic solution.
# %% [markdown]
#
#
# # H.2 (c)

# %%

prop_fun = lambda x_prev: 0.9 * x_prev + 0.1 * np.random.randn(len(x_prev))
obs_lik = lambda y_t, x: norm.pdf(y_t, loc=1.3 * x, scale=0.1)
#obs_lik = np.vectorize(obs_lik)

# %%
def bootstrap(
    y,
    prop_fun,
    obs_lik,
    N,
):
    T = len(y)
    x = np.zeros((T, N)) * 0.0
    w_curl = np.zeros((T, N)) * 0.0
    w = np.zeros((T, N)) * 0.0
    a = np.zeros((T, N), dtype="int")
    w[0, ] = 1 / N
    w_curl[0, ] = 1 / N

    # Init x0
    x[0, ] = norm.rvs(size=N, loc=0, scale=1.0)  # same as prior
    for t in range(1, T):
        # Resample
        a[t, ] = np.random.choice(np.arange(N), size=N, p=w[t - 1])

        # Propagate
        x[t, ] = prop_fun(x_prev=x[t - 1, a[t, ]])

        # Weights
        # w_t  = p(y_i | x_i)
        w_curl[t, ] = obs_lik(y_t=y[t, ], x=x[t, ])
        w[t, ] = w_curl[t, ] / w_curl[t, ].sum()

    loglik = (np.log(w_curl.sum(1)) - np.log(N)).sum()
    return dict_of(loglik, x, a, w, w_curl)


# %%
N_samples = np.array([10, 50, 100, 2000, 5000])
mae = np.zeros(len(N_samples))
var = np.zeros(len(N_samples))

for i, N in enumerate(N_samples):
    results = bootstrap(y=y_true, prop_fun=prop_fun, obs_lik=obs_lik, N=N)
    mae[i] = np.mean(np.abs(results['x'].mean(1) - x_true))
    var[i] = np.mean((results['x'].mean(1) - x_true)**2)


df = pd.DataFrame({'N': N_samples, 'mae': mae, 'var': var})
print(df)

# %% [markdown]
# # H.2 (d)
# We have a gaussian linear model and $P(y_t | x_t)$ is conjugate to $P(x_t|x_{t-1})$ (normal-normal).
# Hence, from the notes (and using the constants as in (b), we get that 
# $$P(y_t|x_{t-1}) = N(y_t | CA x_{t-1}, C^2Q + R)$$
# $$P(x_t|x_{t-1}, y_t) = N(x_t | Ax_{t-1} + K(y_t - CAx_{t-1}), \Sigma) $$
# where
# $$K:= QC(C^2Q+R)^{-1} $$
# $$\Sigma = (1-KC)Q $$
# %% Implementation
# W is really nu in this implementation


y = y_true
# constants:
K = Q*C/(C**2 * Q + R)
sigma = (1-K*C)*Q

prop_fun = lambda x_prev, y_t, N : A*x_prev + K*(y_t-C*A*x_prev) + norm.rvs(size=N)*sigma
obs_lik = lambda y_t, x_prev, N: norm.pdf(y_t, loc=C*A*x_prev, scale=C**2 * Q+R)

#%%
def fully_adapted(y,prop_fun,obs_lik,N,):
    T = len(y)
    N_eff = np.zeros((T,))
    x = np.zeros((T, N)) * 0.0
    w_curl = np.zeros((T, N)) * 0.0
    w = np.zeros((T, N)) * 0.0
    a = np.zeros((T, N), dtype="int")
    w[0, ] = 1 / N
    w_curl[0, ] = 1 / N

    # Init x0
    x[0, ] = norm.rvs(size=N, loc=0, scale=1.0)
    for t in range(1, T):
        # Resample
        N_eff[t] = 1/np.sum(w[t-1]**2)
        a[t, ] = np.random.choice(np.arange(N), size=N, p=w[t - 1])

        # Propagate
        x[t, ] = prop_fun(x_prev=x[t - 1, a[t, ]], y_t=y[t,], N=N)

        # Weights
        # w_t  = p(y_i | x_i)
        w_curl[t, ] = obs_lik(y_t=y[t, ], x_prev=x[t-1, ], N=N)
        w[t, ] = w_curl[t, ] / w_curl[t, ].sum()

    loglik = (np.log(w_curl.sum(1)) - np.log(N)).sum()
    return dict_of(loglik, x, a, w, w_curl, N_eff)
# %%
N_samples = np.array([10, 50, 100, 2000, 5000])
mae = np.zeros(len(N_samples))
var = np.zeros(len(N_samples))

for i, N in enumerate(N_samples):
    results = fully_adapted(y=y_true, prop_fun=prop_fun, obs_lik=obs_lik, N=N)
    mae[i] = np.mean(np.abs(results['x'].mean(1) - x_true))
    var[i] = np.mean((results['x'].mean(1) - x_true)**2)
df = pd.DataFrame({'N': N_samples, 'mae': mae, 'var': var})
print(df)
# %% [markdown]
# The fully adapted particle filter is on par with the kalman filter, 
# whereas the bootstrap is much more noisy.


#%% [markdown]
# # H.2 (e)
# %%
results = fully_adapted(y=y_true, prop_fun=prop_fun, obs_lik=obs_lik, N=100)
print(f"Mean absolute error of Fully Adaptive: {np.mean(np.abs(x_true-results['x'].mean(1))):.3f}")
x_paths = np.zeros((2000,100))
t = 1999
next_a = results['a'][t]
for t in range(1999,1, -1):
    current_a = results['a'][t-1][next_a]
    x_paths[t,] = results['x'][t-1,current_a]
    next_a = current_a

plt.subplots(figsize=(15, 15))
_ = plt.plot(x_paths[1700:], linewidth=1.0, marker="o", markersize=5)
plt.title("Trajectory of the last 300 time steps")
plt.show()
plt.subplots(figsize=(15, 5))
plt.plot(x_paths.std(1))
_ = plt.title("Standard deviation of trajectories across the different particles at each time")
plt.show()
plt.subplots(figsize=(15, 5))
plt.plot(results['N_eff'])
plt.title("Effective samples over time")

#%% [markdown]
# # H.2 (f) Systematic sampling
# We see that the variation is constant over the 2000 timesteps. 
# Although there is little noise in the system, 
# the fully adaptive resampling algorithm will at some point find a nu that is dominating in the resampling and put samples on that ancestor.
# Then the remaining particles will be dropped.
# With systematic resampling, we will instead choose "uniformly" over the cumulative distribution, reducing the problem.
# In fact, it seems to be completely gone in our example.
# 
#%%
def fully_adapted_systematic_sampling(y,prop_fun, obs_lik, N,):
    T = len(y)
    x = np.zeros((T, N)) * 0.0
    w_curl = np.zeros((T, N)) * 0.0
    w = np.zeros((T, N)) * 0.0
    a = np.zeros((T, N), dtype="int")
    w[0, ] = 1 / N
    w_curl[0, ] = 1 / N

    # Init x0
    x[0, ] = norm.rvs(size=N, loc=0, scale=1.0)
    for t in range(1, T):
        # Resample
        offset = np.random.rand()
        idx = np.sort( (np.linspace(0,0.99,100)+offset) %  1 )
        cum = w[t-1].cumsum()
        for i in range(99):
            a[t,i] = (idx[i] < cum).argmax()

        # Propagate
        x[t, ] = prop_fun(x_prev=x[t - 1, a[t, ]], y_t=y[t,], N=N)

        # Weights
        # w_t  = p(y_i | x_i)
        w_curl[t, ] = obs_lik(y_t=y[t, ], x_prev=x[t-1, ], N=N)
        w[t, ] = w_curl[t, ] / w_curl[t, ].sum()

    loglik = (np.log(w_curl.sum(1)) - np.log(N)).sum()
    return dict_of(loglik, x, a, w, w_curl)

results = fully_adapted_systematic_sampling(y=y_true, prop_fun=prop_fun, obs_lik=obs_lik, N=100)
print(f"Mean absolute error of Fully Adaptive with Systematic Sampling: {np.mean(np.abs(x_true-results['x'].mean(1))):.3f}")
x_paths = np.zeros((2000,100))
t = 1999
next_a = results['a'][t]
for t in range(1999,1, -1):
    current_a = results['a'][t-1][next_a]
    x_paths[t,] = results['x'][t-1,current_a]
    next_a = current_a

plt.subplots(figsize=(15, 15))
_ = plt.plot(x_paths[1700:], linewidth=1.0, marker="o", markersize=5)
plt.title("Trajectory of the last 300 time steps")
plt.show()
plt.subplots(figsize=(15, 5))
plt.plot(x_paths.std(1))
_ = plt.title("Standard deviation of trajectories across the different particles at each time")
# %% [markdown]
# # H.2 (g) Adaptive resampling
# We adapt the fully adapted algorithm to do adaptive sampling.
# Interestingly, the effective sample size is very high for most timesteps. 
# In fact, it seems the algorithm only does a resampling step at $t=1$.
# For all remaining time steps, the effective sampling size is close to 100.
# Rerunning the fully adapted filter above, we see that this is the same in the normal fully adaptive case.

# %%
def fully_adapted_adaptive_sampling(y,prop_fun,obs_lik,N,):
    T = len(y)
    N_eff = np.zeros((T,))
    x = np.zeros((T, N)) * 0.0
    w_curl = np.zeros((T, N)) * 0.0
    w = np.zeros((T, N)) * 0.0
    a = np.zeros((T, N), dtype="int")
    w[0, ] = 1 / N
    w_curl[0, ] = 1 / N

    # Init x0
    x[0, ] = norm.rvs(size=N, loc=0, scale=1.0)
    for t in range(1, T):
        # Resample
        N_eff[t] = 1/np.sum(w[t-1]**2)
        if N_eff[t] < 50:
            a[t, ] = np.random.choice(np.arange(N), size=N, p=w[t - 1])
        else:
            a[t, ] = np.arange(0,100)

        # Propagate
        x[t, ] = prop_fun(x_prev=x[t - 1, a[t, ]], y_t=y[t,], N=N)

        # Weights
        # w_t  = p(y_i | x_i)
        w_curl[t, ] = obs_lik(y_t=y[t, ], x_prev=x[t-1, ], N=N)
        w[t, ] = w_curl[t, ] / w_curl[t, ].sum()

    loglik = (np.log(w_curl.sum(1)) - np.log(N)).sum()
    return dict_of(loglik, x, a, w, w_curl, N_eff)

results = fully_adapted_adaptive_sampling(y=y_true, prop_fun=prop_fun, obs_lik=obs_lik, N=100)
print(f"Mean absolute error of Fully Adaptive with Adaptive Sampling: {np.mean(np.abs(x_true-results['x'].mean(1))):.3f}")
x_paths = np.zeros((2000,100))
t = 1999
next_a = results['a'][t]
for t in range(1999,1, -1):
    current_a = results['a'][t-1][next_a]
    x_paths[t,] = results['x'][t-1,current_a]
    next_a = current_a

plt.subplots(figsize=(15, 15))
_ = plt.plot(x_paths[1700:], linewidth=1.0, marker="o", markersize=5)
plt.title("Trajectory of the last 300 time steps")
plt.show()
plt.subplots(figsize=(15, 5))
plt.plot(x_paths.std(1))
_ = plt.title("Standard deviation of trajectories across the different particles at each time")
plt.show()
plt.subplots(figsize=(15, 5))
plt.plot(results['N_eff']/100)
_ = plt.title("Effective samples over time")
# %%

#%% [markdown]
####################
# # H.3 Stochastic Volatility
####################

# # H.3 (a)
# %%
y = pd.read_csv("seOMXlogreturns2012to2014.csv", header=None).values.flatten()
#theta = [0.98, 0.16, 0.70] # phi, sigma, beta
phi, sigma, beta = 0.985, 0.16, 0.70
N = 100
# %%
def stocVol_bootstrap(y, N, phi, sigma, beta):
    T = len(y)
    x = np.zeros((T,N))*0.0
    w_curl = np.zeros((T,N))*0.0
    w = np.zeros((T,N))*0.0
    a = np.zeros((T,N), dtype="int")
    w[0,] = 1/N
    w_curl[0,] = 1/N
    # Make a empirical base type of prior:
    init_mu = 0# , #np.log(y.std()/(beta**2))
    x[0,] = norm.rvs(size=N, loc = init_mu, scale=1.0)

    for t in range(1,T):
        # Resample
        a[t,] = np.random.choice(np.arange(N), size=N, p=w[t-1])

        # Propagate
        # x_t = N(phi*x_t-1, sigma**2):
        x[t,] = norm.rvs(size=N, loc = phi*x[t-1,a[t,]], scale=sigma)

        # Weights
        # w_t  = p(y_i | x_i)
        w_curl[t,] = norm.pdf(y[t], loc=0, scale = beta**2 * np.exp(x[t,]))
        w[t,] = w_curl[t,]/w_curl[t,].sum()

    loglik = (np.log(w_curl.sum(1)) - np.log(N)).sum()
    return {'y' : y, 'x' : x, 'w' : w, 'a' : a, 'w' : w, 'loglik' : loglik, 'w_curl' : w_curl}

num_steps=10
phi_grid = np.linspace(0,1,num_steps).round(2)
loglik = np.zeros((len(phi_grid), 10))
for r, phi_c in enumerate(phi_grid):
    for c in range(10):
        loglik[r,c] = stocVol_bootstrap(y, N, phi=phi_c, sigma=sigma, beta=beta)['loglik']


_ = plt.boxplot(loglik.T, labels = phi_grid)
_ = plt.title("Ten loglik estimates (y-axis) per phi value (x-axis)")
# %% [markdown]
# # H.3 (b)
# According to this webpage we get the equivalent InvGamma distribution by writing the following:
# `scipy.stats.invgamma(alpha, loc=0, scale=beta)`  
# https://distribution-explorer.github.io/continuous/inverse_gamma.html

#%%
logprior_sigma2 = lambda sigma2 :invgamma.logpdf(sigma2, 0.01, loc=0, scale=0.01)
logprior_beta2 = lambda beta2 :invgamma.logpdf(beta2, 0.01, loc=0, scale=0.01)

def compute_logprob(sigma2, beta2, N):
    loglik = stocVol_bootstrap(y, N, phi=phi, sigma=np.sqrt(sigma2), beta=np.sqrt(beta2))['loglik']
    return loglik + logprior_sigma2(sigma2) + logprior_beta2(beta2)

num_samples = 2000
N = 100
step_size = 0.008
trace = {
    'sigma2' : np.zeros((num_samples,)),
    'beta2' : np.zeros((num_samples,)),
    'posterior' : np.zeros((num_samples,)),
    'accept_ratio' : np.zeros((num_samples,)),
    }

# initial values
trace['sigma2'][0], trace['beta2'][0] = 0.1, 0.9

n = 0
trace['posterior'][n] = compute_logprob(trace['sigma2'][n], trace['beta2'][n], N=N)

#for n in range(num_samples):
for n in tqdm(range(1,num_samples)):
    sigma2_cand = trace['sigma2'][n-1] + norm.rvs()*step_size
    beta2_cand = trace['beta2'][n-1] + norm.rvs()*step_size

    if ((sigma2_cand < 1e-3) | (beta2_cand < 1e-3)):
        accept_ratio=0
    else:
        posterior_cand = compute_logprob(sigma2_cand, beta2_cand, N=N)
        accept_ratio = np.exp(posterior_cand - trace['posterior'][n-1])
    
    if np.random.rand() < accept_ratio:
        trace['sigma2'][n] = sigma2_cand
        trace['beta2'][n] = beta2_cand
        trace['posterior'][n] = posterior_cand
    else:
        trace['sigma2'][n] = trace['sigma2'][n-1]
        trace['beta2'][n] = trace['beta2'][n-1]
        trace['posterior'][n] = trace['posterior'][n-1]
    trace['accept_ratio'][n] = accept_ratio
#%%
plt.plot(trace['posterior'])
plt.title("Trace of posterior probability estimate")
plt.show()
plt.plot(np.sqrt(trace['sigma2']))
plt.plot(np.sqrt(trace['beta2']))
_ = plt.title("Trace plots of sigma and beta")
plt.show()
print(f"Average Accept probability {np.clip(trace['accept_ratio'], 0,1).mean()}")

burn=500

plt.hist(trace['sigma2'][200:],bins=50)
plt.title("Histogram of sigma2")
plt.show()
plt.hist(trace['beta2'][200:],bins=50)
_ = plt.title("histogram of beta**2")

# %%
# %% [markdown]
# # H.3 (c) Particle Gibbs
#  We iterate between producing estimates of $\sigma^2$, $\beta^2$ and $x_{1:T}$ (using gibbs PF).
# I see that the gibbs mcmc converges faster than the MH-mcmc.
#%%
T = len(y)
def stocVol_gibbs(y, N, phi, sigma, beta, x_prev):
    T = len(y)
    x = np.zeros((T,N))*0.0
    
    w_curl = np.zeros((T,N))*0.0
    w = np.zeros((T,N))*0.0
    a = np.zeros((T,N), dtype="int")
    w[0,] = 1/N
    w_curl[0,] = 1/N
    # Make a empirical base type of prior:
    init_mu = 0# , #np.log(y.std()/(beta**2))
    x[0,] = norm.rvs(size=N, loc = init_mu, scale=1.0)

    # Set conditional x path:
    x[:,-1] = x_prev
    a[:,-1] = N-1
    for t in range(1,T):
        # Resample
        a[t,:-1] = np.random.choice(np.arange(N), size=N-1, p=w[t-1])
        
        # Propagate
        # x_t = N(phi*x_t-1, sigma**2):
        x[t,:-1] = norm.rvs(size=N-1, loc = phi*x[t-1,a[t,:-1]], scale=sigma)

        # Weights
        # w_t  = p(y_i | x_i)
        w_curl[t,] = norm.pdf(y[t], loc=0, scale = beta**2 * np.exp(x[t,]))
        w[t,] = w_curl[t,]/w_curl[t,].sum()
    
    # Sample one of the paths:
    b = np.random.choice(np.arange(N), size=1, p=w[t,])[0]
    x_new = np.zeros((T,))
    a_idx = b
    for t in range(T-1,-1, -1):        
        x_new[t,] = x[t,a_idx]
        a_idx = a[t][a_idx]
    return x_new

def posterior_sample_sigma2(x, y, phi):
    alpha = 0.01 + T/2
    beta = 0.01 + 0.5 * np.sum((x[1:] - phi*x[:-1])**2)
    return invgamma.rvs(alpha, loc=0, scale = beta)

def posterior_sample_beta2(x, y):
    alpha = 0.01 + T/2
    beta = 0.01 + 0.5 * np.sum(np.exp(-x)*(y**2))
    return invgamma.rvs(alpha, loc=0, scale = beta)

# %%
trace = {
    'sigma2' : np.zeros((num_samples,)),
    'beta2' : np.zeros((num_samples,)),
    'x' : np.zeros((num_samples, len(y)))
    }
# Inits:
trace['sigma2'][0], trace['beta2'][0] = 0.1, 0.9
trace['x'][0] = np.random.randn(500)*0.1
# Sample X

for n in tqdm(range(num_samples)):
    trace['sigma2'][n] = posterior_sample_sigma2(trace['x'][n-1],y,phi)
    trace['beta2'][n] = posterior_sample_beta2(trace['x'][n-1],y)
    trace['x'][n] = stocVol_gibbs(y, N, phi=phi, sigma=np.sqrt(trace['sigma2'][n]), beta=np.sqrt(trace['beta2'][n]), x_prev = trace['x'][n-1])
#%%
plt.plot(np.sqrt(trace['sigma2']))
plt.plot(np.sqrt(trace['beta2']))
_ = plt.title("Trace plots of sigma and beta")
plt.show()
plt.hist(trace['sigma2'][200:],bins=50)
plt.title("Histogram of sigma2")
plt.show()
plt.hist(trace['beta2'][200:],bins=50)
_ = plt.title("histogram of beta**2")
# %%

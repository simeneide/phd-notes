# Recent on Simulator 12 dec 2019
- Recently I rewrote the model to make it adaptable to the step by step time style of the simulator.
This resulted in a huge mess of code (the likelihood computation now looks horribly complex).
- Had to wrap all kinds of functions in a pyro context, however, got feedback from pyro dev today that I can [use a special decorator for it](https://forum.pyro.ai/t/trace-with-pyromodule-using-different-function/1458).
- Plan: After reviewing the work yesterday, I realized I dont really need to be that effective when rolling out timesteps. Playing 15 timesteps for 1000 users doesnt take more than some millisecs. If we instead focus on clean code that computes all timesteps up until that point, there will be less clutter in the code. Need to change the simulator from doing stepwise to receiving the full user history up until that point though. But then it can also be stateless(!).

Positive results:
- I found a bug where I hadnt removed the gradients of the "padding item vector. (used where the no. of slate views are less than maximum, we pad to keep code fast and  vectorized). 
  - Before I just reduced the relevance score by 1000, now I replace the value (so that the program has no gradient back to the parameters).
  - Did not seem to have any huge effect on model.
- Made a small example of time dependent systems. Specifically I tried to see whether we could use pyro.plate over time steps, or if the gradients would not work. Script [here](../general_notes/linear_time_dep_in_pyro.ipynb).

# Cascade-inspired softmax model
The cascade model [e.g. Lattimore, Bandit Book p. 370] is a heavily used model for slate recommendations.
The standard framework assumes that the user scrolls through the whole list.
Usually this is not what happens in reality, and the user do stop scrolling at various steps in the list ($\rho_t^u$).

Assume the setup  is equal to our framework (section 2). 
Remember that a list of recommendations to user $u$ at time $t$ is given as permutation over all items $I$:

$$a_t^u := [\alpha_t^u(1), \alpha_t^u(2), ..., \alpha_t^u(|I|)]$$

We want to build a parameterized model for the click probability of an item in the slate given the user's click history: $P(C_t^u = \alpha_t^u(k), \rho_t^u | C_{1:t-1}^u, a_t^u, \theta)$. 
Note that we have not determined $\rho_t^u$ yet.
For simplicity, I will abuse notation and assume that all future probability statements are dependent on some global model parameters $\theta$ and the user history $C_{1:t-1}^u$. I.e. we can write the above as $P(C_t^u = \alpha_t^u(k), \rho_t^u | a_t^u)$.


### The "standard" cascade model
Let $logistic(x) = \frac{e^x}{e^x+1}$
For a specific k, let the click probability of the item $\alpha_t^u(k) \in a_t^u$ be defined as 
$$
P(C_t^u = \alpha_t^u(k) | a_t^u) = logistic(z_t^u v_{\alpha_t^u(k)}) * \chi(\alpha_t^u(k))
$$

where

$$
\chi(\alpha_t^u(k))=
\begin{cases}
    1  & \text{if } k=1 \\
    0  & \text{if } k>|I| \\
    \prod_{j=1}^{k-1} (1-logistic(z_t^u v_{\alpha_t^u(k)}))  & \text{if } k >1
\end{cases}
$$

This model assumes that the user will scroll until she finds something, possibly to the last item $\alpha_t^u(|I|)$.
That is a very unrealistic model. However, simply making the list shorter is also unrealistic and is contradicted by our data. Users do actually scroll variable lengths!

Therefore, we want to introduce a cascade model with some leaving risk in between each viewed item:

### Cascade model with leaving risk

Assume that the user, after having viewed each item, considers whether to continue to look through the list or not.

For (alot of) simplicity we assume that this consideration is independent of the items she has seen so far.

For a specific k, let the click probability of the item $\alpha_t^u(k) \in a_t^u$ be defined as 
$$
P(C_t^u = \alpha_t^u(k) | a_t^u) = logistic(z_t^u v_{c_t^u}) * \chi(\alpha_t^u(k))
$$

where

$$
\chi(\alpha_t^u(k))=
\begin{cases}
    1  & \text{if } k=1 \\
    0  & \text{if } k>|I| \\
    \prod_{j=1}^{k-1} (1-logistic(z_t^u v_{\alpha_t^u(k)})) (1-l_j)  & \text{if } k >1
\end{cases}
$$
and $l_j$ is the probability that the user will stop scrolling after she have viewed $j$ items.

A couple of (non-important?) comments:
- We can omit "$0 \text{ if } k>|I|$" in the equation above by assuming that $l_{|I|} = 1$.
- We can get our $P(\rho_t^u | a_t^u)$ by manipulating $\chi()$.

#### How does this likelihood compare to our softmax model?
Now we will simplify it further to make it comparable to our current softmax-model:
Assume that all leaving risks for all positions are equal: $l_k = l, \forall k$ (this assumtion could easily be relaxed).
The full likelihood is then:

$$
P(C_t^u = \alpha_t^u(k) | a_t^u) = logistic(z_t^u v_{\alpha_t^u(k)}) * (1-l)^{k-1} \prod_{j=1}^{k-1} (1-logistic(z_t^u v_{c_t^u}))
$$ 

Comparing with the likelihood of softmax model (eq.5):
$$
    P(C^u_t = c_t^u | \rho_t^u, a_t^u, c_{1:t-1}^u) = 
    \frac{exp( z_t^u \cdot v_{c^u_t} )}
        { \sum_{i \in \{a_t^u(\rho), 0\}} exp( z_t^u \cdot v_i)}
$$

In both models the probability of click of item $c_t^u$ is positively related to the relevance score $z_t^u v_c_t^u$, and negatively related to all others viewed.
A key difference in the cascade and in the softmax model, is that in the cascade model the relevance score of the no click option $(1-l)^{k-1}$ is dependent on the total scroll length.
This makes also intuitively more sense: the user has only so much patience to scroll through a slate of items. 

To elaborate on: The cost of this modeling choice is that the $\rho_t^u$ now depends on the items presented $a_t^u$ (need to write this out!). 
But does this matter? Since the leaving prob at each step is independent of previous items, it would be just as simple to find an optimal policy. Sorting by $z_t^u v_i, \forall i$ would still be optimal(?).

# Two item vector parameters [WORK IN PROGRESS]

Use a separate set of item vectors $\tilde{v}$

Result: No improvement

# Summary of Arnoldo from Turing meeting

Here is a ,list of what we think are, or at least can become, our contributions with this paper. Some depend on solving current problems, some depend on doing more work in specific directions.
1. We introduce groups of items. These are currently fixed, each containing up to about a few hundred items. This introduces a set of group parameters v_g, in addition to the item parameters v_i. The prior of the items in the same group g are Gaussian and have the same mean v_g. We think that this hierarchical structure of the prior will help in estimating the v_i’s and therefore lead to better recommendations.  We should think more on possible interesting priors, possibly involving the w2v estimates of the v_i’s: for example the prior mean of v_g could be the coordinate median of the w2v of the items in the group. We should also check if the posterior estimates of v_i at the end are such that the within group difference is smaller than the between group difference, ie that the estimated v_i fit with the group structure. A different aspect is to check if we should assume a priori some special property a prior for the W’s, for example that they have all eigenvalues smaller than 1 on absolute value, so that the RNN does not explode. Might not be essential.
2. A second contribution is that of being Bayesian. The posteriors of the v’s and W’s lead to a posterior of the score z_u times v_i, which s used by the Bandit to make a recommendation policy. Thompsom is an option, but we could think alternatives. The exploration should be able to make use of the uncertainty of these scores. We should also think here to the group structure: maybe it is better to recommend (explore) an item in a different group,  because the uncertainty allows it, instead than just items in the best scoring group. This has to do with diversity also. In synthesis: exploit more the posterior uncertainty in the policy.
3. We should show that the model with the actually seen slate (a_t^u) is better than the model which ignores this information. When we use the actually seen items, we need to also introduce the zero item, no click. These aspects are new. 
4. We should prepare an offline benchmark data, and make it available.
5. We make an online simulator, which models also the users and their clicking. This is maybe not so new, but still quite new, because the existing simulators are very complicated. 
6. Maybe: distinguish the slates which come from the recommender and those that come from the search. This requires an additional model, and maybe it is for another paper. 
--------------------------------
Where are we now:

- A. We see that the optimisation of the elbow (KL) is not doing as it should. It could be a bug, or something with the “restart” which is done after 40 times the whole data have been seen, or in the ministeps in combination with this. The data change, we think not too much, so the jumps, which are visible quite soon are strange. Or it is a mistake in the logic of the algorithm.  
B) We also see that the accuracy first improves and then deteriorates. This means that the estimated v’s and W’s are first good, but then get worse, and therefore the recommendations get bad. Why can this happen?
a) q is not a good variational model. Imagine that the posterior is very fine and fits the data, but the q is not able to approximate it. Therefore will the estimates of v and W be bad, or quite random, because the KL distance is quite flat (and bad) in v and W. Why is the accuracy first good and then gets worse? The optimisation is started in w2v, which is a good estimate of v’s. The steps of change of the v’s in the optimisation are small in the start, so the v’s remain good; and the W’s are estimated better and better, so that recommendations improve; but at a certain point, maybe when the stepsize of the v’s is increased, the v’s start to go their way, away from the w2v, because the q is bad, and does not guide them in the right direction. If this is the case, we would need to improve q. Maybe this has to do also with the items that are very little clicked on, where the q has to approximate the prior, while for the items where there is lots of clicks, the q has to approximate the likelihood. 
B. The likelihood is not appropriate for the data. The RNN is not good. Maybe because of the search, maybe because of the assumption we make that the user looks to the slate not sequentially (we could improve this with s cascade model where the user can click 0 at every step). Maybe the likelihood is too non-identifiable for the data, so that v’s and W’s are not uniquely estimable, even if we would run an MCMC (instead than variational). Here we would need to improve the likelihood. 
c) In the online experiments, the recommendations are made (99%) by another algorithm. If the suggested items are not the ones that our algorithms needs to see, to estimate the v’s and W’s well, then these parameters will not be estimated well. In the extreme, if the a_u^t are too similar (only popular items), then the model will not be able to estimate the v’s and W’s. Maybe this reason is not the most likely one, because the other algorithm is quite good. On the other hand, we have the group structure, and exploring various groups might be needed for us, to get estimates of the v_g. If all slates have only items from one group we would be in trouble.
d) We have slates that come from search, maybe quite many, but we do not model them. Maybe this is making the fit of the data to our likelihood difficult. Imagine an item=sofa that comes in the search v_i and is clicked while the recommender sofa’s are not…
e) In our prior of v_g we should have a variance that is larger than the prior of the v_i. This smaller prior, would keep the v_i’s more in the right place…. should be tried. 
We mentioned also one further possible test:
- keep v_i fixed to the w2v, instead of estimating them, and see if still the accuracy first goes up and then down
----------
The Simulation
1. Generating the data from our model. 
- use the w2v as the v_i’s (the v_g are not needed in the data generation)
- alternative: use the v_i that you get from the on-line runs, when the accuracy is best.
- use the same item as in the finn expereriement. take only a part of it , maybe 2 groups and 20 item per group, for example.
- For the a_u^t we have two options: one is to use the one we have on finn. Here you need to isolate only the few items you have decided to use. 
- alternative is to generate also the a_u^t. Here you can use either the true v’s and W’s (z) that you used above (so the w2v or the ones from the online experiment) and keep them fixed. Alternative is to use the current estimate of v’s and W’s in the data generation iterations. This second options has to be thought through, if it make sense. 
Then you can generate with our model the C_u^t. This gives the data. When you simulate the users, you use our model to flip their coins (the likelihood) that decides what to click. These probabilities should use the true v’s and W’s. (This is why i think also the a_u^t should be generated using the same true values). 
You need to keep some of them as training and some as test. Use our model on the training data to estimate the v’s and W’s and see if the estimates resembles the true ones used to generate the data.
When the data are generated, you estimate the v’s and W’s with our model (i) either using mcmc on the posterior or (ii) using the variational model with the q. And we see if (i) goes well, and if (ii) goes well. 
We should also implement with and without all your tricks (minibatches, 40 times etc) to see if they have an impact on estimates and accuracy. 
You should then run the model with the estimated parameters on the test data and see accuracy, using our policy, or another policy. (I think this is right, please check the logic of this task)
The simulation should scale with number of items and users, so that you can experiment with small and larger numbers, rare items etc. 


# Experiment: fixing item vectors and adding variable no click bias

- grey: included bias, fixed $v_i$
- orange: included bias, learning $v_i$
- Red: No bias, learning $v_i$
- Pink: No Bias, fixed $v_i$

![](assets/2019-11-27-11-28-08.png)

## Result on fixed $v_i$:
When fixing the $v_i$ vectors to their word2vec initializations, we no longer see the deteriorating performance after a number of iterations.

## Adding a no Click bias depending on scroll length:

Seems to give good results.
[todo]: write exactly how this was done.

$b_0(\rho_t^u), \rho_t^u \in \{0, 1, ..., max(\rho_t^u) \}$

i.e. we have a parameter that describes the bias for not clicking for each length of $\rho$.
- Assumes $max(\rho_t^u) = 42$.
  

## Online improvements due to fixing v and new bias term:
![](assets/2019-11-29-13-02-59.png)

# Simulator
![](assets/2019-11-29-01-31-26.png)

![](assets/2019-11-29-01-31-44.png)

![](assets/2019-11-29-01-31-55.png)
# Investigation of new pyro release

- Do I really understand the conditional independence (pyro.plate)? Could this affect the full model? INVESTIGATE!

# Re-read of minimin Chen: Top-K policy..

The model was much different than expected:
- Does not include no-clicks
- Uses separate '$v_i$'s in the RNN and in the candidate set: $v_i$ is in the softmax, but $u_i$ goes into the RNN function.
- The softmax is over all possible items $I$, not only the ones seen by the user: This should cause some weird biases.

# 
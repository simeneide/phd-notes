from torch.distributions import MultivariateNormal as MNorm
import torch
import torch.nn as nn
from scipy.stats import wishart
from spotlight.evaluation import rmse_score
class BayesianMF(nn.Module):
    def __init__(self, train, 
                 test,
                 num_users, 
                 num_items,
                 embedding_dim,
                 alpha = 2,
                 beta0 = 2
                ):
        
        super(BayesianMF, self).__init__()
        self.train = train
        if test:
            self.test = test
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.beta0 = beta0
        self.mu0 = torch.zeros(embedding_dim).view(-1,1) # defined in 3.1. think this actually is a vector, but can be treated as scalar due to broadcasting
        self.W0_inv = torch.eye(embedding_dim).inverse() # defined in 3.1
        
        self.U = torch.zeros(self.num_users, self.embedding_dim) # torch.randn((ds.num_users,d))
        self.V = torch.zeros(self.num_items, self.embedding_dim) #torch.randn((ds.num_items,d))
        
    def sample_hyperparam(self, M):
        K = M.size()[0] #  NUM_USERS/NUM_ITEMS
        df0_star = self.embedding_dim + K # eq 14
        beta0_star = self.beta0 + K # eq 14
        M_avg = M.mean(0).view(-1,1)
        mu0_star = (beta0_star*self.mu0 + K * M_avg) / (beta0_star) # eq 14
        mu0_star = mu0_star.double().view(-1)

        S_avg = M.transpose(0,1).mm(M)/K # eq 14

        W0_star_inv =  self.W0_inv + K*S_avg + self.beta0*K/(beta0_star) * ((self.mu0-M_avg) * (self.mu0 - M_avg).transpose(0,1)) # eq 14
        W0_star = W0_star_inv.inverse()

        lambda_M = wishart.rvs(df = df0_star, scale = W0_star)
        lambda_M = torch.tensor(lambda_M).double()
        #print(mu0_star[:2])

        covar = (lambda_M*beta0_star).inverse()

        mulvarNormal = MNorm(mu0_star, covariance_matrix=covar)


        mu_M = mulvarNormal.sample()
        return mu_M.float().view(-1,1), lambda_M.float()

    
    def get_vector_params(self, idx, mode, mu_K, lambda_K):
        if mode == "user":
            O = self.V
            data_idx = self.train.user_ids==idx
            vec = self.U[idx]
            other_obj = self.train.item_ids[data_idx]
        if mode == "item":
            O = self.U
            data_idx = self.train.item_ids==idx
            vec = self.V[idx]
            other_obj = self.train.user_ids[data_idx]

        Oj = O[other_obj,]
        r = self.train.ratings[data_idx]
        r = torch.tensor(r).view(-1,1)
        
        if len(r) == 0: # if no data return hyperparameters
            return mu_K, lambda_K
        ratings_for_idx = (vec * Oj).sum(1)

        # Calc lambda i star (eq 12)
        lambda_istar =  lambda_K + self.alpha*(Oj.transpose(0,1).mm(Oj))
        covar = lambda_istar.inverse()

        # Calc mu i star (eq 13)
        scoresum = self.alpha*(Oj*r).sum(0).view(-1,1)
        mu_istar = covar.mm(scoresum + lambda_K.mm(mu_K))
        return mu_istar, lambda_istar

    def sample_useritem_vector(self, idx, mode, mu_K, lambda_K):
        mu, prec = self.get_vector_params(idx, mode, mu_K, lambda_K)
        mulvarNormal = MNorm(mu.view(-1), precision_matrix=prec)
        return mulvarNormal.sample()
    
    def step_mcmc(self):
        mu_v, lambda_v = self.sample_hyperparam(self.V)
        mu_u, lambda_u = self.sample_hyperparam(self.U)
        
        for idx in range(self.num_users):
            self.U[idx,] = self.sample_useritem_vector(idx, "user", mu_u, lambda_u)
            
        for idx in range(self.num_items):
            self.V[idx,] = self.sample_useritem_vector(idx, "item", mu_v, lambda_v)
            
    def posterior_given_object(self, idx, mode):
        if mode == "user":
            mu_u, lambda_u = self.sample_hyperparam(self.U)
            return self.sample_useritem_vector(idx, "user", mu_u, lambda_u)
        if mode == "item":
            mu_v, lambda_v = self.sample_hyperparam(self.V)
            return self.sample_useritem_vector(idx, "item", mu_v, lambda_v)
        
    def posterior_score(self, userId, itemId, samples = 500):
        uservecs = torch.zeros((samples,self.embedding_dim))
        itemvecs = torch.zeros((samples,self.embedding_dim))
        
        for k in range(samples):
            uservecs[k] = self.posterior_given_object(userId, "user")
            itemvecs[k] = self.posterior_given_object(itemId, "item")
            
        scores = (uservecs*itemvecs).sum(1)
        return scores, uservecs, itemvecs
        
    def predict(self, user_ids, item_ids):
        uservec = self.U[user_ids]
        itemvec = self.V[item_ids]
        return (uservec * itemvec).sum(1)
    
    def fit(self, num_epochs, report_int=1):
        for t in range(num_epochs):
            self.step_mcmc()
                
            ## REPORTING ###
            if t%report_int == 0:
                rmse_train = rmse_score(self, self.train)
                rmse_test = rmse_score(self, self.test)
                print(f'step: {t} \t rmse train: {rmse_train:.2f}, test: {rmse_test:.2f}')
        return rmse_train, rmse_test
    

import torch
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pyro
import logging
import simulator
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
import pyro
import pyro.distributions as dist
import os
from torch.utils.tensorboard import SummaryWriter
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')
from pyro import poutine
import copy
import json

class PyroTrainer:
    def __init__(self,
                 model,
                 guide,
                 dataloaders,
                 max_epoch=100,
                 before_training_callbacks = [],
                 after_training_callbacks = [],
                 step_callbacks = [],
                 phase_end_callbacks = [],
                 **kwargs):
        self.model = model
        self.guide = guide
        self.dataloaders = dataloaders
        self.max_epoch = max_epoch
        self.step_callbacks = step_callbacks
        self.phase_end_callbacks = phase_end_callbacks
        self.after_training_callbacks = after_training_callbacks
        self.before_training_callbacks = before_training_callbacks

        for key, val in kwargs.items():
            setattr(self,key,val)

        self.step = 0  # global step counter (counts datapoints)
        self.epoch = 0

        self.writer = SummaryWriter(
            f'tensorboard/{kwargs.get("name", "default")}')

    def run_before_training_callbacks(self):
        for cb in self.before_training_callbacks:
            l = cb(self)

    def run_after_training_callbacks(self):
        for cb in self.after_training_callbacks:
            l = cb(self)

        self.writer.flush() #flush all logging to disk before we stop

    def run_step_callbacks(self, phase, batch):
        batch = {key: val.long().to(self.device) for key, val in batch.items()}
        # Special masking operation (move to dataloader?)
        if phase == "train":
            batch['phase_mask'] = batch['mask_train']
        else:
            batch['phase_mask'] = 1-batch['mask_train']

        # EXECUTES ALL CALLBACKS RELATED TO EACH STEP BATCH
        tmp_log = {}
        tmp_log['num_obs'] = (batch['phase_mask']).sum()

        for cb in self.step_callbacks:
            l = cb(self, phase=phase, batch=batch)

            for key, val in l.items():
                tmp_log[key] = val

        return tmp_log

    def run_phase_end_callbacks(self, phase, logs):
        for cb in self.phase_end_callbacks:
            l = cb(self, phase=phase, logs=logs)


    def fit(self):
        # Initialize an epoch log
        self.epoch_log = list()
        self.run_before_training_callbacks()

        while self.epoch <= self.max_epoch:
            self.epoch_log.append({'epoch': self.epoch, 'stop': False})
            logging.info("")
            logging.info('-' * 10)
            logging.info(f'Epoch {self.epoch}/{self.max_epoch} \t Step {self.step}')

            for phase, dl in self.dataloaders.items():
                logs = []
                batch_size = dl.batch_size  # assume that batch size is constant (small error in last step)

                for batch in dl:
                    tmp_log = self.run_step_callbacks(phase, batch)
                    
                    if phase == "train":
                        self.step += batch_size

                    # Add tmp log to log list for phase
                    logs.append(tmp_log)

                self.run_phase_end_callbacks(phase=phase, logs=logs)

            
            if self.epoch_log[-1]['stop']:
                logging.info('Training stopped: A callback wanted early stopping.')
                break
            self.epoch += 1

        self.run_after_training_callbacks()

#######
## CALLBACKS
#######

class VisualizeEmbeddings:
    def __init__(self, sim=None):
        self.sim = sim

    def __call__(self, trainer, **kwargs):
        # Visualize item vectors:
        # %% PLOT OF H0 parameters of users
        if trainer.model.user_init:
            num_plot_users = 1000
            h0 = pyro.param("h0-mean").detach().cpu()[:num_plot_users]  
            
            if self.sim:
                usergroup=self.sim.env.user_init_group[:num_plot_users].cpu()
            else:
                usergroup=None

            trainer.writer.add_embedding(tag="h0",mat= h0, metadata=usergroup, global_step=trainer.step)

        # %% PLOT OF item vector parameters
        V = pyro.param('item_model.itemvec.weight-mean').detach().cpu()
        trainer.writer.add_embedding(tag="V",mat= V, metadata=trainer.model.item_model.item_group.cpu(), global_step=trainer.step)

class RewardComputation:
    def __init__(self, param, test_sim):
        self.param = param
        self.calc_footrule=False
        self.test_sim = test_sim

    def __call__(self, trainer):
        ### SIMULATE REWARDS..:
        rec_types = {
            'thompson' : trainer.model.recommend,
            'inslate' : trainer.model.recommend_inslate
            }

        for rectype, recommend_func in rec_types.items():
            logging.info(f"compute {rectype} reward..")
            t_start = self.param.get("t_testsplit")
            current_data = copy.deepcopy(trainer.dataloaders['train'].dataset.data)
            
            # zero pad the period we want to test:
            for key, val in current_data.items():
                if key not in ["userId", "mask_train"]:
                    current_data[key][:,t_start:] = 0

            self.test_sim.reset_data(data=current_data)
            all_rewards = self.test_sim.play_game(
                recommend_func,
                par=lambda batch: trainer.guide(batch, temp=1.0),
                userIds = current_data['userId'],
                t_start=t_start
                )

            train_mask = current_data['mask_train'][:,t_start:]

            reward_train_timestep = (all_rewards * train_mask).sum(0) / train_mask.sum(0)
            reward_test_timestep = (all_rewards * (1 - train_mask)).sum(0) / (1 - train_mask).sum(0)
            trainer.epoch_log[-1][f'train/reward-{rectype}'] = reward_train_timestep.mean()

            trainer.epoch_log[-1][f'valid/reward-{rectype}'] = reward_test_timestep.mean()

            # log per timestep:
            for i in range(all_rewards.size()[1]):
                trainer.writer.add_scalar(f"reward_time/train-{rectype}", reward_train_timestep[i], global_step=i+t_start)
            for i in range(all_rewards.size()[1]):
                trainer.writer.add_scalar(f"reward_time/test-{rectype}", reward_test_timestep[i], global_step=i+t_start)

            u = 1
            anipath = AnimatePath(sim = self.test_sim, model=trainer.model, guide = trainer.guide, num_samples=100)
            for t in [0, 5, 10, 15, 19]:
                anipath.step(t=t, u = u)
                trainer.writer.add_figure(f"visualize-path-{rectype}", plt.gcf(), global_step=t)

        ### CALC FOOTRULE DISTANCE BETWEEN REAL AND ESTIMATED RECS:
        # calc for all data at last timestep:
        if self.calc_footrule:
            logging.info("Compute FOOTRULE distance..")
            num_items = self.param['num_items']-3
            argsort_real = trainer.model.recommend(trainer.dataloaders['train'].dataset.data, par="real", num_rec = num_items)
            argsort_estimated = trainer.model.recommend(trainer.dataloaders['train'].dataset.data, par=trainer.guide, num_rec = num_items)
            rank_real = argsort2rank_matrix(argsort_real.long(), num_items = num_items+3)
            rank_estimated = argsort2rank_matrix(argsort_estimated.long(), num_items = num_items+3)
            
            train_idx = train_mask[:,-1]
            trainer.epoch_log[-1][f'train/footrule'] = dist_footrule(rank_real[train_idx.bool()], rank_estimated[train_idx.bool()])
            trainer.epoch_log[-1][f'valid/footrule'] = dist_footrule(rank_real[(1-train_idx).bool()], rank_estimated[(1-train_idx).bool()])

class ReportHparam:
    def __init__(self, hyperparam):
        self.hyperparam = hyperparam

    def __call__(self, trainer, **kwargs):
        ### Add hyperparameters and final metrics to hparam:
        serialized_param = json.loads(json.dumps(self.hyperparam, default=str))
        trainer.writer.add_hparams(hparam_dict=serialized_param,
                                metric_dict=trainer.epoch_log[-1])



@torch.no_grad()
def checksum_data(trainer, **kwargs):
    checksum = sum([val.float().mean() for key, val in trainer.dataloaders['train'].dataset.data.items()])
    trainer.writer.add_scalar("data_checksum", checksum, global_step=0)

class ReportPyroParameters:
    def __init__(self, report_param_histogram=False):
        self.report_param_histogram = report_param_histogram

    def __call__(self, trainer, phase, **kwargs):
        if phase == "train":
            # Report all parameters
            try:
                gamma = pyro.param("gamma-mean")
                if len(gamma)>1:
                    for i in range(len(gamma)):
                        trainer.writer.add_scalar(tag=f"param/gamma_{self.ind2val['displayType'][i]}", scalar_value = gamma[i], global_step=self.step)
            except:
                pass
            for name, par in pyro.get_param_store().items():
                trainer.writer.add_scalar(tag=f"param/{name}-l1",
                                        scalar_value=par.abs().mean(),
                                        global_step=trainer.step)
                if self.report_param_histogram:
                    trainer.writer.add_histogram(tag=f"param/{name}",
                                                values=par,
                                                global_step=trainer.step)

class Simulator_batch_stats:
    def __init__(self, sim, **kwargs):
        self.sim = sim

    @torch.no_grad()
    def __call__(self, trainer, phase, batch):
        stats = {}
        res_hat = pyro.condition(lambda batch: trainer.model(batch),
                                data=trainer.guide(batch, temp=0.0001))(batch)


        res = self.sim.env.likelihood(batch)

        # Compute probabilities
        score2prob = lambda s: s.exp() / (s.exp().sum(2, keepdims=True))
        res['prob'] = score2prob(res['score'])
        res_hat['prob_hat'] = score2prob(res_hat['score'])

        prob_mae_unmasked = (res['prob'] - res_hat['prob_hat'])
        masked_prob = (batch['phase_mask']).unsqueeze(2)*prob_mae_unmasked
        stats['prob-mae'] = masked_prob.abs().sum()
        return stats

@torch.no_grad()
def calc_batch_stats(trainer, phase, batch):
    stats = {}
    
    ## LIKELIHOODS
    
    guide_mode = lambda *args, **kwargs: trainer.guide(temp=0.0001, *args, **kwargs)
    guide_trace = poutine.trace(guide_mode).get_trace(batch)
    model_with_guidepar = poutine.replay(trainer.model, trace=guide_trace)
    model_trace = poutine.trace(model_with_guidepar).get_trace(batch)
    model_trace.compute_log_prob()
    guide_trace.compute_log_prob()
    logguide = guide_trace.log_prob_sum().item()
    stats['loglik'] = model_trace.nodes['obs']['log_prob'].sum()
    totlogprob = model_trace.log_prob_sum().item()
    logprior = totlogprob - stats['loglik']
    stats['KL_pq'] = logguide - logprior
    stats['elbo'] = stats['loglik'] - stats['KL_pq']
    return stats


# 
@torch.no_grad()
def report_phase_end(trainer, phase, logs, **kwargs):
    """ Function that reports all values (scalars) in logs to trainer.writer"""
    keys = logs[0].keys()  # take elements of first dict and they are all equal
    summed_stats = {key: sum([l[key] for l in logs]) for key in keys}

    # If there exist a num obs, use that. Otherwise use number of batches
    num = summed_stats.get("num_obs", len(logs)).float()
    # Add summed stats to epoch_log:
    for key, val in summed_stats.items():
        if key =="num_obs":
            trainer.epoch_log[-1][f"{phase}/{key}"] = val
        else:
            trainer.epoch_log[-1][f"{phase}/{key}"] = val / num

    # Report epoch log to tensorboard:
    for key, val in trainer.epoch_log[-1].items():
        if phase in key: # only report if correct phase
            trainer.writer.add_scalar(tag=key,
                                    scalar_value=val,
                                    global_step=trainer.step)
    
    logging.info(f"phase: {phase} \t loss: {trainer.epoch_log[-1][f'{phase}/loss']:.1f}")


class SviStep:
    def __init__(
        self, 
        model, 
        guide,
        learning_rate=1e-2,
        device = "cpu",
        **kwargs
        ):
        self.model = model
        self.guide = guide
        self.learning_rate = learning_rate
        self.device = device

        self.init_opt()

    def init_opt(self):
        logging.info(f"Initializing default Adam optimizer with lr={self.learning_rate}")
        self.svi = pyro.infer.SVI(model=self.model,
                                  guide=self.guide,
                                  optim=pyro.optim.Adam({"lr": self.learning_rate}),
                                  loss=pyro.infer.TraceMeanField_ELBO())
        return True

    def __call__(self, trainer, phase, batch):
        stats = {}
        if phase == "train":
            stats['loss'] = self.svi.step(batch)
        else:
            stats['loss'] = self.svi.evaluate_loss(batch)
        
        stats['num_obs'] = (batch['phase_mask']).sum()
        return stats

class EarlyStoppingAndCheckpoint:
    def __init__(
        self, 
        stopping_criteria,
        patience=1, 
        save_dir="checkpoints", 
        name = "parameters", 
        **kwargs):
        self.stopping_criteria = stopping_criteria
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.best_loss = None
        self.patience = patience
        self.counter = 0
        self.path = f"{self.save_dir}/{name}.pyro"

    def __call__(self, trainer, phase, logs, **kwargs):
        if phase != "train":
            loss = trainer.epoch_log[-1][self.stopping_criteria]
            if (self.best_loss is None):
                self.best_loss = loss
                self.save_checkpoint()
                self.counter = 0
                return False
            elif loss < self.best_loss:
                self.best_loss = loss
                self.save_checkpoint()
                self.counter = 0
                return False
                
            elif loss >= self.best_loss:
                self.counter += 1
                if self.counter > self.patience:
                    logging.info(f"REACHED EARLY STOPPING ON EPOCH {trainer.epoch}")
                    self.load_checkpoint()
                    trainer.epoch_log[-1]['stop'] = True
                    return True

    def save_checkpoint(self):
        logging.info(f"Saving model to {self.path}..")
        pyro.get_param_store().save(self.path)

    def load_checkpoint(self):
        logging.info(f"Loading latest checkpoint from {self.path}.. (+ clear param store first)")
        pyro.clear_param_store()
        pyro.get_param_store().load(self.path)


class PlotFinnAdsRecommended:
    def __init__(self, ind2val, epoch_interval=10):
        import FINNPlot
        self.ind2val = ind2val
        self.idx = 10
        self.num_recs=5
        self.num_time=10
        self.epoch_interval = epoch_interval

    def __call__(self,trainer, phase, **kwargs):
        
        if (phase == "train") & (trainer.epoch % self.epoch_interval == 0):
            import FINNPlot
            smallbatch = {key: val[self.idx].unsqueeze(0).to(trainer.device).long() for key, val in trainer.dataloaders['train'].dataset.data.items()}
            M = torch.zeros(self.num_recs+1, self.num_time)
            M[0,:] = smallbatch['click'].flatten()[:self.num_time] # add view to first row
            guide = lambda *args, **kwargs: trainer.guide(*args, **kwargs, temp=0)
            for t_rec in range(self.num_time):
                M[1:, t_rec] = trainer.model.recommend(smallbatch, par= guide, num_rec=self.num_recs, t_rec=t_rec)

            nrow = M.size()[1]
            finnkoder = [self.ind2val['itemId'][r.item()] for r in M.flatten()]
            img_tensor=FINNPlot.add_image_line(finnkoder, nrow=nrow)
            trainer.writer.add_image("recs",img_tensor=img_tensor, global_step=trainer.step)        

#%%

def plotM(M, type = "scatter", **kwargs):
    if type == "line":
        p = sns.lineplot
    else:
        p = sns.scatterplot
    return p(M[:,0], M[:,1], **kwargs)


class AnimatePath:
    def __init__(self, sim, model, guide, num_samples=1):
        self.sim = sim
        self.num_samples = num_samples
        self.sim.data['phase_mask'] = torch.ones_like(sim.data['click'])
        self.pars = []
        for _ in range(self.num_samples):
            output = model.likelihood(self.sim.data, par=guide(self.sim.data, temp=1.0))
            output = {key : val.detach().cpu() for key, val in output.items()}
            self.pars.append(output)

    def step(self, t, u):

        # for each time step:
        action = self.sim.data['action'][u,t].cpu()
        click = self.sim.data['click'][u,t].unsqueeze(0).cpu()

        # Plot all items:
        p = plotM(self.pars[0]['V'], alpha = 0.1)
        # Plot all recommended actions:
        #plotM(self.pars[0]['V'][action], color =['yellow'], alpha = 0.5)
        #print(action.size())
        for i in range(self.num_samples):
            plotM(self.pars[i]['V'][action], color =['yellow'], alpha = 1.0 if i==0 else 0.5)

            plotM(self.pars[i]['V'][click], color = ['red'], alpha=0.9)

            zt = self.pars[i]['zt'][u,t].unsqueeze(0).cpu()
            plotM(zt, color = ['blue'], alpha=0.5)

        # Plot corresponding click:
        
        plt.legend(labels=['all items', 'recommended items', 'click item', 'zt (estimated)'])
        plt.text(x = 0.8, y=0.8, s = f"t={t}")


### FOOTRULE FUNCTIONS
def argsort2rank(idx, num_items=None):
    # create rank vector from the indicies that returns from torch.argsort()
    if num_items is None:
        num_items = len(idx)
    rank = torch.zeros((num_items,)).long()
    rank[idx] = torch.arange(0,len(idx))
    return rank

def argsort2rank_matrix(idx_matrix, num_items=None):
    batch_size, nc = idx_matrix.size()
    if num_items is None:
        num_items = nc

    rank = torch.zeros((batch_size, num_items)).long()

    for i in range(batch_size):
        rank[i,:] = argsort2rank(idx_matrix[i,:], num_items=num_items)
    return rank

def dist_footrule(r1, r2):
    return (r1-r2).abs().float().mean()

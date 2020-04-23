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

class PyroTrainer:
    def __init__(self,
                 model,
                 guide,
                 dataloaders,
                 learning_rate=1e-2,
                 max_epoch=100,
                 param=None,
                 report_param_histogram=False,
                 calc_footrule=True,
                 **kwargs):
        self.model = model
        self.guide = guide
        self.dataloaders = dataloaders
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.param = param
        self.report_param_histogram = report_param_histogram
        self.calc_footrule = calc_footrule
        self.init_opt(lr=self.learning_rate)
        self.step = 0  # global step counter (counts datapoints)
        self.earlystop = EarlyStoppingAndCheckpoint(**kwargs)
        self.writer = SummaryWriter(
            f'tensorboard/{kwargs.get("name", f"lr-{self.learning_rate}/")}')

    def init_opt(self, lr=1e-2):
        logging.info(f"Initializing default Adam optimizer with lr{lr}")
        self.svi = pyro.infer.SVI(model=self.model,
                                  guide=self.guide,
                                  optim=pyro.optim.Adam({"lr": lr}),
                                  loss=pyro.infer.TraceMeanField_ELBO())
        return True

    def training_step(self, batch):
        loss = self.svi.step(batch)
        return {'loss': loss}

    @torch.no_grad()
    def validation_step(self, batch):
        loss = self.svi.evaluate_loss(batch)
        return {'loss': loss}

    def phase_end(self, logs, ep, phase, **kwargs):
        keys = logs[0].keys()  # take elements of first dict and they are all equal
        summed_stats = {key: sum([l[key] for l in logs]) for key in keys}

        # If there exist a num obs, use that. Otherwise use number of batches
        num = summed_stats.get("num_obs", len(logs)).float()
        # Add summed stats to epoch_log:
        for key, val in summed_stats.items():
            if key =="num_obs":
                self.epoch_log[-1][f"{phase}/{key}"] = val
            else:
                self.epoch_log[-1][f"{phase}/{key}"] = val / num

        # Report epoch log to tensorboard:
        for key, val in self.epoch_log[-1].items():
            self.writer.add_scalar(tag=key,
                                   scalar_value=val,
                                   global_step=self.step)

        logging.info(f"phase: {phase} \t loss: {summed_stats['loss']:.1f}")

        if phase == "train":
            # Report all parameters
            for name, par in pyro.get_param_store().items():
                self.writer.add_scalar(tag=f"param/{name}-l1",
                                       scalar_value=par.abs().mean(),
                                       global_step=self.step)
                if self.report_param_histogram:
                    self.writer.add_histogram(tag=f"param/{name}",
                                              values=par,
                                              global_step=self.step)

    def end_of_training(self, *args, **kwargs):
        pass

    def run_training_epochs(self, dataloaders):
        for ep in range(1, self.max_epoch + 1):
            self.epoch_log.append({'epoch': ep})
            logging.info("")
            logging.info('-' * 10)
            logging.info(f'Epoch {ep}/{self.max_epoch} \t Step {self.step}')
            #

            for phase, dl in dataloaders.items():
                logs = []
                batch_size = dl.batch_size  # assume that batch size is constant (small error in last step)

                for batch in dl:
                    if phase == "train":
                        tmp_log = self.training_step(batch)
                        self.step += batch_size
                    else:
                        tmp_log = self.validation_step(batch)
                    # Add tmp log to log list for phase
                    logs.append(tmp_log)

                self.phase_end(logs=logs, phase=phase, ep=ep)

            # EARLY STOPPING CHECK
            stop = self.earlystop(epoch=ep,
                                  loss=self.epoch_log[-1]['train/loss'])
            if stop:
                logging.info('Early stopping criteria reached.')
                return None

    def fit(self):
        """ 
        Dataloaders is a dict of dataloaders:
        {'train' : Dataloader, 'valid': Dataloader}
        """
        # Initialize an epoch log
        self.epoch_log = list()

        self.run_training_epochs(self.dataloaders)

        self.end_of_training()


class RecTrainer(PyroTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sim = kwargs.get("sim")
        self.device = kwargs.get("device", "cpu")
        if self.sim:
            self.checksum_data()

    @torch.no_grad()
    def checksum_data(self):
        checksum = sum([val.float().mean() for key, val in self.dataloaders['train'].dataset.data.items()])
        self.writer.add_scalar("data_checksum", checksum, global_step=0)

    def training_step(self, batch):
        batch = {key: val.long().to(self.device) for key, val in batch.items()}
        batch['phase_mask'] = batch['mask_train']
        loss = self.svi.step(batch)
        stats = self.calc_stats(batch)

        stats['loss'] = loss
        stats['num_obs'] = (batch['phase_mask']).sum()
        return stats

    @torch.no_grad()
    def validation_step(self, batch):
        batch = {key: val.long().to(self.device) for key, val in batch.items()}
        batch['phase_mask'] = 1-batch['mask_train']
        loss = self.svi.evaluate_loss(batch)
        stats = self.calc_stats(batch)
        stats['loss'] = loss
        stats['num_obs'] = (batch['phase_mask']).sum()
        return stats

    @torch.no_grad()
    def calc_stats(self, batch):
        stats = {}

        if self.sim: # only compute if simulator exist
            
            res_hat = pyro.condition(lambda batch: self.model(batch),
                                    data=self.guide(batch, temp=0.00))(batch)


            res = self.sim.env.likelihood(batch)

            # Compute probabilities
            score2prob = lambda s: s.exp() / (s.exp().sum(2, keepdims=True))
            res['prob'] = score2prob(res['score'])
            res_hat['prob_hat'] = score2prob(res_hat['score'])

            # report stats on batch:
            #{f"{key}-L1": val.abs().mean() for key, val in res_hat.items()}
            
            
            prob_mae_unmasked = (res['prob'] - res_hat['prob_hat'])
            masked_prob = (batch['phase_mask']).unsqueeze(2)*prob_mae_unmasked
            stats['prob-mae'] = masked_prob.abs().sum()

        ## LIKELIHOODS
        
    
        guide_trace = poutine.trace(self.guide).get_trace(batch)
        model_with_guidepar = poutine.replay(self.model, trace=guide_trace)
        model_trace = poutine.trace(model_with_guidepar).get_trace(batch)
        model_trace.compute_log_prob()
        guide_trace.compute_log_prob()
        stats['logguide'] = guide_trace.log_prob_sum().item()
        stats['loglik'] = model_trace.nodes['obs']['log_prob'].sum()
        stats['totlogprob'] = model_trace.log_prob_sum().item()
        stats['logprior'] = stats['totlogprob'] - stats['loglik']
        stats['KL_pq'] = stats['logprior'] - stats['logguide']
        stats['elbo'] = stats['loglik'] + stats['KL_pq']
        return stats

    def end_of_training(self):
        logging.info("Running end of training stats..")
        # calculate rewards:
        if self.sim:
            ### SIMULATE REWARDS..:
            rec_types = {
                'inslate' : self.model.recommend,
                'thompson' : self.model.recommend_inslate}

            for rectype, recommend_func in rec_types.items():
                logging.info(f"compute {rectype} reward..")
                t_start = 10
                current_data = copy.deepcopy(self.dataloaders['train'].dataset.data)
                
                # zero pad the period we want to test:
                for key, val in current_data.items():
                    if key not in ["userId", "mask_train"]:
                        current_data[key][:,t_start:] = 0

                self.sim.reset_data(data=current_data)
                all_rewards = self.sim.play_game(
                    recommend_func,
                    par=self.guide,
                    userIds = current_data['userId'],
                    t_start=t_start
                    )

                train_mask = current_data['mask_train'][:,t_start:]
                self.epoch_log[-1][f'train/reward-{rectype}'] = (
                    all_rewards * train_mask).sum() / train_mask.sum()

                self.epoch_log[-1][f'valid/reward-{rectype}'] = (
                    all_rewards * (1 - train_mask)).sum() / (1 - train_mask).sum()

            ### CALC FOOTRULE DISTANCE BETWEEN REAL AND ESTIMATED RECS:
            # calc for all data at last timestep:
            if self.calc_footrule:
                logging.info("Compute FOOTRULE distance..")
                num_items = self.param['num_items']-3
                argsort_real = self.model.recommend(self.dataloaders['train'].dataset.data, par="real", num_rec = num_items)
                argsort_estimated = self.model.recommend(self.dataloaders['train'].dataset.data, par=self.guide, num_rec = num_items)
                rank_real = argsort2rank_matrix(argsort_real.long(), num_items = num_items+3)
                rank_estimated = argsort2rank_matrix(argsort_estimated.long(), num_items = num_items+3)
                
                train_idx = train_mask[:,-1]
                self.epoch_log[-1][f'train/footrule'] = dist_footrule(rank_real[train_idx.bool()], rank_estimated[train_idx.bool()])
                self.epoch_log[-1][f'valid/footrule'] = dist_footrule(rank_real[(1-train_idx).bool()], rank_estimated[(1-train_idx).bool()])


        ### Add hyperparameters and final metrics to hparam:
        self.writer.add_hparams(hparam_dict=self.param,
                                metric_dict=self.epoch_log[-1])

        # Visualize item vectors:
        # %% PLOT OF H0 parameters of users
        num_plot_users = 1000
        h0 = pyro.param("h0-mean").detach().cpu()[:num_plot_users]  
        
        fig = plt.figure()
        plt.scatter(h0[:, 0],
                    h0[:, 1],
                    c=self.sim.env.user_init[:num_plot_users].cpu(),
                    alpha=0.1)
        self.writer.add_figure('h0', fig, 0)
        
        # %% PLOT OF item vector parameters
        V = pyro.param('item_model.itemvec.weight-mean').detach().cpu()
        fig = plt.figure()
        plt.scatter(V[:, 0], V[:, 1], c=self.model.item_model.item_group.cpu())
        self.writer.add_figure('V', fig, 0)




class EarlyStoppingAndCheckpoint:
    def __init__(self, patience=1, save_dir="checkpoints", name = "parameters", **kwargs):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.best_loss = None
        self.patience = patience
        self.counter = 0
        self.path = f"{self.save_dir}/{name}.pyro"

    def __call__(self, epoch, loss):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint()
            return False
        elif loss >= self.best_loss:
            self.counter += 1
            if self.counter > self.patience:
                logging.info(f"REACHED EARLY STOPPING ON EPOCH {epoch}")
                self.load_checkpoint()
                return True
        elif loss < self.best_loss:
            self.best_loss = loss
            self.save_checkpoint()
            self.counter = 0
            return False

    def save_checkpoint(self):
        logging.info(f"Saving model to {self.path}..")
        pyro.get_param_store().save(self.path)

    def load_checkpoint(self):
        logging.info(f"Loading latest checkpoint from {self.path}.. (+ clear param store first)")
        pyro.clear_param_store()
        pyro.get_param_store().load(self.path)



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

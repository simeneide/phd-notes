import torch
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pyro
import logging
import simulator
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.nn import functional as F
import pyro
import pyro.distributions as dist
import os
from torch.utils.tensorboard import SummaryWriter
logging.basicConfig(format='%(asctime)s %(message)s', level='INFO')

class PyroTrainer:
    def __init__(self, model, guide, learning_rate = 1e-2, max_epoch=100, param=None, **kwargs):
        self.model = model
        self.guide = guide
        self.learning_rate = learning_rate
        self.max_epoch  = max_epoch
        self.param = param

        self.init_opt(self.learning_rate)
        self.step = 0 # global step counter (counts datapoints)
    
        self.earlystop = EarlyStoppingAndCheckpoint(**kwargs)
        self.writer = SummaryWriter( f'tensorboard/{kwargs.get("name", f"lr-{self.learning_rate}/")}')

    def init_opt(self, lr=1e-2):
        logging.info(f"Initializing default Adam optimizer with lr{lr}")
        self.svi = pyro.infer.SVI(
            model=self.model,
            guide=self.guide,
            optim=pyro.optim.Adam({"lr": lr}),
            loss=pyro.infer.TraceMeanField_ELBO())
        return True

    def training_step(self, batch):
        loss = self.svi.step(batch)
        return {'loss' : loss}

    @torch.no_grad()
    def validation_step(self, batch):
        loss = self.svi.evaluate_loss(batch)
        return {'loss' : loss}

    def phase_end(self, logs, ep, phase, **kwargs):
        keys = logs[0].keys() # take elements of first dict and they are all equal
        summed_stats = {key : sum([l[key] for l in logs]) for key in keys}
        
        num_batches = len(logs)
        # Add summed stats to epoch_log:
        for key, val in summed_stats.items():
            self.epoch_log[-1][f"{phase}/{key}"] = val / num_batches

        # Report epoch log to tensorboard:
        for key, val in self.epoch_log[-1].items():
            self.writer.add_scalar(tag=key, scalar_value= val, global_step=self.step)
        
        logging.info(f"phase: {phase} \t loss: {summed_stats['loss']:.1f}")

        if phase == "train":
            # Report all parameters
            for name, par in pyro.get_param_store().items():
                self.writer.add_histogram(tag=f"param/{name}", values=par, global_step=self.step)
                self.writer.add_scalar(tag=f"param/{name}-l1", scalar_value = par.abs().mean(), global_step = self.step)

    def end_of_training(self, *args, **kwargs):
        pass

    def run_training_epochs(self, dataloaders):
        for ep in range(1, self.max_epoch+1):
            self.epoch_log.append({'epoch' : ep})
            logging.info("")
            logging.info('-' * 10)
            logging.info(f'Epoch {ep}/{self.max_epoch} \t Step {self.step}')
            #

            for phase, dl in dataloaders.items():
                logs = []
                batch_size = dl.batch_size # assume that batch size is constant (small error in last step)

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
            stop = self.earlystop(epoch=ep, loss= self.epoch_log[-1]['train/loss'])
            if stop:
                logging.info('Early stopping criteria reached.')
                return None

    def fit(self, dataloaders):
        """ 
        Dataloaders is a dict of dataloaders:
        {'train' : Dataloader, 'valid': Dataloader}
        """
        # Initialize an epoch log
        self.epoch_log = list()
        
        self.run_training_epochs(dataloaders)

        self.end_of_training()




class RecTrainer(PyroTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def training_step(self, batch):
        self.model.train()
        loss = self.svi.step(batch)
        stats = self.calc_stats(batch)
        stats['loss'] = loss
        return stats

    @torch.no_grad()
    def validation_step(self, batch):
        self.model.eval()
        loss = self.svi.evaluate_loss(batch)
        stats = self.calc_stats(batch)
        stats['loss'] = loss
        return stats

    def calc_stats(self, batch):
        res_hat = pyro.condition(
            lambda batch: self.model.forward(batch),
            data = self.guide(batch))(batch)

        res = self.model.likelihood(batch)

        # Compute probabilities
        score2prob = lambda s: s.exp()/(s.exp().sum(2,keepdims=True))
        res['prob'] = score2prob(res['score'])
        res_hat['prob_hat'] = score2prob(res_hat['score'])

        # report stats on batch:
        stats = {f"{key}-L1" : val.abs().mean() for key,val in res_hat.items()}
        stats['prob-mae'] = (res['prob']-res_hat['prob_hat']).abs().mean()
        return stats

    def end_of_training(self):
        logging.info("Running end of training stats..")
        # Add hyperparameters and final metrics to hparam:
        self.writer.add_hparams(hparam_dict=self.param, metric_dict=self.epoch_log[-1])

        # Visualize item vectors:
        # %% PLOT OF H0 parameters of users
        h0 = self.guide.get_parameters()['h0-batch']['mean'].detach()
        fig = plt.figure()
        plt.scatter(h0[:,0], h0[:,1], c=self.model.user_init, alpha = 0.2)
        self.writer.add_figure('h0', fig, 0)

        # %% PLOT OF item vector parameters
        V = self.guide.get_parameters()['item_model.itemvec.weight']['mean'].detach()
        fig = plt.figure()
        plt.scatter(V[:,0], V[:,1], c = self.model.item_model.item_group)
        self.writer.add_figure('V', fig, 0)

class EarlyStoppingAndCheckpoint:
    def __init__(self, patience = 1, save_dir="checkpoints", **kwargs):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.best_loss = None
        self.patience = patience
        self.counter = 0

    def __call__(self, epoch, loss):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint()
            return False
        elif loss >= self.best_loss:
            self.counter += 1
            if self.counter > self.patience:
                logging.info(f"REACHED EARLY STOPPING ON EPOCH {epoch}")
                return True
        elif loss < self.best_loss:
            self.best_loss = loss
            self.save_checkpoint()
            self.counter = 0
            return False

    def save_checkpoint(self):
        path = f"{self.save_dir}/parameters.pyro"
        logging.info(f"Saving model to {path}..")
        pyro.get_param_store().save(path)
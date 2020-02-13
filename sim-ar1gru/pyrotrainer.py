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
    def __init__(self, model, guide, learning_rate = 1e-2, max_epoch=100, **kwargs):
        self.model = model
        self.guide = guide
        self.learning_rate = learning_rate
        self.max_epoch  = max_epoch

        self.init_opt(self.learning_rate)
        self.step = 0 # global step counter (counts datapoints)
    
        self.earlystop = EarlyStoppingAndCheckpoint(**kwargs)
        self.writer = SummaryWriter(kwargs.get("tensorboard_dir", f"tensorboard/lr-{self.learning_rate}/"))

    def init_opt(self, lr=1e-2):
        logging.info(f"Initializing default Adam optimizer with lr{lr}")
        self.svi = pyro.infer.SVI(
            model=self.model,
            guide=self.guide,
            optim=pyro.optim.Adam({"lr": lr}),
            loss=pyro.infer.Trace_ELBO())

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

        # Add summed stats to epoch_log:
        for key, val in summed_stats.items():
            self.epoch_log[ep][f"{phase}/{key}"] = val

        # Report epoch log to tensorboard:
        for key, val in self.epoch_log[ep].items():
            self.writer.add_scalar(tag=key, scalar_value= val, global_step=self.step)
            print(key,val)
        
        logging.info(f"phase: {phase} \t loss: {summed_stats['loss']:.1f}")

    def fit(self, dataloaders):
        """ 
        Dataloaders is a dict of dataloaders:
        {'train' : Dataloader, 'valid': Dataloader}
        """
        # Initialize an epoch log
        self.epoch_log = {}

        for ep in range(1, self.max_epoch+1):
            self.epoch_log[ep] = {}
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
            stop = self.earlystop(epoch=ep, loss= self.epoch_log[ep]['valid/loss'])
            if stop:
                logging.info('Early stopping criteria reached.')
                return None


class RecTrainer(PyroTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def training_step(self, batch):
        loss = self.svi.step(batch)
        return {'loss' : loss}

    @torch.no_grad()
    def validation_step(self, batch):
        loss = self.svi.evaluate_loss(batch)
        return {'loss' : loss}
    def calc_stats(self):
        res_hat = pyro.condition(
            lambda batch: self.model.forward(batch),
            data= system.guide(batch))(batch)

        res = self.model.likelihood(batch)
        
        # general mean values:
        stats = {f"{key}-L1" : val.abs().mean() for key,val in res_hat.items()}
        
        score2prob = lambda s: (s.exp()/s.exp().sum(2,keepdims=True))
        stats['outputs/score-mae'] = (score2prob(res['score'])-score2prob(res_hat['score'])).abs().mean()
        return stats

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
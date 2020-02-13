import torch
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pyro
import logging
import model_gamma
import simulator
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.nn import functional as F
import pyro
import pyro.distributions as dist

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
    def init_opt(self, lr=1e-2):
        logging.info(f"Initializing default Adam optimizer with lr{lr}")
        self.svi = pyro.infer.SVI(
            model=self.model,
            guide=self.guide,
            optim=pyro.optim.Adam({"lr": lr}),
            loss=pyro.infer.Trace_ELBO())

        return [self.svi]

    def training_step(self, batch):
        loss = self.svi.step(batch)
        return {'loss' : loss}

    @torch.no_grad()
    def validation_step(self, batch):
        loss = self.svi.evaluate_loss(batch)
        return {'loss' : loss}

    def training_end(self, logs, **kwargs):
        keys = logs[0].keys()
        summed_stats = {key : sum([l[key] for l in logs]) for key in keys}

        logging.info(
            f"{kwargs.get('phase')} epoch {kwargs.get('ep')} \t loss: {summed_stats['loss']:.1f}"
        )

    def fit(self, dataloaders):
        """ 
        Dataloaders is a dict of dataloaders:
        {'train' : Dataloader, 'valid': Dataloader}
        """
        for ep in self.max_epochs:
            print()
            logging.info(f'Epoch {ep}/{self.max_epochs} \t Step {step}')
            print('-' * 10)

            for phase, dl in dataloaders.keys():
                logs = []
                batch_size = dl.batch_size

                for batch in dl:
                    if phase == "train":
                        tmp_log = self.training_step(batch)
                        self.step += batch_size
                    else:
                        tmp_log = self.validation_step(batch)

                    logs.append(tmp_log)

            self.training_end(logs)
            # EARLY STOPPING CHECK
            stop = earlystop(epoch=epoch, score=-stats_epoch['loss'])
            if stop:
                logging.info('Early stopping criteria reached.')
                return None

class EarlyStoppingAndCheckpoint:
    def __init__(self, patience = 1, save_dir="checkpoints", **kwargs):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.best_score = None
        self.patience = patience
        self.counter = 0

    def __call__(self, epoch, score):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint()
            return False
        elif score <= self.best_score:
            self.counter += 1
            if self.counter > self.patience:
                logging.info(f"REACHED EARLY STOPPING ON EPOCH {epoch}")
                return True
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint()
            self.counter = 0
            return False

    def save_checkpoint(self):
        path = f"{self.save_dir}/parameters.pyro"
        logging.info(f"Saving model to {path}..")
        pyro.get_param_store().save(path)
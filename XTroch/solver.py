import torch
import yaml
from abc import abstractmethod
from XTroch.utils import AttriDict
from XTroch.scheduler import SchedulerBuilder
from XTroch.logger import Logger


class Solver(object):
    def __init__(self, cfg, session=None, save=True):
        with open(cfg, 'r', encoding='utf-8') as f:
            self.cfg = AttriDict(yaml.load(f))
        self.model = self.build_model()
        self.optim = self._create_optim(self.model.parameters())
        self.lr_scheduler = self._create_lr_scheduler()
        self.data_loader = self._create_loader()
        self.start_epoch = 0

        if 'RESUME' in self.cfg:
            chech_point = torch.load(self.cfg.RESUME)
            if not session:
                session = chech_point['session']
            self.start_epoch = chech_point['epoch']
        if 'SESSION' in self.cfg:
            session = self.cfg.SESSION
        self.logger = Logger(self.cfg.NAME, self.cfg.SAVE_PATH, self.cfg.VISDOM, self.cfg.PORT, save, session)

    def _create_optim(self, params, lr=None):
        optim_cofig = self.cfg.TRAIN.OPTIMIZATION
        if lr is None:
            lr = self.cfg.TRAIN.LR
        if optim_cofig.TYPE == 'SGD':
            return torch.optim.SGD(params, lr, optim_cofig.MOMENTUM, weight_decay=optim_cofig.WEIGHT_DECAY)
        elif optim_cofig.TYPE == "ADAM":
            return torch.optim.Adam(params, lr, weight_decay=optim_cofig.WEIGHT_DECAY)
        else:
            return None

    def _create_lr_scheduler(self):
        builder = SchedulerBuilder()
        return builder(self.optim, self.cfg.TRAIN.OPTIMIZATION.SCHEDULER)

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def _create_loader(self):
        pass



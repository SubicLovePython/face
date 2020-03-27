import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import StepLR


class LineStep(_LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        self.gamma = gamma

    def get_lr(self):
        return [base_lr - self.gamma*self.last_epoch for base_lr in self.base_lrs]


class CosRestart(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        self.last_epoch = self.last_epoch % self.T_max
        return [self.eta_min + (base_lr-self.eta_min)*(1+math.cos(math.pi*self.last_epoch/self.T_max))/2 for base_lr in self.base_lrs]


class SchedulerBuilder(object):
    def __init__(self):
        pass

    def __call__(self, optim, args):
        if args.TYPE == "STEP":
            return StepLR(optim, args.STEP_SIZE, args.GAMMA)
        elif args.TYPE == "COS":
            return CosRestart(optim, T_max=args.EPOCH)
        else:
            return TypeError("Unsupported scheduler")

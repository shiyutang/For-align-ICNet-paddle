"""Popular Learning Rate Schedulers"""
from __future__ import division

import paddle
# from paddle.optimizer.lr import PolynomialDecay

__all__ = ['IterationPolyLR']

class IterationPolyLR(paddle.optimizer.lr.LRScheduler):
    def __init__(self, optimizer, target_lr, max_iters, power, last_epoch=-1):
        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        self.last_epoch = last_epoch
        super(IterationPolyLR, self).__init__(target_lr, last_epoch)
        
    def get_lr(self):
        N = self.max_iters 
        T = self.last_epoch
        factor = pow(1 - T / N, self.power)
        # https://blog.csdn.net/mieleizhi0522/article/details/83113824
        return self.target_lr + (self.base_lr - self.target_lr) * factor
        self.ba
        # return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class LRScheduler(object):
    """Base class of a learning rate scheduler.

    A scheduler returns a new learning rate based on the number of updates that have
    been performed.

    """

    def __init__(self, base_lr=0.01, step=(30, 60), factor=0.1,
                 warmup_epoch=0, warmup_begin_lr=0., warmup_mode='linear'):
        """
        Args:
            base_lr : float, optional
                The initial learning rate.
            step: tuple or list
                Steps or epochs at which learning rate can be decayed
            factor: float
                Learning decay rate at each of given step epochs, for instance, in step = (30, 60)
            warmup_epoch: int
                number of warmup steps used before this scheduler starts decay
            warmup_begin_lr: float
                if using warmup, the learning rate from which it starts warming up
            warmup_mode: string
                warmup can be done in two modes.
                'linear' mode gradually increases lr with each step in equal increments
                'constant' mode keeps lr at warmup_begin_lr for warmup_steps
        """
        self.base_lr = base_lr
        self.learning_rate = base_lr
        if isinstance(step, tuple) or isinstance(step, list):
            self.step = step
        else:
            self.step = [step*(i+1) for i in range(20)]
        self.factor = factor
        assert isinstance(warmup_epoch, int)
        self.warmup_epoch = warmup_epoch

        self.warmup_final_lr = base_lr
        self.warmup_begin_lr = warmup_begin_lr
        if self.warmup_begin_lr > self.warmup_final_lr:
            raise ValueError("Base lr has to be higher than warmup_begin_lr")
        if self.warmup_epoch < 0:
            raise ValueError("Warmup steps has to be positive or 0")
        if warmup_mode not in ['linear', 'constant']:
            raise ValueError("Supports only linear and constant modes of warmup")
        self.warmup_mode = warmup_mode

    def update(self, num_epoch):
        if self.warmup_epoch > num_epoch:
            # warmup strategy
            if self.warmup_mode == 'linear':
                self.learning_rate = self.warmup_begin_lr + (self.warmup_final_lr - self.warmup_begin_lr) * \
                                     num_epoch / self.warmup_epoch
            elif self.warmup_mode == 'constant':
                self.learning_rate = self.warmup_begin_lr

        else:
            count = sum([1 for s in self.step if s <= num_epoch])
            self.learning_rate = self.base_lr * pow(self.factor, count)
        return self.learning_rate


# Test
if __name__ == '__main__':
    lr_scheduler = LRScheduler(base_lr=0.0008, step=[40, 60],
                               factor=0.5, warmup_epoch=10,
                               warmup_begin_lr=0.000008)

    num_epochs = 70
    for epoch in range(num_epochs):
        lr = lr_scheduler.update(epoch)
        print(epoch, lr)

    print('Done!')



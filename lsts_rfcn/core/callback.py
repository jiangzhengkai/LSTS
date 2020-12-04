# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import time
import logging
import mxnet as mx


class Speedometer(object):
    def __init__(self, batch_size, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                s = ''
                if param.eval_metric is not None:
                    name, value = param.eval_metric.get()
                    s = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-" % (param.epoch, count, speed)
                    for n, v in zip(name, value):
                        s += "%s=%f,\t" % (n, v)
                else:
                    s = "Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (param.epoch, count, speed)

                logging.info(s)
                print(s)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


def do_checkpoint(prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        weight = arg['rfcn_bbox_weight']
        bias = arg['rfcn_bbox_bias']
        repeat = bias.shape[0] / means.shape[0]

        arg['rfcn_bbox_weight_test'] = weight * mx.nd.repeat(mx.nd.array(stds), repeats=repeat).reshape((bias.shape[0], 1, 1, 1))
        arg['rfcn_bbox_bias_test'] = arg['rfcn_bbox_bias'] * mx.nd.repeat(mx.nd.array(stds), repeats=repeat) + mx.nd.repeat(mx.nd.array(means), repeats=repeat)
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        arg.pop('rfcn_bbox_weight_test')
        arg.pop('rfcn_bbox_bias_test')
    return _callback

def module_checkpoint_iter(prefix, iteration_list='10000,20000'):
    """Callback to checkpoint Module to prefix every epoch.

    Parameters
    ----------
    mod : subclass of BaseModule
        The module to checkpoint.
    prefix : str
        The file prefix for this checkpoint.
    period : int
        How many epochs to wait before checkpointing. Defaults to 1.
    save_optimizer_states : bool
        Indicates whether or not to save optimizer states for continued training.

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_end_callback to fit.
    """

    # pylint: disable=unused-argument
    def _callback(epoch_no, iter_no, sym=None, arg=None, aux=None):
        """The checkpoint function."""
        import numpy as np
        iters_list = np.array([int(i) for i in iteration_list.split(',')])

        if sum((iter_no + 1) == iters_list) == 1:
            mx.model.save_checkpoint(prefix+'_%d' % (iter_no+1), epoch_no + 1, sym, arg, aux)
    return _callback
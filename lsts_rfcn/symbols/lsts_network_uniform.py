# --------------------------------------------------------
# LSTS networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zhengkai Jiang
# --------------------------------------------------------
import cPickle
import mxnet as mx

from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *
from operator_py.rpn_inv_normalize import *
from operator_py.tile_as import *


class lsts_network_uniform(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3)  # use for 101
        self.filter_list = [256, 512, 1024, 2048]

    def get_resnet_low_feature(self, data):
        conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
                                      no_bias=True)
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=self.use_global_stats,
                                       eps=self.eps, fix_gamma=False)
        scale_conv1 = bn_conv1
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1, act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pad=(1, 1), kernel=(3, 3), stride=(2, 2),
                                  pool_type='max')
        res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1, num_filter=256, pad=(0, 0),
                                              kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1,
                                           use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2a_branch1 = bn2a_branch1
        res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2a_branch2a = bn2a_branch2a
        res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a, act_type='relu')
        res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu, num_filter=64,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2a_branch2b = bn2a_branch2b
        res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b, act_type='relu')
        res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu, num_filter=256,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2a_branch2c = bn2a_branch2c
        res2a = mx.symbol.broadcast_add(name='res2a', *[scale2a_branch1, scale2a_branch2c])
        res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a, act_type='relu')
        res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2b_branch2a = bn2b_branch2a
        res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a, act_type='relu')
        res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu, num_filter=64,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2b_branch2b = bn2b_branch2b
        res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b, act_type='relu')
        res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu, num_filter=256,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2b_branch2c = bn2b_branch2c
        res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu, scale2b_branch2c])
        res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b, act_type='relu')
        res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2c_branch2a = bn2c_branch2a
        res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a, act_type='relu')
        res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu, num_filter=64,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2c_branch2b = bn2c_branch2b
        res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b, act_type='relu')
        res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu, num_filter=256,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale2c_branch2c = bn2c_branch2c
        res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu, scale2c_branch2c])
        res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c, act_type='relu')
        res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu, num_filter=512, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1,
                                           use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3a_branch1 = bn3a_branch1
        res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3a_branch2a = bn3a_branch2a
        res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a, act_type='relu')
        res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu, num_filter=128,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3a_branch2b = bn3a_branch2b
        res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b, act_type='relu')
        res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu, num_filter=512,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3a_branch2c = bn3a_branch2c
        res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1, scale3a_branch2c])
        res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a, act_type='relu')
        res3b1_branch2a = mx.symbol.Convolution(name='res3b1_branch2a', data=res3a_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2a = mx.symbol.BatchNorm(name='bn3b1_branch2a', data=res3b1_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b1_branch2a = bn3b1_branch2a
        res3b1_branch2a_relu = mx.symbol.Activation(name='res3b1_branch2a_relu', data=scale3b1_branch2a,
                                                    act_type='relu')
        res3b1_branch2b = mx.symbol.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b1_branch2b = mx.symbol.BatchNorm(name='bn3b1_branch2b', data=res3b1_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b1_branch2b = bn3b1_branch2b
        res3b1_branch2b_relu = mx.symbol.Activation(name='res3b1_branch2b_relu', data=scale3b1_branch2b,
                                                    act_type='relu')
        res3b1_branch2c = mx.symbol.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2c = mx.symbol.BatchNorm(name='bn3b1_branch2c', data=res3b1_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b1_branch2c = bn3b1_branch2c
        res3b1 = mx.symbol.broadcast_add(name='res3b1', *[res3a_relu, scale3b1_branch2c])
        res3b1_relu = mx.symbol.Activation(name='res3b1_relu', data=res3b1, act_type='relu')
        res3b2_branch2a = mx.symbol.Convolution(name='res3b2_branch2a', data=res3b1_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2a = mx.symbol.BatchNorm(name='bn3b2_branch2a', data=res3b2_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b2_branch2a = bn3b2_branch2a
        res3b2_branch2a_relu = mx.symbol.Activation(name='res3b2_branch2a_relu', data=scale3b2_branch2a,
                                                    act_type='relu')
        res3b2_branch2b = mx.symbol.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b2_branch2b = mx.symbol.BatchNorm(name='bn3b2_branch2b', data=res3b2_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b2_branch2b = bn3b2_branch2b
        res3b2_branch2b_relu = mx.symbol.Activation(name='res3b2_branch2b_relu', data=scale3b2_branch2b,
                                                    act_type='relu')
        res3b2_branch2c = mx.symbol.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2c = mx.symbol.BatchNorm(name='bn3b2_branch2c', data=res3b2_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b2_branch2c = bn3b2_branch2c
        res3b2 = mx.symbol.broadcast_add(name='res3b2', *[res3b1_relu, scale3b2_branch2c])
        res3b2_relu = mx.symbol.Activation(name='res3b2_relu', data=res3b2, act_type='relu')
        res3b3_branch2a = mx.symbol.Convolution(name='res3b3_branch2a', data=res3b2_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2a = mx.symbol.BatchNorm(name='bn3b3_branch2a', data=res3b3_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b3_branch2a = bn3b3_branch2a
        res3b3_branch2a_relu = mx.symbol.Activation(name='res3b3_branch2a_relu', data=scale3b3_branch2a,
                                                    act_type='relu')
        res3b3_branch2b = mx.symbol.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b3_branch2b = mx.symbol.BatchNorm(name='bn3b3_branch2b', data=res3b3_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b3_branch2b = bn3b3_branch2b
        res3b3_branch2b_relu = mx.symbol.Activation(name='res3b3_branch2b_relu', data=scale3b3_branch2b,
                                                    act_type='relu')
        res3b3_branch2c = mx.symbol.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2c = mx.symbol.BatchNorm(name='bn3b3_branch2c', data=res3b3_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale3b3_branch2c = bn3b3_branch2c
        res3b3 = mx.symbol.broadcast_add(name='res3b3', *[res3b2_relu, scale3b3_branch2c])
        res3b3_relu = mx.symbol.Activation(name='res3b3_relu', data=res3b3, act_type='relu')
        res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3b3_relu, num_filter=1024, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1,
                                           use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4a_branch1 = bn4a_branch1
        res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3b3_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4a_branch2a = bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a, act_type='relu')
        res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu, num_filter=256,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4a_branch2b = bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b, act_type='relu')
        res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu, num_filter=1024,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4a_branch2c = bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1, scale4a_branch2c])
        res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a, act_type='relu')
        res4b1_branch2a = mx.symbol.Convolution(name='res4b1_branch2a', data=res4a_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2a = mx.symbol.BatchNorm(name='bn4b1_branch2a', data=res4b1_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b1_branch2a = bn4b1_branch2a
        res4b1_branch2a_relu = mx.symbol.Activation(name='res4b1_branch2a_relu', data=scale4b1_branch2a,
                                                    act_type='relu')
        res4b1_branch2b = mx.symbol.Convolution(name='res4b1_branch2b', data=res4b1_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b1_branch2b = mx.symbol.BatchNorm(name='bn4b1_branch2b', data=res4b1_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b1_branch2b = bn4b1_branch2b
        res4b1_branch2b_relu = mx.symbol.Activation(name='res4b1_branch2b_relu', data=scale4b1_branch2b,
                                                    act_type='relu')
        res4b1_branch2c = mx.symbol.Convolution(name='res4b1_branch2c', data=res4b1_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2c = mx.symbol.BatchNorm(name='bn4b1_branch2c', data=res4b1_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b1_branch2c = bn4b1_branch2c
        res4b1 = mx.symbol.broadcast_add(name='res4b1', *[res4a_relu, scale4b1_branch2c])
        res4b1_relu = mx.symbol.Activation(name='res4b1_relu', data=res4b1, act_type='relu')
        res4b2_branch2a = mx.symbol.Convolution(name='res4b2_branch2a', data=res4b1_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2a = mx.symbol.BatchNorm(name='bn4b2_branch2a', data=res4b2_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b2_branch2a = bn4b2_branch2a
        res4b2_branch2a_relu = mx.symbol.Activation(name='res4b2_branch2a_relu', data=scale4b2_branch2a,
                                                    act_type='relu')
        res4b2_branch2b = mx.symbol.Convolution(name='res4b2_branch2b', data=res4b2_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b2_branch2b = mx.symbol.BatchNorm(name='bn4b2_branch2b', data=res4b2_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b2_branch2b = bn4b2_branch2b
        res4b2_branch2b_relu = mx.symbol.Activation(name='res4b2_branch2b_relu', data=scale4b2_branch2b,
                                                    act_type='relu')
        res4b2_branch2c = mx.symbol.Convolution(name='res4b2_branch2c', data=res4b2_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2c = mx.symbol.BatchNorm(name='bn4b2_branch2c', data=res4b2_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b2_branch2c = bn4b2_branch2c
        res4b2 = mx.symbol.broadcast_add(name='res4b2', *[res4b1_relu, scale4b2_branch2c])
        res4b2_relu = mx.symbol.Activation(name='res4b2_relu', data=res4b2, act_type='relu')
        res4b3_branch2a = mx.symbol.Convolution(name='res4b3_branch2a', data=res4b2_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2a = mx.symbol.BatchNorm(name='bn4b3_branch2a', data=res4b3_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b3_branch2a = bn4b3_branch2a
        res4b3_branch2a_relu = mx.symbol.Activation(name='res4b3_branch2a_relu', data=scale4b3_branch2a,
                                                    act_type='relu')
        res4b3_branch2b = mx.symbol.Convolution(name='res4b3_branch2b', data=res4b3_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b3_branch2b = mx.symbol.BatchNorm(name='bn4b3_branch2b', data=res4b3_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b3_branch2b = bn4b3_branch2b
        res4b3_branch2b_relu = mx.symbol.Activation(name='res4b3_branch2b_relu', data=scale4b3_branch2b,
                                                    act_type='relu')
        res4b3_branch2c = mx.symbol.Convolution(name='res4b3_branch2c', data=res4b3_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2c = mx.symbol.BatchNorm(name='bn4b3_branch2c', data=res4b3_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b3_branch2c = bn4b3_branch2c
        #res4b3 = mx.symbol.broadcast_add(name='res4b3', *[res4b2_relu, scale4b3_branch2c])
        res4b3 = mx.symbol.broadcast_add(name='low_level_feat_key', *[res4b2_relu, scale4b3_branch2c])
        #res4b3_relu = mx.symbol.Activation(name='low_level_feat_key', data=res4b3, act_type='relu')
        return res4b3

    def get_resnet_high_feature(self, data):
        res4b3_relu = mx.symbol.Activation(name='res4b3_relu', data=data, act_type='relu')
        res4b4_branch2a = mx.symbol.Convolution(name='res4b4_branch2a', data=res4b3_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2a = mx.symbol.BatchNorm(name='bn4b4_branch2a', data=res4b4_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b4_branch2a = bn4b4_branch2a
        res4b4_branch2a_relu = mx.symbol.Activation(name='res4b4_branch2a_relu', data=scale4b4_branch2a,
                                                    act_type='relu')
        res4b4_branch2b = mx.symbol.Convolution(name='res4b4_branch2b', data=res4b4_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b4_branch2b = mx.symbol.BatchNorm(name='bn4b4_branch2b', data=res4b4_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b4_branch2b = bn4b4_branch2b
        res4b4_branch2b_relu = mx.symbol.Activation(name='res4b4_branch2b_relu', data=scale4b4_branch2b,
                                                    act_type='relu')
        res4b4_branch2c = mx.symbol.Convolution(name='res4b4_branch2c', data=res4b4_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2c = mx.symbol.BatchNorm(name='bn4b4_branch2c', data=res4b4_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b4_branch2c = bn4b4_branch2c
        res4b4 = mx.symbol.broadcast_add(name='res4b4', *[res4b3_relu, scale4b4_branch2c])
        res4b4_relu = mx.symbol.Activation(name='res4b4_relu', data=res4b4, act_type='relu')
        res4b5_branch2a = mx.symbol.Convolution(name='res4b5_branch2a', data=res4b4_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2a = mx.symbol.BatchNorm(name='bn4b5_branch2a', data=res4b5_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b5_branch2a = bn4b5_branch2a
        res4b5_branch2a_relu = mx.symbol.Activation(name='res4b5_branch2a_relu', data=scale4b5_branch2a,
                                                    act_type='relu')
        res4b5_branch2b = mx.symbol.Convolution(name='res4b5_branch2b', data=res4b5_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b5_branch2b = mx.symbol.BatchNorm(name='bn4b5_branch2b', data=res4b5_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b5_branch2b = bn4b5_branch2b
        res4b5_branch2b_relu = mx.symbol.Activation(name='res4b5_branch2b_relu', data=scale4b5_branch2b,
                                                    act_type='relu')
        res4b5_branch2c = mx.symbol.Convolution(name='res4b5_branch2c', data=res4b5_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2c = mx.symbol.BatchNorm(name='bn4b5_branch2c', data=res4b5_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b5_branch2c = bn4b5_branch2c
        res4b5 = mx.symbol.broadcast_add(name='res4b5', *[res4b4_relu, scale4b5_branch2c])
        res4b5_relu = mx.symbol.Activation(name='res4b5_relu', data=res4b5, act_type='relu')
        res4b6_branch2a = mx.symbol.Convolution(name='res4b6_branch2a', data=res4b5_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2a = mx.symbol.BatchNorm(name='bn4b6_branch2a', data=res4b6_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b6_branch2a = bn4b6_branch2a
        res4b6_branch2a_relu = mx.symbol.Activation(name='res4b6_branch2a_relu', data=scale4b6_branch2a,
                                                    act_type='relu')
        res4b6_branch2b = mx.symbol.Convolution(name='res4b6_branch2b', data=res4b6_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b6_branch2b = mx.symbol.BatchNorm(name='bn4b6_branch2b', data=res4b6_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b6_branch2b = bn4b6_branch2b
        res4b6_branch2b_relu = mx.symbol.Activation(name='res4b6_branch2b_relu', data=scale4b6_branch2b,
                                                    act_type='relu')
        res4b6_branch2c = mx.symbol.Convolution(name='res4b6_branch2c', data=res4b6_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2c = mx.symbol.BatchNorm(name='bn4b6_branch2c', data=res4b6_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b6_branch2c = bn4b6_branch2c
        res4b6 = mx.symbol.broadcast_add(name='res4b6', *[res4b5_relu, scale4b6_branch2c])
        res4b6_relu = mx.symbol.Activation(name='res4b6_relu', data=res4b6, act_type='relu')
        res4b7_branch2a = mx.symbol.Convolution(name='res4b7_branch2a', data=res4b6_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2a = mx.symbol.BatchNorm(name='bn4b7_branch2a', data=res4b7_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b7_branch2a = bn4b7_branch2a
        res4b7_branch2a_relu = mx.symbol.Activation(name='res4b7_branch2a_relu', data=scale4b7_branch2a,
                                                    act_type='relu')
        res4b7_branch2b = mx.symbol.Convolution(name='res4b7_branch2b', data=res4b7_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b7_branch2b = mx.symbol.BatchNorm(name='bn4b7_branch2b', data=res4b7_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b7_branch2b = bn4b7_branch2b
        res4b7_branch2b_relu = mx.symbol.Activation(name='res4b7_branch2b_relu', data=scale4b7_branch2b,
                                                    act_type='relu')
        res4b7_branch2c = mx.symbol.Convolution(name='res4b7_branch2c', data=res4b7_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2c = mx.symbol.BatchNorm(name='bn4b7_branch2c', data=res4b7_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b7_branch2c = bn4b7_branch2c
        res4b7 = mx.symbol.broadcast_add(name='res4b7', *[res4b6_relu, scale4b7_branch2c])
        res4b7_relu = mx.symbol.Activation(name='res4b7_relu', data=res4b7, act_type='relu')
        res4b8_branch2a = mx.symbol.Convolution(name='res4b8_branch2a', data=res4b7_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2a = mx.symbol.BatchNorm(name='bn4b8_branch2a', data=res4b8_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b8_branch2a = bn4b8_branch2a
        res4b8_branch2a_relu = mx.symbol.Activation(name='res4b8_branch2a_relu', data=scale4b8_branch2a,
                                                    act_type='relu')
        res4b8_branch2b = mx.symbol.Convolution(name='res4b8_branch2b', data=res4b8_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b8_branch2b = mx.symbol.BatchNorm(name='bn4b8_branch2b', data=res4b8_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b8_branch2b = bn4b8_branch2b
        res4b8_branch2b_relu = mx.symbol.Activation(name='res4b8_branch2b_relu', data=scale4b8_branch2b,
                                                    act_type='relu')
        res4b8_branch2c = mx.symbol.Convolution(name='res4b8_branch2c', data=res4b8_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2c = mx.symbol.BatchNorm(name='bn4b8_branch2c', data=res4b8_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b8_branch2c = bn4b8_branch2c
        res4b8 = mx.symbol.broadcast_add(name='res4b8', *[res4b7_relu, scale4b8_branch2c])
        res4b8_relu = mx.symbol.Activation(name='res4b8_relu', data=res4b8, act_type='relu')
        res4b9_branch2a = mx.symbol.Convolution(name='res4b9_branch2a', data=res4b8_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2a = mx.symbol.BatchNorm(name='bn4b9_branch2a', data=res4b9_branch2a,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b9_branch2a = bn4b9_branch2a
        res4b9_branch2a_relu = mx.symbol.Activation(name='res4b9_branch2a_relu', data=scale4b9_branch2a,
                                                    act_type='relu')
        res4b9_branch2b = mx.symbol.Convolution(name='res4b9_branch2b', data=res4b9_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b9_branch2b = mx.symbol.BatchNorm(name='bn4b9_branch2b', data=res4b9_branch2b,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b9_branch2b = bn4b9_branch2b
        res4b9_branch2b_relu = mx.symbol.Activation(name='res4b9_branch2b_relu', data=scale4b9_branch2b,
                                                    act_type='relu')
        res4b9_branch2c = mx.symbol.Convolution(name='res4b9_branch2c', data=res4b9_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2c = mx.symbol.BatchNorm(name='bn4b9_branch2c', data=res4b9_branch2c,
                                             use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b9_branch2c = bn4b9_branch2c
        res4b9 = mx.symbol.broadcast_add(name='res4b9', *[res4b8_relu, scale4b9_branch2c])
        res4b9_relu = mx.symbol.Activation(name='res4b9_relu', data=res4b9, act_type='relu')
        res4b10_branch2a = mx.symbol.Convolution(name='res4b10_branch2a', data=res4b9_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2a = mx.symbol.BatchNorm(name='bn4b10_branch2a', data=res4b10_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b10_branch2a = bn4b10_branch2a
        res4b10_branch2a_relu = mx.symbol.Activation(name='res4b10_branch2a_relu', data=scale4b10_branch2a,
                                                     act_type='relu')
        res4b10_branch2b = mx.symbol.Convolution(name='res4b10_branch2b', data=res4b10_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b10_branch2b = mx.symbol.BatchNorm(name='bn4b10_branch2b', data=res4b10_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b10_branch2b = bn4b10_branch2b
        res4b10_branch2b_relu = mx.symbol.Activation(name='res4b10_branch2b_relu', data=scale4b10_branch2b,
                                                     act_type='relu')
        res4b10_branch2c = mx.symbol.Convolution(name='res4b10_branch2c', data=res4b10_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2c = mx.symbol.BatchNorm(name='bn4b10_branch2c', data=res4b10_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b10_branch2c = bn4b10_branch2c
        res4b10 = mx.symbol.broadcast_add(name='res4b10', *[res4b9_relu, scale4b10_branch2c])
        res4b10_relu = mx.symbol.Activation(name='res4b10_relu', data=res4b10, act_type='relu')
        res4b11_branch2a = mx.symbol.Convolution(name='res4b11_branch2a', data=res4b10_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2a = mx.symbol.BatchNorm(name='bn4b11_branch2a', data=res4b11_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b11_branch2a = bn4b11_branch2a
        res4b11_branch2a_relu = mx.symbol.Activation(name='res4b11_branch2a_relu', data=scale4b11_branch2a,
                                                     act_type='relu')
        res4b11_branch2b = mx.symbol.Convolution(name='res4b11_branch2b', data=res4b11_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b11_branch2b = mx.symbol.BatchNorm(name='bn4b11_branch2b', data=res4b11_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b11_branch2b = bn4b11_branch2b
        res4b11_branch2b_relu = mx.symbol.Activation(name='res4b11_branch2b_relu', data=scale4b11_branch2b,
                                                     act_type='relu')
        res4b11_branch2c = mx.symbol.Convolution(name='res4b11_branch2c', data=res4b11_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2c = mx.symbol.BatchNorm(name='bn4b11_branch2c', data=res4b11_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b11_branch2c = bn4b11_branch2c
        res4b11 = mx.symbol.broadcast_add(name='res4b11', *[res4b10_relu, scale4b11_branch2c])
        res4b11_relu = mx.symbol.Activation(name='res4b11_relu', data=res4b11, act_type='relu')
        res4b12_branch2a = mx.symbol.Convolution(name='res4b12_branch2a', data=res4b11_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2a = mx.symbol.BatchNorm(name='bn4b12_branch2a', data=res4b12_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b12_branch2a = bn4b12_branch2a
        res4b12_branch2a_relu = mx.symbol.Activation(name='res4b12_branch2a_relu', data=scale4b12_branch2a,
                                                     act_type='relu')
        res4b12_branch2b = mx.symbol.Convolution(name='res4b12_branch2b', data=res4b12_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b12_branch2b = mx.symbol.BatchNorm(name='bn4b12_branch2b', data=res4b12_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b12_branch2b = bn4b12_branch2b
        res4b12_branch2b_relu = mx.symbol.Activation(name='res4b12_branch2b_relu', data=scale4b12_branch2b,
                                                     act_type='relu')
        res4b12_branch2c = mx.symbol.Convolution(name='res4b12_branch2c', data=res4b12_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2c = mx.symbol.BatchNorm(name='bn4b12_branch2c', data=res4b12_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b12_branch2c = bn4b12_branch2c
        res4b12 = mx.symbol.broadcast_add(name='res4b12', *[res4b11_relu, scale4b12_branch2c])
        res4b12_relu = mx.symbol.Activation(name='res4b12_relu', data=res4b12, act_type='relu')
        res4b13_branch2a = mx.symbol.Convolution(name='res4b13_branch2a', data=res4b12_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2a = mx.symbol.BatchNorm(name='bn4b13_branch2a', data=res4b13_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b13_branch2a = bn4b13_branch2a
        res4b13_branch2a_relu = mx.symbol.Activation(name='res4b13_branch2a_relu', data=scale4b13_branch2a,
                                                     act_type='relu')
        res4b13_branch2b = mx.symbol.Convolution(name='res4b13_branch2b', data=res4b13_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b13_branch2b = mx.symbol.BatchNorm(name='bn4b13_branch2b', data=res4b13_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b13_branch2b = bn4b13_branch2b
        res4b13_branch2b_relu = mx.symbol.Activation(name='res4b13_branch2b_relu', data=scale4b13_branch2b,
                                                     act_type='relu')
        res4b13_branch2c = mx.symbol.Convolution(name='res4b13_branch2c', data=res4b13_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2c = mx.symbol.BatchNorm(name='bn4b13_branch2c', data=res4b13_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b13_branch2c = bn4b13_branch2c
        res4b13 = mx.symbol.broadcast_add(name='res4b13', *[res4b12_relu, scale4b13_branch2c])
        res4b13_relu = mx.symbol.Activation(name='res4b13_relu', data=res4b13, act_type='relu')
        res4b14_branch2a = mx.symbol.Convolution(name='res4b14_branch2a', data=res4b13_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2a = mx.symbol.BatchNorm(name='bn4b14_branch2a', data=res4b14_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b14_branch2a = bn4b14_branch2a
        res4b14_branch2a_relu = mx.symbol.Activation(name='res4b14_branch2a_relu', data=scale4b14_branch2a,
                                                     act_type='relu')
        res4b14_branch2b = mx.symbol.Convolution(name='res4b14_branch2b', data=res4b14_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b14_branch2b = mx.symbol.BatchNorm(name='bn4b14_branch2b', data=res4b14_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b14_branch2b = bn4b14_branch2b
        res4b14_branch2b_relu = mx.symbol.Activation(name='res4b14_branch2b_relu', data=scale4b14_branch2b,
                                                     act_type='relu')
        res4b14_branch2c = mx.symbol.Convolution(name='res4b14_branch2c', data=res4b14_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2c = mx.symbol.BatchNorm(name='bn4b14_branch2c', data=res4b14_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b14_branch2c = bn4b14_branch2c
        res4b14 = mx.symbol.broadcast_add(name='res4b14', *[res4b13_relu, scale4b14_branch2c])
        res4b14_relu = mx.symbol.Activation(name='res4b14_relu', data=res4b14, act_type='relu')
        res4b15_branch2a = mx.symbol.Convolution(name='res4b15_branch2a', data=res4b14_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2a = mx.symbol.BatchNorm(name='bn4b15_branch2a', data=res4b15_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b15_branch2a = bn4b15_branch2a
        res4b15_branch2a_relu = mx.symbol.Activation(name='res4b15_branch2a_relu', data=scale4b15_branch2a,
                                                     act_type='relu')
        res4b15_branch2b = mx.symbol.Convolution(name='res4b15_branch2b', data=res4b15_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b15_branch2b = mx.symbol.BatchNorm(name='bn4b15_branch2b', data=res4b15_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b15_branch2b = bn4b15_branch2b
        res4b15_branch2b_relu = mx.symbol.Activation(name='res4b15_branch2b_relu', data=scale4b15_branch2b,
                                                     act_type='relu')
        res4b15_branch2c = mx.symbol.Convolution(name='res4b15_branch2c', data=res4b15_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2c = mx.symbol.BatchNorm(name='bn4b15_branch2c', data=res4b15_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b15_branch2c = bn4b15_branch2c
        res4b15 = mx.symbol.broadcast_add(name='res4b15', *[res4b14_relu, scale4b15_branch2c])
        res4b15_relu = mx.symbol.Activation(name='res4b15_relu', data=res4b15, act_type='relu')
        res4b16_branch2a = mx.symbol.Convolution(name='res4b16_branch2a', data=res4b15_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2a = mx.symbol.BatchNorm(name='bn4b16_branch2a', data=res4b16_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b16_branch2a = bn4b16_branch2a
        res4b16_branch2a_relu = mx.symbol.Activation(name='res4b16_branch2a_relu', data=scale4b16_branch2a,
                                                     act_type='relu')
        res4b16_branch2b = mx.symbol.Convolution(name='res4b16_branch2b', data=res4b16_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b16_branch2b = mx.symbol.BatchNorm(name='bn4b16_branch2b', data=res4b16_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b16_branch2b = bn4b16_branch2b
        res4b16_branch2b_relu = mx.symbol.Activation(name='res4b16_branch2b_relu', data=scale4b16_branch2b,
                                                     act_type='relu')
        res4b16_branch2c = mx.symbol.Convolution(name='res4b16_branch2c', data=res4b16_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2c = mx.symbol.BatchNorm(name='bn4b16_branch2c', data=res4b16_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b16_branch2c = bn4b16_branch2c
        res4b16 = mx.symbol.broadcast_add(name='res4b16', *[res4b15_relu, scale4b16_branch2c])
        res4b16_relu = mx.symbol.Activation(name='res4b16_relu', data=res4b16, act_type='relu')
        res4b17_branch2a = mx.symbol.Convolution(name='res4b17_branch2a', data=res4b16_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2a = mx.symbol.BatchNorm(name='bn4b17_branch2a', data=res4b17_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b17_branch2a = bn4b17_branch2a
        res4b17_branch2a_relu = mx.symbol.Activation(name='res4b17_branch2a_relu', data=scale4b17_branch2a,
                                                     act_type='relu')
        res4b17_branch2b = mx.symbol.Convolution(name='res4b17_branch2b', data=res4b17_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b17_branch2b = mx.symbol.BatchNorm(name='bn4b17_branch2b', data=res4b17_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b17_branch2b = bn4b17_branch2b
        res4b17_branch2b_relu = mx.symbol.Activation(name='res4b17_branch2b_relu', data=scale4b17_branch2b,
                                                     act_type='relu')
        res4b17_branch2c = mx.symbol.Convolution(name='res4b17_branch2c', data=res4b17_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2c = mx.symbol.BatchNorm(name='bn4b17_branch2c', data=res4b17_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b17_branch2c = bn4b17_branch2c
        res4b17 = mx.symbol.broadcast_add(name='res4b17', *[res4b16_relu, scale4b17_branch2c])
        res4b17_relu = mx.symbol.Activation(name='res4b17_relu', data=res4b17, act_type='relu')
        res4b18_branch2a = mx.symbol.Convolution(name='res4b18_branch2a', data=res4b17_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2a = mx.symbol.BatchNorm(name='bn4b18_branch2a', data=res4b18_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b18_branch2a = bn4b18_branch2a
        res4b18_branch2a_relu = mx.symbol.Activation(name='res4b18_branch2a_relu', data=scale4b18_branch2a,
                                                     act_type='relu')
        res4b18_branch2b = mx.symbol.Convolution(name='res4b18_branch2b', data=res4b18_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b18_branch2b = mx.symbol.BatchNorm(name='bn4b18_branch2b', data=res4b18_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b18_branch2b = bn4b18_branch2b
        res4b18_branch2b_relu = mx.symbol.Activation(name='res4b18_branch2b_relu', data=scale4b18_branch2b,
                                                     act_type='relu')
        res4b18_branch2c = mx.symbol.Convolution(name='res4b18_branch2c', data=res4b18_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2c = mx.symbol.BatchNorm(name='bn4b18_branch2c', data=res4b18_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b18_branch2c = bn4b18_branch2c
        res4b18 = mx.symbol.broadcast_add(name='res4b18', *[res4b17_relu, scale4b18_branch2c])
        res4b18_relu = mx.symbol.Activation(name='res4b18_relu', data=res4b18, act_type='relu')
        res4b19_branch2a = mx.symbol.Convolution(name='res4b19_branch2a', data=res4b18_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2a = mx.symbol.BatchNorm(name='bn4b19_branch2a', data=res4b19_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b19_branch2a = bn4b19_branch2a
        res4b19_branch2a_relu = mx.symbol.Activation(name='res4b19_branch2a_relu', data=scale4b19_branch2a,
                                                     act_type='relu')
        res4b19_branch2b = mx.symbol.Convolution(name='res4b19_branch2b', data=res4b19_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b19_branch2b = mx.symbol.BatchNorm(name='bn4b19_branch2b', data=res4b19_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b19_branch2b = bn4b19_branch2b
        res4b19_branch2b_relu = mx.symbol.Activation(name='res4b19_branch2b_relu', data=scale4b19_branch2b,
                                                     act_type='relu')
        res4b19_branch2c = mx.symbol.Convolution(name='res4b19_branch2c', data=res4b19_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2c = mx.symbol.BatchNorm(name='bn4b19_branch2c', data=res4b19_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b19_branch2c = bn4b19_branch2c
        res4b19 = mx.symbol.broadcast_add(name='res4b19', *[res4b18_relu, scale4b19_branch2c])
        res4b19_relu = mx.symbol.Activation(name='res4b19_relu', data=res4b19, act_type='relu')
        res4b20_branch2a = mx.symbol.Convolution(name='res4b20_branch2a', data=res4b19_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2a = mx.symbol.BatchNorm(name='bn4b20_branch2a', data=res4b20_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b20_branch2a = bn4b20_branch2a
        res4b20_branch2a_relu = mx.symbol.Activation(name='res4b20_branch2a_relu', data=scale4b20_branch2a,
                                                     act_type='relu')
        res4b20_branch2b = mx.symbol.Convolution(name='res4b20_branch2b', data=res4b20_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b20_branch2b = mx.symbol.BatchNorm(name='bn4b20_branch2b', data=res4b20_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b20_branch2b = bn4b20_branch2b
        res4b20_branch2b_relu = mx.symbol.Activation(name='res4b20_branch2b_relu', data=scale4b20_branch2b,
                                                     act_type='relu')
        res4b20_branch2c = mx.symbol.Convolution(name='res4b20_branch2c', data=res4b20_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2c = mx.symbol.BatchNorm(name='bn4b20_branch2c', data=res4b20_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b20_branch2c = bn4b20_branch2c
        res4b20 = mx.symbol.broadcast_add(name='res4b20', *[res4b19_relu, scale4b20_branch2c])
        res4b20_relu = mx.symbol.Activation(name='res4b20_relu', data=res4b20, act_type='relu')
        res4b21_branch2a = mx.symbol.Convolution(name='res4b21_branch2a', data=res4b20_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2a = mx.symbol.BatchNorm(name='bn4b21_branch2a', data=res4b21_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b21_branch2a = bn4b21_branch2a
        res4b21_branch2a_relu = mx.symbol.Activation(name='res4b21_branch2a_relu', data=scale4b21_branch2a,
                                                     act_type='relu')
        res4b21_branch2b = mx.symbol.Convolution(name='res4b21_branch2b', data=res4b21_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b21_branch2b = mx.symbol.BatchNorm(name='bn4b21_branch2b', data=res4b21_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b21_branch2b = bn4b21_branch2b
        res4b21_branch2b_relu = mx.symbol.Activation(name='res4b21_branch2b_relu', data=scale4b21_branch2b,
                                                     act_type='relu')
        res4b21_branch2c = mx.symbol.Convolution(name='res4b21_branch2c', data=res4b21_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2c = mx.symbol.BatchNorm(name='bn4b21_branch2c', data=res4b21_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b21_branch2c = bn4b21_branch2c
        res4b21 = mx.symbol.broadcast_add(name='res4b21', *[res4b20_relu, scale4b21_branch2c])
        res4b21_relu = mx.symbol.Activation(name='res4b21_relu', data=res4b21, act_type='relu')
        res4b22_branch2a = mx.symbol.Convolution(name='res4b22_branch2a', data=res4b21_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2a = mx.symbol.BatchNorm(name='bn4b22_branch2a', data=res4b22_branch2a,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b22_branch2a = bn4b22_branch2a
        res4b22_branch2a_relu = mx.symbol.Activation(name='res4b22_branch2a_relu', data=scale4b22_branch2a,
                                                     act_type='relu')
        res4b22_branch2b = mx.symbol.Convolution(name='res4b22_branch2b', data=res4b22_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b22_branch2b = mx.symbol.BatchNorm(name='bn4b22_branch2b', data=res4b22_branch2b,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b22_branch2b = bn4b22_branch2b
        res4b22_branch2b_relu = mx.symbol.Activation(name='res4b22_branch2b_relu', data=scale4b22_branch2b,
                                                     act_type='relu')
        res4b22_branch2c = mx.symbol.Convolution(name='res4b22_branch2c', data=res4b22_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2c = mx.symbol.BatchNorm(name='bn4b22_branch2c', data=res4b22_branch2c,
                                              use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale4b22_branch2c = bn4b22_branch2c
        res4b22 = mx.symbol.broadcast_add(name='res4b22', *[res4b21_relu, scale4b22_branch2c])
        res4b22_relu = mx.symbol.Activation(name='res4b22_relu', data=res4b22, act_type='relu')
        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=res4b22_relu, num_filter=2048, pad=(0, 0),
                                              kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1,
                                           use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5a_branch1 = bn5a_branch1
        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=res4b22_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5a_branch2a = bn5a_branch2a
        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')
        res5a_branch2b = mx.symbol.Convolution(name='res5a_branch2b', data=res5a_branch2a_relu, num_filter=512,
                                               pad=(2, 2), dilate=(2, 2), kernel=(3, 3), stride=(1, 1), no_bias=True,
                                               cudnn_off=True)
        bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b, act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu, num_filter=2048,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5a_branch2c = bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1, scale5a_branch2c])
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a, act_type='relu')
        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a, act_type='relu')
        res5b_branch2b = mx.symbol.Convolution(name='res5b_branch2b', data=res5b_branch2a_relu, num_filter=512,
                                               pad=(2, 2), dilate=(2, 2), kernel=(3, 3), stride=(1, 1), no_bias=True,
                                               cudnn_off=True)
        bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b, act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu, num_filter=2048,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5b_branch2c = bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu, scale5b_branch2c])
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b, act_type='relu')
        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a, act_type='relu')
        res5c_branch2b = mx.symbol.Convolution(name='res5c_branch2b', data=res5c_branch2a_relu, num_filter=512,
                                               pad=(2, 2), dilate=(2, 2), kernel=(3, 3), stride=(1, 1), no_bias=True,
                                               cudnn_off=True)
        bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b, act_type='relu')
        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu, num_filter=2048,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c,
                                            use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        scale5c_branch2c = bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu, scale5c_branch2c])
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c, act_type='relu')
        feat_conv_3x3 = mx.sym.Convolution(data=res5c_relu, kernel=(3, 3), pad=(6, 6), dilate=(6, 6), num_filter=1024,
                                           name="feat_conv_3x3", cudnn_off=True)
        feat_conv_3x3_relu = mx.sym.Activation(data=feat_conv_3x3, act_type="relu", name="feat_conv_3x3_relu")
        return feat_conv_3x3_relu

    # feature quality for feat1 and feat2
    def quality_network(self, feat1, feat2, suffix='', output_name=''):
        feature_concate = mx.sym.Concat(*[feat1, feat2], dim=0)
        quality_conv1 = mx.sym.Convolution(data=feature_concate, kernel=(1, 1), pad=(0, 0), num_filter=256,
                                           name='quality_conv1_%s' % (suffix))
        quality_conv2 = mx.sym.Convolution(data=quality_conv1, kernel=(3, 3), pad=(1, 1), num_filter=16,
                                           name='quality_conv2_%s' % (suffix))
        quality_conv3 = mx.sym.Convolution(data=quality_conv2, kernel=(3, 3), pad=(1, 1), num_filter=1,
                                           name='quality_conv3_%s' % (suffix))
        quality_weights = mx.sym.SliceChannel(data=quality_conv3, axis=0, num_outputs=2)
        quality_weights = mx.sym.SoftmaxActivation(data=mx.sym.Concat(*[quality_weights[0], quality_weights[1]], dim=1),
                                                   mode='channel', name='quality_weights_softmax_%s' % (suffix))
        quality_weights_slice = mx.sym.SliceChannel(data=quality_weights, axis=1, num_outputs=2)
        quality_weight1 = mx.sym.tile(data=quality_weights_slice[0], reps=(1, 1024, 1, 1),
                                      name='quality_weight1_%s' % (suffix))
        quality_weight2 = mx.sym.tile(data=quality_weights_slice[1], reps=(1, 1024, 1, 1),
                                      name='quality_weight2_%s' % (suffix))
        feat_result = mx.sym.elemwise_add(quality_weight1 * feat1, quality_weight2 * feat2, name=output_name)
        return feat_result
    def align_conv_feat_dynamic_local(self, offset, conv_feat_oldkey, conv_feat_newkey, neighbor=4, prefix='key2key'):
        embedding_conv_feat_oldkey = mx.sym.Convolution(data=conv_feat_oldkey, kernel=(1, 1), pad=(0, 0),
                                                        num_filter=256, name='embedding_conv_feat_oldkey_%s' % prefix)
        embedding_conv_feat_newkey = mx.sym.Convolution(data=conv_feat_newkey, kernel=(1, 1), pad=(0, 0),
                                                        num_filter=256, name='embedding_conv_feat_newkey_%s' % prefix)
        # local correlation

        # embedding_conv_feat_oldkey_norm = mx.sym.L2Normalization(data=embedding_conv_feat_oldkey,mode='channel',name='embedding_conv_feat_oldkey_l2norm_%s' % prefix)

        # embedding_conv_feat_newkey_norm = mx.sym.L2Normalization(data=embedding_conv_feat_newkey,mode='channel',name='embedding_conv_feat_newkey_l2norm_%s' % prefix)

        local_correlation = mx.sym.Correlation(data1=embedding_conv_feat_newkey, data2=embedding_conv_feat_oldkey,
                                               stride1=1, stride2=1,
                                               is_multiply=True, max_displacement=neighbor, pad_size=neighbor,
                                               kernel_size=1, name='local_correlation_%s' % prefix) * 256.0
        local_weights = mx.sym.softmax(data=local_correlation, axis=1, name='softmax_%s' % prefix)
        # align using weight
        align_feat_newkey = mx.sym.contrib.WeightPropagation(data=conv_feat_oldkey, weights=local_weights,
                                                             weight_height=2 * neighbor + 1,
                                                             weight_width=2 * neighbor + 1, name='align_feat_local')
        return align_feat_newkey


    def align_conv_feat_dynamic(self, offset, conv_feat_oldkey, conv_feat_newkey, prefix='key2key'):
        embedding_conv_feat_oldkey = mx.sym.Convolution(data=conv_feat_oldkey, kernel=(1, 1), pad=(0, 0),
                                                        num_filter=256, name='embedding_conv_feat_oldkey_%s' % prefix)
        embedding_conv_feat_newkey = mx.sym.Convolution(data=conv_feat_newkey, kernel=(1, 1), pad=(0, 0),
                                                        num_filter=256, name='embedding_conv_feat_newkey_%s' % prefix)
        # N * K * C * H * W
        bilinear_sample_k_embedding_oldkey = mx.sym.contrib.BilinearSampleK(data=embedding_conv_feat_oldkey,
                                                                            offset=offset,
                                                                            name='bilinear_sample_k_embedding_oldkey_%s' % prefix)

        weight_dot_k = mx.sym.contrib.WeightGenerateK(data=embedding_conv_feat_newkey, data_ref=bilinear_sample_k_embedding_oldkey,
                                                      name='weight_dot_k_%s' % prefix)
        weight_softmax_k = mx.sym.softmax(data=weight_dot_k, axis=1, name='weight_softmax_k_%s' % prefix)

        # N * K * C2 * H * W
        oldkey_bilinear = mx.sym.contrib.BilinearSampleK(data=conv_feat_oldkey, offset=offset,
                                                         name='bilinear_sample_k_oldkey_%s' % prefix)
        align_feat = mx.sym.contrib.AlignData(data=oldkey_bilinear, weight=weight_softmax_k,name='align_feat')
        return align_feat

    def low2high(self, low_feat):
        highlevel_feat1 = mx.sym.Convolution(data=low_feat, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_filter=256,
                                             no_bias=False, name='low2high_conv1')
        highlevel_feat1_relu = mx.sym.relu(data=highlevel_feat1, name='highlevel_feat1_relu')
        highlevel_feat2 = mx.sym.Convolution(data=highlevel_feat1_relu, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                             num_filter=256, no_bias=False, name='low2high_conv2')
        highlevel_feat2_relu = mx.sym.relu(data=highlevel_feat2, name='highlevel_feat2_relu')
        highlevel_feat3 = mx.sym.Convolution(data=highlevel_feat2_relu, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                             num_filter=1024, no_bias=False, name='low2high_conv3')
        return highlevel_feat3

    def get_train_symbol_impression(self, cfg):
        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS
        N = cfg.default.num_locations
        data_oldkey = mx.sym.Variable(name="data_oldkey")
        data_newkey = mx.sym.Variable(name="data_newkey")
        data_cur = mx.sym.Variable(name="data_cur")
        eq_flag_key2cur = mx.sym.Variable(name="eq_flag_key2cur")
        eq_flag_key2key = mx.sym.Variable(name="eq_flag_key2key")
        im_info = mx.sym.Variable(name="im_info")
        gt_boxes = mx.sym.Variable(name="gt_boxes")
        rpn_label = mx.sym.Variable(name='label')
        rpn_bbox_target = mx.sym.Variable(name='bbox_target')
        rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')

        # offset params
        #init_offset = mx.sym.Variable(name='init_offset', shape=(N, 2))
        #offset = mx.sym.Variable(name='offset', shape=(N, 2))
        offset = mx.sym.Variable(name='init_offset', shape=(N, 2), lr_mult=10000*cfg.TRAIN.lr)

        #offset = mx.sym.elemwise_add(offset, init_offset, name='offset_add'
        # shared convolution layers
        features_low_level = self.get_resnet_low_feature(mx.sym.Concat(*[data_oldkey, data_newkey, data_cur], dim=0))
        features_low_level_slice = mx.sym.SliceChannel(features_low_level, num_outputs=3, axis=0)

        features = self.get_resnet_high_feature(mx.sym.Concat(*[features_low_level_slice[0], features_low_level_slice[1]], dim=0))
        features = mx.sym.SliceChannel(features, num_outputs=2, axis=0)
        conv_feat_oldkey = features[0]
        conv_feat_newkey = features[1]

        # align conv_feat_oldkey to conv_feat_newkey
        align_conv_feat_newkey = self.align_conv_feat_dynamic(offset, conv_feat_oldkey, conv_feat_newkey, prefix='key2key')
        # key2key quality networks
        feat_task = self.quality_network(conv_feat_newkey, align_conv_feat_newkey, suffix='key2key',
                                         output_name='feat_task')
        # task_feat = conv_feat_newkey
        feat_task = mx.sym.take(mx.sym.Concat(*[feat_task, conv_feat_oldkey], dim=0), eq_flag_key2key)

        # transform current low-level feat to high-level feat
        high_feat_cur = self.low2high(features_low_level_slice[2])
        # align feat_task to current high-level feat
        align_conv_feat_cur = self.align_conv_feat_dynamic(offset, feat_task, high_feat_cur, prefix='key2cur')
        # key2cur quality networks
        align_conv_feat_task_cur = self.quality_network(high_feat_cur, align_conv_feat_cur, suffix='key2cur',
                                                        output_name='feat_task_cur')
        # current frame == new keyframe
        align_conv_feat_task = mx.sym.take(mx.sym.Concat(*[align_conv_feat_task_cur, feat_task], dim=0),
                                           eq_flag_key2cur)

        conv_feats = mx.sym.SliceChannel(data=align_conv_feat_task, axis=1, num_outputs=2)
        # RPN layers
        rpn_feat = conv_feats[0]
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
        # prepare rpn data
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
        # classification
        rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                            normalization='valid', use_ignore=True, ignore_label=-1,
                                            name="rpn_cls_prob")
        # bounding box regression
        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=1.0,
                                                                data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)
        else:
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0,
                                                                data=(rpn_bbox_pred - rpn_bbox_target))
        rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                        grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)
        # ROI proposal
        rpn_cls_act = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
        rpn_cls_act_reshape = mx.sym.Reshape(
            data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')

        if cfg.TRAIN.CXX_PROPOSAL:
            rois = mx.contrib.sym.Proposal(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
        else:
            rois = mx.sym.Custom(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)

            # ROI proposal target
        gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
        rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape,
                                                              op_type='proposal_target',
                                                              num_classes=num_reg_classes,
                                                              batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                              batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                              cfg=cPickle.dumps(cfg),
                                                              fg_fraction=cfg.TRAIN.FG_FRACTION)

        # res5
        rfcn_feat = conv_feats[1]
        rfcn_cls = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7 * 7 * num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7 * 7 * 4 * num_reg_classes,
                                       name="rfcn_bbox")
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois,
                                                           group_size=7,
                                                           pooled_size=7,
                                                           output_dim=num_classes, spatial_scale=0.0625)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois,
                                                           group_size=7,
                                                           pooled_size=7,
                                                           output_dim=8, spatial_scale=0.0625)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg',
                                   global_pool=True,
                                   kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg',
                                   global_pool=True,
                                   kernel=(7, 7))
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        # classification
        if cfg.TRAIN.ENABLE_OHEM:
            print 'use ohem!'
            labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                           num_reg_classes=num_reg_classes,
                                                           roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                           cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                           bbox_targets=bbox_target, bbox_weights=bbox_weight)
            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid',
                                            use_ignore=True, ignore_label=-1)
            bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                              data=(bbox_pred - bbox_target))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
            rcnn_label = labels_ohem
        else:
            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid')
            bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
            rcnn_label = label

        # reshape output
        rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                  name='cls_prob_reshape')
        bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                   name='bbox_loss_reshape')

        group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        self.sym = group
        return group

    def get_first_test_symbol_impression(self, cfg):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        data_oldkey = mx.sym.Variable(name="data_oldkey")
        data_newkey = mx.sym.Variable(name="data_newkey")
        data_cur = mx.sym.Variable(name="data_cur")
        im_info = mx.sym.Variable(name="im_info")
        impression = mx.sym.Variable(name="impression")
        key_feat_task = mx.sym.Variable(name="key_feat_task")


        # shared convolutional layers
        conv_feat_low_level = self.get_resnet_low_feature(data_newkey)
        conv_feat = self.get_resnet_high_feature(conv_feat_low_level)
        conv_feats = mx.sym.SliceChannel(conv_feat, axis=1, num_outputs=2)
        # RPN
        rpn_feat = conv_feats[0]
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)

        # ROI Proposal
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
        rpn_cls_prob_reshape = mx.sym.Reshape(
            data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
        if cfg.TEST.CXX_PROPOSAL:
            rois = mx.contrib.sym.Proposal(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
        else:
            rois = mx.sym.Custom(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
        # res5
        rfcn_feat = conv_feats[1]
        rfcn_cls = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7 * 7 * num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7 * 7 * 4 * num_reg_classes,
                                       name="rfcn_bbox")
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois,
                                                           group_size=7, pooled_size=7,
                                                           output_dim=num_classes, spatial_scale=0.0625)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois,
                                                           group_size=7, pooled_size=7,
                                                           output_dim=8, spatial_scale=0.0625)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg',
                                   global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg',
                                   global_pool=True, kernel=(7, 7))

        # classification
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
        # bounding box regression
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))
        # reshape output
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                  name='cls_prob_reshape')
        bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                   name='bbox_pred_reshape')
        # group output
        group = mx.sym.Group([data_oldkey, data_newkey, data_cur, impression, key_feat_task, conv_feat, rois, cls_prob, bbox_pred])
        self.sym = group
        return group

    def get_key_test_symbol_impression(self, cfg):
        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        data_oldkey = mx.sym.Variable(name="data_oldkey")
        data_newkey = mx.sym.Variable(name="data_newkey")
        data_cur = mx.sym.Variable(name="data_cur")
        im_info = mx.sym.Variable(name="im_info")
        impression = mx.sym.Variable(name="impression")
        key_feat_task = mx.sym.Variable(name="key_feat_task")

        N = cfg.default.num_locations
        #init_offset = mx.sym.Variable(name='init_offset', shape=(N, 2))
        #offset = mx.sym.Variable(name='offset', shape=(N, 2))
        offset = mx.sym.Variable(name='init_offset', shape=(N, 2))

        #offset = mx.sym.elemwise_add(offset, init_offset, name='offset_add')

        # weight predicor
        feat_low_level_newkey = self.get_resnet_low_feature(data_newkey)
        conv_feat_newkey = self.get_resnet_high_feature(feat_low_level_newkey)

        # aligin conv_feat_oldkey to conv_feat_newkey
        align_conv_feat_newkey = self.align_conv_feat_dynamic(offset, impression, conv_feat_newkey, prefix='key2key')
        # key2key quality networks
        feat_task = self.quality_network(conv_feat_newkey, align_conv_feat_newkey, suffix='key2key',
                                         output_name='feat_task')

        # update impression
        impression_update = mx.sym.elemwise_add(cfg.TEST.MEMORY_UPDATE_RATIO * feat_task,
                                                (1 - cfg.TEST.MEMORY_UPDATE_RATIO) * conv_feat_newkey,
                                                name='impression_update')
        conv_feats = mx.sym.SliceChannel(feat_task, axis=1, num_outputs=2)
        # RPN
        rpn_feat = conv_feats[0]
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)

        # ROI Proposal
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
        rpn_cls_prob_reshape = mx.sym.Reshape(
            data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
        if cfg.TEST.CXX_PROPOSAL:
            rois = mx.contrib.sym.Proposal(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
        else:
            rois = mx.sym.Custom(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

        # res5
        rfcn_feat = conv_feats[1]
        rfcn_cls = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7 * 7 * num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7 * 7 * 4 * num_reg_classes,
                                       name="rfcn_bbox")
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois,
                                                           group_size=7, pooled_size=7,
                                                           output_dim=num_classes, spatial_scale=0.0625)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois,
                                                           group_size=7, pooled_size=7,
                                                           output_dim=8, spatial_scale=0.0625)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg',
                                   global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg',
                                   global_pool=True, kernel=(7, 7))

        # classification
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
        # bounding box regression
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        # reshape output
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                  name='cls_prob_reshape')
        bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                   name='bbox_pred_reshape')
        # group output
        group = mx.sym.Group([data_oldkey, data_newkey, data_cur, impression, key_feat_task, feat_task, impression_update, rois, cls_prob, bbox_pred])
        self.sym = group
        return group

    def get_cur_test_symbol_impression(self, cfg):
        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        data_oldkey = mx.sym.Variable(name="data_oldkey")
        data_newkey = mx.sym.Variable(name="data_newkey")
        data_cur = mx.sym.Variable(name="data_cur")
        im_info = mx.sym.Variable(name="im_info")
        impression = mx.sym.Variable(name="impression")
        key_feat_task = mx.sym.Variable(name="key_feat_task")

        N = cfg.default.num_locations
        # offset params
        #init_offset = mx.sym.Variable(name='init_offset', shape=(N, 2))
        #offset = mx.sym.Variable(name='offset', shape=(N, 2))
        offset = mx.sym.Variable(name='init_offset', shape=(N, 2))

        #offset = mx.sym.elemwise_add(offset, init_offset, name='offset_add')

        features_low_level_cur = self.get_resnet_low_feature(data_cur)

        # transform current low-level feat to high-level feat
        high_feat_cur = self.low2high(features_low_level_cur)
        # align feat_task to current high-level feat
        conv_feat_cur = self.align_conv_feat_dynamic(offset, key_feat_task, high_feat_cur, prefix='key2cur')
        # key2cur quality networks
        align_conv_feat_cur = self.quality_network(high_feat_cur, conv_feat_cur, suffix='key2cur',
                                                   output_name='feat_task_cur')

        conv_feats = mx.sym.SliceChannel(data=align_conv_feat_cur, axis=1, num_outputs=2)
        # RPN
        rpn_feat = conv_feats[0]
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)

        # ROI Proposal
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
        rpn_cls_prob_reshape = mx.sym.Reshape(
            data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
        if cfg.TEST.CXX_PROPOSAL:
            rois = mx.contrib.sym.Proposal(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
        else:
            rois = mx.sym.Custom(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

        # res5
        rfcn_feat = conv_feats[1]
        rfcn_cls = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7 * 7 * num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7 * 7 * 4 * num_reg_classes,
                                       name="rfcn_bbox")
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois,
                                                           group_size=7, pooled_size=7,
                                                           output_dim=num_classes, spatial_scale=0.0625)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois,
                                                           group_size=7, pooled_size=7,
                                                           output_dim=8, spatial_scale=0.0625)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg',
                                   global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg',
                                   global_pool=True, kernel=(7, 7))

        # classification
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
        # bounding box regression
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        # reshape output
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                  name='cls_prob_reshape')
        bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                   name='bbox_pred_reshape')

        # group output
        group = mx.sym.Group([data_oldkey, data_newkey, data_cur, impression, key_feat_task, rois, cls_prob, bbox_pred])
        self.sym = group
        return group

    def init_weight(self, cfg, arg_params, aux_params):
        arg_params['feat_conv_3x3_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['feat_conv_3x3_weight'])
        arg_params['feat_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['feat_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])
        arg_params['rfcn_cls_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_cls_weight'])
        arg_params['rfcn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_bias'])
        arg_params['rfcn_bbox_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_bbox_weight'])
        arg_params['rfcn_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_bias'])

        # quality key2key
        arg_params['quality_conv1_key2key_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'quality_conv1_key2key_weight'])
        arg_params['quality_conv1_key2key_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['quality_conv1_key2key_bias'])
        arg_params['quality_conv2_key2key_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'quality_conv2_key2key_weight'])
        arg_params['quality_conv2_key2key_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['quality_conv2_key2key_bias'])
        arg_params['quality_conv3_key2key_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'quality_conv3_key2key_weight'])
        arg_params['quality_conv3_key2key_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['quality_conv3_key2key_bias'])

        # quality key2cur
        arg_params['quality_conv1_key2cur_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'quality_conv1_key2cur_weight'])
        arg_params['quality_conv1_key2cur_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['quality_conv1_key2cur_bias'])
        arg_params['quality_conv2_key2cur_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'quality_conv2_key2cur_weight'])
        arg_params['quality_conv2_key2cur_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['quality_conv2_key2cur_bias'])
        arg_params['quality_conv3_key2cur_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'quality_conv3_key2cur_weight'])
        arg_params['quality_conv3_key2cur_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['quality_conv3_key2cur_bias'])

        # embedding conv_feat
        arg_params['embedding_conv_feat_oldkey_key2key_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'embedding_conv_feat_oldkey_key2key_weight'])
        arg_params['embedding_conv_feat_oldkey_key2key_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['embedding_conv_feat_oldkey_key2key_bias'])

        arg_params['embedding_conv_feat_oldkey_key2cur_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'embedding_conv_feat_oldkey_key2cur_weight'])
        arg_params['embedding_conv_feat_oldkey_key2cur_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['embedding_conv_feat_oldkey_key2cur_bias'])

        arg_params['embedding_conv_feat_newkey_key2key_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'embedding_conv_feat_newkey_key2key_weight'])
        arg_params['embedding_conv_feat_newkey_key2key_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['embedding_conv_feat_newkey_key2key_bias'])

        arg_params['embedding_conv_feat_newkey_key2cur_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'embedding_conv_feat_newkey_key2cur_weight'])
        arg_params['embedding_conv_feat_newkey_key2cur_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['embedding_conv_feat_newkey_key2cur_bias'])

        # low2high
        arg_params['low2high_conv1_weight'] = mx.random.normal(0, 0.01,
                                                               shape=self.arg_shape_dict['low2high_conv1_weight'])
        arg_params['low2high_conv1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['low2high_conv1_bias'])
        arg_params['low2high_conv2_weight'] = mx.random.normal(0, 0.01,
                                                               shape=self.arg_shape_dict['low2high_conv2_weight'])
        arg_params['low2high_conv2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['low2high_conv2_bias'])
        arg_params['low2high_conv3_weight'] = mx.random.normal(0, 0.01,
                                                               shape=self.arg_shape_dict['low2high_conv3_weight'])
        arg_params['low2high_conv3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['low2high_conv3_bias'])

        neighbor = cfg.default.neighboor
        K = cfg.default.num_locations
        init_offset = []
        for x in range(-neighbor,neighbor+1):
            for y in range(-neighbor,neighbor+1):
               init_offset.append([x, y])

        init_offset = np.array(init_offset)
        arg_params['init_offset'] = mx.nd.array(init_offset)
        #arg_params['offset'] = mx.nd.zeros(shape=(K, 2))




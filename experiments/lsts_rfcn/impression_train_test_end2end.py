# --------------------------------------------------------
# Online Memory Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zhengkai Jiang
# --------------------------------------------------------
import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

#os.environ['MXNET_PROFILER_AUTOSTART'] = '1'
#os.environ['MXNET_EXEC_BULK_EXEC_TRAIN'] = '0'
#os.environ['MXNET_EXEC_BULK_EXEC_INFERENCE'] = 'false'


this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'impression_rfcn'))

# import matplotlib
# matplotlib.use('Agg')
import impression_train_end2end
import impression_test

if __name__ == "__main__":
    impression_train_end2end.main()
    #impression_test.main()

import os.path as osp
import sys
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..', '3rdparty')
add_path(lib_path)
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:26:35 2018

@author: kaikaijiang.jzk
"""
import mxnet as mx
import numpy as np
import os
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
class SpatiallyConvolution(mx.operator.CustomOp):
    """
    pad: pad in default set padding  value =0
    num_filter: num_filter
    kernel_size: (kernel,kernel)
    ctx: context
    """
    def __init__(self, pad, kernel):
        super(SpatiallyConvolution, self).__init__()
        self.pad = int(pad)
        self.kernel_size = int(kernel)
    def forward(self, is_train, req, in_data, out_data, aux):
        #data shape (1,c,h,w)
        data = in_data[0].asnumpy()
        #kernels shape (1,k*k,h,w)
        kernels = in_data[1].asnumpy()
        
        channel = kernels.shape[1] # k*k
        height = kernels.shape[2] # h
        width = kernels.shape[3] # w
        # padding 0 value to make shape insistently with output
        data = np.pad(data,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)),'constant')
        #print('data',data)
        kernels = kernels.reshape((channel,height*width)) # (k*k,h*w)
        i = 0 # kernel flags 
        out = np.zeros((data.shape[0],data.shape[1],height,width))
        for n in range(data.shape[0]):
            for c in range(data.shape[1]):
                for h in range(height):
                    for w in range(width):
                        i = h*width+w
                        kernel = kernels[:,i].reshape((self.kernel_size,self.kernel_size))
                        out[n,c,h,w] = np.sum(data[n,c,h:self.kernel_size+h,w:self.kernel_size+w]*kernel)
        self.assign(out_data[0],req[0],mx.nd.array(out))
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        data = in_data[0].asnumpy() # 1*c*h*w
        kernels = in_data[1].asnumpy() # 1*(k*k)*h*w
        batch = data.shape[0]
        channel = data.shape[1]
        height = data.shape[2]
        width = data.shape[3]
        data = np.pad(data,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)),'constant')
        out_grad = out_grad[0].asnumpy()
        data_grad = np.zeros((batch,channel,height+2*self.pad,width+2*self.pad))
        kernel_grad = np.zeros((batch,kernels.shape[1],height*width))
        i = 0
        for n in range(batch):
            for c in range(channel):
                for h in range(height):
                    for w in range(width):
                        i = h*width+w
                        kernel = kernels[:,i].reshape((self.kernel_size,self.kernel_size))
                        data_grad_value=out_grad[n,c,h,w]*kernel
                        data_grad[n,c,h:h+self.kernel_size,w:w+self.kernel_size] += data_grad_value
                        kernel_grad_value=out_grad[n,c,h,w]*data[n,c,h:self.kernel_size+h,w:self.kernel_size+w]
                        #print('data_grad_value',data[n,c,h:self.kernel_size+h,w:self.kernel_size+w])
                        kernel_grad[n,:,i]+=kernel_grad_value.reshape(self.kernel_size*self.kernel_size)
        self.assign(in_grad[0],req[0],mx.nd.array(data_grad[:,:,self.pad:self.pad+height,self.pad:self.pad+width]))
        self.assign(in_grad[1],req[1],mx.nd.array(kernel_grad.reshape(batch,kernels.shape[1],height,width)))
@mx.operator.register('SpatiallyConvolution')
class SpatiallyConvolutionProp(mx.operator.CustomOpProp):
    def __init__(self, pad, kernel):
        super(SpatiallyConvolutionProp, self).__init__(need_top_grad=True)
        self.pad = pad
        self.kernel = kernel
    def list_arguments(self):
        return ['data', 'kernels']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        kernel_shape = in_shape[1]
        output_shape = in_shape[0]
        return [data_shape, kernel_shape], [output_shape], []
    def infer_type(self, in_type):
        return in_type, [in_type[0]], []
    def create_operator(self, ctx, shapes, dtypes):
        return SpatiallyConvolution(self.pad,self.kernel)
    
    
    



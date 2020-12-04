/*!
 * Copyright (c) 2015 by Contributors
 * \file weight_generate-inl.h
 * \brief
 * \author ZhengKai Jiang
*/
#include "./align_data-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"


namespace mshadow {
namespace cuda {

inline __device__ int offset5d(int n, int k, int c, int h, int w, int N, int K, int C, int H, int W) {
    return n*K*C*H*W + k*C*H*W + c*H*W + h*W + w;
}

inline __device__ int offset(int n, int c, int h, int w, int N, int C, int H, int W) {
    return n*C*H*W + c*H*W + h*W + w;
}


template<typename DType>
__global__ void AlignDataForwardKernel(const int count, int N, int K, int C, int H, int W,
                                             const DType* bottom_data, const DType* bottom_weight, DType* top_data ) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
      index < count;
      index += blockDim.x * gridDim.x * gridDim.y) {

    const int w = index % W;
    const int h = (index / W) % H;
    const int c = (index / (H * W)) % C;
    const int n = (index / (C * H * W));

    for (int i=0;i<K;i++) {
      *(top_data + index) += bottom_data[offset5d(n,i,c,h,w,N,K,C,H,W)]*bottom_weight[offset(n,i,h,w,N,K,H,W)];
    }

  } // cuda_kernel_loop
}

template<typename DType>
inline void AlignDataPointsForward(const Tensor<gpu, 4, DType> &out,
                                   const Tensor<gpu, 5, DType> &data,
                                   const Tensor<gpu, 4, DType> &weight) {
  const DType *bottom_data = data.dptr_;
  const DType *bottom_weight = weight.dptr_;
  DType *top_data = out.dptr_;
  const int count = out.shape_.Size(); // the number of threads
  int N = data.size(0);
  int K = data.size(1);
  int C = data.size(2);
  int H = data.size(3);
  int W = data.size(4);

  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "Align Data Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  AlignDataForwardKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(count, N, K, C, H, W, bottom_data, bottom_weight, top_data);
}


template<typename DType>
__global__ void AlignDataBackwardKernel(const int count, int N, int K, int C, int H, int W,
                                              const DType* grad_out, const DType* bottom_data,
                                              const DType* bottom_weight, DType* grad_data, DType* grad_weight) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    const int w = index % W;
    const int h = (index / W) % H;
    const int c = (index / (H * W)) % C;
    const int n = (index / (C * H * W));

    for (int i=0;i<K;i++) {
      atomicAdd(grad_data+offset5d(n,i,c,h,w,N,K,C,H,W),grad_out[index]*bottom_weight[offset(n,i,h,w,N,K,H,W)]);
      atomicAdd(grad_weight+offset(n,i,h,w,N,K,H,W),grad_out[index]*bottom_data[offset5d(n,i,c,h,w,N,K,C,H,W)]);
      }
    }
}


template<typename DType>
inline void AlignDataBackward(const Tensor<gpu, 5, DType> &grad_data,
                              const Tensor<gpu, 4, DType> &grad_weight,
                              const Tensor<gpu, 5, DType> &data,
                              const Tensor<gpu, 4, DType> &weight,
                              const Tensor<gpu, 4, DType> &grad_out) {
  const DType *top_grad = grad_out.dptr_;
  const DType *bottom_data = data.dptr_;
  const DType *bottom_weight = weight.dptr_;
  DType *bottom_grad_data = grad_data.dptr_;
  DType *bottom_grad_weight = grad_weight.dptr_;

  const int count = grad_out.shape_.Size(); // the number of threads

  int N = data.size(0);
  int K = data.size(1);
  int C = data.size(2);
  int H = data.size(3);
  int W = data.size(4);


  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "Align Data Backward");
  cudaStream_t stream_data = Stream<gpu>::GetStream(grad_data.stream_);

  AlignDataBackwardKernel<DType><<<dimGrid, dimBlock, 0, stream_data>>>(count, N, K, C, H, W,
                                        top_grad, bottom_data, bottom_weight, bottom_grad_data, bottom_grad_weight);

}



} // namespace cuda

template<typename DType>
inline void AlignDataForward(const Tensor<gpu, 4, DType> &out,
                             const Tensor<gpu, 5, DType> &data,
                             const Tensor<gpu, 4, DType> &weight) {
  cuda::AlignDataPointsForward(out, data, weight);
}

template<typename DType>
inline void AlignDataBackward(const Tensor<gpu, 5, DType> &grad_data,
                              const Tensor<gpu, 4, DType> &grad_weight,
                              const Tensor<gpu, 5, DType> &data,
                              const Tensor<gpu, 4, DType> &weight,
                              const Tensor<gpu, 4, DType> &grad_out) {
  cuda::AlignDataBackward(grad_data, grad_weight, data, weight, grad_out);
}


} //namespace mshadow


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(AlignDataParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new AlignDataOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
} // namespace mxnet

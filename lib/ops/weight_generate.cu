/*!
 * Copyright (c) 2015 by Contributors
 * \file weight_generate-inl.h
 * \brief
 * \author ZhengKai Jiang
*/
#include "./weight_generate-inl.h"
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
__device__ bool between(DType value, int lowerBound, int upperBound) {
  return (value >= lowerBound && value <= upperBound);
}

template<typename DType>
__global__ void WeightGenerateKForwardKernel(const int count, int N, int K, int C, int H, int W,
                                             const DType* bottom_data, const DType* bottom_data_ref, DType* top_data ) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
      index < count;
      index += blockDim.x * gridDim.x * gridDim.y) {

    const int w = index % W;
    const int h = (index / W) % H;
    const int k = (index / (H * W)) % K;
    const int n = (index / (K * H * W));

    for (int i=0;i<C;i++) {
      *(top_data + index) += bottom_data[offset(n,i,h,w,N,C,H,W)]*bottom_data_ref[offset5d(n,k,i,h,w,N,K,C,H,W)];
    }

  } // cuda_kernel_loop
}

template<typename DType>
inline void WeightGenerateKPointsForward(const Tensor<gpu, 4, DType> &out,
                                         const Tensor<gpu, 4, DType> &data,
                                         const Tensor<gpu, 5, DType> &data_ref) {
  const DType *bottom_data = data.dptr_;
  const DType *bottom_data_ref = data_ref.dptr_;
  DType *top_data = out.dptr_;
  const int count = out.shape_.Size(); // the number of threads
  int N = data_ref.size(0);
  int K = data_ref.size(1);
  int C = data_ref.size(2);
  int H = data_ref.size(3);
  int W = data_ref.size(4);

  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "Weight GenerateK Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  WeightGenerateKForwardKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(count, N, K, C, H, W, bottom_data, bottom_data_ref, top_data);
}


template<typename DType>
__global__ void WeightGenerateKBackwardKernel(const int count, int N, int K, int C, int H, int W,
                                              const DType* grad_out, const DType* bottom_data,
                                              const DType* bottom_data_ref, DType* grad_data, DType* grad_data_ref) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    const int w = index % W;
    const int h = (index / W) % H;
    const int k = (index / (H * W)) % K;
    const int n = (index / (K * H * W));

    for (int i=0;i<C;i++) {
      atomicAdd(grad_data+offset(n,i,h,w,N,C,H,W),grad_out[index]*bottom_data_ref[offset5d(n,k,i,h,w,N,K,C,H,W)]);
      atomicAdd(grad_data_ref+offset5d(n,k,i,h,w,N,K,C,H,W),grad_out[index]*bottom_data[offset(n,i,h,w,N,C,H,W)]);
      }
    }
}


template<typename DType>
inline void WeightGenerateKBackward(const Tensor<gpu, 4, DType> &grad_data,
                                    const Tensor<gpu, 5, DType> &grad_data_ref,
                                    const Tensor<gpu, 4, DType> &data,
                                    const Tensor<gpu, 5, DType> &data_ref,
                                    const Tensor<gpu, 4, DType> &grad_out) {
  const DType *top_grad = grad_out.dptr_;
  const DType *bottom_data = data.dptr_;
  const DType *bottom_data_ref = data_ref.dptr_;
  DType *bottom_grad_data = grad_data.dptr_;
  DType *bottom_grad_data_ref = grad_data_ref.dptr_;

  const int count = grad_out.shape_.Size(); // the number of threads

  int N = data_ref.size(0);
  int K = data_ref.size(1);
  int C = data_ref.size(2);
  int H = data_ref.size(3);
  int W = data_ref.size(4);


  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "Weight GenerateK Backward");
  cudaStream_t stream_data = Stream<gpu>::GetStream(grad_data.stream_);

  WeightGenerateKBackwardKernel<DType><<<dimGrid, dimBlock, 0, stream_data>>>(count, N, K, C, H, W,
                                        top_grad, bottom_data, bottom_data_ref, bottom_grad_data, bottom_grad_data_ref);

}



} // namespace cuda

template<typename DType>
inline void WeightGenerateKForward(const Tensor<gpu, 4, DType> &out,
                                   const Tensor<gpu, 4, DType> &data,
                                   const Tensor<gpu, 5, DType> &data_ref) {
  cuda::WeightGenerateKPointsForward(out, data, data_ref);
}

template<typename DType>
inline void WeightGenerateKBackward(const Tensor<gpu, 4, DType> &grad_data,
                                    const Tensor<gpu, 5, DType> &grad_data_ref,
                                    const Tensor<gpu, 4, DType> &data,
                                    const Tensor<gpu, 5, DType> &data_ref,
                                    const Tensor<gpu, 4, DType> &grad_out) {
  cuda::WeightGenerateKBackward(grad_data, grad_data_ref, data, data_ref, grad_out);
}


} //namespace mshadow


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(WeightGenerateKParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new WeightGenerateKOp<gpu, DType>(param);
  });
  return op;
}


}  // namespace op
} // namespace mxnet

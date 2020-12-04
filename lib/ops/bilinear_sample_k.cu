 /*!
 * Copyright (c) 2015 by Contributors
 * \file bilinear_sample_k-inl.h
 * \brief
 * \author ZhengKai Jiang
*/
#include "./bilinear_sample_k-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"


namespace mshadow {
namespace cuda {

inline __device__ int offset(int n, int c, int h, int w, int N, int C, int H, int W) {
    return n*C*H*W + c*H*W + h*W + w;
}

inline __device__ int offset5d(int n, int k, int c, int h, int w, int N, int K, int C, int H, int W) {
    return n*K*C*H*W + k*C*H*W + c*H*W + h*W + w;
}

template<typename DType>
__device__ bool between(DType value, int lowerBound, int upperBound) {
  return (value >= lowerBound && value <= upperBound);
}

template <typename Dtype>
__device__ Dtype roialign_bilinear_interp(const Dtype* data,
                                          Dtype x,
                                          Dtype y,
                                          int width,
                                          int height) {
  int x1 = floor(x);
  int y1 = floor(y);
  int x2 = x1 + 1;
  int y2 = y1 + 1;

  Dtype dist_x = static_cast<Dtype>(x - x1);
  Dtype dist_y = static_cast<Dtype>(y - y1);

  Dtype value11 = 0;
  Dtype value12 = 0;
  Dtype value21 = 0;
  Dtype value22 = 0;

  if (between(x1, 0, width-1) && between(y1, 0, height-1))
    value11 = data[y1*width + x1];
  if (between(x1, 0, width-1) && between(y2, 0, height-1))
    value12 = data[y2*width + x1];
  if (between(x2, 0, width-1) && between(y1, 0, height-1))
    value21 = data[y1*width + x2];
  if (between(x2, 0, width-1) && between(y2, 0, height-1))
    value22 = data[y2*width + x2];
  Dtype value = (1 - dist_x)*(1 - dist_y)*value11 + (1 - dist_x)*dist_y*value12 +
                 dist_x*(1 - dist_y)*value21 + dist_x*dist_y*value22;
  return value;
}

template<typename DType>
__global__ void BilinearSampleKForwardKernel(const int count, int N, int K, int C, int H, int W,
                                             const DType* bottom_data, const DType* bottom_offset, DType* top_data ) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
      index < count;
      index += blockDim.x * gridDim.x * gridDim.y) {

    int C1 = C / 8;
    const int w = index % W;
    const int h = (index / W) % H;
    const int c = (index / (H * W)) % C1;
    const int k = (index / (C1 * H * W)) % K;
    const int n = (index / (K* C1 * H * W));

    for(int i=0;i<8;i++){
        DType x_real = bottom_offset[2 * k] + w;
        DType y_real = bottom_offset[2 * k + 1] + h;

        const DType* data = bottom_data + n * (C * H * W) + (8*c+i) * (H * W);

        int x1 = floor(x_real);
        int y1 = floor(y_real);
        int x2 = x1 + 1;
        int y2 = y1 + 1;

        DType dist_x = static_cast<DType>(x_real - x1);
        DType dist_y = static_cast<DType>(y_real - y1);

        DType value11 = 0;
        DType value12 = 0;
        DType value21 = 0;
        DType value22 = 0;

        if (between(x1, 0, W-1) && between(y1, 0, H-1))
            value11 = *(data + y1*W + x1);

        if (between(x1, 0, W-1) && between(y2, 0, H-1))
            value12 = *(data + y2*W + x1);

        if (between(x2, 0, W-1) && between(y1, 0, H-1))
            value21 = *(data + y1*W + x2);

        if (between(x2, 0, W-1) && between(y2, 0, H-1))
            value22 = *(data + y2*W + x2);

        DType value = (1 - dist_x)*(1 - dist_y)*value11 + (1 - dist_x)*dist_y*value12 + dist_x*(1 - dist_y)*value21 + dist_x*dist_y*value22;

        top_data[offset5d(n,k,8*c+i,h,w,N,K,C,H,W)] = value;
    }
  } // cuda_kernel_loop
}

template<typename DType>
inline void BilinearSampleKForward(const Tensor<gpu, 5, DType> &out,
                                   const Tensor<gpu, 4, DType> &data,
                                   const Tensor<gpu, 2, DType> &offset) {
  const DType *bottom_data = data.dptr_;
  const DType *bottom_offset = offset.dptr_;
  DType *top_data = out.dptr_;
  int N = out.size(0);
  int K = out.size(1);
  int C = out.size(2);
  int H = out.size(3);
  int W = out.size(4);
  const int count = out.shape_.Size() / 8; // the number of threads

  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "Bilinear Sample K Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  BilinearSampleKForwardKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(count, N, K, C, H, W, bottom_data, bottom_offset, top_data);
}


template <typename Dtype>
__device__ Dtype bilinear_function(int x,
                                   int y,
                                   Dtype i,
                                   Dtype j) {
  Dtype dist_x = static_cast<Dtype>(i - x);
  Dtype dist_y = static_cast<Dtype>(j - y);
  Dtype value = max(0.0,1-abs(dist_x)) * max(0.0, 1-abs(dist_y));
  return value;
}

template<typename DType>
__global__ void BilinearSampleKBackwardDataKernel(const int count, int N, int K, int C, int H, int W,
                                              const DType* grad_out, const DType* bottom_data,
                                              const DType* bottom_offset, DType* grad_data) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    const int w = index % W;
    const int h = (index / W) % H;
    const int c = (index / (H * W)) % C;
    const int k = (index / (C * H * W)) % K;
    const int n = (index / (K * C * H * W));

    DType x_real = bottom_offset[2 * k] + w;
    DType y_real = bottom_offset[2 * k + 1] + h;

    int x1 = floor(x_real);
    int y1 = floor(y_real);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    if (between(x1, 0, W-1) && between(y1, 0, H-1)) {
      atomicAdd(grad_data + offset(n,c,y1,x1,N,C,H,W),grad_out[index]*bilinear_function(x1,y1,x_real,y_real));
    }
    if (between(x1, 0, W-1) && between(y2, 0, H-1)) {
      atomicAdd(grad_data + offset(n,c,y2,x1,N,C,H,W),grad_out[index]*bilinear_function(x1,y2,x_real,y_real));
    }
    if (between(x2, 0, W-1) && between(y1, 0, H-1)) {
      atomicAdd(grad_data + offset(n,c,y1,x2,N,C,H,W),grad_out[index]*bilinear_function(x2,y1,x_real,y_real));
    }
    if (between(x2, 0, W-1) && between(y2, 0, H-1)) {
      atomicAdd(grad_data + offset(n,c,y2,x2,N,C,H,W),grad_out[index]*bilinear_function(x2,y2,x_real,y_real));
    }
  }
}

template<typename DType>
__global__ void BilinearSampleKBackwardOffsetKernel(const int count, int N, int K, int C, int H, int W,
                                              const DType* grad_out, const DType* bottom_data,
                                              const DType* bottom_offset, DType* grad_offset) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    const int w = index % W;
    const int h = (index / W) % H;
    const int k = (index / (H * W)) % K;
    const int n = (index / (K * H * W));

    DType x_real = bottom_offset[2 * k] + w;
    DType y_real = bottom_offset[2 * k + 1] + h;

    int x1 = floor(x_real);
    int y1 = floor(y_real);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    DType grad_x = 0;
    DType grad_y = 0;

    for(int c=0;c<C;c++) {
      if (between(x1, 0, W-1) && between(y1, 0, H-1)) {
        grad_x += bottom_data[offset(n,c,y1,x1,N,C,H,W)]*(-1)*(1- abs(y1-y_real))*grad_out[offset5d(n,k,c,h,w,N,K,C,H,W)];
        grad_y += bottom_data[offset(n,c,y1,x1,N,C,H,W)]*(-1)*(1- abs(x1-x_real))*grad_out[offset5d(n,k,c,h,w,N,K,C,H,W)];
      }
      if (between(x1, 0, W-1) && between(y2, 0, H-1)) {
        grad_x += bottom_data[offset(n,c,y2,x1,N,C,H,W)]*(-1)*(1- abs(y2-y_real))*grad_out[offset5d(n,k,c,h,w,N,K,C,H,W)];
        grad_y += bottom_data[offset(n,c,y2,x1,N,C,H,W)]*(1- abs(x1-x_real))*grad_out[offset5d(n,k,c,h,w,N,K,C,H,W)];
      }
      if (between(x2, 0, W-1) && between(y1, 0, H-1)) {
        grad_x += bottom_data[offset(n,c,y1,x2,N,C,H,W)]*(1- abs(y1-y_real))*grad_out[offset5d(n,k,c,h,w,N,K,C,H,W)];
        grad_y += bottom_data[offset(n,c,y1,x2,N,C,H,W)]*(-1)*(1- abs(x2-x_real))*grad_out[offset5d(n,k,c,h,w,N,K,C,H,W)];
      }
      if (between(x2, 0, W-1) && between(y2, 0, H-1)) {
        grad_x += bottom_data[offset(n,c,y2,x2,N,C,H,W)]*(1- abs(y2-y_real))*grad_out[offset5d(n,k,c,h,w,N,K,C,H,W)];
        grad_y += bottom_data[offset(n,c,y2,x2,N,C,H,W)]*(1- abs(x2-x_real))*grad_out[offset5d(n,k,c,h,w,N,K,C,H,W)];
      }
    }
    atomicAdd(grad_offset + 2 * k, grad_x);
    atomicAdd(grad_offset + 2 * k + 1, grad_y);
  }
}


template<typename DType>
inline void BilinearSampleKBackward(const Tensor<gpu, 4, DType> &grad_data,
                                    const Tensor<gpu, 2, DType> &grad_offset,
                                    const Tensor<gpu, 4, DType> &data,
                                    const Tensor<gpu, 2, DType> &offset,
                                    const Tensor<gpu, 5, DType> &grad_out) {
  const DType *top_grad = grad_out.dptr_;
  const DType *bottom_data = data.dptr_;
  const DType *bottom_offset = offset.dptr_;
  DType *bottom_grad_data = grad_data.dptr_;
  DType *bottom_grad_offset = grad_offset.dptr_;

  const int count = grad_out.shape_.Size(); // the number of threads

  int N = grad_out.size(0);
  int K = grad_out.size(1);
  int C = grad_out.size(2);
  int H = grad_out.size(3);
  int W = grad_out.size(4);


  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "WeightPropagation Backward");
  cudaStream_t stream_data = Stream<gpu>::GetStream(grad_data.stream_);

  BilinearSampleKBackwardDataKernel<DType><<<dimGrid, dimBlock, 0, stream_data>>>(count, N, K, C, H, W,
                                        top_grad, bottom_data, bottom_offset, bottom_grad_data);

  int count_offset = N * K * H * W;

  BilinearSampleKBackwardOffsetKernel<DType><<<dimGrid, dimBlock, 0, stream_data>>>(count_offset, N, K, C, H, W,
                                        top_grad, bottom_data, bottom_offset, bottom_grad_offset);

}



} // namespace cuda
template<typename DType>
inline void BilinearSampleKForward(const Tensor<gpu, 5, DType> &out,
                                   const Tensor<gpu, 4, DType> &data,
                                   const Tensor<gpu, 2, DType> &offset) {
  cuda::BilinearSampleKForward(out, data, offset);
}

template<typename DType>
inline void BilinearSampleKBackward(const Tensor<gpu, 4, DType> &grad_data,
                                    const Tensor<gpu, 2, DType> &grad_offset,
                                    const Tensor<gpu, 4, DType> &data,
                                    const Tensor<gpu, 2, DType> &offset,
                                    const Tensor<gpu, 5, DType> &grad_out) {
  cuda::BilinearSampleKBackward(grad_data, grad_offset, data, offset, grad_out);
}


} //namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(BilinearSampleKParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BilinearSampleKOp<gpu, DType>(param);
  });
  return op;
}


}  // namespace op
} // namespace mxnet

/*!
 * Copyright (c) 2017 by Contributors
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file bilinear_sample-inl.h
 * \brief bilinear sample operator and symbol
 * \author Jiang ZhengKai
*/
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>
#include "./bilinear_sample_k-inl.h"


namespace mshadow {

template<typename DType>
inline void BilinearSampleKForward(const Tensor<cpu, 5, DType> &out,
                                   const Tensor<cpu, 4, DType> &data,
                                   const Tensor<cpu, 2, DType> &offset) {
  // NOT_IMPLEMENTED
  return;
}


template<typename DType>
inline void BilinearSampleKBackward(const Tensor<cpu, 4, DType> &grad_data,
                                    const Tensor<cpu, 2, DType> &grad_offset,
                                    const Tensor<cpu, 4, DType> &data,
                                    const Tensor<cpu, 2, DType> &offset,
                                    const Tensor<cpu, 5, DType> &grad_out) {
  // NOT_IMPLEMENTED
  return;
}


} //namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(BilinearSampleKParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BilinearSampleKOp<cpu, DType>(param);
  });
  return op;
}

Operator *BilinearSampleKProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(BilinearSampleKParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_BilinearSampleK, BilinearSampleKProp)
.describe("bilinear sample according to the data and offset")
.add_argument("data", "Symbol", "Input data to the pooling operator, a 4D Feature maps")
.add_argument("offset", "Symbol", "Offsets, a 2D array ")
.add_arguments(BilinearSampleKParam::__FIELDS__());

} //namespace op
} //namespace mxnet

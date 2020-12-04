/*!
 * Copyright (c) 2017 by Contributors
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file dynamic_correlation-inl.h
 * \brief weight generate operator and symbol
 * \author Jiang ZhengKai
*/
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>
#include "./align_data-inl.h"


namespace mshadow {

template<typename DType>
inline void AlignDataForward(const Tensor<cpu, 4, DType> &out,
                             const Tensor<cpu, 5, DType> &data,
                             const Tensor<cpu, 4, DType> &weight) {
  // NOT_IMPLEMENTED
  return;
}


template<typename DType>
inline void AlignDataBackward(const Tensor<cpu, 5, DType> &grad_data,
                              const Tensor<cpu, 4, DType> &grad_weight,
                              const Tensor<cpu, 5, DType> &data,
                              const Tensor<cpu, 4, DType> &weight,
                              const Tensor<cpu, 4, DType> &grad_out) {
  // NOT_IMPLEMENTED
  return;
}


} //namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(AlignDataParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new AlignDataOp<cpu, DType>(param);
  });
  return op;
}

Operator *AlignDataProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(AlignDataParam);
MXNET_REGISTER_OP_PROPERTY(_contrib_AlignData, AlignDataProp)
.describe("weight generate according to the data and data_ref")
.add_argument("data", "Symbol", "Input data to the pooling operator, a 5D Feature maps")
.add_argument("weight", "Symbol", "Offsets, a 4D array ")
.add_arguments(AlignDataParam::__FIELDS__());

} //namespace op
} //namespace mxnet

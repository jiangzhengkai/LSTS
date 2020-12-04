/*!
 * Copyright (c) 2017 by Contributors
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file bilinear_sample-inl.h
 * \brief bilinear_sample operator and symbol
 * \author Jiang ZhengKai
*/
#ifndef MXNET_OPERATOR_CONTRIB_BILINEAR_SAMPLE_K_INL_H_
#define MXNET_OPERATOR_CONTRIB_BILINEAR_SAMPLE_K_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive
namespace BilinearSampleK {
enum BilinearSampleKOpInputs {kData, kOffset};
enum BilinearSampleKOpOutputs {kOut};
} // Bilinear Sample

struct BilinearSampleKParam : public dmlc::Parameter<BilinearSampleKParam> {
  // Tshape
  // int n;
  DMLC_DECLARE_PARAMETER(BilinearSampleKParam) {
  //  DMLC_DECLARE_FIELD(n)
  //  .describe("bilinear sample n points")
  //  .set_default(0);
  }
};

template<typename xpu, typename DType>
class BilinearSampleKOp : public Operator {
 public:
  explicit BilinearSampleKOp(BilinearSampleKParam p) {
    this->param_ = p;
    }
  // forward
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t expected_in = 2;
    size_t expected_out = 1;
    CHECK_EQ(in_data.size(), expected_in);
    CHECK_EQ(out_data.size(), expected_out);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[BilinearSampleK::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> offset = in_data[BilinearSampleK::kOffset].get<xpu, 2, DType>(s);

    Tensor<xpu, 5, DType> out = out_data[BilinearSampleK::kOut].get<xpu, 5, DType>(s);

    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(offset.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);

    out = 0.0;
    BilinearSampleKForward(out, data, offset);
  }
  // backward
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {

    using namespace mshadow;
    size_t expected_in = 2;
    size_t expected_out = 1;
    CHECK_EQ(in_data.size(), expected_in);
    CHECK_EQ(out_data.size(), expected_out);

    CHECK_NE(req[BilinearSampleK::kData], kWriteInplace) <<
      "BilinearSampleK: Backward doesn't support kWriteInplace.";
    CHECK_NE(req[BilinearSampleK::kOffset], kWriteInplace) <<
      "BilinearSampleK: Backward doesn't support kWriteInplace.";

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 5, DType> grad_out = out_grad[BilinearSampleK::kOut].get<xpu, 5, DType>(s);
    Tensor<xpu, 4, DType> data = in_data[BilinearSampleK::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> offset = in_data[BilinearSampleK::kOffset].get<xpu, 2, DType>(s);

    Tensor<xpu, 4, DType> grad_data = in_grad[BilinearSampleK::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> grad_offset = in_grad[BilinearSampleK::kOffset].get<xpu, 2, DType>(s);


    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(offset.CheckContiguous(), true);
    CHECK_EQ(grad_data.CheckContiguous(), true);
    CHECK_EQ(grad_offset.CheckContiguous(), true);

    grad_data = 0.0;
    grad_offset = 0.0;
    BilinearSampleKBackward(grad_data, grad_offset, data, offset, grad_out);
  }

 private:
    BilinearSampleKParam param_;
}; // class dynamic correlation op


// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(BilinearSampleKParam param, int dtype);

#if DMLC_USE_CXX11
class BilinearSampleKProp : public OperatorProperty {
 public:
    std::vector<std::string> ListArguments() const override {
    return {"data", "offset"};
  }
  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }
  int NumOutputs() const override {
    return 1;
  }
  int NumVisibleOutputs() const override {
    return 1;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }
  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, offset]";
    // data1: [batch_size, c, h, w]
    TShape dshape = in_shape->at(BilinearSampleK::kData);
    CHECK_EQ(dshape.ndim(), 4) << "data should be a 4D tensor";

    TShape oshape = in_shape->at(BilinearSampleK::kOffset);
    CHECK_EQ(oshape.ndim(), 2) << "offset should be a 2D tensor";

    // out: [batch_size, k, c, h , w]
    out_shape->clear();
    out_shape->push_back(Shape5(dshape[0], oshape[0], dshape[1], dshape[2], dshape[3]));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 2);
    int dtype = (*in_type)[0];
    (*in_type)[1] = dtype;
    CHECK_NE(dtype, -1) << "Input must have specified type";

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new BilinearSampleKProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_BilinearSampleK";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[BilinearSampleK::kOut], in_data[BilinearSampleK::kData], in_data[BilinearSampleK::kOffset]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;
 private:
  BilinearSampleKParam param_;
};  // class BILINEAR SAMPLE K
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BILINEAR_SAMPLE_K_INL_H_
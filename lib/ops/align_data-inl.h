/*!
 * Copyright (c) 2017 by Contributors
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file bilinear_sample-inl.h
 * \brief bilinear_sample operator and symbol
 * \author Jiang ZhengKai
*/
#ifndef MXNET_OPERATOR_CONTRIB_ALIGN_DATA_INL_H_
#define MXNET_OPERATOR_CONTRIB_ALIGN_DATA_INL_H_

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
namespace AlignData {
enum AlignDataOpInputs {kData, kWeight};
enum AlignDataOpOutputs {kOut};
} // Bilinear Sample

struct AlignDataParam : public dmlc::Parameter<AlignDataParam> {
  // Tshape
  // int n;
  DMLC_DECLARE_PARAMETER(AlignDataParam) {
  //  DMLC_DECLARE_FIELD(n)
  //  .describe("weight generate n points")
  //  .set_default(0);
  }
};

template<typename xpu, typename DType>
class AlignDataOp : public Operator {
 public:
  explicit AlignDataOp(AlignDataParam p) {
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

    Tensor<xpu, 5, DType> data = in_data[AlignData::kData].get<xpu, 5, DType>(s);
    Tensor<xpu, 4, DType> weight = in_data[AlignData::kWeight].get<xpu, 4, DType>(s);

    Tensor<xpu, 4, DType> out = out_data[AlignData::kOut].get<xpu, 4, DType>(s);

    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(weight.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);

    out = 0.0;
    AlignDataForward(out, data, weight);
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

    CHECK_NE(req[AlignData::kData], kWriteInplace) <<
      "AlignData: Backward doesn't support kWriteInplace.";
    CHECK_NE(req[AlignData::kWeight], kWriteInplace) <<
      "AlignData: Backward doesn't support kWriteInplace.";

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> grad_out = out_grad[AlignData::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 5, DType> data = in_data[AlignData::kData].get<xpu, 5, DType>(s);
    Tensor<xpu, 4, DType> weight = in_data[AlignData::kWeight].get<xpu, 4, DType>(s);

    Tensor<xpu, 5, DType> grad_data = in_grad[AlignData::kData].get<xpu, 5, DType>(s);
    Tensor<xpu, 4, DType> grad_weight = in_grad[AlignData::kWeight].get<xpu, 4, DType>(s);


    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(weight.CheckContiguous(), true);
    CHECK_EQ(grad_data.CheckContiguous(), true);
    CHECK_EQ(grad_weight.CheckContiguous(), true);

    grad_data = 0.0;
    grad_weight = 0.0;
    AlignDataBackward(grad_data, grad_weight, data, weight, grad_out);
  }

 private:
    AlignDataParam param_;
}; // class dynamic correlation op


// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(AlignDataParam param, int dtype);

#if DMLC_USE_CXX11
class AlignDataProp : public OperatorProperty {
 public:
    std::vector<std::string> ListArguments() const override {
    return {"data", "weight"};
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
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, data_ref]";
    // data1: [batch_size, c, h, w]
    TShape dshape = in_shape->at(AlignData::kData);
    CHECK_EQ(dshape.ndim(), 5) << "data1 should be a 4D tensor";

    TShape oshape = in_shape->at(AlignData::kWeight);
    CHECK_EQ(oshape.ndim(), 4) << "offset should be a 2D tensor";

    // out: [batch_size, k, c, h , w]
    out_shape->clear();
    out_shape->push_back(Shape4(dshape[0], dshape[2], dshape[3], dshape[4]));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 2);
    int dtype = (*in_type)[0];
    CHECK_EQ(dtype, (*in_type)[1]);
    CHECK_NE(dtype, -1) << "Input must have specified type";

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new AlignDataProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_AlignData";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[AlignData::kOut], in_data[AlignData::kData], in_data[AlignData::kWeight]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;
 private:
  AlignDataParam param_;
};
#endif
}  // namespace op
}  // namespace mxnet
#endif

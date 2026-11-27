/*!
 * \file tl/hexagon/layout/panel.cc
 */
#include "hexagon/layout/panel.h"

#include "layout/reflection.h"
#include <tvm/tirx/op.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

int64_t StaticExtent(const PrimExpr &extent, const char *what) {
  const auto *n = extent.as<IntImmNode>();
  ICHECK(n) << "a panel layout needs a static " << what << ", got " << extent;
  return n->value;
}

// Map each logical element to its coordinates in ordinary or paired panels.
Array<PrimExpr> PanelPosition(const Array<PrimExpr> &shape, int64_t lane,
                              bool paired) {
  size_t n = shape.size();
  ICHECK_GE(n, 2) << "a panel layout needs at least two dimensions";
  ICHECK(lane == 4 || lane == 8)
      << "a panel lane holds 4 (8-bit) or 8 (4-bit) columns, got " << lane;
  int64_t rows = StaticExtent(shape[n - 2], "row count");
  int64_t cols = StaticExtent(shape[n - 1], "column count");
  ICHECK(rows > 0 && cols > 0 && rows % kPanelRows == 0 && cols % lane == 0)
      << "a panel matrix has rows in multiples of " << kPanelRows
      << " and columns in multiples of " << lane << ", got " << rows << "x"
      << cols;
  PrimExpr batch = make_zero(DataType::Int(32));
  for (size_t i = 0; i + 2 < n; ++i)
    batch = batch * shape[i] + InputPlaceholder(i);
  PrimExpr r = InputPlaceholder(n - 2), c = InputPlaceholder(n - 1);
  PrimExpr p = make_const(DataType::Int(32), kPanelRows);
  PrimExpr l = make_const(DataType::Int(32), lane);
  if (paired) {
    ICHECK(lane == 4 && rows % 64 == 0 && cols % 32 == 0)
        << "paired panels require 8-bit elements, rows in multiples of 64 "
           "and columns in multiples of 32";
    return {batch * static_cast<int>(rows / 64) + floordiv(r, 64),
            floordiv(c, 32),
            floormod(floordiv(r, 32), 2),
            floordiv(floormod(c, 32), l),
            floormod(r, p),
            floormod(c, l)};
  }
  return {batch * static_cast<int>(rows / kPanelRows) + floordiv(r, p),
          floordiv(c, l), floormod(r, p), floormod(c, l)};
}

} // namespace

PanelLayoutNode::PanelLayoutNode(Array<PrimExpr> shape, int64_t lane,
                                 bool paired)
    : LayoutNode(shape, PanelPosition(shape, lane, paired)), lane(lane),
      paired(paired) {}

int64_t PanelLayoutNode::Rows() const {
  return InputShape()[InputDim() - 2].as<IntImmNode>()->value;
}

int64_t PanelLayoutNode::Cols() const {
  return InputShape()[InputDim() - 1].as<IntImmNode>()->value;
}

PanelLayout::PanelLayout(Array<PrimExpr> shape, int64_t lane, bool paired) {
  data_ = make_object<PanelLayoutNode>(shape, lane, paired);
}

void PanelLayoutNode::RestoreLegacyIndex() {
  // Earlier JSON stored the forward expression without its binding Vars.
  if (placeholder_vars_.empty()) {
    InitForwardIndex(PanelPosition(input_size_, lane, paired));
  }
}

void PanelLayoutNode::RegisterReflection() {
  namespace refl = reflection;
  refl::ObjectDef<PanelLayoutNode>()
      .def_ro("paired", &PanelLayoutNode::paired, refl::default_value(false))
      .def_ro("lane", &PanelLayoutNode::lane,
              RestoreAfterSet<&PanelLayoutNode::lane,
                              &PanelLayoutNode::RestoreLegacyIndex>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  PanelLayoutNode::RegisterReflection();
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tl.hexagon.PanelLayout", [](PackedArgs args,
                                                            Any *rv) {
    ICHECK(args.size() == 2 || args.size() == 3);
    *rv = PanelLayout(args[0].cast<Array<PrimExpr>>(), args[1].cast<int64_t>(),
                      args.size() == 3 && args[2].cast<bool>());
  });
}

} // namespace tl
} // namespace tvm

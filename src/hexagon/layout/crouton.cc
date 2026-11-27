/*!
 * \file tl/hexagon/layout/crouton.cc
 */
#include "hexagon/layout/crouton.h"

#include "layout/reflection.h"
#include <tvm/tirx/op.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

int64_t StaticExtent(const PrimExpr &extent, const char *what) {
  const auto *n = extent.as<IntImmNode>();
  ICHECK(n) << "a Crouton layout needs a static " << what << ", got " << extent;
  return n->value;
}

// The physical position of every buffer index: [tile, register, lane]. Leading
// dimensions pick a matrix, whose tiles follow it.
Array<PrimExpr> CroutonPosition(const Array<PrimExpr> &shape, bool column_tiles,
                                bool transposed, int bits) {
  ICHECK(bits == 8 || bits == 16) << "Crouton elements must be 8 or 16 bits";
  ICHECK(bits != 8 || (!column_tiles && !transposed))
      << "8-bit Crouton requires row-ordered, non-transposed tiles";
  int rows_per_register = 32 / bits;
  int tile_rows = 16 * rows_per_register;
  size_t n = shape.size();
  ICHECK_GE(n, 2) << "a Crouton layout needs at least two dimensions";
  int row_axis = static_cast<int>(transposed ? n - 1 : n - 2);
  int col_axis = static_cast<int>(transposed ? n - 2 : n - 1);
  int64_t rows = StaticExtent(shape[row_axis], "row count");
  int64_t cols = StaticExtent(shape[col_axis], "column count");
  ICHECK(rows > 0 && cols > 0 && rows % tile_rows == 0 &&
         cols % kCroutonTile == 0)
      << "Crouton matrix dimensions must be positive multiples of " << tile_rows
      << " rows and " << kCroutonTile << " columns, got " << rows << "x"
      << cols;
  PrimExpr batch = make_zero(DataType::Int(32));
  for (size_t i = 0; i + 2 < n; ++i)
    batch = batch * shape[i] + InputPlaceholder(i);
  PrimExpr r = InputPlaceholder(row_axis), c = InputPlaceholder(col_axis);
  PrimExpr t = make_const(DataType::Int(32), kCroutonTile);
  PrimExpr tile =
      column_tiles
          ? floordiv(c, t) * static_cast<int>(rows / tile_rows) +
                floordiv(r, tile_rows)
          : floordiv(r, tile_rows) * static_cast<int>(cols / kCroutonTile) +
                floordiv(c, t);
  int64_t tiles = (rows / tile_rows) * (cols / kCroutonTile);
  return {batch * static_cast<int>(tiles) + tile,
          floordiv(floormod(r, tile_rows), rows_per_register),
          floormod(c, t) * rows_per_register + floormod(r, rows_per_register)};
}

} // namespace

CroutonLayoutNode::CroutonLayoutNode(Array<PrimExpr> shape, bool column_tiles,
                                     bool transposed, int bits)
    : LayoutNode(shape, CroutonPosition(shape, column_tiles, transposed, bits)),
      column_tiles(column_tiles), transposed(transposed), bits(bits) {}

int CroutonLayoutNode::RowAxis() const {
  int n = static_cast<int>(InputDim());
  return transposed ? n - 1 : n - 2;
}

int CroutonLayoutNode::ColAxis() const {
  int n = static_cast<int>(InputDim());
  return transposed ? n - 2 : n - 1;
}

int64_t CroutonLayoutNode::Rows() const {
  return InputShape()[RowAxis()].as<IntImmNode>()->value;
}

int64_t CroutonLayoutNode::Cols() const {
  return InputShape()[ColAxis()].as<IntImmNode>()->value;
}

CroutonLayout::CroutonLayout(Array<PrimExpr> shape, bool column_tiles,
                             bool transposed, int bits) {
  data_ = make_object<CroutonLayoutNode>(shape, column_tiles, transposed, bits);
}

void CroutonLayoutNode::RestoreLegacyIndex() {
  // Earlier JSON stored the forward expression without its binding Vars.
  if (placeholder_vars_.empty()) {
    InitForwardIndex(
        CroutonPosition(input_size_, column_tiles, transposed, bits));
  }
}

void CroutonLayoutNode::RegisterReflection() {
  namespace refl = reflection;
  refl::ObjectDef<CroutonLayoutNode>()
      .def_ro("column_tiles", &CroutonLayoutNode::column_tiles)
      .def_ro("bits", &CroutonLayoutNode::bits, refl::default_value(16))
      .def_ro("transposed", &CroutonLayoutNode::transposed,
              RestoreAfterSet<&CroutonLayoutNode::transposed,
                              &CroutonLayoutNode::RestoreLegacyIndex>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  CroutonLayoutNode::RegisterReflection();
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      "tl.hexagon.CroutonLayout", [](PackedArgs args, Any *rv) {
        ICHECK(args.size() == 3 || args.size() == 4);
        *rv = CroutonLayout(args[0].cast<Array<PrimExpr>>(),
                            args[1].cast<bool>(), args[2].cast<bool>(),
                            args.size() == 4 ? args[3].cast<int>() : 16);
      });
}

} // namespace tl
} // namespace tvm

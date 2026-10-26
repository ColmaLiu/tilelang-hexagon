/*!
 * \file tl/hexagon/target_utils.cc
 * \brief Hexagon target attribute helpers.
 */

#include "hexagon/target_utils.h"

#include <tvm/ffi/reflection/registry.h>

#include "dlpack/dlpack.h"

namespace tvm {
namespace tl {

bool TargetIsHexagon(Target target) {
  return target->GetTargetDeviceType() == kDLHexagon;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.TargetIsHexagon",
                        [](Target target) { return TargetIsHexagon(target); });
}

} // namespace tl
} // namespace tvm

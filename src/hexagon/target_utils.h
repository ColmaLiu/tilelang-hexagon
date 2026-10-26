/*!
 * \file tl/hexagon/target_utils.h
 * \brief Hexagon target attribute helpers.
 */

#ifndef TVM_TL_HEXAGON_TARGET_UTILS_H_
#define TVM_TL_HEXAGON_TARGET_UTILS_H_

#include <tvm/target/target.h>

namespace tvm {
namespace tl {

bool TargetIsHexagon(Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HEXAGON_TARGET_UTILS_H_

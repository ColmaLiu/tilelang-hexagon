/*!
 * \file tl/hexagon/op/builtin.h
 * \brief Hexagon-specific TileLang intrinsic Ops.
 */

#ifndef TVM_TL_HEXAGON_OP_BUILTIN_H_
#define TVM_TL_HEXAGON_OP_BUILTIN_H_

#include "op/builtin.h"

namespace tvm {
namespace tl {

/*! \brief Acquire exclusive use of the implicit HMX accumulator state. */
TVM_DLL const Op &hexagon_hmx_acquire();

/*! \brief Release exclusive use of the implicit HMX accumulator state. */
TVM_DLL const Op &hexagon_hmx_release();

/*! \brief Load the HMX bias/scale register from a VTCM memory operand. */
TVM_DLL const Op &hexagon_hmx_load_bias();

/*! \brief Clear the implicit HMX accumulator. */
TVM_DLL const Op &hexagon_hmx_clear();

/*!
 * \brief Execute one HMX deep multiply packet for 1 to 32 contiguous K tiles.
 *
 * The memory operands are activation and weight Crouton tiles in VTCM.  The
 * result accumulates into the implicit accumulator state token.
 */
TVM_DLL const Op &hexagon_hmx_mma();

/*! \brief Convert the implicit HMX accumulator into convert state. */
TVM_DLL const Op &hexagon_hmx_convert();

/*! \brief Store HMX convert state to an output Crouton tile in VTCM. */
TVM_DLL const Op &hexagon_hmx_store();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HEXAGON_OP_BUILTIN_H_

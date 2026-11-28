#pragma once
// Whole HVX registers at any address, parts of registers, and lane broadcasts.
//
// Included by hvx_math.h and hvx_reduce.h.
#include <hexagon_types.h>
#include <stdint.h>
#include <tl_templates/hexagon/common.h>

#ifndef TL_DEVICE
#define TL_DEVICE static inline __attribute__((always_inline))
#endif

// A whole register at any address: `aligned(1)` makes hexagon-clang emit the
// unaligned vmemu form, where a plain HVX_Vector access is an aligned vmem,
// which ignores the low address bits.
typedef HVX_Vector tl_hvx_uvector __attribute__((aligned(1)));
TL_DEVICE HVX_Vector tl_hvx_loadu(const void *p) {
  return *(const tl_hvx_uvector *)p;
}
TL_DEVICE void tl_hvx_storeu(void *p, HVX_Vector v) {
  *(tl_hvx_uvector *)p = v;
}

// A vector (vec_type) at any address, for the vector accesses the codegen
// cannot prove aligned: vmemu register by register, and for a part of a
// register only the aligned blocks that hold its bytes (defined below).
template <typename V> TL_DEVICE V tl_hvx_vec_loadu(const void *p);
template <typename V> TL_DEVICE void tl_hvx_vec_storeu(void *p, const V &v);

// A whole register at `p`: aligned vmem when the caller knows `p` is on a
// register boundary, else vmemu.
template <bool Aligned> TL_DEVICE HVX_Vector tl_hvx_load(const void *p) {
  if constexpr (Aligned)
    return *(const HVX_Vector *)p;
  else
    return tl_hvx_loadu(p);
}
template <bool Aligned> TL_DEVICE void tl_hvx_store(void *p, HVX_Vector v) {
  if constexpr (Aligned)
    *(HVX_Vector *)p = v;
  else
    tl_hvx_storeu(p, v);
}

// The register-aligned block that holds `p`, and p's offset in it. The block
// address is `p` minus the offset, not `p` with its low bits masked: LLVM then
// keeps the pointer's provenance, so it can tell the block from the stores of
// other buffers, reorder around them and hoist invariant loads.
TL_DEVICE int tl_hvx_block_offset(const void *p) {
  return static_cast<int>(reinterpret_cast<uintptr_t>(p) & 127);
}
TL_DEVICE const HVX_Vector *tl_hvx_block_of(const void *p) {
  return reinterpret_cast<const HVX_Vector *>(static_cast<const char *>(p) -
                                              tl_hvx_block_offset(p));
}

// The `nbytes` (1..128) bytes at `p` in the low bytes of a register; the other
// bytes are undefined. Reads only the aligned blocks that hold those bytes, so
// it never touches memory past them.
TL_DEVICE HVX_Vector tl_hvx_load_bytes(const void *p, int nbytes) {
  int off = tl_hvx_block_offset(p);
  const HVX_Vector *block = tl_hvx_block_of(p);
  HVX_Vector lo = block[0];
  HVX_Vector hi = off + nbytes > 128 ? block[1] : lo;
  return Q6_V_valign_VVR(hi, lo, off);
}

// Stores the low `nbytes` (1..128) bytes of `v` at `p`, leaving the bytes
// around them unchanged.
TL_DEVICE void tl_hvx_store_bytes(void *p, HVX_Vector v, int nbytes) {
  int off = tl_hvx_block_offset(p);
  HVX_Vector *block = const_cast<HVX_Vector *>(tl_hvx_block_of(p));
  // Byte i of v goes to byte off + i of the two blocks.
  HVX_Vector rotated = Q6_V_vror_VR(v, (128 - off) & 127);
  int end = off + nbytes;
  HVX_VectorPred first =
      Q6_Q_and_QQn(Q6_Q_vsetq2_R(end < 128 ? end : 128), Q6_Q_vsetq_R(off));
  Q6_vmem_QRIV(first, block, rotated);
  if (end > 128)
    Q6_vmem_QRIV(Q6_Q_vsetq_R(end - 128), block + 1, rotated);
}

template <typename V> TL_DEVICE V tl_hvx_vec_loadu(const void *p) {
  constexpr int S = static_cast<int>(sizeof(V));
  const char *src = static_cast<const char *>(p);
  V out;
#pragma clang loop unroll(full)
  for (int b = 0; b < S; b += 128) {
    int n = S - b < 128 ? S - b : 128;
    HVX_Vector v =
        n == 128 ? tl_hvx_loadu(src + b) : tl_hvx_load_bytes(src + b, n);
    __builtin_memcpy(reinterpret_cast<char *>(&out) + b, &v, n);
  }
  return out;
}

template <typename V> TL_DEVICE void tl_hvx_vec_storeu(void *p, const V &v) {
  constexpr int S = static_cast<int>(sizeof(V));
  char *dst = static_cast<char *>(p);
#pragma clang loop unroll(full)
  for (int b = 0; b < S; b += 128) {
    int n = S - b < 128 ? S - b : 128;
    HVX_Vector r = Q6_V_vzero();
    __builtin_memcpy(&r, reinterpret_cast<const char *>(&v) + b, n);
    if (n == 128)
      tl_hvx_storeu(dst + b, r);
    else
      tl_hvx_store_bytes(dst + b, r, n);
  }
}

// The `count` (0..N) elements at `p` in the low lanes of an N-lane vector of
// at most one register; the other lanes are undefined.
template <int N, typename T>
TL_DEVICE vec_type<T, N> tl_hvx_load_lanes(const T *p, int count) {
  static_assert(N * sizeof(T) <= 128, "one register at most");
  HVX_Vector v = count > 0
                     ? tl_hvx_load_bytes(p, count * static_cast<int>(sizeof(T)))
                     : Q6_V_vzero();
  vec_type<T, N> out;
  __builtin_memcpy(&out, &v, sizeof(out));
  return out;
}

// The delta-network control that copies the low B bytes of a register over
// all of it: stage k of vdelta takes byte j from byte j ^ 2^k where bit k of
// control byte j is set, so clearing the bits below B copies byte j % B.
template <int B> TL_DEVICE HVX_Vector tl_hvx_replicate_control() {
  alignas(128) static const uint8_t control[128] = {
#define TL_C4(k)                                                               \
  (uint8_t)((k) & ~(B - 1) & 0x7f), (uint8_t)((k + 1) & ~(B - 1) & 0x7f),      \
      (uint8_t)((k + 2) & ~(B - 1) & 0x7f),                                    \
      (uint8_t)((k + 3) & ~(B - 1) & 0x7f)
#define TL_C16(k) TL_C4(k), TL_C4(k + 4), TL_C4(k + 8), TL_C4(k + 12)
#define TL_C64(k) TL_C16(k), TL_C16(k + 16), TL_C16(k + 32), TL_C16(k + 48)
      TL_C64(0), TL_C64(64)
#undef TL_C64
#undef TL_C16
#undef TL_C4
  };
  return *reinterpret_cast<const HVX_Vector *>(control);
}

// Lane group i of `x` (at most one register), the G lanes from lane i * G,
// repeated over an N-lane vector: a rotate brings the group to lane 0, and the
// delta network copies it over the register. G = 1 broadcasts lane i.
template <int N, int G, typename T, int M>
TL_DEVICE vec_type<T, N> tl_hvx_broadcast_lane(const vec_type<T, M> &x, int i) {
  static_assert(M * sizeof(T) <= 128, "one register at most");
  constexpr int B = G * static_cast<int>(sizeof(T));
  constexpr int bytes = N * static_cast<int>(sizeof(T));
  HVX_Vector v;
  __builtin_memcpy(&v, &x, sizeof(x));
  v = Q6_V_vdelta_VV(Q6_V_vror_VR(v, i * B), tl_hvx_replicate_control<B>());
  vec_type<T, N> out;
#pragma clang loop unroll(full)
  for (int b = 0; b < bytes; b += 128)
    __builtin_memcpy(reinterpret_cast<char *>(&out) + b, &v,
                     bytes - b < 128 ? bytes - b : 128);
  return out;
}

// The B bytes at p (a power of two of at most one register) in the low bytes
// of a register. Aligned: p is a multiple of B, so the window lies in one
// aligned block, which one load and a rotate bring down without a branch.
template <int B, bool Aligned>
TL_DEVICE HVX_Vector tl_hvx_load_window(const void *p) {
  static_assert((B & (B - 1)) == 0 && B <= 128, "at most one register");
  if constexpr (Aligned) {
    return Q6_V_vror_VR(*tl_hvx_block_of(p), tl_hvx_block_offset(p));
  } else {
    return tl_hvx_load_bytes(p, B);
  }
}

// An N-lane vector (at most two registers) of the P elements at p, repeated:
// lane l is p[l % P]. The window, P * sizeof(T) bytes, is a power of two of at
// most one register.
template <int P, int N, bool Aligned, typename T>
TL_DEVICE vec_type<T, N> tl_hvx_load_replicate(const T *p) {
  constexpr int B = P * static_cast<int>(sizeof(T));
  constexpr int bytes = N * static_cast<int>(sizeof(T));
  static_assert(bytes <= 256, "two registers at most");
  // Not inside the intrinsic: it is a macro, and the template arguments'
  // comma would split its arguments.
  HVX_Vector window = tl_hvx_load_window<B, Aligned>(p);
  HVX_Vector v = Q6_V_vdelta_VV(window, tl_hvx_replicate_control<B>());
  vec_type<T, N> out;
#pragma clang loop unroll(full)
  for (int b = 0; b < bytes; b += 128)
    __builtin_memcpy(reinterpret_cast<char *>(&out) + b, &v,
                     bytes - b < 128 ? bytes - b : 128);
  return out;
}

// An N-lane vector of the N / K elements at p, each
// repeated K times: lane l is p[l / K]. Shuffling a register with itself
// repeats each of its chunks twice, so log2(K) shuffles repeat each element
// K times.
template <int K, int N, bool Aligned, typename T>
TL_DEVICE vec_type<T, N> tl_hvx_load_repeat(const T *p) {
  constexpr int B = static_cast<int>(sizeof(T));
  static_assert(K >= 2 && (K & (K - 1)) == 0 && N % K == 0 &&
                    (N * B <= 256 || (N * B == 512 && K == 4)),
                "the repeated window must fit its register shuffles");
  HVX_Vector v = tl_hvx_load_window<N / K * B, Aligned>(p);
  if constexpr (N * B == 512) {
    HVX_VectorPair twice = Q6_W_vshuff_VVR(v, v, -B);
    HVX_Vector low = Q6_V_lo_W(twice), high = Q6_V_hi_W(twice);
    HVX_VectorPair parts[2] = {Q6_W_vshuff_VVR(low, low, -2 * B),
                               Q6_W_vshuff_VVR(high, high, -2 * B)};
    vec_type<T, N> out;
    __builtin_memcpy(&out, parts, sizeof(out));
    return out;
  } else {
    HVX_VectorPair w;
#pragma clang loop unroll(full)
    for (int chunk = B; chunk < K * B; chunk *= 2) {
      w = Q6_W_vshuff_VVR(v, v, -chunk);
      v = Q6_V_lo_W(w);
    }
    vec_type<T, N> out;
    __builtin_memcpy(&out, &w, N * B);
    return out;
  }
}

// Scalar reads also handle the high nibble of an odd logical index.
template <typename T, typename Index>
TL_DEVICE T tl_load_nibble(const T *data, Index i) {
  int n = (static_cast<uint8_t>(data[i / 2]) >> (4 * (i & 1))) & 15;
  if constexpr (std::is_signed_v<T>)
    n = (n ^ 8) - 8;
  return static_cast<T>(n);
}

template <int N, typename T>
TL_DEVICE vec_type<T, N> tl_hvx_unpack4(const vec_type<T, N / 2> &packed) {
  HVX_Vector x = Q6_V_vzero();
  __builtin_memcpy(&x, &packed, N / 2);
  HVX_Vector mask = Q6_Vb_vsplat_R(15);
  HVX_Vector lo = Q6_V_vand_VV(x, mask);
  HVX_Vector hi = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(x, 4), mask);
  HVX_VectorPair halves = Q6_W_vshuff_VVR(hi, lo, -1);
  if constexpr (std::is_signed_v<T>) {
    HVX_Vector sign = Q6_Vb_vsplat_R(8);
    auto extend = [&](HVX_Vector v) {
      return Q6_Vb_vsub_VbVb(Q6_V_vxor_VV(v, sign), sign);
    };
    halves =
        Q6_W_vcombine_VV(extend(Q6_V_hi_W(halves)), extend(Q6_V_lo_W(halves)));
  }
  vec_type<T, N> out;
  __builtin_memcpy(&out, &halves, N);
  return out;
}

// Each word of a panel register holds eight nibbles from one row. Look up
// their signed or unsigned halfword values once for two adjacent windows.
// The byte parity selects the two lookup registers. The halfword parity
// selects the first or second group of four columns in each row.
template <typename T>
TL_DEVICE vec_type<int16_t, 128> tl_hvx_load_widened_nibbles(const T *base,
                                                             int high) {
  struct Table {
    int16_t values[64]{};
    constexpr Table() {
      for (int i = 0; i < 64; ++i) {
        int code = (i / 2) & 15;
        values[i] = std::is_signed_v<T> ? (code ^ 8) - 8 : code;
      }
    }
  };
  alignas(128) static constexpr Table table{};
  HVX_Vector lut = *reinterpret_cast<const HVX_Vector *>(&table);
  HVX_Vector x = *reinterpret_cast<const HVX_Vector *>(base);
  HVX_VectorPair lo = Q6_Wh_vlut16_VbVhR_nomatch(x, lut, 0);
  HVX_VectorPair hi =
      Q6_Wh_vlut16_VbVhR_nomatch(Q6_Vuh_vlsr_VuhR(x, 4), lut, 0);
  HVX_Vector a, b;
  if (high) {
    a = Q6_Vh_vshuffo_VhVh(Q6_V_lo_W(hi), Q6_V_lo_W(lo));
    b = Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(hi), Q6_V_hi_W(lo));
  } else {
    a = Q6_Vh_vshuffe_VhVh(Q6_V_lo_W(hi), Q6_V_lo_W(lo));
    b = Q6_Vh_vshuffe_VhVh(Q6_V_hi_W(hi), Q6_V_hi_W(lo));
  }
  HVX_VectorPair ordered = Q6_W_vshuff_VVR(b, a, -4);
  vec_type<int16_t, 128> out;
  __builtin_memcpy(&out, &ordered, sizeof(out));
  return out;
}

template <typename Byte, typename Halfword>
TL_DEVICE vec_type<Byte, 128>
tl_hvx_high_bytes(const vec_type<Halfword, 128> &x) {
  static_assert(sizeof(Byte) == 1 && sizeof(Halfword) == 2);
  HVX_VectorPair pair;
  __builtin_memcpy(&pair, &x, sizeof(pair));
  HVX_Vector packed = Q6_Vb_vpacko_VhVh(Q6_V_hi_W(pair), Q6_V_lo_W(pair));
  vec_type<Byte, 128> out;
  __builtin_memcpy(&out, &packed, sizeof(out));
  return out;
}

// Four consecutive elements per group. Nibbles occupy one half of each
// 32-bit lane. A byte-panel row instead supplies a word from each register.
template <int Bits, int N, typename T>
TL_DEVICE auto tl_hvx_load_groups(const T *p) {
  if constexpr (Bits == 4) {
    int high = reinterpret_cast<uintptr_t>(p) & 2;
    HVX_Vector v = tl_hvx_load_window<N, true>(p - high);
    v = Q6_Vuw_vlsr_VuwR(v, high * 8);
    HVX_Vector halves = Q6_Vh_vpacke_VwVw(v, v);
    vec_type<T, N / 2> packed;
    __builtin_memcpy(&packed, &halves, N / 2);
    return packed;
  } else {
    vec_type<T, N> out;
#pragma clang loop unroll(full)
    for (int i = 0; i < N / 4; ++i)
      __builtin_memcpy(reinterpret_cast<char *>(&out) + i * 4, p + i * 128, 4);
    return out;
  }
}

// An N-lane vector (at most two registers) of two runs of N / 2 elements, at
// p and stride elements on, interleaved: lane l is p[(l % 2) * stride + l / 2],
// as a Crouton tile register's row pair reads two rows of a row-major matrix.
// One load of each run, and a shuffle. Aligned: both runs start on multiples
// of their size.
template <int E, int N, bool Aligned, typename T>
TL_DEVICE vec_type<T, N> tl_hvx_load_interleave(const T *p, int stride) {
  constexpr int B = static_cast<int>(sizeof(T));
  constexpr int run = N / 2 * B;
  static_assert(E == 2 && N % 2 == 0 && run <= 128 && (run & (run - 1)) == 0,
                "two runs of a power of two of at most one register");
  HVX_Vector even = tl_hvx_load_window<run, Aligned>(p);
  HVX_Vector odd = tl_hvx_load_window<run, Aligned>(p + stride);
  // Not inside the intrinsic: it is a macro (see tl_hvx_load_replicate).
  constexpr int lanes = -B;
  HVX_VectorPair w = Q6_W_vshuff_VVR(odd, even, lanes);
  vec_type<T, N> out;
  __builtin_memcpy(&out, &w, N * B);
  return out;
}

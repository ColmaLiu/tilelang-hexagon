#pragma once

#include <tl_templates/hexagon/hmx_gemm.h>

// Register j in each byte plane holds four rows, interleaved by column.
// Gather even and odd rows while joining bytes, then join halfwords to
// leave one row of 32 accumulator words in each register.
TL_HMX_INLINE void tl_hmx_int_store(int32_t *c, int ldc,
                                    const HVX_Vector *planes, bool clear) {
  for (int j = 0; j < 16; ++j) {
    HVX_Vector p0 = planes[j], p1 = planes[16 + j];
    HVX_Vector p2 = planes[32 + j], p3 = planes[48 + j];
    HVX_Vector even_lo = Q6_Vb_vshuffe_VbVb(p1, p0);
    HVX_Vector odd_lo = Q6_Vb_vshuffo_VbVb(p1, p0);
    HVX_Vector even_hi = Q6_Vb_vshuffe_VbVb(p3, p2);
    HVX_Vector odd_hi = Q6_Vb_vshuffo_VbVb(p3, p2);
    HVX_Vector rows[] = {
        Q6_Vh_vshuffe_VhVh(even_hi, even_lo),
        Q6_Vh_vshuffe_VhVh(odd_hi, odd_lo),
        Q6_Vh_vshuffo_VhVh(even_hi, even_lo),
        Q6_Vh_vshuffo_VhVh(odd_hi, odd_lo),
    };
#pragma clang loop unroll(full)
    for (int r = 0; r < 4; ++r) {
      int32_t *dst = c + (4 * j + r) * ldc;
      HVX_Vector v = rows[r];
      if (!clear)
        v = Q6_Vw_vadd_VwVw(v, tl_hvx_loadu(dst));
      tl_hvx_storeu(dst, v);
    }
  }
}

TL_HMX_INLINE void tl_hmx_int_bias(const HVX_Vector *config, int plane,
                                   int bank) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(config + 2 * plane) | bank;
  asm volatile("bias = mxmem2(%0)" ::"r"(addr) : "memory");
}

template <bool Shared, bool Paired = false>
TL_HMX_GEMM void
tl_hexagon_hmx_gemm_int(int M, int N, int K, int32_t *c, int ldc,
                        const uint8_t *a, int a_row_tile_bytes, const int8_t *b,
                        int panel_bytes, bool clear, uint8_t *stage,
                        uint32_t *config, bool reload_banks = false) {
  auto *banks = reinterpret_cast<HVX_Vector *>(config);
  for (int i = 0; i < 4; ++i) {
    banks[2 * i] = Q6_V_vsplat_R((24 - 8 * i) << 10);
    banks[2 * i + 1] = Q6_V_vzero();
  }
  int32_t *previous = nullptr;
  auto acquire = [&]() {
    tl_hmx_unit_acquire();
#if __HVX_ARCH__ >= 79
    // Banks belong to the unit. Another worker may run an fp16 GEMM
    // between this worker's tiles and replace the conversion scales.
    if (!previous || reload_banks)
      for (int i = 0; i < 4; ++i)
        tl_hmx_int_bias(banks, i, i);
#endif
  };
  if constexpr (!Shared)
    acquire();

  auto *planes = reinterpret_cast<HVX_Vector *>(stage);
  constexpr int groups = Paired ? 2 : 1;
  constexpr int width = groups * 32;
  constexpr int weight_tile_bytes = width * 32;
  int slot = 0;
  auto store_previous = [&]() {
    for (int group = 0; group < groups; ++group)
      tl_hmx_int_store(previous + group * 32, ldc,
                       planes + ((slot ^ 1) * groups + group) * 64, clear);
  };
  for (int mt = 0; mt < M / 64; ++mt) {
    for (int nt = 0; nt < N / width; ++nt, slot ^= 1) {
      if constexpr (Shared)
        acquire();
      asm volatile("mxclracc" ::: "memory");
      for (int k = 0; k < K / 32; k += 32) {
        int n = K / 32 - k < 32 ? K / 32 - k : 32;
        const uint8_t *ap = a + mt * a_row_tile_bytes + k * 2048;
        const int8_t *bp = b + nt * panel_bytes + k * weight_tile_bytes;
        unsigned ar = n * 2048u - 1;
        unsigned wr = ((n * weight_tile_bytes - 128u) & ~127u) | 0x7f;
        if constexpr (Paired)
          asm volatile("{ activation.ub = mxmem(%0, %1):deep\n"
                       "  weight.b = mxmem(%2, %3):deep }" ::"r"(ap),
                       "r"(ar), "r"(bp), "r"(wr)
                       : "memory");
        else
          asm volatile("{ activation.ub = mxmem(%0, %1):deep\n"
                       "  weight.b = mxmem(%2, %3) }" ::"r"(ap),
                       "r"(ar), "r"(bp), "r"(wr)
                       : "memory");
      }
      if constexpr (!Shared)
        if (previous)
          store_previous();
      // Releasing the fourth byte clears this accumulator and selects the next.
      for (int group = 0; group < groups; ++group) {
        for (int i = 0; i < 4; ++i) {
#if __HVX_ARCH__ >= 79
          int bank = i;
#else
          int bank = 0;
          tl_hmx_int_bias(banks, i, bank);
#endif
          int control = (1 << 15) | (bank << 12) | 2 | (i < 3);
          auto *out = planes + (slot * groups + group) * 64 + i * 16;
          asm volatile("cvt.ub = acc(%0)\n mxmem(%1, %2) = cvt" ::"r"(control),
                       "r"(out), "r"(0xf00)
                       : "memory");
        }
      }
      if constexpr (Shared) {
        tl_hmx_unit_release();
        if (previous)
          store_previous();
      }
      previous = c + mt * 64 * ldc + nt * width;
    }
  }
  if constexpr (!Shared)
    tl_hmx_unit_release();
  store_previous();
}

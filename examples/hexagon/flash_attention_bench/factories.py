"""Current example and matched row-major control, using the same DSL/runtime."""

import importlib.util
from pathlib import Path

from tilelang.hexagon import language as T

EXAMPLE = Path(__file__).resolve().parents[1] / "example_flash_attention.py"
_spec = importlib.util.spec_from_file_location("flash_attention_bench_example", EXAMPLE)
_example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_example)
make_crouton = _example.make_flash
NEG = -3.0e38


def make_staged(M, SEQ, BN, D):
    @T.prim_func
    def flash(
        Q: T.Tensor((M, D), "float16"), K: T.Tensor((SEQ, D), "float16"), V: T.Tensor((SEQ, D), "float16"), O: T.Tensor((M, D), "float16")
    ):
        with T.Kernel(1) as _:
            K_row = T.alloc_shared((BN, D), "float16")
            V_row = T.alloc_shared((BN, D), "float16")
            Q_sh = T.alloc_shared((M, D), "float16")
            K_sh = T.alloc_shared((BN, D), "float16")
            V_sh = T.alloc_shared((BN, D), "float16")
            S_hmx = T.alloc_shared((M, BN), "float16")
            S = T.alloc_shared((M, BN), "float16")  # row-major softmax
            temp_hmx = T.alloc_shared((M, D), "float16")
            temp = T.alloc_shared((M, D), "float16")  # row-major accumulation
            acc_o = T.alloc_shared((M, D), "float16")
            m = T.alloc_shared((M,), "float32")  # running max
            l = T.alloc_shared((M,), "float32")  # running denominator
            rmax = T.alloc_shared((M,), "float32")  # reduce dst (also the new max)
            rsum = T.alloc_shared((M,), "float32")  # reduce dst
            scale = T.alloc_shared((M,), "float32")
            T.copy(Q, Q_sh)
            for i, j in T.Parallel(M, D):
                acc_o[i, j] = T.float16(0)
            for i in T.Parallel(M):
                m[i] = T.float32(NEG)
                l[i] = T.float32(0)
            for kv in T.serial(SEQ // BN):
                T.copy(K[kv * BN, 0], K_row)
                T.copy(V[kv * BN, 0], V_row)
                T.copy(K_row, K_sh)
                T.copy(V_row, V_sh)
                T.gemm(Q_sh, K_sh, S_hmx, transpose_B=True, clear_accum=True)  # S = Q@Kᵀ (HMX)
                T.copy(S_hmx, S)  # Crouton -> row-major
                T.reduce_max(S, rmax, dim=1)  # rmax = rowmax(S) (HVX)
                for i in T.Parallel(M):  # running max + rescale
                    rmax[i] = T.max(m[i], rmax[i])
                    scale[i] = T.exp(m[i] - rmax[i])
                for i, j in T.Parallel(M, BN):  # P = exp(S − m)
                    S[i, j] = T.cast(T.exp(T.cast(S[i, j], "float32") - rmax[i]), "float16")
                T.reduce_sum(S, rsum, dim=1)  # rsum = rowsum(P) (HVX)
                for i in T.Parallel(M):  # denominator update
                    l[i] = l[i] * scale[i] + rsum[i]
                    m[i] = rmax[i]
                for i, j in T.Parallel(M, D):  # rescale acc_o
                    acc_o[i, j] = T.cast(T.cast(acc_o[i, j], "float32") * scale[i], "float16")
                T.copy(S, S_hmx)  # row-major -> Crouton
                T.gemm(S_hmx, V_sh, temp_hmx, clear_accum=True)  # temp = P@V (HMX)
                T.copy(temp_hmx, temp)  # Crouton -> row-major
                for i, j in T.Parallel(M, D):  # acc_o += temp
                    acc_o[i, j] = T.cast(T.cast(acc_o[i, j], "float32") + T.cast(temp[i, j], "float32"), "float16")
            for i, j in T.Parallel(M, D):  # O = acc_o / l
                acc_o[i, j] = T.cast(T.cast(acc_o[i, j], "float32") / l[i], "float16")
            T.copy(acc_o, O)

    return flash

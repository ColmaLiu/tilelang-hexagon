"""Original 19 factories; fixed read-only inputs; kernel/full SDK timers."""

from pathlib import Path
import re
from dsp_power import with_power
from frozen_ops import make_operator, input_layout


def make_timed(case, warmups, samples):
    import tilelang
    import tilelang.language as T
    from tilelang.utils.target import determine_target
    import example_demo_ops as ops

    s = case["spec"]
    m, n = s["m"], s["n"]
    total = warmups + samples
    func = make_operator(case, ops)
    target = determine_target("hexagon", return_object=True)
    with target:
        artifact = tilelang.lower(
            func.with_attr("global_symbol", "original19_op"), target=target, enable_host_codegen=False, enable_device_compile=False
        )
    original = str(artifact.kernel_source)
    entry = "original19_op_kernel"
    signature = re.search(r"\bint32_t\s+" + entry + r"\(([^)]*)\)", original)
    if not signature:
        raise RuntimeError("Unexpected generated signature")
    names = [re.search(r"(\w+)\s*$", a).group(1) for a in signature[1].split(",")]
    source, count = re.subn(r'#ifdef __cplusplus\s+extern "C"\s+#endif\s+(?=int32_t\s+' + entry + r"\()", "", original)
    if count != 1:
        raise RuntimeError("Expected one operator entry")
    source = source.replace("int32_t " + entry + "(", "static int32_t " + entry + "(", 1)
    gemm = case["group"] == "gemm"
    unary = len(input_layout(case)) == 1
    mapping = {"A": "a", "W": "w", "C": "o"} if gemm else {"X": "x", "O": "o"}
    if not gemm and not unary:
        mapping["Y"] = "y"
    if set(names) != set(mapping):
        raise RuntimeError("Unexpected operator arguments: " + str(names))
    decl = "float *a, half *w, float *o" if gemm else "float *x, float *o" if unary else "float *x, float *y, float *o"
    helper_args = "a,w,o" if gemm else "x,o" if unary else "x,y,o"
    call = ",".join(mapping[n] for n in names)
    # Fixed read-only benchmark contract only. Preserve DMA hardware options,
    # descriptors and waits, omitting repeated per-input cache publication.
    dma = r"""
template <uint32_t Capacity>
static inline int readonly_input_copy(tl_hexagon_dma_async_context<Capacity> &ctx,
    void *dst, const void *src, uint32_t bytes, int direction) {
  if (direction != TL_HEXAGON_DMA_DDR_TO_VTCM)
    return ctx.copy_1d(dst,src,bytes,direction);
  tl_hexagon_dma_queue *q=&ctx.queue;
  tl_hexagon_dma_descriptor *d=tl_hexagon_dma_queue_next_descriptor(q);
  if (!d) return TL_HEXAGON_DMA_ERR_QUEUE_FULL;
  int rc=tl_hexagon_dma_descriptor_init_1d(d,dst,src,bytes,
      tl_hexagon_dma_options_for_direction(direction));
  if (rc) return rc;
  tl_hexagon_dma_transfer transfer={dst,src,bytes,bytes,bytes,
      bytes==0 ? 0u : 1u,TL_HEXAGON_DMA_NO_CACHE};
  return tl_hexagon_dma_queue_submit_prepared(q,transfer);
}
"""
    dma_include = "#include <tl_templates/hexagon/dma.h>"
    if "tl_dma_async.copy_1d(" in source:
        if source.count(dma_include) != 1:
            raise RuntimeError("Unexpected DMA include")
        source = source.replace("tl_dma_async.copy_1d(", "readonly_input_copy(tl_dma_async, ")
        source = source.replace(dma_include, dma_include + "\n" + dma, 1)
    else:
        source = dma_include + "\n" + source
    if "tl_dma_async.copy_2d(" in source:
        raise RuntimeError("2D input contract not implemented")
    inputs = [("a", m * s["k"] * 4), ("w", n * s["k"] * 2)] if gemm else [("x", m * n * 4)]
    if not gemm and not unary:
        inputs.append(("y", m * n * 4))
    publish = "\n".join(
        f"  if ((publish_rc=tl_hexagon_dma_cache_clean({p},{b}u,QURT_MEM_CACHE_FLUSH))) return publish_rc;" for p, b in inputs
    )
    source = (
        "#include <HAP_perf.h>\n"
        + (Path(__file__).parent / "cache_policy.h").read_text()
        + "\n"
        + source
        + f"""
static inline int32_t collect_ticks({decl}, int64_t *ticks) {{
  for(int i=0;i<{2 * total};++i) ticks[i]=-1;
  int publish_rc=0;
{publish}
  for(int i=0;i<{total};++i) {{
    asm volatile("" ::: "memory");
    uint64_t full_start=HAP_perf_get_qtimer_count();
    uint64_t kernel_start=HAP_perf_get_qtimer_count();
    int32_t rc={entry}({call});
    asm volatile("" ::: "memory");
    uint64_t kernel_end=HAP_perf_get_qtimer_count();
    if(!rc) rc=bench_publish_output(o,{m * n * 4}u);
    asm volatile("" ::: "memory");
    uint64_t full_end=HAP_perf_get_qtimer_count();
    ticks[i]=(int64_t)(kernel_end-kernel_start);
    ticks[{total}+i]=(int64_t)(full_end-full_start);
    if(rc) return rc;
  }}
  return 0;
}}
"""
    )
    source = with_power(source, "collect_ticks", decl, helper_args, 2 * total, "max")
    if gemm:
        k = s["k"]

        @T.prim_func
        def timed_gemm(
            A: T.Tensor((m, k), "float32"),
            W: T.Tensor((n, k), "float16"),
            O: T.Tensor((m, n), "float32"),
            ticks: T.Tensor((2 * total,), "int64"),
            status: T.Tensor((22,), "int32"),
        ):
            with T.Kernel(1, threads=1):
                T.import_source(source)
                status[0] = T.call_extern(
                    "int32",
                    "collect_ticks",
                    T.access_ptr(A[0, 0], "r", m * k),
                    T.access_ptr(W[0, 0], "r", n * k),
                    T.access_ptr(O[0, 0], "w", m * n),
                    T.access_ptr(ticks[0], "w", 2 * total),
                    T.access_ptr(status[0], "w", 22),
                )

        return timed_gemm, original, source
    if unary:

        @T.prim_func
        def timed_unary(
            X: T.Tensor((m, n), "float32"),
            O: T.Tensor((m, n), "float32"),
            ticks: T.Tensor((2 * total,), "int64"),
            status: T.Tensor((22,), "int32"),
        ):
            with T.Kernel(1, threads=1):
                T.import_source(source)
                status[0] = T.call_extern(
                    "int32",
                    "collect_ticks",
                    T.access_ptr(X[0, 0], "r", m * n),
                    T.access_ptr(O[0, 0], "w", m * n),
                    T.access_ptr(ticks[0], "w", 2 * total),
                    T.access_ptr(status[0], "w", 22),
                )

        return timed_unary, original, source

    @T.prim_func
    def timed_binary(
        X: T.Tensor((m, n), "float32"),
        Y: T.Tensor((m, n), "float32"),
        O: T.Tensor((m, n), "float32"),
        ticks: T.Tensor((2 * total,), "int64"),
        status: T.Tensor((22,), "int32"),
    ):
        with T.Kernel(1, threads=1):
            T.import_source(source)
            status[0] = T.call_extern(
                "int32",
                "collect_ticks",
                T.access_ptr(X[0, 0], "r", m * n),
                T.access_ptr(Y[0, 0], "r", m * n),
                T.access_ptr(O[0, 0], "w", m * n),
                T.access_ptr(ticks[0], "w", 2 * total),
                T.access_ptr(status[0], "w", 22),
            )

    return timed_binary, original, source

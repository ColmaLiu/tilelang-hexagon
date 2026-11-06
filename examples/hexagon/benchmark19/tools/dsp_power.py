"""Opt-in v73 benchmark power votes inside the standard TileLang invocation.

No transport/session ownership. Votes, queries and cleanup are outside every
sample's qtimer interval. The default path adds no code or power requests.
"""

POWER_INFO_WORDS = 22


def status_words(policy):
    if policy not in ("default", "max"):
        raise ValueError("Unknown power policy: " + policy)
    return POWER_INFO_WORDS if policy == "max" else 1


def with_power(source, helper, declarations, arguments, total, policy):
    status_words(policy)
    if policy == "default":
        return source
    signature = "int32_t " + helper + "("
    if source.count(signature) != 1:
        raise RuntimeError("Expected one timer helper: " + helper)
    inner = helper + "_without_power"
    source = source.replace(signature, "int32_t " + inner + "(", 1)
    runtime = ""
    if "tl_vtcm" in source:
        runtime += "  info[18] = (int32_t)tl_vtcm_total;\n"
    if "tl_parallel(" in source:
        runtime += "  info[19] = tl_num_workers();\n  info[20] = tl_pool.spawned;\n  info[21] = tl_pool.nw > 0 ? tl_pool.nw : 1;\n"
    return (
        source
        + r"""
#include <HAP_power.h>
static int tl_bench_power_ctx;
static inline void tl_bench_query(int32_t *info, int offset,
                                  HAP_Power_response_type type) {
  HAP_power_response_t response;
  memset(&response, 0, sizeof(response));
  response.type = type;
  int rc = HAP_power_get(&tl_bench_power_ctx, &response);
  info[offset] = rc;
  if (rc == 0) info[offset + 1] = type == HAP_power_get_dcvsEnabled
      ? (int32_t)response.dcvsEnabled : (int32_t)response.clkFreqHz;
}
"""
        + f"""
static inline int32_t {helper}({declarations}, int64_t *ticks, int32_t *info) {{
  for (int i = 1; i < {POWER_INFO_WORDS}; ++i) info[i] = -1;
  for (int i = 0; i < {total}; ++i) ticks[i] = -1;
  bool dcvs_attempted = false, hvx_attempted = false;
  HAP_power_request_t request;
  memset(&request, 0, sizeof(request));
  request.type = HAP_power_set_apptype;
  request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
  int rc = HAP_power_set(&tl_bench_power_ctx, &request);
  info[1] = rc;
  if (rc == 0) {{
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_DCVS_v3;
    request.dcvs_v3.set_dcvs_enable = TRUE;
    request.dcvs_v3.dcvs_enable = FALSE;
    request.dcvs_v3.set_core_params = TRUE;
    request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.set_bus_params = TRUE;
    request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.set_sleep_disable = TRUE;
    request.dcvs_v3.sleep_disable = TRUE;
    // The protected bus-corner extension is v79+, not used on this v73 board.
    dcvs_attempted = true;
    rc = HAP_power_set(&tl_bench_power_ctx, &request);
    info[2] = rc;
  }}
  if (rc == 0) {{
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = TRUE;
    hvx_attempted = true;
    rc = HAP_power_set(&tl_bench_power_ctx, &request);
    info[3] = rc;
  }}
  if (rc == 0) {{
    tl_bench_query(info, 4, HAP_power_get_clk_Freq);
    tl_bench_query(info, 6, HAP_power_get_dma_core_clk_Freq);
    tl_bench_query(info, 8, HAP_power_get_dcvsEnabled);
    rc = {inner}({arguments}, ticks);
    tl_bench_query(info, 10, HAP_power_get_clk_Freq);
    tl_bench_query(info, 12, HAP_power_get_dma_core_clk_Freq);
    tl_bench_query(info, 14, HAP_power_get_dcvsEnabled);
{runtime}  }}
  // Remove only this benchmark context's votes, including on kernel failure.
  if (hvx_attempted) {{
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = FALSE;
    info[16] = HAP_power_set(&tl_bench_power_ctx, &request);
    if (rc == 0 && info[16] != 0) rc = info[16];
  }}
  if (dcvs_attempted) {{
    HAP_power_set_dcvs_v3_init(&request);
    info[17] = HAP_power_set(&tl_bench_power_ctx, &request);
    if (rc == 0 && info[17] != 0) rc = info[17];
  }}
  return rc;
}}
"""
    )


def decode_power(values, policy):
    expected = status_words(policy)
    if len(values) != expected:
        raise RuntimeError(f"Expected {expected} status words, got {len(values)}")
    data = dict(
        requested=policy,
        custom_votes=policy == "max",
        actual_bus_ddr_frequency_hz=None,
        frequency_scope="before warmups / after samples; not continuous telemetry",
    )
    if policy == "default":
        return data

    def reading(offset, boolean=False):
        value = None
        if values[offset] == 0:
            value = bool(values[offset + 1]) if boolean else values[offset + 1] & 0xFFFFFFFF
        return dict(rc=values[offset], value=value)

    data.update(
        request_rc=dict(compute_client=values[1], dcvs_core_bus_max=values[2], hvx_on=values[3]),
        cleanup_rc=dict(hvx_off=values[16], dcvs_reset=values[17]),
        core_hz_before=reading(4),
        core_hz_after=reading(10),
        dma_hz_before=reading(6),
        dma_hz_after=reading(12),
        global_dcvs_enabled_before=reading(8, True),
        global_dcvs_enabled_after=reading(14, True),
        vtcm_bytes=values[18] if values[18] >= 0 else None,
        max_hw_threads=values[19] if values[19] >= 0 else None,
        spawned_pool_workers=values[20] if values[20] >= 0 else None,
        last_parallel_dispatch_workers=values[21] if values[21] >= 0 else None,
        raw_status_words=values,
    )
    return data


def describe_power(data):
    if not data["custom_votes"]:
        return "POWER default: unchanged standard TileLang runtime"
    before = data["core_hz_before"]["value"]
    after = data["core_hz_after"]["value"]
    return (
        "POWER max: request_rc="
        + str(data["request_rc"])
        + f"; core_hz before={before}, after={after}; cleanup_rc="
        + str(data["cleanup_rc"])
        + "; actual bus/DDR frequency unverified"
    )

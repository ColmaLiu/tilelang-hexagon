// Benchmark-only output publication. Matches frozen stock hex_l2flush.
// SDK operations only; no llama.cpp runtime or kernel is linked into TileLang.
#include <hexagon_protos.h>
#include <qurt.h>
static inline int bench_publish_output(void *addr, uint32_t size) {
  if (size > 128u * 1024u)
    return qurt_mem_cache_clean((qurt_addr_t)0, 0,
                                QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL,
                                QURT_MEM_DCACHE);
  const uint32_t start = (uint32_t)addr;
  const uint32_t end = start + size;
  for (uint32_t i = start; i < end; i += 64 * 4) {
    Q6_dccleaninva_A((void *)(i + 64 * 0));
    Q6_dccleaninva_A((void *)(i + 64 * 1));
    Q6_dccleaninva_A((void *)(i + 64 * 2));
    Q6_dccleaninva_A((void *)(i + 64 * 3));
  }
  return 0;
}

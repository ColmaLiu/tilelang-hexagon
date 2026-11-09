#pragma once
// HW-thread worker pool for tilelang Hexagon kernels.
//
// The Hexagon DSP has a few HW threads, each able to issue HVX (and,
// time-shared, HMX) instructions.  tl_parallel runs a callback on `nw` workers:
// worker 0 on the CALLING thread, 1..nw-1 on a PERSISTENT pool of qurt threads.
// QuRT assigns HVX contexts implicitly. The hardware-thread count can exceed
// the HVX-context count. Such coresident grids block participants
// waiting at a contended barrier.
//
// The pool is spawned ONCE (lazily, on the first dispatch — FastRPC _run is
// serial, so no init race) and the workers BLOCK on a per-worker semaphore when
// idle (no CPU, no HVX held), so the per-dispatch cost is a semaphore up/down,
// not a thread spawn+join (which dominated tiny ops).  Big buffers live in
// VTCM, so the per-worker stack stays small.  The pool lives for the FastRPC
// session and is reclaimed when the PD unloads.
//
// Generated kernels reach the pool through tl_parallel_for, the form of a
// parallel grid loop: one call runs every tile program of the grid.
//
// Compiled into the FastRPC skel with qurt.h on the include path.
#include <qurt.h>
#include <stdint.h>
#include <stdlib.h>
#include <tl_templates/hexagon/common.h>
#include <tl_templates/hexagon/vtcm.h>
#include <type_traits>

#define TL_MAX_WORKERS 6
// 64 KiB a worker: a megakernel's program body and the out-of-line GEMMs it
// calls take tens of KiB of frames.
#define TL_WORKER_STACK_SZ (64 * 1024)

// Worker callback: (ctx, worker id in [0,nw), total worker count nw).  Returns
// a status code (0 = ok); tl_parallel returns the code of the lowest-numbered
// failing worker, so a per-worker device failure (e.g. HMX enable) propagates
// to the kernel entry and the host.
typedef int (*tl_worker_fn)(void *ctx, int wid, int nw);

// Maximum hardware-thread count, clamped to [1, TL_MAX_WORKERS]. This is
// independent of the number of hardware HVX contexts.
static inline int tl_num_workers(void) {
  qurt_sysenv_max_hthreads_t h;
  if (qurt_sysenv_get_max_hw_threads(&h) != QURT_EOK)
    return 1;
  int n = (int)h.max_hthreads;
  if (n > TL_MAX_WORKERS)
    n = TL_MAX_WORKERS;
  if (n < 1)
    n = 1;
  return n;
}

// Persistent pool: workers 1..TL_MAX_WORKERS-1 are spawned once; each blocks on
// its `go` semaphore until the dispatcher releases it for a job, runs the
// shared (fn,ctx,nw), then `up`s `done`.  The dispatcher is the ONLY writer of
// fn/ctx/nw and writes them only between a worker's previous completion and its
// next release, so no extra lock is needed: qurt_sem_up/down are out-of-line
// futex calls (opaque compiler barriers + kernel-side ordering), so the writes
// happen-before the worker's reads. (Do NOT replace them with an inlined
// lockless path without an explicit barrier.)
static struct {
  qurt_sem_t
      go[TL_MAX_WORKERS]; // dispatcher ups go[w]; worker w blocks on down
  qurt_sem_t done; // worker ups; dispatcher downs once per released worker
  tl_worker_fn fn;
  void *ctx;
  int nw;
  int spawned; // highest worker id spawned (0 = none)
  int tried;   // spawn attempted (so a failed spawn isn't retried/leaked)
  qurt_thread_t th[TL_MAX_WORKERS];
  void *blob;                 // one stack blob for the spawned workers
  int status[TL_MAX_WORKERS]; // worker w's return; dispatcher reads it after
                              // the join
} tl_pool;

static void tl_pool_worker(void *arg) {
  int wid = (int)(intptr_t)arg; // 1..spawned, fixed for this thread's life
  for (;;) {
    qurt_sem_down(&tl_pool.go[wid]); // BLOCK until released for a job
    // Publish status[wid] BEFORE the done-up: the dispatcher reads it only
    // after the matching done-down, so the up/down pair orders this write to
    // its read.
    tl_pool.status[wid] = tl_pool.fn(tl_pool.ctx, wid, tl_pool.nw);
    qurt_sem_up(&tl_pool.done);
  }
}

// Spawn the pool once.  Single-threaded (FastRPC _run is serial).  On a spawn
// failure we stop; tl_parallel runs the un-spawned workers' slices inline.
static void tl_pool_spawn(void) {
  tl_pool.tried =
      1; // attempt once; a failed spawn must not be retried (would leak)
  int n = TL_MAX_WORKERS - 1; // worker 0 is always the caller
  tl_pool.blob = malloc((size_t)TL_WORKER_STACK_SZ * n);
  if (!tl_pool.blob)
    return; // spawned stays 0 -> tl_parallel falls back to serial inline
  qurt_sem_init_val(&tl_pool.done, 0);
  for (int w = 1; w <= n; ++w)
    qurt_sem_init_val(&tl_pool.go[w],
                      0); // init ALL go[] before any create, so a
                          // partial-spawn failure leaves none undefined
  int prio = qurt_thread_get_priority(qurt_thread_get_id());
  for (int w = 1; w <= n; ++w) {
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(
        &attr, (char *)tl_pool.blob + (size_t)TL_WORKER_STACK_SZ * (w - 1));
    qurt_thread_attr_set_stack_size(&attr, TL_WORKER_STACK_SZ);
    qurt_thread_attr_set_priority(&attr, prio);
    if (qurt_thread_create(&tl_pool.th[w], &attr, tl_pool_worker,
                           (void *)(intptr_t)w) != QURT_EOK)
      break; // spawned stays at the last success
    tl_pool.spawned = w;
  }
  if (tl_pool.spawned ==
      0) { // not one worker started -> free the blob, run serial
    free(tl_pool.blob);
    tl_pool.blob = 0;
  }
}

// Run `fn` on `nw` workers and block until all finish.  Worker 0 runs inline on
// the calling thread; 1..nw-1 go to the persistent pool (any beyond the spawned
// count run inline too).  Falls back to fully serial if the pool couldn't be
// created.  With `coresident`, the workers must all run at once (they meet at
// tl_grid_sync), so a pool too small to hold them fails with TL_ERR_GRID
// instead.
static inline int tl_parallel(tl_worker_fn fn, void *ctx, int nw,
                              bool coresident = false) {
  if (nw < 1)
    nw = 1;
  if (nw > TL_MAX_WORKERS)
    nw = TL_MAX_WORKERS;
  if (nw == 1)
    return fn(ctx, 0, 1);
  if (!tl_pool.tried)
    tl_pool_spawn(); // lazy, once per session (tried set even on failure, no
                     // retry)
  int pooled = nw - 1;
  if (pooled > tl_pool.spawned) {
    if (coresident)
      return TL_ERR_GRID;
    pooled = tl_pool.spawned;
  }
  tl_pool.fn = fn;
  tl_pool.ctx = ctx;
  tl_pool.nw = nw;
  for (int w = 1; w <= pooled; ++w)
    qurt_sem_up(&tl_pool.go[w]); // release the pooled workers
  int status[TL_MAX_WORKERS];
  status[0] = fn(ctx, 0, nw); // worker 0 on the calling thread
  for (int w = pooled + 1; w < nw; ++w)
    status[w] = fn(ctx, w, nw); // overflow (fewer spawned than requested)
  for (int w = 1; w <= pooled; ++w)
    qurt_sem_down(&tl_pool.done); // wait for the pooled workers
  for (int w = 1; w <= pooled; ++w)
    status[w] = tl_pool.status[w]; // the join established happens-before
  for (int w = 0; w < nw; ++w) {
    if (status[w] != 0)
      return status[w];
  }
  return 0;
}

// --- parallel grid loops ---------------------------------------------------
// tl_parallel_for(extent, max_workers, body) runs body(i) for every tile
// program i in [0, extent) on up to min(max_workers, tl_num_workers(), extent)
// workers. Worker w runs i = w, w + nw, ... in order and stops at its first
// nonzero status. Given per-worker arenas, body(i, arena) also receives its
// worker's private VTCM arena, and no more workers run than arenas fit.
//
// `Scope` holds per-worker resources: every worker constructs one on its own
// thread before its first program and destroys it after its last. A scope
// whose status is nonzero after construction fails the worker.
//
// A `Coresident` grid runs every program at once, one per worker, so that its
// programs can meet at tl_grid_sync (T.sync_grid). It fails with TL_ERR_GRID
// when fewer workers can start than it has programs.
// Its body receives a final compile-time parking policy, selected once at
// dispatch. Ordinary grids therefore keep blocking calls out of their bodies.
struct tl_worker_scope {
  int status = TL_OK;
};

// The kernel arena split into `count` per-worker arenas of `bytes` each.
struct tl_worker_arenas {
  uint8_t *base;
  unsigned int bytes;
  unsigned int count;
};

// Per-worker arenas for a tile program whose arena takes `bytes`, after the
// first `grid` bytes of VTCM, the buffers all the programs share (whole HMX
// tiles). Each starts on a whole HMX tile, like the region itself, so that it
// keeps the alignment the program's buffers were laid out with.
static inline tl_worker_arenas tl_vtcm_worker_arenas(unsigned int bytes,
                                                     unsigned int grid = 0) {
  bytes = (bytes + TL_VTCM_HMX_ALIGNMENT - 1u) & ~(TL_VTCM_HMX_ALIGNMENT - 1u);
  uint8_t *base = tl_vtcm_arena();
  unsigned int total = tl_vtcm_arena_bytes();
  unsigned int count =
      base && bytes && total > grid ? (total - grid) / bytes : 0u;
  return {base ? base + grid : base, bytes, count};
}

// The barrier of the running coresident grid. Launches run one at a time per
// session, so one instance serves them all.
static struct {
  int32_t programs;               // programs that meet at each barrier
  int32_t arrived;                // programs waiting at the current barrier
  int32_t completed;              // barriers every program has passed
  int32_t failed;                 // a program stopped with an error
  int32_t passed[TL_MAX_WORKERS]; // barriers program i has reached, its own
  bool oversubscribed; // more participants than reported 128-byte HVX contexts
  uint32_t event;      // changes after every completion or failure publication
  int32_t waiters;     // participants registered in the blocking path
} tl_grid;

// Dispatch is serial within a Session, as for pool initialization. Cache the
// hardware configuration outside worker/poll loops; it does not report how
// many contexts are currently free. An unavailable query keeps the spin path.
static int tl_grid_hvx_contexts = -1;
static inline bool tl_grid_is_oversubscribed(int programs) {
  if (tl_grid_hvx_contexts < 0) {
    int units = qurt_hvx_get_units();
    tl_grid_hvx_contexts = units > 0 ? (units >> 8) & 255 : 0;
  }
  return tl_grid_hvx_contexts > 0 && programs > tl_grid_hvx_contexts;
}

// Both completion and failure change the word on which participants park.
// The sequentially consistent event/count handshake avoids a lost wake when
// registration races with publication. Only a registered waiter needs waking.
template <bool Parking> static inline void tl_grid_notify(void) {
  static_assert(Parking);
  __atomic_fetch_add(&tl_grid.event, 1u, __ATOMIC_SEQ_CST);
  if (__atomic_load_n(&tl_grid.waiters, __ATOMIC_SEQ_CST))
    qurt_futex_wake(&tl_grid.event, TL_MAX_WORKERS);
}

// Keep the whole blocking wait out of line so live vector values need saving
// only once. Read the event before checking the condition: publication before
// parking then makes the expected value mismatch, rather than losing a wake.
static __attribute__((noinline, cold)) int32_t
tl_grid_wait_blocking(int32_t barrier) {
  __atomic_add_fetch(&tl_grid.waiters, 1, __ATOMIC_SEQ_CST);
  int32_t status;
  for (;;) {
    uint32_t event = __atomic_load_n(&tl_grid.event, __ATOMIC_SEQ_CST);
    if (__atomic_load_n(&tl_grid.completed, __ATOMIC_ACQUIRE) == barrier) {
      status = TL_OK;
      break;
    }
    if (__atomic_load_n(&tl_grid.failed, __ATOMIC_RELAXED)) {
      status = TL_ERR_GRID;
      break;
    }
    qurt_futex_wait(&tl_grid.event, static_cast<int32_t>(event));
  }
  __atomic_sub_fetch(&tl_grid.waiters, 1, __ATOMIC_SEQ_CST);
  return status;
}

// Program `program` of the running coresident grid waits until all its
// programs reach this call. Memory written before the call, including HVX
// stores, is visible to every program after it. Ordinary grids retain the
// pause loop. An oversubscribed grid pauses briefly before blocking to give
// the scheduler an opportunity to run its remaining participants.
// A program that fails never arrives; others leave with TL_ERR_GRID.
template <bool Parking> static inline int32_t tl_grid_sync(int32_t program) {
  int32_t barrier = ++tl_grid.passed[program];
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  if (__atomic_add_fetch(&tl_grid.arrived, 1, __ATOMIC_ACQ_REL) ==
      tl_grid.programs) {
    // The last program resets the count before it releases the others.
    __atomic_store_n(&tl_grid.arrived, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&tl_grid.completed, barrier, __ATOMIC_RELEASE);
    if constexpr (Parking)
      tl_grid_notify<Parking>();
    return TL_OK;
  }
  if constexpr (!Parking) {
    while (__atomic_load_n(&tl_grid.completed, __ATOMIC_ACQUIRE) != barrier) {
      if (__atomic_load_n(&tl_grid.failed, __ATOMIC_RELAXED))
        return TL_ERR_GRID;
      asm volatile("pause(#8)");
    }
    return TL_OK;
  } else {
    for (unsigned polls = 0; polls < 128; ++polls) {
      if (__atomic_load_n(&tl_grid.completed, __ATOMIC_ACQUIRE) == barrier)
        return TL_OK;
      if (__atomic_load_n(&tl_grid.failed, __ATOMIC_RELAXED))
        return TL_ERR_GRID;
      asm volatile("pause(#8)");
    }
    return tl_grid_wait_blocking(barrier);
  }
}

template <typename Scope, typename Body, bool Coresident>
struct tl_parallel_for_job {
  Body *body;
  int32_t extent;
  tl_worker_arenas arenas;

  template <bool Parking> static int run(void *ctx, int wid, int nw) {
    tl_parallel_for_job *job = static_cast<tl_parallel_for_job *>(ctx);
    Scope scope;
    int status = scope.status;
    uint8_t *arena = job->arenas.base + (size_t)wid * job->arenas.bytes;
    for (int32_t i = wid; status == TL_OK && i < job->extent; i += nw) {
      if constexpr (Coresident)
        status =
            (*job->body)(i, arena, std::integral_constant<bool, Parking>{});
      else
        status = (*job->body)(i, arena);
    }
    // Programs waiting at a barrier for a failed one stop (tl_grid_sync).
    if (status != TL_OK) {
      __atomic_store_n(&tl_grid.failed, 1, __ATOMIC_RELAXED);
      if constexpr (Coresident && Parking)
        tl_grid_notify<Parking>();
    }
    return status;
  }
};

template <typename Scope = tl_worker_scope, bool Coresident = false,
          typename Body>
static inline int tl_parallel_for(int32_t extent, int max_workers,
                                  tl_worker_arenas arenas, Body &&body) {
  if (extent <= 0)
    return TL_OK;
  if (arenas.count < 1)
    return TL_ERR_VTCM; // not even one worker's arena fits the VTCM grant
  int nw = tl_num_workers();
  if (nw > max_workers)
    nw = max_workers;
  if ((unsigned int)nw > arenas.count)
    nw = (int)arenas.count;
  if (nw > extent)
    nw = (int)extent;
  // No participant from a previous dispatch remains after tl_parallel joins.
  if (Coresident) {
    tl_grid.oversubscribed = tl_grid_is_oversubscribed(extent);
    if (nw < extent)
      return TL_ERR_GRID;
    tl_grid.programs = extent;
    tl_grid.arrived = 0;
    tl_grid.completed = 0;
    tl_grid.failed = 0;
    tl_grid.event = 0;
    tl_grid.waiters = 0;
    for (int i = 0; i < TL_MAX_WORKERS; ++i)
      tl_grid.passed[i] = 0;
  }
  // Select one specialization for every participant before releasing workers.
  // Keep the shared job alive until tl_parallel has joined all participants.
  using Job =
      tl_parallel_for_job<Scope, typename std::remove_reference<Body>::type,
                          Coresident>;
  Job job{&body, extent, arenas};
  tl_worker_fn fn = &Job::template run<false>;
  if constexpr (Coresident) {
    if (tl_grid.oversubscribed)
      fn = &Job::template run<true>;
  }
  return tl_parallel(fn, &job, nw, Coresident);
}

template <typename Scope = tl_worker_scope, bool Coresident = false,
          typename Body>
static inline int tl_parallel_for(int32_t extent, int max_workers,
                                  Body &&body) {
  auto program = [&body](int32_t i, uint8_t *, auto... policy) {
    return body(i, policy...);
  };
  return tl_parallel_for<Scope, Coresident>(
      extent, max_workers, tl_worker_arenas{nullptr, 0u, TL_MAX_WORKERS},
      program);
}

// Exercise worker.h with host threads, not a copy of its barrier algorithm.
// Fake QuRT implements compare-and-enqueue/wake; it models no DSP scheduling.
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// External dependencies must not start a pool or request a resource grant.
enum { TL_OK = 0, TL_ERR_VTCM = 1, TL_ERR_GRID = 4, QURT_EOK = 0 };
constexpr unsigned TL_VTCM_HMX_ALIGNMENT = 2048;
struct qurt_sem_t {};
struct qurt_thread_attr_t {};
struct qurt_sysenv_max_hthreads_t {
  int max_hthreads;
};
using qurt_thread_t = uintptr_t;
static int qurt_sysenv_get_max_hw_threads(qurt_sysenv_max_hthreads_t *) {
  std::abort();
}
static void qurt_sem_init_val(qurt_sem_t *, int) { std::abort(); }
static void qurt_sem_up(qurt_sem_t *) { std::abort(); }
static void qurt_sem_down(qurt_sem_t *) { std::abort(); }
static qurt_thread_t qurt_thread_get_id() { std::abort(); }
static int qurt_thread_get_priority(qurt_thread_t) { std::abort(); }
static void qurt_thread_attr_init(qurt_thread_attr_t *) { std::abort(); }
static void qurt_thread_attr_set_stack_addr(qurt_thread_attr_t *, void *) {
  std::abort();
}
static void qurt_thread_attr_set_stack_size(qurt_thread_attr_t *, unsigned) {
  std::abort();
}
static void qurt_thread_attr_set_priority(qurt_thread_attr_t *, int) {
  std::abort();
}
static int qurt_thread_create(qurt_thread_t *, qurt_thread_attr_t *,
                              void (*)(void *), void *) {
  std::abort();
}
static uint8_t *tl_vtcm_arena() { std::abort(); }
static unsigned tl_vtcm_arena_bytes() { std::abort(); }
static int hardware_units, hardware_queries;
static int qurt_hvx_get_units() {
  ++hardware_queries;
  return hardware_units;
}

static std::mutex control_mutex;
static std::condition_variable control;
static bool hold_entry, hold_return;
static std::atomic<unsigned> pauses{0}, entries{0}, parked{0}, mismatches{0};
static unsigned spurious_budget, spurious_returns;
static std::atomic<unsigned> wake_calls{0};
static void announce() {
  std::lock_guard<std::mutex> lock(control_mutex);
  control.notify_all();
}
static void test_pause() {
  if (pauses.fetch_add(1) == 0)
    announce();
  std::this_thread::yield();
}
struct Waiter {
  void *word;
  bool notified = false;
  std::condition_variable ready;
};
static std::mutex futex_mutex;
static std::vector<Waiter *> waiters;
static int qurt_futex_wait(void *word, int expected) {
  {
    std::unique_lock<std::mutex> lock(control_mutex);
    ++entries;
    control.notify_all();
    control.wait(lock, [] { return !hold_entry; });
  }
  std::unique_lock<std::mutex> lock(futex_mutex);
  // Comparison and enrollment are atomic with respect to every fake wake.
  if (__atomic_load_n(static_cast<uint32_t *>(word), __ATOMIC_SEQ_CST) !=
      static_cast<uint32_t>(expected)) {
    ++mismatches;
    return QURT_EOK;
  }
  if (spurious_budget) {
    --spurious_budget;
    ++spurious_returns;
    return QURT_EOK;
  }
  Waiter waiter{word};
  waiters.push_back(&waiter);
  ++parked;
  announce();
  // Test watchdog only: a lost wake fails rather than hanging pytest. The
  // production futex API has no such timeout, and polling cannot wake this CV.
  assert(waiter.ready.wait_for(lock, std::chrono::seconds(3),
                               [&] { return waiter.notified; }));
  for (auto it = waiters.begin(); it != waiters.end(); ++it) {
    if (*it == &waiter) {
      waiters.erase(it);
      break;
    }
  }
  --parked;
  lock.unlock();
  std::unique_lock<std::mutex> control_lock(control_mutex);
  control.wait(control_lock, [] { return !hold_return; });
  return QURT_EOK;
}
static int qurt_futex_wake(void *word, int limit) {
  ++wake_calls;
  std::lock_guard<std::mutex> lock(futex_mutex);
  int count = 0;
  for (auto waiter : waiters) {
    if (waiter->word == word && !waiter->notified && count < limit) {
      waiter->notified = true;
      waiter->ready.notify_one();
      ++count;
    }
  }
  return count;
}

#include <tl_templates/hexagon/worker.h>

static void initialize(int participants, int contexts) {
  hardware_units = contexts << 8;
  tl_grid = {};
  tl_grid.programs = participants;
  tl_grid.oversubscribed = tl_grid_is_oversubscribed(participants);
}
template <typename Predicate> static void await(Predicate predicate) {
  std::unique_lock<std::mutex> lock(control_mutex);
  assert(control.wait_for(lock, std::chrono::seconds(3), predicate));
}
static void release_control() {
  std::lock_guard<std::mutex> lock(control_mutex);
  hold_entry = hold_return = false;
  control.notify_all();
}
template <bool Parking>
static void fail_program(int program, int participants) {
  std::array<uint8_t, 6> arenas{};
  auto fail = [&](int index, uint8_t *arena, auto policy) {
    static_assert(decltype(policy)::value == Parking);
    assert(index == program && arena == arenas.data() + program);
    return 37;
  };
  tl_parallel_for_job<tl_worker_scope, decltype(fail), true> job{
      &fail, participants, {arenas.data(), 1, 6}};
  assert(decltype(job)::template run<Parking>(&job, program, participants) ==
         37);
}
static void check_gate(int contexts) {
  hardware_units = contexts << 8;
  for (int n = 1; n <= 6; ++n)
    assert(tl_grid_is_oversubscribed(n) == (contexts > 0 && n > contexts));
  assert(hardware_queries == 1);
  hardware_units = contexts == 4 ? 6 << 8 : 4 << 8;
  for (int n = 1; n <= 6; ++n)
    assert(tl_grid_is_oversubscribed(n) == (contexts > 0 && n > contexts));
  assert(hardware_queries == 1 && pauses == 0 && entries == 0);
}
template <bool Parking>
static void check_visibility(int participants, int contexts,
                             bool spurious = false) {
  initialize(participants, contexts);
  const bool blocking = participants > contexts;
  assert(blocking == Parking && tl_grid.oversubscribed == Parking);
  spurious_budget = spurious ? 7 : 0;
  std::array<int, 6> values{};
  std::vector<std::thread> threads;
  auto body = [&](int worker) {
    for (int round = 1; round <= 64; ++round) {
      values[worker] = round * 100 + worker;
      assert(tl_grid_sync<Parking>(worker) == TL_OK);
      for (int other = 0; other < participants; ++other)
        assert(values[other] == round * 100 + other);
      // Keep the next round's writes after every reader has finished.
      assert(tl_grid_sync<Parking>(worker) == TL_OK);
    }
  };
  for (int worker = 0; worker < participants - 1; ++worker)
    threads.emplace_back(body, worker);
  await([blocking] { return blocking ? parked != 0 : pauses != 0; });
  threads.emplace_back(body, participants - 1);
  for (auto &thread : threads)
    thread.join();
  assert(tl_grid.completed == 128 && tl_grid.failed == 0);
  assert((entries != 0) == blocking);
  assert(tl_grid.waiters == 0 && waiters.empty());
  assert(spurious_returns == (spurious ? 7 : 0));
}
static void check_spin_failure() {
  initialize(6, 6);
  std::array<int, 5> status{};
  std::vector<std::thread> threads;
  for (int worker = 0; worker < 5; ++worker)
    threads.emplace_back(
        [&, worker] { status[worker] = tl_grid_sync<false>(worker); });
  await([] { return pauses != 0; });
  fail_program<false>(5, 6);
  for (auto &thread : threads)
    thread.join();
  for (int result : status)
    assert(result == TL_ERR_GRID);
  assert(tl_grid.completed == 0 && entries == 0);
}
static void check_publication(bool before_park, bool failure,
                              bool completed_first = false,
                              uint32_t initial_event = 0) {
  initialize(2, 1);
  tl_grid.event = initial_event;
  hold_entry = before_park;
  hold_return = completed_first;
  int status = -1;
  std::thread waiter([&] { status = tl_grid_sync<true>(0); });
  await([before_park] { return before_park ? entries != 0 : parked != 0; });
  // The waiter has registered in the production helper, but either has not
  // entered the atomic compare-and-enqueue yet or is already asleep there.
  assert(__atomic_load_n(&tl_grid.waiters, __ATOMIC_SEQ_CST) == 1);
  if (!failure || completed_first)
    assert(tl_grid_sync<true>(1) == TL_OK);
  if (failure)
    fail_program<true>(1, 2);
  release_control();
  waiter.join();
  assert(status == (failure && !completed_first ? TL_ERR_GRID : TL_OK));
  assert(tl_grid.waiters == 0 && waiters.empty());
  if (before_park)
    assert(mismatches == 1 && parked == 0);
  const unsigned publications = completed_first && failure ? 2u : 1u;
  assert(tl_grid.event == initial_event + publications);
  assert(wake_calls == publications);
}
// Model a waiter delayed between its fast check and slow registration. Use
// the actual last-arrival and job-failure publishers with no registered waiter,
// then call the actual slow helper after publication. No event change or wake
// is needed: its SC condition checks must observe the published result.
static void check_late_publication(bool completed, bool failed) {
  initialize(1, 1);
  constexpr uint32_t initial_event = 0x31415926u;
  tl_grid.event = initial_event;
  if (completed)
    assert(tl_grid_sync<true>(0) == TL_OK);
  if (failed)
    fail_program<true>(0, 1);
  assert(tl_grid.event == initial_event && tl_grid.waiters == 0);
  assert(wake_calls == 0 && entries == 0);
  assert(tl_grid_wait_blocking(1) == (completed ? TL_OK : TL_ERR_GRID));
  assert(tl_grid.event == initial_event && tl_grid.waiters == 0);
  assert(wake_calls == 0 && entries == 0 && waiters.empty());
}
static void check_completed_spin() {
  initialize(6, 6);
  tl_grid.completed = 1;
  tl_grid.failed = 1;
  assert(tl_grid_sync<false>(0) == TL_OK && entries == 0);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  const std::string test = argv[1];
  if (test == "gate-unknown")
    check_gate(0);
  else if (test == "gate-four")
    check_gate(4);
  else if (test == "gate-six")
    check_gate(6);
  else if (test == "visible-four")
    check_visibility<false>(4, 4);
  else if (test == "visible-six-spin")
    check_visibility<false>(6, 6);
  else if (test == "visible-six-blocking")
    check_visibility<true>(6, 4);
  else if (test == "spurious-returns")
    check_visibility<true>(6, 4, true);
  else if (test == "failed-spin")
    check_spin_failure();
  else if (test == "failed-before-park")
    check_publication(true, true);
  else if (test == "failed-parked")
    check_publication(false, true);
  else if (test == "completed-before-park")
    check_publication(true, false);
  else if (test == "completed-parked")
    check_publication(false, false);
  else if (test == "late-completed")
    check_late_publication(true, false);
  else if (test == "late-failed")
    check_late_publication(false, true);
  else if (test == "late-completed-and-failed")
    check_late_publication(true, true);
  else if (test == "wrapped-completed-before-park")
    check_publication(true, false, false, UINT32_MAX);
  else if (test == "wrapped-failed-before-park")
    check_publication(true, true, false, UINT32_MAX);
  else if (test == "completed-spin")
    check_completed_spin();
  else if (test == "completed-blocking")
    check_publication(false, true, true);
  else
    std::abort();
}

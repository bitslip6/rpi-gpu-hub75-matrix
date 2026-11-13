# Pin-free RIO schedule ownership tests (implementation phase 1)

This implements **deliverable 1** in `RIO_USERSPACE_PLAN.md`: schedule ownership
foundations, counter definitions, and tests. It is not Phase U1's dedicated
output process or kernel isolation work.

## Run

```sh
make test-rio-scan
make BUILD=debug test-rio-scan

# Separate build directories avoid reusing executables with different flags.
make BUILDDIR=build/rio-ndebug EXTRA_CFLAGS='-DNDEBUG=1 -Werror' test-rio-scan
make BUILD=debug BUILDDIR=build/rio-asan \
    EXTRA_CFLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer -Werror' \
    test-rio-scan
make BUILD=debug BUILDDIR=build/rio-tsan \
    EXTRA_CFLAGS='-fsanitize=thread -fno-omit-frame-pointer -Werror' \
    test-rio-scan
```

No root, GPIO mapping, display library, GPU/FFmpeg libraries, or ARM-specific
compiler flags are needed. The test is POSIX C with pthreads; checks remain active
with `NDEBUG`. A 60-second process alarm detects hangs. Test-only yields deliberately
increase interleaving opportunities; the ownership primitive does not yield.

On the current aarch64 host, GCC's ThreadSanitizer builds the test but exits
before testing with `unsupported VMA range` (`Found 47 - Supported 39, 42 and 48`).
This is **not** a successful race-detector run. Run TSan again on a supported host.

## Scope and memory contract

- `include/rio_scan.h` / `src/rio_scan.c` provide a reusable internal primitive,
  compiled directly into the tests. They are **not linked into the production
  display libraries or installed as public API** yet.
- Exactly one preparer/producer and one emitter/consumer use a pool, with a
  controller allowed to request cooperative stop. Worker ownership can be handed
  over before thread creation or after joining, never concurrently.
- The caller allocates a pool and 2–4 schedule buffers before starting workers.
  `rio_scan_storage_words()` validates geometry, padding, capacity, and byte-size
  overflow. `rio_scan_init()` checks storage size/alignment and lock-free atomics.
  Storage is 64-byte aligned and each schedule has a padded cache-line stride.
  The pool type is also over-aligned: stack/static objects respect this naturally;
  dynamically allocated pools must use an aligned allocator, not plain `malloc`.
- State slots, producer-local state, consumer-local state, and the stop flag start
  on separate 64-byte cache-line boundaries. The active payload does not share
  cache lines with a buffer being prepared.
- The words are opaque test data, not a validated electrical/protocol schedule.
  This first format holds one word per pixel, with
  `word_count = width * scanned_rows * planes`. Parallel RGB ports do not multiply
  this count. Padding is allocation space, never emitted data.
- No allocation, locks, semaphores, syscalls, logging, clocks, MMIO, or floating
  point occur in the ownership functions. Setup checks lock-free support before
  workers start. Memory locking/prefaulting, cross-process shared-memory setup,
  off-core telemetry transport, protocol preparation, and timing remain later work.

## Lifecycle

Normal ownership is:

```text
FREE -> BUILDING -> READY -> ACTIVE -> FREE
```

There are two intentional additional transitions: `BUILDING -> FREE` on producer
cancellation, and `READY -> FREE` when a complete but superseded generation is
dropped. Quiescent shutdown reclamation can discard any residual state.

1. The producer calls `rio_scan_build_begin()` once, fills the entire returned
   buffer, and calls `rio_scan_build_publish()` with a nonzero, strictly increasing
   generation ID. IDs cannot wrap; reinitialize a quiescent pool to start over.
   Failed publication retains the build for correction or explicit cancellation.
   A full pool gives backpressure, never permission to overwrite ACTIVE or READY.
   Writable pointers expire at publish/cancel.
2. At a safe boundary, the consumer calls `rio_scan_cycle_begin()`. It samples
   the bounded set of READY slots once, acquiring their immutable words and IDs,
   and selects the newest observed generation newer than the current ACTIVE.
   This is not a globally instantaneous snapshot: a racing publication may wait
   until the next boundary. Older observed complete generations are dropped.
3. `NEW` or `REPEAT` returns a read-only view and opens one full cycle. A nested
   begin returns `BUSY` without changing the view or releasing ownership. The
   caller emits all rows for all planes, then calls `rio_scan_cycle_end()` exactly
   once. End is a caller declaration of completion, not a verifier of GPIO output.
   No per-pixel or per-plane ownership operation is needed.
4. End records completed work but **retains ACTIVE**, so the producer still cannot
   recycle it. If no new schedule is observed next time, the consumer deliberately
   repeats it. With no initial schedule, begin returns `EMPTY` instead.
5. A view expires at the next successful begin or a `STOPPED` result. Never retain
   it for reads after the consumer releases its slot.

A subtle interleave is covered explicitly: a producer can publish an older frame
into an already-sampled slot, then publish a newer frame into a not-yet-sampled
slot. The consumer may select the newer one first. At the next boundary it must
**drop the older one**, not regress to it.

Publication uses a release READY store and acquire READY load; release FREE
happens only after the consumer's final payload read, and the producer acquires
FREE before reuse. No compare/exchange loop is needed: only the single producer
takes FREE slots, and only the single consumer takes/drops READY slots. `volatile`
is not used as synchronization.

## Counters and frame definitions

Counters are plain **single-owner** data. Read them on the owning worker or after
joining both workers, as the tests do. Do not read them concurrently for live
telemetry; a boundary snapshot transport will be needed for off-core aggregation.

| Counter | Meaning |
| --- | --- |
| `published_generations` | Producer successfully published a complete schedule. Not a renderer FPS count. |
| `no_free_slots` | Producer begin attempts rejected by full-pool backpressure. |
| `cancelled_builds` | Explicitly cancelled unfinished builds. |
| `content_generations` | Distinct producer-supplied generations acquired by the consumer. May skip dropped IDs. |
| `last_generation` | Most recently acquired ID; zero before first acquisition. |
| `plane_sweeps` | Completed full-cycle count multiplied by configured planes. One sweep covers all scanned rows for one `pwm`. |
| `full_cycles` | Completed emissions of all scanned rows for all configured planes. This is the initial service-rate unit, not content FPS. |
| `repeats` | Completed cycles for which begin deliberately retained the previous ACTIVE schedule. |
| `underruns` | Boundary attempts with neither ACTIVE nor an observed READY schedule. Repeats are not underruns. |
| `dropped_generations` | Observed complete READY schedules discarded as superseded. |
| `service_boundaries` | Accepted begin attempts, including initial empty attempts and first stop observation. Nested `BUSY` and subsequent `STOPPED` calls do not count. |

Service time, deadline misses, and shutdown latency need actual external clock
measurements; no pretend measurements or rate guarantees are supplied here.
Likewise these tests do not change the existing production loop's refresh labels.

## Shutdown

`rio_scan_request_stop()` only publishes a flag. If a cycle is open it remains
owned and must finish; the next begin releases ACTIVE and returns `STOPPED` without
acquiring a new schedule. Producer publication after observing stop fails and an
unfinished build must be cancelled. A publication racing the request can succeed.
The consumer does not free a producer-owned build or attempt a concurrent queue
reset. After **both workers have joined**, `rio_scan_reclaim()` discards any
remaining READY/BUILDING buffers and invalidates all pointers. Reinitialize to
restart. Reclaim is not a substitute for joining workers.

This is cooperative memory ownership only. Actual OE/CLK/LAT blanking, safe MMIO
completion, signal routing, shutdown latency measurement, and failure recovery
are not implemented here; SIGKILL/panic safety is not promised.

## Coverage and limits

- Geometry examples from the plan; zero/overflow/size/alignment validation;
  schedule padding and allocation canaries.
- Incomplete-build invisibility, invalid/double begin/commit/end/cancel,
  monotonic generation IDs including `UINT64_MAX`, full-pool backpressure,
  ACTIVE retention across every simulated plane, latest-complete selection,
  superseded-frame drops, repeats, and initial underruns.
- Sequential and deterministic threaded stop with ACTIVE, READY, and BUILDING
  states simultaneously live; quiescent reclamation.
- Seeded randomized sequential reference-model tests (30,000 operations each)
  and real producer/consumer stress (30,000 publications each), for 2, 3, and 4
  buffers. Every acquired payload is checked word-for-word, including after
  deliberate scheduling interleaves, with counter conservation checks after join.

These are software correctness tests, not a hardware baseline or timing proof.
The known queue misuse in `hub75_display_run()` is deliberately untouched until
a separate RIO baseline has been saved. No GPIO is driven and no scheduler,
CPU affinity, boot setting, interrupt routing, or electrical configuration changes.

# Pin-free benchmark and soak tools

Install the selected package first; run from the checkout or unpacked sdist.
These tools always use `offscreen=True` and never activate GPIO.

```sh
python benchmarks/build_reference.py       # when inside python/
# From the checkout root:
python python/benchmarks/build_reference.py
python python/benchmarks/benchmark.py --workload lines --compare-unbatched --output /tmp/lines.json
python python/benchmarks/benchmark.py --workload upload --output /tmp/upload.json
python python/benchmarks/benchmark.py --workload scene --output /tmp/scene.json
python python/benchmarks/benchmark.py --workload scene --shadows --width 128 --height 128 --output /tmp/shadows.json
python python/benchmarks/benchmark.py --workload text --font assets/roboto --output /tmp/text.json
python python/benchmarks/benchmark.py --workload lines --retained --frames 200
# GPU installation required; choose a local shader/video:
python python/benchmarks/benchmark.py --workload shader --file shaders/cartoon.glsl --frames 200
python python/benchmarks/benchmark.py --workload video --file /path/to/local.mp4 --frames 100
python python/benchmarks/soak.py --seconds 60 --output /tmp/soak.json
```

`build_reference.py` compiles the shared C demo against the **exact installed
package library**, with a reference-binary RUNPATH pointing there. It must be
rebuilt after switching backends/installations. Managed system linkage needs its
explicit `--libdir`. No separate renderer is compiled into the Python extension.

The benchmark performs warmup followed by repeated matched C/Python runs.
Manual workloads are unpaced throughput tests. Retained/media workloads are
natively paced. `p50_ns`/`p95_ns`/`p99_ns` measure manual acquire/render/submit call
latency, **not** electrical scanout. `latency_ns` and `mean_latency_ns` measure
native submission to logical RGBA consumption (including mapper backpressure on
physical sessions, but never proof of panel visibility). `cpu_ns` is total process
CPU during the measured interval; mean CPU cores is CPU time / wall time.
`shutdown_ns` times runtime close after content stop/flush. `render_ns`, `copy_ns`,
`map_ns`, `scan_ns` and queue waits are coarse inclusive native totals. Offscreen
map/scan totals are zero. No per-thread/core profiling is claimed by these tools;
use e.g. `pidstat -t`, `perf` and hardware instrumentation on deployment systems.

Manual image hashes must match. Time-driven retained/shader/video final frames
need not be at identical timestamps, so those hashes are reported without a
parity assertion. Setup/marshalling is outside steady-state measurements.
`--compare-unbatched` additionally measures a deliberately Python-call-heavy line
loop; that is a convenience-path comparison, not the recommended API.

Use identical dimensions, bit depth, FPS, queue depth and native library/flags.
Do not run compiler jobs or other benchmarks concurrently. Report absolute time
as well as percentages for tiny operations. Offscreen results do not qualify the
proposed Pi panel refresh, ≤5%/≤10% throughput, or eight-hour release gates.
See `../VALIDATION.md` for actual observations and remaining qualification, and
`results/local-aarch64-cp313.json` for the recorded local build snapshot.

The soak repeatedly creates/closes native sessions, mixes manual uploads with
retained scene/text rendering, mutates native state, and checks bounded RSS and
file/thread counts after warmup. Optional `--shader` / `--video` add local media
cycles. `--seconds 28800` requests the proposed eight-hour test; the default is
only 60 seconds. Samples are bounded to the last 4096 entries; inspect trends,
not just the configured `--rss-growth-mib` allowance. It is not a hardware soak.

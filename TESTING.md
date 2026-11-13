# Tests, CI, and protecting `main`

All automated CI tests are **pin-free**. GitHub-hosted ARM64 runners exercise
ARM/NEON code, but are not Raspberry Pis and do not validate panel timing,
wiring, physical brightness, refresh jitter, or electrical safety.

## Test inventory

| Suite | Command | Coverage / requirements |
| --- | --- | --- |
| Native C | `make test` | Runtime lifecycle, cancellation/failure paths, queues, fake GPIO affinity, legacy scanners against RAM, weighted BCM LUTs/bit planes/dithering, weighted RIO delivery, row-address policies, compositor, sampler, dashboard metrics, and video-fit geometry. CI runs on ARM64. |
| Portable host/controller | `make CFLAGS_CPU= test-host` | RIO ownership and RAM scope models; scope CLI/collector; fake CPU-isolation, boot/runtime setup, rollback and restoration; synthetic kernel packaging; video-fit geometry. No display library or GPU dependencies. |
| Native CLI | `make CFLAGS_CPU= test-runtime-cli` | Offscreen argument validation, topology, brightness/dithering, video dispatch/playback, signal handling and terminal dashboard. Builds GPU/FFmpeg support; actual EGL shader test is opt-in. |
| Python bindings | `python -X faulthandler -m unittest discover -s python/tests -v` | Installed-package ABI checks, lifecycle, buffers, resource ownership, retained rendering, content, CLI and invalid-input behavior. GPU-only tests skip for a CPU package. |
| Python packaging | `python python/tests/clean_install.py` | Build wheel and sdist; install each into a fresh venv outside the checkout; check private native libraries, RUNPATH, ABI rejection and CPU dependency isolation; rerun binding tests. |
| Type checking | See below | Python implementation, public stubs and CLI compatibility. |
| Resource soak | `python python/benchmarks/soak.py --seconds 2` | Short offscreen session/resource recycling check; not a long-duration qualification or throughput benchmark. |

`make test` includes `test-sampler` and `test-video-fit`. `test-host` also runs
`test-configure-system` and the RIO Python suites through `test-rio-scope`.
Kernel packaging tests use fake tools/files and deliberately skip when run as
root; run tests as an ordinary user, as CI does.

### Local prerequisites

Use a Linux ARM64 environment for the full native library suite. The portable
host suite also runs on Linux x86_64. Compiler defaults use `gnu2x`; CI uses
Ubuntu 24.04. Pass `CFLAGS_CPU=` to avoid host-specific tuning in reproducible
builds. Do not use `sudo make test`.

CPU build dependencies on Ubuntu:

```sh
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libpng-dev libjpeg-dev
make CFLAGS_CPU= test -j2
```

For GPU/media compilation and the offscreen native CLI suite, also install:

```sh
sudo apt-get install -y libgles2-mesa-dev libgbm-dev libegl1-mesa-dev \
  libavcodec-dev libavformat-dev libswscale-dev libavutil-dev
make CFLAGS_CPU= all test-runtime-cli -j2
```

For Python, use a virtual environment and its Python development headers:

```sh
python -m pip install --upgrade pip build setuptools wheel mypy numpy Pillow
HUB75_BACKEND=cpu HUB75_CFLAGS_CPU= python -m pip install ./python
python -X faulthandler -m unittest discover -s python/tests -v
mypy python/hub75gpu/__init__.py python/hub75gpu/__main__.py --follow-imports=skip --ignore-missing-imports
mypy python/hub75gpu/__init__.pyi python/hub75gpu/__main__.py --ignore-missing-imports
python python/tests/clean_install.py
```

NumPy/Pillow adapters are optional locally; CI installs both to exercise them.

### Sanitizers

CI runs the native suite against an ASan/UBSan-instrumented debug library:

```sh
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
make CFLAGS_CPU= BUILD=debug \
  EXTRA_CFLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  LDFLAGS='-fsanitize=address,undefined' test -j2
```

Debug and release outputs are separate. Do not reuse debug objects from a
non-sanitized build for this command: Make does not detect changed compiler
flags. Use a fresh checkout/build directory (as CI does), or clean your build
outputs first. This job does not instrument the host Python interpreter and
makes no ThreadSanitizer coverage claim.

## GitHub Actions

Workflow: [`.github/workflows/python.yml`](.github/workflows/python.yml)
(display name **CI**).

| Job | Hosted runner | Checks |
| --- | --- | --- |
| Portable host tests | `ubuntu-24.04` (x86_64) | `make test-host` |
| Native tests and GPU build | `ubuntu-24.04-arm` | Both native libraries, `make test`, offscreen native CLI regressions |
| Native ASan and UBSan | `ubuntu-24.04-arm` | Debug native suite with memory/UB checks and leak detection |
| Python 3.10 / 3.13 | `ubuntu-24.04-arm` | CPU package API, optional adapters, CLI and short soak; 3.13 also checks typing and clean wheel/sdist installation |
| **CI required** | `ubuntu-24.04` | Fails unless every prerequisite job/matrix entry succeeds |

The workflow runs on pushes, pull requests, merge-queue groups and manual
requests. There are no path filters that could leave a required check pending
for documentation-only changes. The aggregate job runs even after a test job
fails and rejects failed, cancelled or skipped prerequisite jobs.

Jobs use read-only repository permissions and no secrets. Tests do not open
physical GPIO, install persistent host tuning, or require a self-hosted runner.
Hosted ARM64 runner availability must be confirmed for your repository/plan;
if unavailable, use a supported hosted ARM alternative rather than running
untrusted pull requests on a Pi attached to panels.

### Enable branch protection / a ruleset

1. Commit and push the workflow **together with its source/test inputs**. In the
   current checkout, `.github/`, `tests/`, `examples/`, `rpihub75.pc.in` and this
   document include untracked files; merely pushing the previously merged branch
   commits will not include them. Review and include the relevant inputs, not
   generated `build/`, `tests/results/`, scope captures or shared libraries.
2. Let a GitHub Actions run finish successfully. The check must have been
   reported before GitHub can offer it for selection.
3. Open **Settings → Rules → Rulesets** and create an active branch ruleset for
   `main` (or use **Settings → Branches → Add branch protection rule**).
4. Require a pull request before merging and **require status checks to pass**.
   Select the check emitted by the job named **`CI required`** (GitHub may show
   it grouped under **CI**). Keep that job name stable.
5. Require the branch to be up to date before merging, or enable a merge queue;
   the workflow supports `merge_group` for the latter.
6. Block force pushes and branch deletion. Optionally require a review and apply
   the rules to administrators / restrict bypass permissions.

The workflow provides the check; adding the YAML does **not** itself activate
GitHub branch protection. Repository-administrator access is needed for that
setting. Do not make an optional physical-hardware job a required PR check.

## Hardware-only / separately authorized checks

- Actual EGL shader tests: GPU package plus explicit `HUB75_TEST_GPU=1`; the
  native demo shader-speed test separately uses `H75_TEST_GPU=1`.
- Physical output smoke test: `python/tests/hardware_smoke.py`, with its explicit
  hardware authorization and wiring/topology requirements.
- RIO/PIO scope and DMA tools, hardware baselines, and CPU-isolation audits under
  `tests/`: follow each accompanying README before running. They are not ordinary
  unit tests and may access devices or modify host settings when opted in.
- Benchmarks and long resource soaks: see
  [`python/VALIDATION.md`](python/VALIDATION.md). Short CI checks are not a substitute
  for an eight-hour soak or scope/logic-analyzer acceptance.

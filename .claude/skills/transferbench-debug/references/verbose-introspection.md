# Debug-flavored env vars and built-in introspection

This is the curated debug-side counterpart to the run-side `env-vars.md`. Variables here are the ones you reach for **when something is wrong**, not when you're tuning. For the authoritative complete list compiled into your binary, run:

```bash
./TransferBench envvars
```

## Built-in info commands (no env vars needed)

```bash
./TransferBench                  # banner: detected GPUs, NUMA, NICs, compiled features
./TransferBench help             # config-file syntax + examples
./TransferBench presets          # presets compiled into THIS build
./TransferBench envvars          # full env-var list with descriptions
./TransferBench dryrun "..."     # parse + expand wildcards, no execution
```

The first one is especially valuable when debugging — its banner names every executor and feature compiled in (`NIC_EXEC_ENABLED`, `POD_COMM_ENABLED`, `NVML_ENABLED`, etc.). If a feature you expect isn't there, you've found the problem before you even ran a test.

## Verbose & lifecycle logging

| Variable | What it does | When to use |
|---|---|---|
| `HIDE_ENV=0` (default) | Print env-var summary at startup | Always leave at default when debugging — you want to see what TransferBench thinks the env says |
| `TB_VERBOSE=1` | Verbose lifecycle logging in newer execution paths (anvil/SDMA) | When recent-feature execution paths hang or behave oddly |
| `SHOW_ITERATIONS=1` | Print every iteration's time | First step when "BW seems too low" or "BW is unstable" |
| `SHOW_PERCENTILES=50,75,90,99` | Add percentile columns | When the mean is fine but tail latency is the suspected issue |
| `SHOW_BORDERS=0` | Strip table borders | Easier to diff two runs' raw output |

## Validation / correctness

| Variable | Effect | Notes |
|---|---|---|
| `ALWAYS_VALIDATE=1` | Validate dst after every iteration | Catches data-corruption regressions; significantly slows runs |
| `VALIDATE_DIRECT=1` | Validate by reading dst directly | Skips copy-back; isolates "is the validation path itself buggy?" |
| `VALIDATE_SOURCE=1` | Confirm src wasn't overwritten | Catches kernels that aliased into src |
| `FILL_PATTERN=0xDEADBEEF` | Custom hex source fill pattern | Makes corruption signatures recognizable |
| `FILL_COMPRESS=1` | Use compressible source data | Useful when debugging compression-aware paths |
| `BYTE_OFFSET=N` | Offset (bytes) into allocated buffers | Useful when alignment is suspected |
| `BLOCK_BYTES=256` | Block granularity for transfers | Try larger / smaller when validation fails on edge sizes |

## Iteration / timing isolation

| Variable | Effect | Use case |
|---|---|---|
| `NUM_ITERATIONS=1` | Run exactly one iteration | "Validation fails on iter N>0" → reduce to 1 to confirm cold case |
| `NUM_WARMUPS=0` | No warmups | Forces iter-0 to be the only iteration; useful with `NUM_ITERATIONS=1` |
| `NUM_ITERATIONS=-30` | Timed mode, 30 seconds | When you want to study perf over time, not iterations |
| `NUM_SUBITERATIONS=N` | N sub-iterations per outer iteration | Reduce if sub-iter is the granularity at which a bug appears |

## Capture / reproducibility

| Variable | Effect | Use case |
|---|---|---|
| `TB_DUMP_CFG_FILE=out.cfg` | Dump executed Transfers to a config file | "What is this preset *actually* running?" — crucial when a preset behaves unexpectedly |
| `TB_DUMP_LINES=N` | Limit dumped lines | Quick peek at the start of a large preset |
| `OUTPUT_TO_CSV=1` | CSV output | Easier to diff two runs programmatically |

## Interactive / breakpoint-friendly

| Variable | Effect | Use case |
|---|---|---|
| `USE_INTERACTIVE=1` | Pause for stdin between tests | Attach `gdb` / `cuda-gdb` / `rocgdb` mid-run, set breakpoints, hit Enter to continue |

## NIC / RDMA debug

| Variable | Effect | Use case |
|---|---|---|
| `IB_GID_INDEX=N` | Force a specific GID index | Almost always the first thing to set when NIC presets hang |
| `IB_PORT_NUMBER=N` | Force a specific port | When the active port isn't 1 |
| `ROCE_VERSION=1\|2` | RoCE version | Both ends must agree |
| `IP_ADDRESS_FAMILY=4\|6` | IPv4 or IPv6 | When dual-stack hosts pick the wrong one |
| `TB_NIC_FILTER=name1,name2` | Restrict to listed NICs | Localize which NIC is misbehaving |
| `NIC_CHUNK_BYTES=N` | NIC transfer chunk size | Reduce to confirm a size-dependent bug |
| `NIC_CQ_POLL_BATCH=N` | CQ poll batch size | Reduce to 1 to expose race conditions |
| `NIC_RELAX_ORDER=0\|1` | Relaxed ordering on the NIC | Disable when ordering bugs suspected |

## Pod / multi-rank fallbacks

| Variable | Effect | Use case |
|---|---|---|
| `TB_FORCE_SINGLE_POD=1` | Treat all ranks as one pod | Workaround when AMD-SMI / NVML pod detection is broken |
| `TB_RANK`, `TB_NUM_RANKS`, `TB_MASTER_ADDR` | Socket-mode rank coordination | Alternative bootstrap when MPI isn't available or is the suspect |

## Wallclock / timing edge cases

| Variable | Effect | Use case |
|---|---|---|
| `TB_WALLCLOCK_RATE=<hz>` | Override GPU wallclock rate | Some firmware reports 0 for the rate; this lets you bypass that |
| `USE_HIP_EVENTS=0` | Use host clock instead of HIP/CUDA events | When HIP-event timing is suspected (rare) |

## Combined recipes

A few canned env-var combinations that come up often:

```bash
# "What does this preset actually run?"
TB_DUMP_CFG_FILE=p2p_dump.cfg ./TransferBench p2p

# "Is the slowness in iter 0 only, or every iter?"
NUM_WARMUPS=0 NUM_ITERATIONS=20 SHOW_ITERATIONS=1 ./TransferBench cmdline "1 4 (G0->G0->G1)" 256M

# "Validate every iteration, with a custom pattern"
ALWAYS_VALIDATE=1 VALIDATE_SOURCE=1 FILL_PATTERN=0xDEADBEEF ./TransferBench p2p

# "Quietest possible run for diff'ing two builds"
HIDE_ENV=1 SHOW_BORDERS=0 OUTPUT_TO_CSV=1 NUM_WARMUPS=10 NUM_ITERATIONS=20 ./TransferBench p2p > run_A.csv

# "Pause between tests so I can gdb attach"
USE_INTERACTIVE=1 ./TransferBench p2p
```

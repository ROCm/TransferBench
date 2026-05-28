---
name: transferbench-debug
description: Use when a TransferBench (ROCm/CUDA bandwidth-benchmark) run fails, hangs, crashes, validates incorrectly, or produces unexpected/misleading results — i.e. the user is troubleshooting rather than ramping up usage. Covers reading error output, isolating hangs (single-rank vs. multi-rank, NIC vs. POD detection), validation failures, performance regressions, and the binary's built-in verbose / dump / dryrun introspection. Does NOT cover writing new configs from scratch (use the run-side skill) or modifying TransferBench source.
---

# TransferBench debugging

This skill kicks in when something is **wrong** with a TransferBench run. The goal is always: turn a vague "it doesn't work" into a specific failure mode with a known fix or workaround.

## Triage flow

Always run these three steps **first** before guessing:

1. **Reproduce with the smallest possible config.** Replace presets with a single-line `cmdline` if possible; halve rank count; drop to one Transfer.
2. **Confirm the binary parses the input.** Run `dryrun` instead of executing — separates parser bugs from runtime bugs.
3. **Capture what the binary actually saw.** `TB_DUMP_CFG_FILE=out.cfg` for presets; `HIDE_ENV=0` (default) so the env-var summary at startup is visible.

Only after that, branch by symptom — see the table below.

## Symptom → reference

| Symptom | Most likely cause | First thing to try | Deeper |
|---|---|---|---|
| Process hangs at startup, no output | MPI bootstrap or socket-mode env vars wrong | `mpirun --tag-output` to confirm all ranks started; verify `TB_NUM_RANKS` matches `-np` | `references/multi-rank-debug.md` |
| `Pod-aware` preset hangs or errors out before transfers | AMD-SMI / NVML pod detection unavailable | `TB_FORCE_SINGLE_POD=1` | `references/common-failures.md` §pod |
| RDMA preset (nicp2p, nica2a, …) hangs in NIC bring-up | GID index, IB port, or NIC filter wrong | Lower `IB_GID_INDEX` to a known good index; `TB_NIC_FILTER` to a single NIC | `references/multi-rank-debug.md` §rdma |
| Validation failure (`ALWAYS_VALIDATE=1` reports mismatch) | Wrong CU mask, wrong memory type, or actual HW issue | `VALIDATE_DIRECT=1`; rerun with `NUM_ITERATIONS=1` to see if first iter is wrong | `references/common-failures.md` §validation |
| Bandwidth far below expected | Stream/HW-queue serialization, wrong executor, GFX kernel mis-tuned | `USE_SINGLE_STREAM=0` + `GPU_MAX_HW_QUEUES=8`; try `D` (DMA) instead of `G` | `references/common-failures.md` §perf |
| Bandwidth varies wildly run-to-run | Warmup too short, NUMA/clock policy | `NUM_WARMUPS=10`, `SHOW_ITERATIONS=1`, `SHOW_PERCENTILES=50,90,99` | `references/common-failures.md` §perf |
| Crash / segfault | Bad memory code (e.g. `F` on a GPU without fine-grain), bad kernel for arch | Run with `dryrun` first; rebuild without optimization for symbol info | `references/common-failures.md` §crash |
| "Unsupported" / executor missing | Build-time disable (e.g. `DISABLE_NIC_EXEC=1`, `DISABLE_POD_COMM=1`) | `./TransferBench` (no args) — its banner lists which executors are compiled in | `references/common-failures.md` §unsupported |
| Output is garbled / interleaved across ranks | MPI stderr buffering, no per-rank labels | `mpirun --tag-output` or pipe each rank into a per-rank log | `references/multi-rank-debug.md` §output |

## The four "always-on" introspection commands

These four commands are how you **observe** the binary as it actually exists on this host (don't trust any documentation, including this one, when troubleshooting):

```bash
./TransferBench                  # banner: detected GPUs, NUMA, NICs, compiled features
./TransferBench help             # config-file syntax with examples
./TransferBench presets          # list of presets compiled into THIS build
./TransferBench envvars          # complete list of env vars THIS build honors
```

Plus two safe inspections of any preset/config:

```bash
./TransferBench dryrun "<expression>"          # validate parsing, expand wildcards
TB_DUMP_CFG_FILE=dump.cfg ./TransferBench p2p  # dump what a preset actually emits
```

## Verbose / capture env vars

Reach for these when you need more visibility (full table in `references/verbose-introspection.md`):

| Env var | Effect |
|---|---|
| `HIDE_ENV=0` (default) | Print env-var summary at start (shows what was actually set) |
| `SHOW_ITERATIONS=1` | Per-iteration timings — exposes warmup/jitter issues |
| `SHOW_PERCENTILES=50,90,99` | Tail latencies — exposes slow-iteration outliers |
| `ALWAYS_VALIDATE=1` | Validate destination after every iteration (slow, but catches data-corruption regressions) |
| `VALIDATE_DIRECT=1` | Validate by reading the destination directly (skips copy-back path) |
| `VALIDATE_SOURCE=1` | Confirm src was unchanged (catches kernels that overwrite src) |
| `NUM_ITERATIONS=1` | Run exactly one iteration — useful when validation fails on iter N>0 |
| `NUM_WARMUPS=0` | Strip warmups so iter-0 timing is the cold case |
| `USE_INTERACTIVE=1` | Pause between tests — useful for `gdb attach` mid-run |
| `TB_DUMP_CFG_FILE=out.cfg` | Dump executed Transfers from a preset to a config file |
| `TB_DUMP_LINES=N` | Limit number of dumped lines |
| `TB_VERBOSE=1` | Verbose lifecycle logging for newer execution paths (anvil/SDMA in recent builds) |
| `TB_WALLCLOCK_RATE=<hz>` | Override GPU wallclock rate when the GPU returns 0 (debug-only) |

## Multi-rank-specific quick checks

When debugging across nodes, before suspecting TransferBench itself:

1. **Same binary on every node.** `md5sum ./TransferBenchCuda` on each host. A different mtime/checksum is the most common multi-rank gotcha.
2. **Same env on every rank.** Use `mpirun -x VAR` (not just shell-export); without `-x`, only rank 0 sees your shell vars.
3. **Network actually up.** `ibstatus` (RDMA) or a `nc` between hosts on your master port (socket mode).
4. **Hostfile slots = 1 per node.** TransferBench expects one rank per node by default.

## When you're stuck

If the table above and the references didn't help:

1. Build with `-g -O0` (or `-g -O1`) to get usable symbols, run under `gdb` / `cuda-gdb` / `rocgdb`, and `bt` once it hangs or crashes. Hangs in particular are usually obvious from the stuck thread's stack.
2. Strip the build down: pass `DISABLE_*` flags for any executor not under test (`DISABLE_NIC_EXEC=1`, `DISABLE_POD_COMM=1`, etc.). Eliminates whole code paths from suspicion.
3. Compare against a known-good commit. The `git log` on this repo has many tagged commits where features were added — you can check out an older commit, run the same config, and confirm it passes there.

## References

- `references/common-failures.md` — symptom-organized catalog with concrete fixes
- `references/multi-rank-debug.md` — MPI / socket / RDMA-specific issues
- `references/verbose-introspection.md` — every debug-flavored env var + when to reach for it
- `examples/topology-probe.sh` — minimal script that prints what TransferBench sees about the host

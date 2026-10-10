---
name: transferbench-run
description: Use when the user wants to *run* TransferBench (the ROCm/CUDA memory-transfer benchmarking tool from AMD) — benchmarking, profiling, or measuring GPU/CPU/NIC bandwidth and latency. Covers writing config files, picking the right preset (a2a, p2p, sweep, nicp2p, podp2p, etc.), tuning environment variables, and launching single-node or multi-node (MPI / socket) runs. Does NOT cover building the binary from source, modifying its source code, or extending it with new presets/executors — for those, defer to a separate skill or the codebase itself.
---

# TransferBench

TransferBench is a command-line utility for benchmarking simultaneous data transfers between CPU, GPU, and NIC memory locations using GPU kernels, DMA engines, RDMA NICs, or CPU threads. It runs on AMD (ROCm/HIP) and NVIDIA (CUDA) platforms.

The binary is named `TransferBench` (HIP build) or `TransferBenchCuda` (CUDA build). A prebuilt `TransferBenchCuda` may exist at the repo root; otherwise build with `make` (HIP) or `make TransferBenchCuda` (CUDA).

## Mental model

A **Transfer** is one operation: an **Executor** reads values from one or more **SRC** memory locations, sums them, and writes the result to one or more **DST** memory locations. With one SRC and one DST it's a plain copy.

```
SRC 0
SRC 1 -> Executor -> DST 0
SRC X                DST Y
```

A **Test** is one line in a config file — a set of Transfers run in parallel.

A **SubExecutor (SE)** is the unit of parallelism inside an executor:
- CPU executor → CPU thread
- GPU executor → threadblock / Compute Unit (CU)
- DMA / Batched-DMA → stream / batch item (must have a single SRC)
- NIC → Queue Pair

## Invocation

```bash
./TransferBench <config> [N]
```

- `<config>` is one of:
  - A path to a config file
  - A preset name (`a2a`, `p2p`, `sweep`, `nicp2p`, `podp2p`, ...)
  - `cmdline "<transfer expression>"` — run one ad-hoc transfer
  - `dryrun "<transfer expression>"` — parse and print without executing
  - `help`, `presets`, `envvars` — built-in info screens
- `N` (optional) is the number of bytes per Transfer. Defaults if omitted. `0` means sweep over a range. May be suffixed with `K`, `M`, or `G`. Must be a multiple of 4.

Run `./TransferBench` with no args to see usage and detected topology (GPUs, NUMA nodes, NICs).

## Quick recipes

Decide which path the user needs and pick from below. **Always prefer a preset** if it matches the user's intent — presets handle topology discovery and produce well-formatted output.

### "How fast is GPU↔GPU?" → use a preset
```bash
./TransferBench p2p           # peer-to-peer matrix between all GPUs
./TransferBench a2a           # all-to-all simultaneous transfers
./TransferBench scaling       # one GPU to all others, scaled CU counts
./TransferBench sweep 64M     # sweep through transfer combinations
```

### "How fast is HBM / local memory?"
```bash
./TransferBench hbm
```

### "How fast is CPU↔GPU or pinned-memory transfer?" → custom config
Write a small `.cfg` file (see `references/config-format.md`) and pass it as the first argument.

### "Benchmark one specific transfer" → cmdline mode
```bash
./TransferBench cmdline "1 4 (G0->G0->G1)" 256M
./TransferBench dryrun  "2 8 G0->G0->G1 G1->G1->G0"   # validate parsing first
```

### "RDMA / NIC across nodes" → NIC presets
```bash
./TransferBench nicp2p        # NIC peer-to-peer matrix
./TransferBench nica2a        # NIC all-to-all
./TransferBench nicrings      # NIC ring tests
```

### "Pod-aware (multi-rank, single MNNVL/XGMI pod)" → pod presets
```bash
./TransferBench podp2p        # within-pod P2P
./TransferBench poda2a        # within-pod all-to-all
```
For pod presets, set `TB_FORCE_SINGLE_POD=1` if AMD-SMI / NVML pod detection is unavailable.

See `references/presets.md` for the full list.

## Multi-rank execution

TransferBench runs multi-node either via MPI (if compiled with `MPI_PATH` set) or via plain TCP sockets.

### MPI approach
```bash
mpirun -np 4 -host node0,node1,node2,node3 ./TransferBench a2a
```

### Socket approach (no MPI)
On rank 0, set only `TB_NUM_RANKS=N` to print the master address; copy that to other ranks.
```bash
# node0
TB_NUM_RANKS=4 ./TransferBench a2a
# node1, node2, node3 (use the address node0 prints)
TB_NUM_RANKS=4 TB_RANK=1 TB_MASTER_ADDR=<addr> ./TransferBench a2a
TB_NUM_RANKS=4 TB_RANK=2 TB_MASTER_ADDR=<addr> ./TransferBench a2a
TB_NUM_RANKS=4 TB_RANK=3 TB_MASTER_ADDR=<addr> ./TransferBench a2a
```

Recommend **one process per node**. See `examples/multi-node.sh` for a full mpirun launcher script with environment-variable propagation (`-x VAR`).

## Tuning behavior with environment variables

Most tuning is environment-variable driven. The most useful ones:

| Variable | Purpose |
|---|---|
| `NUM_ITERATIONS` | Iterations per test (negative = run for that many seconds in timed mode) |
| `NUM_WARMUPS` | Warmup iterations (default 3) |
| `USE_SINGLE_STREAM` | When 0, each Transfer gets its own stream (may serialize on HW queue limits) |
| `GPU_MAX_HW_QUEUES` | Raise HW-queue cap when `USE_SINGLE_STREAM=0` |
| `OUTPUT_TO_CSV` | Emit CSV output |
| `SHOW_ITERATIONS` | Print per-iteration timings |
| `SHOW_PERCENTILES` | e.g. `50,75,90,99` for tail-latency percentiles |
| `ALWAYS_VALIDATE` | Validate destination data after each iteration |
| `FILL_PATTERN` | Custom source fill pattern |
| `TB_DUMP_CFG_FILE` | Dump executed Transfers (e.g. from a preset) to a config file |
| `TB_FORCE_SINGLE_POD` | Force single-pod membership when AMD-SMI/NVML unavailable |
| `TB_NIC_FILTER` | Restrict which NICs are used |

Run `./TransferBench envvars` for the authoritative list. See `references/env-vars.md` for grouped/annotated reference.

## Reading results

Default output prints one row per Test with bandwidth (GB/s) and time (ms). With `OUTPUT_TO_CSV=1`, results are CSV-formatted. With `SHOW_PERCENTILES=...`, percentile tail-latency columns are appended. Lines beginning with `##` in a config file are echoed back into the output for annotation.

## When you're stuck

1. Run `./TransferBench help` for config-file syntax with examples.
2. Run `./TransferBench presets` for the live list of presets.
3. Run `./TransferBench envvars` for all environment variables.
4. Run `./TransferBench dryrun "..."` to validate a transfer expression without executing.
5. If a preset fails on multi-node, first try `TB_FORCE_SINGLE_POD=1` and confirm rank count matches host count.

## References

- `references/config-format.md` — full grammar for config files (memory codes, executor codes, basic vs. advanced syntax)
- `references/presets.md` — every preset, when to use each, and its key env-var knobs
- `references/env-vars.md` — grouped environment-variable reference
- `examples/basic-p2p.cfg` — minimal GPU↔GPU copy
- `examples/advanced-mixed.cfg` — mixed CPU/GPU/DMA transfers with explicit byte counts
- `examples/multi-node.sh` — MPI launcher template with logging and env-var propagation

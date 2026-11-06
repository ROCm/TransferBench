# TransferBench environment variables

This is a curated guide to the most-used variables. For the authoritative complete list as compiled into your binary, run:

```bash
./TransferBench envvars
```

## Iteration / timing

| Variable | Default | Effect |
|---|---|---|
| `NUM_ITERATIONS` | `10` | Iterations per test. **Negative** = timed mode (run for that many seconds). |
| `NUM_SUBITERATIONS` | `1` | Sub-iterations per outer iteration. |
| `NUM_WARMUPS` | `3` | Warmup iterations before timing. |
| `USE_HIP_EVENTS` | `1` | Use HIP/CUDA events for timing (vs. host clock). |
| `SAMPLING_FACTOR` | `1` | Subsampling factor for sweep presets. |

## Output / reporting

| Variable | Default | Effect |
|---|---|---|
| `OUTPUT_TO_CSV` | `0` | Emit CSV output instead of human-readable tables. |
| `SHOW_BORDERS` | `1` | Draw table borders. |
| `SHOW_ITERATIONS` | `0` | Print per-iteration timings. |
| `SHOW_PERCENTILES` | unset | Comma list, e.g. `50,75,90,99`, to add percentile columns. |
| `HIDE_ENV` | `0` | Suppress the env-var summary printed at startup. |
| `OUTPUT_FORMAT` | preset-specific | `0` = list, `1` = full matrix (used by `podp2p`). |

## Validation / data

| Variable | Default | Effect |
|---|---|---|
| `ALWAYS_VALIDATE` | `0` | Validate destination data after every iteration (slow but safe). |
| `VALIDATE_DIRECT` | `0` | Validation reads memory directly without copy-back. |
| `VALIDATE_SOURCE` | `0` | Validate that source data is unchanged. |
| `FILL_PATTERN` | unset | Custom hex pattern for source initialization. |
| `FILL_COMPRESS` | unset | Use compressible source data. |
| `BYTE_OFFSET` | `0` | Offset (bytes) into allocated buffers. |
| `BLOCK_BYTES` | `256` | Block granularity for transfers. |

## GPU / GFX kernel knobs

| Variable | Default | Effect |
|---|---|---|
| `USE_SINGLE_STREAM` | `1` | When `0`, each Transfer gets its own stream (may serialize on HW-queue cap). |
| `GPU_MAX_HW_QUEUES` | `4` | Hardware-queue cap when `USE_SINGLE_STREAM=0`. Raise for more parallelism. |
| `GFX_KERNEL` | `0` | Choose copy kernel variant. |
| `GFX_BLOCK_ORDER` | `0` | Threadblock dispatch order. |
| `GFX_BLOCK_SIZE` | `256` | Threads per block. |
| `GFX_SE_TYPE` | `0` | SubExecutor mapping strategy. |
| `GFX_SINGLE_TEAM` | `0` | Combine work into a single team. |
| `GFX_TEMPORAL` | `0` | Temporal hints for cache. |
| `GFX_UNROLL` | preset-specific | Loop-unroll factor in the kernel. |
| `GFX_WAVE_ORDER` | `0` | Wavefront iteration order. |
| `GFX_WORD_SIZE` | `4` | Per-thread element size in bytes. |
| `CU_MASK` | unset | Bitmask restricting which CUs are used. |
| `XCC_PREF_TABLE` | unset | XCC preference table for MI300-class GPUs. |
| `USE_HSA_DMA` | `0` | Use HSA DMA path on AMD platforms. |

## Variable SubExecutor sweeps

| Variable | Default | Effect |
|---|---|---|
| `MIN_VAR_SUBEXEC` | `1` | Min SE count when sweeping. |
| `MAX_VAR_SUBEXEC` | `0` | Max SE count when sweeping (`0` = unlimited). |

## NIC / RDMA

| Variable | Default | Effect |
|---|---|---|
| `IB_GID_INDEX` | `-1` | InfiniBand GID index (`-1` = auto). |
| `IB_PORT_NUMBER` | `1` | IB port number. |
| `ROCE_VERSION` | `2` | RoCE version (1 or 2). |
| `IP_ADDRESS_FAMILY` | `4` | `4` = IPv4, `6` = IPv6. |
| `NIC_CHUNK_BYTES` | `1073741824` | Chunk size (bytes) for NIC transfers. |
| `NIC_CQ_POLL_BATCH` | `4` | Completion-queue poll batch size. |
| `NIC_RELAX_ORDER` | `1` | Relaxed ordering on the NIC. |
| `TB_NIC_FILTER` | unset | Restrict which NICs participate. |

## Multi-rank / pod

| Variable | Default | Effect |
|---|---|---|
| `TB_RANK` | unset | Rank ID (0-based) for socket-mode. |
| `TB_NUM_RANKS` | unset | Total ranks for socket-mode. |
| `TB_MASTER_ADDR` | unset | Master address printed by rank 0. |
| `TB_FORCE_SINGLE_POD` | `0` | Force single-pod membership when AMD-SMI/NVML unavailable. |

## Debug / capture

| Variable | Default | Effect |
|---|---|---|
| `TB_DUMP_CFG_FILE` | unset | Dump executed Transfers (e.g. from a preset) to this config file. |
| `TB_DUMP_LINES` | unset | Limit number of dumped lines. |
| `TB_WALLCLOCK_RATE` | unset | Override wallclock rate when GPU returns 0 (debug). |
| `USE_INTERACTIVE` | `0` | Pause for input between tests. |

## Pod-preset specific

Used by `podp2p` and `poda2a`:

| Variable | Used by | Values |
|---|---|---|
| `P2P_MODE` | `podp2p` | `0` both, `1` uni only, `2` bi only |
| `A2A_MODE` | `poda2a` | `0` copy, `1` read-only, `2` write-only, `2:3` custom |
| `A2A_LOCAL` | `poda2a` | `0` exclude same-rank, `1` include |
| `PARALLEL_LVL` | `podp2p` | `0` serial node pairs, `1` parallel |
| `STRIDE` | `poda2a` | Interleave stride |
| `GROUP_SIZE` | `poda2a` | GPUs per group (must divide rank count) |
| `USE_GPU_DMA` | `podp2p` | `0` GFX exec, `1` DMA exec |
| `USE_DMA_EXEC` | `poda2a` | `0` GFX exec, `1` DMA exec (DMA only allowed for `A2A_MODE=0`) |
| `USE_REMOTE_READ` | both | `0` write to remote, `1` read from remote |
| `NUM_GPU_DEVICES` | both | Limit GPUs per rank |

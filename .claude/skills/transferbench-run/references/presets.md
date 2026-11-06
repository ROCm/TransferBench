# TransferBench presets

Presets are built-in configurations that handle topology discovery and produce well-formatted bandwidth tables. Run any of them as the first argument:

```bash
./TransferBench <preset> [N]
```

For the live list on a given build, run `./TransferBench presets`.

## Single-node bandwidth presets

| Preset | Purpose |
|---|---|
| `a2a` | All-to-all parallel transfers between every pair of GPUs. |
| `a2asweep` | GFX-based a2a swept across CU counts and unroll factors (`MEM_TYPE`, `NUM_SUB_EXECS`). |
| `bmasweep` | Compares DMA vs. Batched-DMA for one-to-many copies (HIP 7.1 / CUDA 12.8+). |
| `gfxsweep` | Sweeps GFX kernel options for one Transfer. |
| `hbm` | Local HBM read bandwidth on each GPU. |
| `healthcheck` | Quick correctness/perf health check (AMD MI300 series only). |
| `one2all` | All subsets of parallel transfers from one GPU to all others. |
| `p2p` | Peer-to-peer device-memory matrix between every GPU pair. |
| `pcopy` | Parallel copies from a single GPU to other GPUs. |
| `rsweep` | Random sweep through Transfer combinations. |
| `rwrite` | Parallel remote writes from a single GPU to others. |
| `scaling` | Scaling test: one GPU → all others, varying SEs, mem types (`CPU_MEM_TYPE`, `GPU_MEM_TYPE`). |
| `schmoo` | Local/remote read/write/copy scaling between two GPUs. |
| `smoketest` | Quick DMA/GFX correctness sweep. |
| `sweep` | Ordered sweep through Transfer combinations. |
| `wallclock` | Compares wallclock counters across XCCs within one GPU. |

## Multi-node / NIC presets

Require an MPI launcher or socket-mode environment variables (`TB_NUM_RANKS`, `TB_RANK`, `TB_MASTER_ADDR`).

| Preset | Purpose |
|---|---|
| `a2a_n` | All-to-all over RDMA via each GPU's nearest NIC. |
| `nica2a` | NIC all-to-all using each NIC's closest GPU/CPU endpoint. |
| `nicp2p` | NIC peer-to-peer matrix across all NICs in the world. |
| `nicrings` | Ring transfers across identical NIC indices on each rank. |
| `rings` | Ring transfers within subgroups of pod ranks (also runs single-node). |

## Pod-aware presets (multi-rank, single MNNVL/XGMI pod)

Detect pod membership via AMD-SMI (HIP) or NVML (CUDA). If unavailable, set `TB_FORCE_SINGLE_POD=1`.

| Preset | Purpose | Key knobs |
|---|---|---|
| `podp2p` | P2P across ranks within a pod. | `P2P_MODE`, `PARALLEL_LVL`, `USE_GPU_DMA`, `USE_REMOTE_READ`, `OUTPUT_FORMAT`, `NUM_GPU_DEVICES` |
| `poda2a` | All-to-all across ranks within a pod. | `A2A_MODE`, `A2A_LOCAL`, `STRIDE`, `GROUP_SIZE`, `USE_DMA_EXEC`, `USE_REMOTE_READ`, `NUM_GPU_DEVICES` |

`P2P_MODE`: `0` = both directions, `1` = unidirectional only, `2` = bidirectional only.
`A2A_MODE`: `0` = copy, `1` = read-only, `2` = write-only, `2:3` = custom ratio.
`PARALLEL_LVL`: `0` = serial node pairs, `1` = node pairs in parallel.

## Info-only presets

These print and exit; they don't run transfers.

| Preset | Purpose |
|---|---|
| `help` | Config-file syntax with examples. |
| `presets` | Lists all available presets. |
| `envvars` | Lists every environment variable and its effect. |

## Choosing a preset

- "Quick GPU↔GPU bandwidth" → `p2p`.
- "All-pairs simultaneous" → `a2a`.
- "How does perf scale with CUs?" → `scaling` or `gfxsweep`.
- "Across two nodes via RDMA" → `nicp2p` for matrix, `nica2a` for collective-style.
- "Within an MNNVL pod" → `podp2p` / `poda2a`.
- "I want to capture what a preset does and tweak it" → run with `TB_DUMP_CFG_FILE=out.cfg`, then edit `out.cfg`.

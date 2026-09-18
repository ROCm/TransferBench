# TransferBench config-file format

A config file is plain text. Each non-comment line defines a **Test** — a set of **Transfers** that run in parallel.

- Lines starting with `#` are ignored.
- Lines starting with `##` are echoed verbatim into output (use them as labels for results).
- Round brackets `()` and arrows `->` are decorative and ignored by the parser.

## Two ways to specify a Test

### Basic: same SE count for every Transfer

```
<numTransfers> <SEs> (srcMem1 -> Executor1 -> dstMem1) ... (srcMemN -> ExecutorN -> dstMemN)
```

- `numTransfers` — positive integer, count of parallel Transfers on this line
- `SEs` — number of SubExecutors used by every Transfer on the line
- Each triplet describes one Transfer

Examples:
```
1 4  (G0->G0->G1)                  # 4 CUs on GPU0 copy GPU0 -> GPU1
1 4  (C1->G2->G0)                  # 4 CUs on GPU2 copy CPU1 -> GPU0
2 4  G0->G0->G1  G1->G1->G0        # bidirectional, 4 SEs each
```

### Advanced: per-Transfer SE count and byte count

```
-<numTransfers> (srcMem1 -> Exec1 -> dstMem1 SEs1 Bytes1) ... (srcMemN -> ExecN -> dstMemN SEsN BytesN)
```

- `numTransfers` is **negated** to switch into advanced mode.
- `Bytes` is per-Transfer; `0` means "use the command-line `N`". May be suffixed with `K`, `M`, or `G`. Must be a multiple of 4.

Example:
```
-2 (G0->G0->G1 4 1M) (G1->G1->G0 8 2M)
# Copies 1MiB GPU0->GPU1 with 4 CUs, in parallel with 2MiB GPU1->GPU0 with 8 CUs
```

## Executor codes

`Executor` is one character + a 0-based device index (NICs use a two-part index).

| Code | Executor | Index range | Notes |
|---|---|---|---|
| `C` | CPU | NUMA node | SubExecutor = CPU thread |
| `G` | GPU kernel | GPU device | SubExecutor = threadblock / CU |
| `D` | DMA | GPU device | Single SRC, ≥1 DST |
| `B` | Batched-DMA | GPU device | `hipMemcpyBatchAsync`-based; HIP 7.1 / CUDA 12.8+ |
| `I#.#` | NIC executor | NIC index `.` QP index | e.g. `I0.2` |
| `N#.#` | Nearest-NIC executor | GPU index `.` QP index | Picks each end's closest NIC |

## Memory-location codes

A memory location is `<code><index>`. Multiple locations can be concatenated for multi-SRC / multi-DST (e.g. `G0G1` is "both GPU0 and GPU1 memory").

| Code | Memory type | Indexed by |
|---|---|---|
| `C` | Pinned host (coarse-grained) | NUMA node |
| `P` | Pinned host (closest-GPU NUMA) | GPU index |
| `B` | Coherent pinned host | NUMA node |
| `D` | Non-coherent pinned host | NUMA node |
| `K` | Uncached pinned host | NUMA node |
| `H` | Unpinned host | NUMA node |
| `G` | Global device memory | GPU |
| `F` | Fine-grain device memory | GPU |
| `U` | Uncached device memory | GPU |
| `N` | Null (no read or no write) | ignored |

`N` on the SRC side gives a "memset-like" write benchmark; `N` on the DST side gives a "read-only" benchmark.

## Idiomatic patterns

```
## Memset by GPU0 onto its own memory
1 32 (N0->G0->G0)

## Read-only by CPU0 NUMA node
1 4 (C0->C0->N0)

## Broadcast from GPU0 to GPU0 and GPU1 simultaneously
1 16 (G0->G0->G0G1)

## Fan-in / sum: read from GPU0 and GPU1, write the sum to GPU2
1 16 (G0G1->G2->G2)

## NIC RDMA between two GPUs across NIC0 and NIC2 with 2 QPs
1 2 (F0->I0.2->F1)

## Nearest-NIC RDMA: each side picks its closest NIC
1 1 (F0->N0.1->F1)
```

## Validating a config without running it

```
./TransferBench dryrun "1 4 (G0->G0->G1)"
./TransferBench dryrun my.cfg
```
`dryrun` parses, expands wildcards, and prints what *would* execute — useful when iterating on complex configs.

## Capturing what a preset actually executes

```
TB_DUMP_CFG_FILE=p2p_dump.cfg ./TransferBench p2p
```
Writes the resolved Transfers from the preset to a config file you can edit and rerun.

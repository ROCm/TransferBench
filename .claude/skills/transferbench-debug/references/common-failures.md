# TransferBench common failures — symptoms, causes, fixes

Catalog of the most common "it doesn't work" cases, organized by symptom. Each section has: **symptom signal**, **likely cause(s)**, and **concrete fix or next probe**.

---

## §pod — pod-aware preset hangs / errors

### Symptom
- `podp2p` / `poda2a` / `rings` errors immediately with a pod-detection message.
- Or hangs in startup before any transfer table prints.

### Cause
- AMD-SMI (HIP build) or NVML (CUDA build) is unavailable, blocked, or returns inconsistent pod membership across ranks.
- The build has `POD_COMM_ENABLED` but the host lacks fabricmanager / `nvidia-fabricmanager.service`.

### Fix
1. `TB_FORCE_SINGLE_POD=1` — fastest workaround, treats every rank as one pod.
2. Confirm fabricmanager is running (CUDA): `systemctl status nvidia-fabricmanager`.
3. Confirm NVML works: a one-liner like `nvidia-smi -q | head` should succeed on every rank.
4. If you actually want per-pod awareness, ensure all ranks see the same pod IDs by running a probe script (each rank prints its detected pod ID).

---

## §rdma — NIC / RDMA preset hangs or errors

### Symptom
- Hang during NIC bring-up (no transfer table) on `nicp2p`, `nica2a`, `nicrings`, `a2a_n`.
- Or "QP create failed" / "RDMA connect failed" messages.

### Cause
- Wrong `IB_GID_INDEX` — depends on the host's IB / RoCE configuration.
- `IB_PORT_NUMBER` doesn't match the active port on the chosen NIC.
- More NICs detected than usable; some are unconfigured.
- RoCE version mismatch.

### Fix
1. Find a working GID: `show_gids` or `ibv_devinfo -v` on each host.
2. Set `IB_GID_INDEX=<index>`, `IB_PORT_NUMBER=<port>` to known good values.
3. Restrict to a single NIC with `TB_NIC_FILTER=<nic_name>` to localize the bad one.
4. RoCE: try `ROCE_VERSION=2` (most common) or `ROCE_VERSION=1`.
5. Confirm both ends agree on `IP_ADDRESS_FAMILY` (4 vs 6).
6. If using OpenMPI: `--mca pml ucx --mca btl ^vader,openib` is the canonical setting in this repo.

---

## §validation — `ALWAYS_VALIDATE=1` reports mismatch

### Symptom
- Run completes but reports "Validation failed" / mismatch between expected and actual destination contents.

### Cause (in order of likelihood)
1. Wrong memory-location code (e.g. fine-grain `F` requested on a GPU that doesn't support it → memory backed by global instead, kernel writes go to a different place than reads).
2. Wrong CU mask — kernel uses a CU group that doesn't have the right cache visibility.
3. Multi-Transfer test where two Transfers race on the same destination address.
4. Actual hardware issue (least likely).

### Fix
1. `VALIDATE_DIRECT=1` — read destination directly without copy-back; isolates copy-back-path bugs.
2. `VALIDATE_SOURCE=1` — confirm source data was not overwritten by the kernel; catches `src == dst` issues.
3. `NUM_ITERATIONS=1 NUM_WARMUPS=0 ALWAYS_VALIDATE=1` — confirms it's not a state-leak between iterations.
4. Drop to `cmdline` with **one** Transfer to rule out multi-Transfer races.
5. If still failing: `FILL_PATTERN=0xDEADBEEF` (or any custom pattern) — makes the corruption signature easy to spot in the diff.

---

## §perf — bandwidth far below expected, or wildly variable

### Symptom
- Reported BW is a fraction (e.g. 1/4, 1/8) of the link's theoretical max.
- Or BW jumps 2× between iterations without obvious reason.

### Cause
- Stream-per-Transfer hits HW-queue limit and serializes (`USE_SINGLE_STREAM=0` + low `GPU_MAX_HW_QUEUES`).
- GFX kernel parameters mis-tuned for the size (`GFX_UNROLL`, `GFX_BLOCK_SIZE`, `GFX_WORD_SIZE`).
- Not enough warmup — first few iterations include allocation, paging, clock ramp.
- Wrong executor for the workload: GFX kernel for tiny payload (use DMA), DMA for one-to-many (use Batched-DMA).
- NUMA / pinned-memory mismatch (e.g. CPU-side memory on the wrong NUMA for the chosen GPU).

### Fix
1. `NUM_WARMUPS=10 NUM_ITERATIONS=20 SHOW_ITERATIONS=1` — see whether iter 0–2 are slow and the rest converge.
2. `SHOW_PERCENTILES=50,75,90,99` — exposes outlier iterations.
3. Try the alternate executor on the same memory pair: GFX (`G`) ↔ DMA (`D`) ↔ Batched-DMA (`B`).
4. `USE_SINGLE_STREAM=0 GPU_MAX_HW_QUEUES=8` for many parallel Transfers.
5. Sweep with a preset (`gfxsweep`, `a2asweep`) to find the right kernel options before hand-tuning.

---

## §crash — crash / segfault

### Symptom
- Process exits with SIGSEGV / "memory access fault" / "invalid memory access."

### Cause
- Memory code unsupported by HW (e.g. `F` on a GPU without fine-grain memory; `U` with no uncached path).
- DMA Transfer with multiple SRCs (DMA requires exactly one SRC).
- NIC executor with mismatched index syntax (`I0` instead of `I0.0`).
- Buffer alignment: byte count not a multiple of 4 (parser usually catches this, but custom builds may slip).

### Fix
1. `dryrun "<expression>"` first — most parser-level bugs surface here.
2. Read the banner from `./TransferBench` with no args — confirms which memory types and executors are compiled in for this build.
3. For DMA crashes: confirm exactly one SRC per Transfer.
4. Build with `-g -O0` and run under `cuda-gdb` (NVIDIA) or `rocgdb` (AMD); the stack at the fault tells you which Executor's path failed.

---

## §unsupported — "executor missing" / "feature not compiled in"

### Symptom
- "Unsupported executor" or similar, even though the code seems to allow it.

### Cause
- This build was compiled with one of the `DISABLE_*` Makefile flags (`DISABLE_NIC_EXEC=1`, `DISABLE_POD_COMM=1`, `DISABLE_AMD_SMI=1`, etc.).
- Or `MPI_PATH` was not set, so multi-rank paths were stubbed out.

### Fix
1. Run `./TransferBench` with no arguments. Its banner lists which executors and features are compiled in for this exact binary.
2. If the feature is genuinely missing, rebuild without the corresponding `DISABLE_*` flag. (See the build-side skill — out of scope here.)

---

## §parser — config-file parser rejects a line

### Symptom
- "Failed to parse" / "Invalid config line" / silently runs the wrong thing.

### Cause
- Confused basic vs. advanced syntax (`numTransfers` positive vs. negative).
- Whitespace inside an executor or memory token.
- Quoting issues on the shell side when using `cmdline` (e.g. `G*` getting glob-expanded).

### Fix
1. **Always quote** `cmdline` arguments: `./TransferBench cmdline "1 4 (G0->G0->G1)"`.
2. `dryrun` first to see the parse result without execution.
3. For complex configs, use `##` echo lines liberally — they show in the output and help correlate result rows to test definitions.

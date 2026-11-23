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

## §storage — storage Transfer (`S` with hipFile `F`) or `nvme` preset fails

### Symptom
- "No storage paths are defined on rank N" or the `nvme` preset prints `TB_STORAGE_PATHS` instructions and exits.
- "Storage memory (S) can only be accessed by the hipFile (F) executor".
- "hipFile executor is requested but is not available".
- "hipFile read/write ... failed (...): ... hipFile's fastpath may not be usable here".
- "Storage N (...) is a block device. Writing to it destroys its contents and requires TB_STORAGE_ALLOW_BLOCK_WRITE=1".
- "Storage N (...) is an existing file that holds data. Writing to it destroys its contents and requires TB_STORAGE_ALLOW_OVERWRITE=1", or the `nvme` preset reports `UNVALIDATED` on a second run.
- "numBytes (...) and [data.byteOffset] (...) are not both multiples of 4096.  hipFile will use its compatibility path for unaligned I/O" (warning).
- Suspiciously high bandwidth with I/O path `compat`.

### Cause
- `TB_STORAGE_PATHS` is unset, or the storage index is beyond the list (`S1` with one path).
- A combination outside what storage Transfers allow: exactly one SRC and one DST, one of them `S`, the other `G` on the executor's GPU, same rank.
- The binary was built without hipFile support: `hipfile.h` from the hipFile development package (`amdrocm-hipfile-dev<version>`) was not found at build time. The version line then has no `hipFile`. Rebuilding is the only fix.
- "error while loading shared libraries: libhipfile.so.0" at startup: the binary was built with hipFile support and the hipFile runtime package (`amdrocm-hipfile<version>`) is missing on this host.
- A storage path under `/dev/` that does not exist is rejected ("Paths under /dev/ ... are not created"): almost always a mistyped device name.
- hipFile attempted direct I/O where it can't work (no AIS support in the GPU driver, no PCIe P2PDMA, storage not local NVMe, WSL2 with an older hipFile). It does not always fall back by itself.
- A write targets a raw block device without the opt-in.
- A storage file already holds data (a user's file, or the file an earlier run left behind) and `TB_STORAGE_ALLOW_OVERWRITE=1` is not set.
- Compatibility-path I/O goes through host memory and the page cache; it is not a direct storage ↔ GPU number.

### Fix
1. `./TransferBench` with no args — the storage section lists the paths and whether hipFile is available on this host.
2. `./TransferBench dryrun 1M "<expression>"` to see the parsed Transfer; fix the SRC/EXE/DST combination.
3. `HIPFILE_FORCE_COMPAT_MODE=true` to confirm the Transfer itself is sound; then `ais-check` (ships with hipFile) to see why direct I/O is unavailable.
4. `TB_VERBOSE=1` and look for `storage I/O path:` — `HIPFILE_ALLOW_COMPAT_MODE=false` makes hipFile fail instead of silently using `compat`.
5. For direct I/O: use a byte count that is a multiple of 4096, `BYTE_OFFSET=0`, and a file on xfs / ext4 on a local NVMe drive (or the raw block device).
6. Block devices: **do not** set `TB_STORAGE_ALLOW_BLOCK_WRITE=1` unless the user confirms the device holds nothing of value. Read-only Transfers (`S0->F0->G0`) work without it, but skip validation.
   Existing files: delete TransferBench's own scratch file or set `TB_STORAGE_ALLOW_OVERWRITE=1` to reuse it; **do not** set it for a file that may hold the user's data.
7. `TB_HIPFILE_SKIP_BUF_REGISTER=1` if it hangs or fails while registering the GPU buffer.

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

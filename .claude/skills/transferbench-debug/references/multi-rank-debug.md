# Multi-rank debugging (MPI / socket / RDMA)

Multi-rank TransferBench is the most failure-prone configuration. This guide is organized by the **layer where the failure happens**: launcher → bootstrap → NIC bring-up → transfer execution.

## Pre-flight: things to check before suspecting TransferBench

These rule out 90% of multi-rank "bugs":

```bash
# 1. Same binary on every node
for h in node0 node1; do ssh $h md5sum /home/timhu102/tBench/TransferBenchCuda; done

# 2. NICs functional on every node
for h in node0 node1; do ssh $h "ibstatus | head -20"; done

# 3. Hosts can reach each other on the master port (socket mode)
ssh node1 "nc -zv node0 <port>"

# 4. Same env vars actually propagating to all ranks
mpirun --tag-output -np 2 -host node0,node1 -x SOME_VAR env | grep SOME_VAR
```

## Launcher layer: hangs/errors before any TransferBench output

### Hostfile / `-host` problems

- **Hostnames not resolving:** `mpirun` will hang silently. Try IP addresses instead.
- **More slots requested than declared:** `host node0:1,node1:1 -np 4` is a misconfiguration; use one slot per node.
- **SSH not passwordless:** `mpirun` quietly fails when an `ssh` it spawns prompts for a password. Test with `ssh node1 hostname` first.

### MPI transport problems

The canonical incantation in this repo is:

```bash
mpirun --mca pml ucx --mca btl '^vader,openib' ...
```

If you see "no transport available" / "BTL openib unavailable":
- Confirm UCX is built into your OpenMPI (`ompi_info | grep ucx`).
- The `^vader,openib` is a **negative** filter — it excludes those BTLs to force PML/UCX. Don't drop it without good reason.

### Env-var propagation

OpenMPI does **not** forward your shell env to remote ranks unless you ask it to:

```bash
mpirun -x VAR1 -x VAR2=value -np ... ./TransferBenchCuda ...
```

- `-x VAR` forwards the **current** shell value of `VAR`.
- `-x VAR=value` sets it explicitly.
- Without `-x`, remote ranks see only the env they inherit from the SSH session, which usually does NOT include your interactive shell exports.

This is the most common source of "works on rank 0, fails on rank ≥1" bugs. The `examples/multi-node.sh` template in `transferbench-run` builds the `-x` flags from a list — model after it.

## Bootstrap layer: process started, but no transfer output

### MPI bootstrap

If `mpirun` reports all ranks started but TransferBench prints nothing, the most common cause is the rank-0 → rank-N handshake taking longer than expected because of a stuck NIC or NUMA probe.

- Run `mpirun --tag-output ...` so each rank's output is prefixed `[rank,N]`.
- If rank 0 reaches the banner and others don't, those ranks are stuck in their initialization (often AMD-SMI or NVML).

### Socket-mode bootstrap

```bash
# On rank 0, no TB_RANK set → rank 0 prints master address
TB_NUM_RANKS=4 ./TransferBenchCuda <preset>

# On rank N (N>0), set TB_RANK and TB_MASTER_ADDR
TB_NUM_RANKS=4 TB_RANK=1 TB_MASTER_ADDR=<addr> ./TransferBenchCuda <preset>
```

Common socket-mode issues:
- **`TB_NUM_RANKS` differs across ranks** → silent hang.
- **`TB_MASTER_ADDR` not reachable** (firewall, wrong interface). Test with `nc -zv` first.
- **One rank exited early** → the others wait forever.

## NIC bring-up layer: `nicp2p` / `nica2a` / pod presets hang here

### GID index

The single most common cause of NIC hangs. Find a working GID:

```bash
for d in $(ibv_devices | tail -n +3 | awk '{print $1}'); do
  echo "=== $d ==="
  show_gids $d 2>/dev/null || ibv_devinfo -d $d -v 2>/dev/null | grep -E 'GID|state'
done
```

Pick a GID with a populated address (not `0000:0000:...`) and an active port:

```bash
IB_GID_INDEX=<index> IB_PORT_NUMBER=<port> ./TransferBenchCuda nicp2p
```

### NIC filtering

If you have eight NICs but only four are configured, list the bad ones:

```bash
TB_NIC_FILTER=mlx5_0,mlx5_1,mlx5_2,mlx5_3 ./TransferBenchCuda nicp2p
```

This restricts the world-view of NICs to a subset; useful for narrowing down a bad NIC.

### RoCE version mismatch

Symptom: connection establishment hangs forever.
Fix: `ROCE_VERSION=2` is the modern default; some legacy clusters require `ROCE_VERSION=1`. Both ends must agree.

### POD / MNNVL detection

For `podp2p` / `poda2a`:

- AMD-SMI (HIP) or NVML (CUDA) must be functional, AND fabricmanager must be running on NVIDIA.
- Quick workaround if pod detection is broken: `TB_FORCE_SINGLE_POD=1`.

## Output layer: results are garbled or look wrong

### Interleaved output

Without explicit tagging, MPI lets all ranks write to the same stdout, which interleaves bytes:

```bash
mpirun --tag-output ...                   # each line prefixed with [rank,N]
mpirun --output-filename out -np ...      # per-rank log files
mpirun --merge-stderr-to-stdout ...       # one stream
```

### Apparent zeros / NaNs in the bandwidth table

- Rank-N reported a hardware error and failed silently → check rank-N's stderr (use `--output-filename`).
- Or the `Test` was malformed and one rank parsed it differently → use `dryrun` and `TB_DUMP_CFG_FILE` to confirm both ranks are running the same Transfers.

## When the hang is genuinely TransferBench's fault

If the layers above are clean and TransferBench still hangs:

1. Build with `-g -O0` (or `-g -O1`).
2. Run a tiny config (one Transfer, one rank pair) under `cuda-gdb` / `rocgdb`.
3. When it hangs: `Ctrl-C`, then `bt`. The stuck thread's stack will name the function holding the lock or the queue it's polling.
4. Look for: NIC completion-queue polling without a timeout, a stream-event wait, or an MPI collective that's missing a peer.

If you're hitting this often, the build-side skill (separate skill) covers the recompile workflow.

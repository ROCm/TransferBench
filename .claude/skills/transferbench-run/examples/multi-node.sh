#!/usr/bin/env bash
# Template launcher for multi-node TransferBench runs via mpirun.
# Adjust HOSTS, NP, and the OpenMPI install path for your cluster.
#
# Usage:
#   ./multi-node.sh <preset_or_config> [N]
# Example:
#   ./multi-node.sh nicp2p
#   ./multi-node.sh podp2p
#   ./multi-node.sh my.cfg 256M

set -euo pipefail

BINARY="${BINARY:-./TransferBenchCuda}"   # or ./TransferBench for HIP
PRESET="${1:?usage: $0 <preset_or_config> [N]}"
SIZE="${2:-}"

# --- MPI environment (edit for your cluster) ----------------------------------
export PATH="${HOME}/rdma/ompi/install/bin:${PATH}"
export LD_LIBRARY_PATH="${HOME}/rdma/ompi/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPAL_PREFIX="${HOME}/rdma/ompi/install"

HOSTS="${HOSTS:-node0:1,node1:1}"          # one slot per node is recommended
NP="${NP:-2}"

# --- TransferBench tuning (edit as needed) ------------------------------------
# Forward each var with -x so MPI propagates it to every rank.
TB_ENV=(
  "NUM_ITERATIONS=20"
  "NUM_WARMUPS=3"
  "OUTPUT_TO_CSV=0"
  # "TB_FORCE_SINGLE_POD=1"      # uncomment if AMD-SMI / NVML pod detection fails
  # "USE_REMOTE_READ=1"
  # "TB_DUMP_CFG_FILE=run_dump.cfg"
)

x_flags=()
env_inline=()
for kv in "${TB_ENV[@]}"; do
  key="${kv%%=*}"
  x_flags+=("-x" "$key")
  env_inline+=("$kv")
done

# Export so mpirun can see them when forwarding with -x KEY
for kv in "${TB_ENV[@]}"; do export "$kv"; done

CMD=(mpirun
  --mca pml ucx
  --mca btl '^vader,openib'
  --host "$HOSTS"
  -np "$NP"
  "${x_flags[@]}"
  "$BINARY" "$PRESET"
)
[[ -n "$SIZE" ]] && CMD+=("$SIZE")

echo "# Launching: ${env_inline[*]} ${CMD[*]}"
"${CMD[@]}"

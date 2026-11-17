#!/usr/bin/env bash
# topology-probe.sh — print everything TransferBench can tell you about this host.
#
# Run as the FIRST step of any debugging session. Captures: detected GPUs,
# NUMA, NICs, compiled feature flags, env-var defaults, and (optionally) what
# a single preset actually emits.
#
# Usage:
#   ./topology-probe.sh                 # just probe
#   ./topology-probe.sh p2p             # probe + dump what `p2p` would run
#   ./topology-probe.sh p2p out/        # write output files into out/

set -euo pipefail

BINARY="${BINARY:-./TransferBench}"
[[ -x "./TransferBenchCuda" ]] && BINARY="${BINARY/TransferBench/TransferBenchCuda}"
[[ -x "$BINARY" ]] || { echo "ERROR: $BINARY not found or not executable"; exit 1; }

PRESET="${1:-}"
OUTDIR="${2:-.}"
mkdir -p "$OUTDIR"

echo "=== Binary banner (compiled features + detected hardware) ==="
"$BINARY" 2>&1 | tee "$OUTDIR/banner.txt" | head -60
echo

echo "=== Compiled-in presets ==="
"$BINARY" presets 2>&1 | tee "$OUTDIR/presets.txt"
echo

echo "=== Compiled-in environment variables ==="
"$BINARY" envvars 2>&1 | tee "$OUTDIR/envvars.txt" | head -40
echo "  (full list in $OUTDIR/envvars.txt)"
echo

echo "=== Config-file syntax help ==="
"$BINARY" help 2>&1 | tee "$OUTDIR/help.txt" | head -40
echo "  (full help in $OUTDIR/help.txt)"
echo

if [[ -n "$PRESET" ]]; then
  DUMP="$OUTDIR/${PRESET}_dump.cfg"
  echo "=== Dumping what '$PRESET' actually runs to $DUMP ==="
  TB_DUMP_CFG_FILE="$DUMP" TB_DUMP_LINES=100 "$BINARY" "$PRESET" >/dev/null 2>&1 || true
  if [[ -f "$DUMP" ]]; then
    echo "  First 30 lines:"
    head -30 "$DUMP" | sed 's/^/    /'
  else
    echo "  (TB_DUMP_CFG_FILE produced no output — preset may not support dump)"
  fi
fi

echo
echo "=== Quick parser sanity check ==="
"$BINARY" dryrun "1 4 (G0->G0->G1)" 2>&1 | head -10
echo

echo "Done. Files written to $OUTDIR/"

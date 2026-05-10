#!/bin/bash

# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#
# LaunchTransferBench - Multi-rank TransferBench Socket Execution Script
#
# This script simplifies the execution of socket-based multi-rank TransferBench
# by automatically setting up SSH connections to specified hosts and setting
# the appropriate environment variables.
#
# Usage:
#   ./LaunchTransferBench.sh <hosts> [env_vars...] [-- <transferbench_args>]
#
# Arguments:
#   hosts: Comma-separated list of hostnames/IPs to run on
#   env_vars: Optional environment variables (e.g., NUM_ITERATIONS=10 NUM_SUBITERATIONS=100)
#   transferbench_args: Arguments to pass to TransferBench (after --)
#
# Examples:
#   ./LaunchTransferBench.sh node1,node2,node3,node4 NUM_ITERATIONS=10 NUM_SUBITERATIONS=100 -- a2a
#   ./LaunchTransferBench.sh host1,host2 -- cmdline 1G "1 1 R0G0 R0D0 R1G0"
#   ./LaunchTransferBench.sh server1,server2,server3 TB_MASTER_PORT=30000 -- example.cfg
#

set -e

# Function to display usage information
show_usage() {
    cat << EOF
Usage: $0 <hosts> [env_vars...] [-- <transferbench_args>]

Arguments:
  hosts                Comma-separated list of hostnames/IPs to run on
  env_vars             Optional environment variables (KEY=VALUE format)
  transferbench_args   Arguments to pass to TransferBench (after --)

Environment Variables for TransferBench:
  NUM_ITERATIONS       Number of timed iterations to perform (default: 10)
  NUM_SUBITERATIONS    Number of subiterations to perform (default: 1)
  NUM_WARMUPS          Number of warmup iterations (default: 3)
  TB_MASTER_PORT       Port for rank 0 communication (default: 29500)
  ... and many others (see TransferBench documentation)

Examples:
  $0 node1,node2,node3,node4 NUM_ITERATIONS=10 NUM_SUBITERATIONS=100 -- a2a
  $0 host1,host2 -- cmdline 1G "1 1 R0G0 R0D0 R1G0"
  $0 server1,server2,server3 TB_MASTER_PORT=30000 -- example.cfg

Notes:
  - The first host in the list becomes rank 0 (master)
  - TransferBench must be built in the same directory as this script on all hosts
  - SSH access must be configured for all hosts
EOF
}


# Parse command line arguments
if [[ $# -lt 1 ]]; then
    show_usage
    exit 1
fi

# Parse hosts
hosts_input="$1"
shift

if [[ -z "$hosts_input" ]]; then
    echo "ERROR: No hosts specified"
    show_usage
    exit 1
fi

# Convert comma-separated hosts to array and trim whitespace
IFS=',' read -ra hosts_raw <<< "$hosts_input"
hosts=()
for host in "${hosts_raw[@]}"; do
    # Trim leading and trailing whitespace
    host=$(echo "$host" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    if [[ -z "$host" ]]; then
        echo "ERROR: Empty hostname found in host list"
        exit 1
    fi
    # Check for remaining whitespace in hostname
    if [[ "$host" =~ [[:space:]] ]]; then
        echo "ERROR: Hostname '$host' contains whitespace"
        exit 1
    fi
    hosts+=("$host")
done
num_ranks=${#hosts[@]}

if [[ $num_ranks -lt 2 ]]; then
    echo "ERROR: At least 2 hosts are required for multi-rank execution"
    echo "For single-node execution, run TransferBench directly without this script"
    exit 1
fi

echo "Hosts     : ${hosts[*]}"
echo "Ranks     : $num_ranks"

# Parse environment variables and TransferBench arguments
env_vars=()
tb_args=()
parsing_tb_args=false

while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
        parsing_tb_args=true
        shift
        continue
    fi

    if [[ $parsing_tb_args == true ]]; then
        tb_args+=("$1")
    elif [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then
        env_vars+=("$1")
    else
        echo "ERROR: Invalid environment variable format: $1"
        echo "Environment variables should be in KEY=VALUE format"
        exit 1
    fi
    shift
done

echo "EnvVars   : ${env_vars[*]:-none}"
if [[ ${#tb_args[@]} -eq 0 ]]; then
    echo "Args      : none (will show topology)"
else
    echo "Args      : ${tb_args[*]}"
fi

# Get the absolute directory where this script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
transferbench_path="$script_dir/TransferBench"

echo

# Build properly escaped environment variable string
env_string=""
for env_var in "${env_vars[@]}"; do
    # Validate that env_var is in KEY=VALUE format
    if [[ ! "$env_var" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then
        echo "ERROR: Invalid environment variable format: $env_var" >&2
        exit 1
    fi

    # Split into key and value
    key="${env_var%%=*}"
    value="${env_var#*=}"

    # Escape the value and rebuild the env var
    escaped_value=$(printf '%q' "$value")
    env_string="$env_string $key=$escaped_value"
done

# Cleanup function for interruption
cleanup() {
    echo
    echo "Interrupted! Cleaning up worker processes..." >&2

    # Kill remote TransferBench processes first (only our own processes)
    for ((i=0; i<${#worker_hosts[@]}; i++)); do
        host="${worker_hosts[$i]}"
        echo "Killing TransferBench on $host..." >&2
        # Kill only TransferBench processes owned by the current user
        ssh -q -o LogLevel=ERROR -o ConnectTimeout=2 "$host" "pkill -u \$(whoami) -f TransferBench 2>/dev/null || true" 2>/dev/null &
    done

    # Wait a moment for remote kills to take effect
    sleep 2

    # Now kill local SSH processes
    for pid in "${worker_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done

    # Give local processes a moment to terminate
    sleep 1

    # Force kill any remaining local SSH processes
    for pid in "${worker_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing local SSH PID $pid..." >&2
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    # Wait for all local processes to actually terminate
    for pid in "${worker_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    echo "Cleanup complete" >&2
    exit 130
}

# Set up signal handlers for Ctrl-C and termination
trap cleanup INT TERM

# Start worker ranks in the background
master_host="${hosts[0]}"
worker_pids=()
worker_hosts=()

# Build properly escaped arguments string
tb_args_escaped=""
for arg in "${tb_args[@]}"; do
    tb_args_escaped+=" $(printf '%q' "$arg")"
done

for ((rank=1; rank<num_ranks; rank++)); do
    worker_host="${hosts[$rank]}"
    worker_cmd="TB_NUM_RANKS=$num_ranks TB_RANK=$rank TB_SINGLE_LOG=1 TB_MASTER_ADDR=$master_host $env_string '$transferbench_path'$tb_args_escaped"
    ssh -q -o LogLevel=ERROR "$worker_host" "$worker_cmd" >/dev/null 2>&1 &
    worker_pids+=($!)
    worker_hosts+=("$worker_host")
done

# Start master rank (TransferBench will wait for all workers to connect)
master_cmd="TB_NUM_RANKS=$num_ranks TB_RANK=0 TB_SINGLE_LOG=1 $env_string '$transferbench_path'$tb_args_escaped"
if ! ssh -q -o LogLevel=ERROR "$master_host" "$master_cmd"; then
    echo "ERROR: Master rank failed on $master_host" >&2
    # Clean up worker processes before exiting
    cleanup
    exit 1
fi

# Check worker exit codes
any_worker_failed=false
for ((i=0; i<${#worker_pids[@]}; i++)); do
    if ! wait "${worker_pids[$i]}"; then
        rank=$((i+1))
        echo "ERROR: Worker rank $rank failed on ${worker_hosts[$i]}" >&2
        any_worker_failed=true
    fi
done

if [[ "$any_worker_failed" == "true" ]]; then
    exit 1
fi

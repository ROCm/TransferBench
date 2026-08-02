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
    echo "ERROR: No hosts specified" >&2
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
        echo "ERROR: Empty hostname found in host list" >&2
        exit 1
    fi
    # Check for remaining whitespace in hostname
    if [[ "$host" =~ [[:space:]] ]]; then
        echo "ERROR: Hostname '$host' contains whitespace" >&2
        exit 1
    fi
    hosts+=("$host")
done
num_ranks=${#hosts[@]}

if [[ $num_ranks -lt 2 ]]; then
    echo "ERROR: At least 2 hosts are required for multi-rank execution" >&2
    echo "For single-node execution, run TransferBench directly without this script" >&2
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
        echo "ERROR: Invalid environment variable format: $1" >&2
        echo "Environment variables should be in KEY=VALUE format" >&2
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
    # Split into key and value (validation already done during parsing)
    key="${env_var%%=*}"
    value="${env_var#*=}"

    # Escape the value and rebuild the env var
    escaped_value=$(printf '%q' "$value")
    env_string="$env_string $key=$escaped_value"
done

# Cleanup function for interruption
cleanup() {
    echo >&2
    echo "Interrupted! Cleaning up worker processes..." >&2

    # First kill local SSH processes to stop remote TransferBench
    if [[ ${#worker_pids[@]} -gt 0 ]]; then
        for pid in "${worker_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done

        # Give SSH processes a moment to terminate and clean up remote processes
        sleep 2

        # Force kill any remaining local SSH processes
        for pid in "${worker_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Force killing SSH PID $pid..." >&2
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done

        # Brief wait for cleanup, but don't hang
        for pid in "${worker_pids[@]}"; do
            # Wait with timeout - if process doesn't exit in 1 second, move on
            timeout 1 bash -c "wait $pid" 2>/dev/null || true
        done
    fi

    # Final cleanup: kill any remaining TransferBench processes on all hosts
    if [[ ${#worker_hosts[@]} -gt 0 ]]; then
        for host in "${worker_hosts[@]}"; do
            ssh -q -o LogLevel=ERROR -o ConnectTimeout=1 "$host" "pkill -u \$(whoami) -f TransferBench 2>/dev/null || true" 2>/dev/null || true &
        done
        # Don't wait for these - let them complete in background
    fi

    echo "Cleanup complete" >&2
    exit 130
}

# Set up signal handlers for Ctrl-C and termination
trap cleanup INT TERM

# SSH aliases defined in ssh_config are only understood by the SSH client, so workers
# calling getaddrinfo() on one would fail. Expand to the underlying host and prefer a
# literal IPv4, since the master address is resolved remotely and only over AF_INET.
resolve_master_addr() {
    local host="$1" addr ipv4
    addr=$(ssh -G "$host" 2>/dev/null | awk '/^hostname /{print $2; exit}')
    [[ -z "$addr" ]] && addr="$host"

    if [[ ! "$addr" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        ipv4=$(getent ahostsv4 "$addr" 2>/dev/null | awk 'NR==1{print $1}')
        if [[ -n "$ipv4" ]]; then
            addr="$ipv4"
        else
            echo "WARNING: Could not resolve '$host' to an IPv4 address" >&2
            echo "         Workers must be able to resolve '$addr' themselves" >&2
        fi
    fi
    printf '%s' "$addr"
}

# Start worker ranks in the background
master_host="${hosts[0]}"
master_addr=$(resolve_master_addr "$master_host")
if [[ "$master_addr" != "$master_host" ]]; then
    echo "Master    : $master_host ($master_addr)"
    echo
fi
worker_pids=()
worker_hosts=()

# Build properly escaped arguments string
tb_args_escaped=""
for arg in "${tb_args[@]}"; do
    tb_args_escaped+=" $(printf '%q' "$arg")"
done

for ((rank=1; rank<num_ranks; rank++)); do
    worker_host="${hosts[$rank]}"
    worker_cmd="TB_NUM_RANKS=$num_ranks TB_RANK=$rank TB_SINGLE_LOG=1 TB_MASTER_ADDR=$master_addr $env_string '$transferbench_path'$tb_args_escaped"
    ssh -q -o LogLevel=ERROR "$worker_host" "$worker_cmd" >/dev/null 2>&1 &
    worker_pids+=($!)
    worker_hosts+=("$worker_host")
done

# Start master rank (TransferBench will wait for all workers to connect)
master_cmd="TB_NUM_RANKS=$num_ranks TB_RANK=0 TB_SINGLE_LOG=1 TB_MASTER_ADDR=$master_addr $env_string '$transferbench_path'$tb_args_escaped"
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

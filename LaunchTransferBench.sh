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
#   ./LaunchTransferBench.sh [-h|--help] [-n|--dry-run] <hosts> [env_vars...] [-- <transferbench_args>]
#
# Arguments:
#   -h, --help: Show usage information and exit
#   -n, --dry-run: Show the expanded hosts and the commands that would be run, without running them
#   hosts: Comma-separated list of hostnames/IPs to run on
#   env_vars: Optional environment variables (e.g., NUM_ITERATIONS=10 NUM_SUBITERATIONS=100)
#   transferbench_args: Arguments to pass to TransferBench (after --)
#
# Examples:
#   ./LaunchTransferBench.sh 'node[1..4]' NUM_ITERATIONS=10 NUM_SUBITERATIONS=100 -- a2a
#   ./LaunchTransferBench.sh 'rack[01..02]-node[01..04]' -- a2a
#   ./LaunchTransferBench.sh host1,host2 -- cmdline 1G "1 1 R0G0 R0D0 R1G0"
#   ./LaunchTransferBench.sh server1,server2,server3 TB_MASTER_PORT=30000 -- example.cfg
#   ./LaunchTransferBench.sh --dry-run 'node[1..4]' -- a2a
#

set -e

# Function to display usage information
show_usage() {
    cat << EOF
Usage: $0 [-h|--help] [-n|--dry-run] <hosts> [env_vars...] [-- <transferbench_args>]

Arguments:
  -h, --help           Show this usage information and exit
  -n, --dry-run        Show the expanded hosts and the commands that would be
                       run on each host, without actually running them
  hosts                Comma-separated list of hostnames/IPs to run on.
                       A bracketed range wildcard expands in place:
                         [1..4]    -> 1, 2, 3, 4
                         [01..04]  -> 01, 02, 03, 04
                         [1,5..7]  -> 1, 5, 6, 7
                       Several ranges in one name are combined left to right.
  env_vars             Optional environment variables (KEY=VALUE format)
  transferbench_args   Arguments to pass to TransferBench (after --)

Environment Variables for TransferBench:
  NUM_ITERATIONS       Number of timed iterations to perform (default: 10)
  NUM_SUBITERATIONS    Number of subiterations to perform (default: 1)
  NUM_WARMUPS          Number of warmup iterations (default: 3)
  TB_MASTER_PORT       Port for rank 0 communication (default: 29500)
  ... and many others (see TransferBench documentation)

Examples:
  $0 'node[1..4]' NUM_ITERATIONS=10 NUM_SUBITERATIONS=100 -- a2a
  $0 'rack[01..02]-node[01..04]' -- a2a
  $0 host1,host2 -- cmdline 1G "1 1 R0G0 R0D0 R1G0"
  $0 server1,server2,server3 TB_MASTER_PORT=30000 -- example.cfg
  $0 --dry-run 'node[1..4]' -- a2a

Notes:
  - Quote a host list that contains [] so the shell does not expand it as a glob
  - The first host after expansion becomes rank 0 (master)
  - TransferBench must be built in the same directory as this script on all hosts
  - SSH access must be configured for all hosts
  - -h/--help and -n/--dry-run may appear before or after <hosts>, as long as they
    come before --; anything after -- is passed through to TransferBench unchanged
EOF
}

# Expand host-list range wildcards into the global hosts array.
# Commas inside [] belong to the range; commas outside [] separate hosts.
# A bound written with a leading zero sets the printed width of that range.
# Several ranges in one name combine left to right; the rightmost varies fastest.
HOST_EXPAND_LIMIT=4096

expand_range_spec() {
    local spec="$1"
    local context="$2"
    local token start_str end_str start end width i

    _range_values=()
    spec="${spec},"
    while [[ "$spec" == *,* ]]; do
        token="${spec%%,*}"
        spec="${spec#*,}"
        token="${token#"${token%%[![:space:]]*}"}"
        token="${token%"${token##*[![:space:]]}"}"
        if [[ -z "$token" ]]; then
            echo "ERROR: empty entry in host range '$context'" >&2
            return 1
        fi
        if [[ "$token" =~ ^([0-9]+)[[:space:]]*\.\.[[:space:]]*([0-9]+)$ ]]; then
            start_str="${BASH_REMATCH[1]}"
            end_str="${BASH_REMATCH[2]}"
            if (( ${#start_str} > 9 || ${#end_str} > 9 )); then
                echo "ERROR: host range '$token' in '$context' is too large" >&2
                return 1
            fi
            start=$((10#$start_str))
            end=$((10#$end_str))
            if (( end < start )); then
                echo "ERROR: host range '$token' in '$context' is descending" >&2
                return 1
            fi
            if (( end - start + 1 > HOST_EXPAND_LIMIT )); then
                echo "ERROR: host range '$token' in '$context' expands past $HOST_EXPAND_LIMIT hosts" >&2
                return 1
            fi
            width=0
            if [[ "$start_str" =~ ^0[0-9] || "$end_str" =~ ^0[0-9] ]]; then
                width=${#start_str}
                if (( ${#end_str} > width )); then
                    width=${#end_str}
                fi
            fi
            for ((i = start; i <= end; i++)); do
                if (( width > 0 )); then
                    printf -v token '%0*d' "$width" "$i"
                else
                    printf -v token '%d' "$i"
                fi
                _range_values+=("$token")
            done
        elif [[ "$token" =~ ^[0-9]+$ ]]; then
            if (( ${#token} > 9 )); then
                echo "ERROR: host range entry '$token' in '$context' is too large" >&2
                return 1
            fi
            # Keep the written form, including a leading zero on a single index.
            _range_values+=("$token")
        else
            echo "ERROR: invalid host range entry '$token' in '$context'" >&2
            echo "       expected a number or start..end (example: [1,5..7])" >&2
            return 1
        fi
    done
    if [[ ${#_range_values[@]} -eq 0 ]]; then
        echo "ERROR: empty host range in '$context'" >&2
        return 1
    fi
}

expand_host_pattern() {
    local pat="$1"
    local prefix rest spec suffix
    local -a values=()
    local v

    if [[ "$pat" != *'['* && "$pat" != *']'* ]]; then
        hosts+=("$pat")
        return 0
    fi
    if [[ "$pat" != *'['* || "$pat" != *']'* ]]; then
        echo "ERROR: unmatched bracket in host pattern '$pat'" >&2
        return 1
    fi

    prefix="${pat%%\[*}"
    if [[ "$prefix" == *']'* ]]; then
        echo "ERROR: unmatched ']' in host pattern '$pat'" >&2
        return 1
    fi
    rest="${pat#*\[}"
    spec="${rest%%\]*}"
    suffix="${rest#*\]}"

    expand_range_spec "$spec" "$pat" || return 1
    values=("${_range_values[@]}")
    for v in "${values[@]}"; do
        expand_host_pattern "${prefix}${v}${suffix}" || return 1
    done
}

expand_host_list() {
    local input="$1"
    local current="" depth=0 i c pat stripped rest
    local -a patterns=()

    hosts=()
    for ((i = 0; i < ${#input}; i++)); do
        c="${input:i:1}"
        if [[ "$c" == '[' ]]; then
            depth=$((depth + 1))
            current+="$c"
        elif [[ "$c" == ']' ]]; then
            if (( depth == 0 )); then
                echo "ERROR: unmatched ']' in host list" >&2
                return 1
            fi
            depth=$((depth - 1))
            current+="$c"
        elif [[ "$c" == ',' && depth -eq 0 ]]; then
            patterns+=("$current")
            current=""
        else
            current+="$c"
        fi
    done
    if (( depth != 0 )); then
        echo "ERROR: unmatched '[' in host list" >&2
        return 1
    fi
    patterns+=("$current")

    for pat in "${patterns[@]}"; do
        pat="${pat#"${pat%%[![:space:]]*}"}"
        pat="${pat%"${pat##*[![:space:]]}"}"
        if [[ -z "$pat" ]]; then
            echo "ERROR: Empty hostname found in host list" >&2
            return 1
        fi
        # Whitespace is allowed inside [] and is trimmed per entry. Outside [],
        # it is still a bad hostname (leading/trailing space was already removed).
        stripped="$pat"
        while [[ "$stripped" == *'['* && "$stripped" == *']'* ]]; do
            rest="${stripped#*\[}"
            if [[ "$rest" != *']'* ]]; then
                break
            fi
            stripped="${stripped%%\[*}${rest#*\]}"
        done
        if [[ "$stripped" =~ [[:space:]] ]]; then
            echo "ERROR: Hostname '$pat' contains whitespace" >&2
            return 1
        fi
        expand_host_pattern "$pat" || return 1
    done

    if (( ${#hosts[@]} > HOST_EXPAND_LIMIT )); then
        echo "ERROR: host list expands to ${#hosts[@]} hosts (limit $HOST_EXPAND_LIMIT)" >&2
        return 1
    fi
}


# Parse command line arguments. -h/--help and -n/--dry-run may appear anywhere
# before a literal "--" (before or after the host list), since anything after
# "--" is a TransferBench argument and must be passed through unchanged.
dry_run=false
args=()
parsing_tb_args=false
for arg in "$@"; do
    if [[ "$parsing_tb_args" == false && "$arg" == "--" ]]; then
        parsing_tb_args=true
        args+=("$arg")
        continue
    fi
    if [[ "$parsing_tb_args" == false ]]; then
        case "$arg" in
            -h|--help)
                show_usage
                exit 0
                ;;
            -n|--dry-run)
                dry_run=true
                continue
                ;;
        esac
    fi
    args+=("$arg")
done
set -- "${args[@]}"

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

# Convert comma-separated hosts to an array, expanding range wildcards.
if ! expand_host_list "$hosts_input"; then
    exit 1
fi
num_ranks=${#hosts[@]}

if [[ $num_ranks -lt 2 ]]; then
    echo "ERROR: At least 2 hosts are required for multi-rank execution" >&2
    echo "For single-node execution, run TransferBench directly without this script" >&2
    exit 1
fi

if [[ "$dry_run" == true ]]; then
    echo "Mode      : DRY RUN (no commands will be executed)"
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

# Quote one word for the remote login shell. Single quotes are literal in
# bash, sh, and csh/tcsh, and the '\'' sequence is a quote in all three.
shell_quote() {
    printf "'%s'" "${1//\'/\'\\\'\'}"
}

# The login shell that ssh starts may be csh/tcsh, which rejects a VAR=value
# prefix. Hand the real command to bash. The outer word is single-quoted so
# either login shell passes it through unchanged.
build_remote_cmd() {
    local rank="$1"
    local cmd key value arg

    cmd="TB_NUM_RANKS=$num_ranks TB_RANK=$rank TB_SINGLE_LOG=1"
    cmd+=" TB_MASTER_ADDR=$(shell_quote "$master_addr")"
    for env_var in "${env_vars[@]}"; do
        key="${env_var%%=*}"
        value="${env_var#*=}"
        cmd+=" $key=$(shell_quote "$value")"
    done
    cmd+=" $(shell_quote "$transferbench_path")"
    for arg in "${tb_args[@]}"; do
        cmd+=" $(shell_quote "$arg")"
    done
    printf '%s' "/bin/bash --noprofile --norc -c $(shell_quote "$cmd")"
}

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
            ssh -q -o LogLevel=ERROR -o ConnectTimeout=1 "$host" "pkill -u \`whoami\` -f TransferBench >/dev/null 2>&1 || true" 2>/dev/null || true &
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
    addr=$(ssh -G -- "$host" 2>/dev/null | awk '/^hostname /{print $2; exit}')
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

for ((rank=1; rank<num_ranks; rank++)); do
    worker_host="${hosts[$rank]}"
    worker_cmd=$(build_remote_cmd "$rank")
    if [[ "$dry_run" == true ]]; then
        echo "Rank $rank (worker) on $worker_host:"
        echo "  ssh $worker_host $worker_cmd"
        continue
    fi
    ssh -q -o LogLevel=ERROR "$worker_host" "$worker_cmd" >/dev/null 2>&1 &
    worker_pids+=($!)
    worker_hosts+=("$worker_host")
done

# Start master rank (TransferBench will wait for all workers to connect)
master_cmd=$(build_remote_cmd 0)
if [[ "$dry_run" == true ]]; then
    echo "Rank 0 (master) on $master_host:"
    echo "  ssh $master_host $master_cmd"
    exit 0
fi
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

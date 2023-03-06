# TransferBench

TransferBench is a simple utility capable of benchmarking simultaneous copies between user-specified devices (CPUs/GPUs).

## Requirements

1. ROCm stack installed on the system (HIP runtime)
2. libnuma installed on system

## Building
  To build TransferBench using Makefile:
 ```shell
 $ make
 ```

  To build TransferBench using cmake:
 ```shell 
$ mkdir build
$ cd build
$ CXX=/opt/rocm/bin/hipcc cmake ..
$ make
 ```

  If ROCm is installed in a folder other than `/opt/rocm/`, set ROCM_PATH appropriately

## NVIDIA platform support

TransferBench may also be built to run on NVIDIA platforms via HIP, but requires a HIP-compatible CUDA version installed (e.g. CUDA 11.5)

To build:
```
   CUDA_PATH=<path_to_CUDA> HIP_PLATFORM=nvidia make`
```

## Hints and suggestions
- Running TransferBench with no arguments will display usage instructions and detected topology information
- There are several preset configurations that can be used instead of a configuration file
  including:
  - p2p    - Peer to peer benchmark test
  - sweep  - Sweep across possible sets of Transfers
  - rsweep - Random sweep across possible sets of Transfers
- When using the same GPU executor in multiple simultaneous Transfers, performance may be
  serialized due to the maximum number of hardware queues available.
  - The number of maximum hardware queues can be adjusted via GPU_MAX_HW_QUEUES
  - Alternatively, running in single stream mode (USE_SINGLE_STREAM=1) may avoid this issue
    by launching all Transfers on a single stream instead of individual streams

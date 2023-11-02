# TransferBench

TransferBench is a simple utility capable of benchmarking simultaneous copies between user-specified
CPU and GPU devices.

Documentation for TransferBench is available at
[https://rocm.docs.amd.com/projects/TransferBench/en/latest/index.html](https://rocm.docs.amd.com/projects/TransferBench/en/latest/index.html).

## Requirements

* You must have a ROCm stack installed on your system (HIP runtime)
* You must have `libnuma` installed on your system

## Documentation

To build documentation locally, use the following code:

```shell
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Building TransferBench

You can build TransferBench using Makefile or CMake.

* Makefile:

  ```shell
  make
  ```

* CMake:

  ```shell
  mkdir build
  cd build
  CXX=/opt/rocm/bin/hipcc cmake ..
  make
  ```

  If ROCm is not installed in `/opt/rocm/`, you must set `ROCM_PATH` to the correct location.

## NVIDIA platform support

You can build TransferBench to run on NVIDIA platforms via HIP or native NVCC.

Use the following code to build with HIP for NVIDIA (note that you must have a HIP-compatible CUDA
version installed, e.g., CUDA 11.5):

```shell
CUDA_PATH=<path_to_CUDA> HIP_PLATFORM=nvidia make`
```

Use the following code to build with native NVCC (builds `TransferBenchCuda`):

```shell
make
```

## Things to note

* Running TransferBench with no arguments displays usage instructions and detected topology
  information
* You can use several preset configurations instead of a configuration file:
  * `p2p`: Peer-to-peer benchmark test
  * `sweep`: Sweep across possible sets of transfers
  * `rsweep`: Random sweep across possible sets of transfers
* When using the same GPU executor in multiple simultaneous transfers, performance may be
  serialized due to the maximum number of hardware queues available
  * The number of maximum hardware queues can be adjusted via `GPU_MAX_HW_QUEUES`
  * Alternatively, running in single-stream mode (`USE_SINGLE_STREAM`=1) may avoid this issue
    by launching all transfers on a single stream, rather than on individual streams

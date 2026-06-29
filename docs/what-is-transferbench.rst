.. meta::
  :description: Overview of TransferBench, a utility for benchmarking simultaneous memory transfers between CPUs, GPUs, and NICs.
  :keywords: TransferBench, benchmarking utility, memory transfers, GPU transfers, NIC transfers, memory device, Executor

.. _what-is-transferbench:

=======================
What is TransferBench?
=======================

TransferBench is a utility for benchmarking simultaneous memory transfers between user-specified devices (CPUs, GPUs, and NICs).

Memory transfers
================

A memory transfer is a single operation where an Executor (EXE) reads and adds values from source (SRC) memory devices, then writes the sum to destination (DST) memory devices. With a single SRC or DST, a memory transfer reduces to a simple copy operation. The memory transfer is commonly denoted by the ``(SRC->EXE->DST)`` triplet.

A memory device consists of a location (a specific device that owns the memory) and a memory type (usually some attribute about the memory). For example, fine-grained HBM memory (memory type) on GPU 0 (location), or pinned CPU memory on NUMA node 1 (location).

A set of transfers to be run simultaneously is called a Test.

Executors
=========

An Executor (EXE) is the specific device that performs a transfer — reading from SRC memory devices and writing to DST memory devices. TransferBench supports the following executor types:

- **CPU threads**
- **GPU compute kernels** (GFX)
- **GPU DMA engines** (SDMA)
- **NIC RDMA** (via Remote Direct Memory Access network interface cards)

Some Executors support **SubExecutors**, which allow further partitioning of the data to be transferred across multiple execution units within the same device.

Features
========

TransferBench supports the following features:

- **Multiple Executors:** CPU threads, GPU compute kernels, GPU Direct Memory Access (DMA) or System DMA (SDMA), and Remote Direct Memory Access (RDMA) network interface cards (RNICs).

- **Multi-input or multi-output (MIMO) transfers:** Element-wise sum from multiple SRCs to multiple DSTs.

- **Multinode execution:** Runs across distributed systems using MPI or sockets.

- **Flexible configuration:** Configure benchmarks using config files or presets.

- **Flexible hardware:** Runs HIP and CUDA programs on both AMD and NVIDIA hardware.

Frontend and backend
====================

TransferBench provides a frontend client (the executable) and a backend library (the header-only ``TransferBench.hpp``). Use the backend library to integrate TransferBench into your own applications.

.. meta::
  :description: TransferBench documentation home. TransferBench is a utility for benchmarking simultaneous memory transfers between CPUs, GPUs, and NICs.
  :keywords: TransferBench, benchmarking utility, memory transfers, GPU transfers, NIC transfers, multinode benchmark

****************************
TransferBench documentation
****************************

TransferBench is a utility for benchmarking simultaneous memory transfers between user-specified devices (CPUs, GPUs, and NICs).

A memory transfer is a single operation where an Executor (EXE) reads and adds values from source (SRC) memory devices, then writes the sum to destination (DST) memory devices. When dealing with a single SRC or DST, a memory transfer is similar to a simple copy operation. The memory transfer is commonly denoted by the (SRC->EXE->DST) triplet.

A Memory device consists of a location (a specific device that owns the memory) and a memory type (usually some attribute about the memory). For example, fine-grained HBM memory (memory type) on GPU 0 (location) or pinned CPU memory (memory type) on NUMA node 1 (location).

TransferBench supports the following features:

- **Multiple executors:** CPU threads, GPU compute kernels, GPU Direct Memory Access (DMA) or System DMA (SDMA), and Remote Direct Memory Access (RDMA) NIC or RNIC. Some Executors support SubExecutors, allowing further partitioning of the data to be transferred.

- **Multi-input or multi-output (MIMO) transfers:** Element-wise sum from multiple SRCs to multiple DSTs.

- **Multinode execution:** Using MPI or sockets across distributed systems.

- **Flexible configuration:** Using Config files or presets for common benchmarks.

- **Flexible hardware:** Supports HIP and CUDA programs that can run on both AMD and NVIDIA hardware.

TransferBench provides a frontend client (the executable) and a backend library (the header-only TransferBench.hpp). The backend library can be used to integrate TransferBench into other custom applications.

The code is open and hosted at `<https://github.com/ROCm/TransferBench>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :ref:`install-transferbench`

  .. grid-item-card:: How to

    * :ref:`running-transferbench-customized`

  .. grid-item-card:: Conceptual

    * :ref:`transferbench-workflow`
    * :ref:`transferbench-timing`
    * :ref:`transferbench-data-validation`

  .. grid-item-card:: Reference

    * :ref:`running-presets`
    * :ref:`environment-variables`
    * :ref:`faq`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.

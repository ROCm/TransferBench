.. meta::
  :description: Reference for TransferBench presets, including all-to-all, peer-to-peer, NIC rings, sweep, and scaling tests with supported environment variables and example outputs.
  :keywords: TransferBench presets, TransferBench a2a, TransferBench p2p, TransferBench nicrings, TransferBench nicp2p, TransferBench sweep, TransferBench scaling

.. _running-presets:

=======================
TransferBench presets
=======================

Presets are a predefined series of transfers that can be used instead of manually configuring the transfers.

The following table lists the presets available on TransferBench 1.66.03:

.. list-table::
    :header-rows: 1

    * - Preset name
      - Description
      - Multinode support

    * - :ref:`All-to-all preset (a2a) <a2a>`
      - Tests parallel transfers between all pairs of GPU devices.
      - ✅

    * - :ref:`All-to-all via nearest NIC preset (a2a_n) <a2a_n>`
      - Tests parallel transfers between all pairs of GPU devices using nearest NIC RDMA.
      - ❌

    * - :ref:`All-to-all sweep preset (a2asweep) <a2asweep>`
      - Performs a parameter sweep of GFX-based all-to-all transfers across different SubExecutor counts, unroll factors, and thread block sizes.
      - ❌

    * - :ref:`NIC rings preset (nicrings) <nicrings>`
      - Tests NIC rings created across identical NIC indices across ranks.
      - ✅

    * - :ref:`NIC peer-to-peer preset (nicp2p) <nicp2p>`
      - Tests multinode peer-to-peer RDMA transfer between all NICs across all ranks.
      - ✅

    * - :ref:`One-to-all preset (one2all) <one2all>`
      - Tests all subsets of parallel transfers from one GPU to the others.
      - ❌

    * - :ref:`Peer-to-peer preset (p2p) <p2p>`
      - Tests unidirectional and bidirectional transfers for CPU-to-CPU, CPU-to-GPU, and GPU-to-GPU combinations.
      - ❌

    * - :ref:`Scaling preset (scaling) <scaling>`
      - Runs a scaling test from one GPU to all other devices (CPUs and GPUs).
      - ❌

    * - :ref:`Schmoo preset (schmoo) <schmoo>`
      - Runs scaling tests for local and remote read, write, and copy operations between two GPUs.
      - ❌

    * - :ref:`Sweep or random sweep preset (sweep/rsweep) <sweep>`
      - Tests combinations of source (SRC), Executor, and destination (DST) with varying parallelism.
      - ❌

.. note::

    You can modify a preset using environment variables, which are detailed when running the preset.

.. _a2a:

All-to-all preset (a2a)
========================

The a2a preset tests parallel transfers between all pairs of GPU devices. It measures bidirectional bandwidth across every GPU-to-GPU combination on a single node or multinode system. It supports GFX (compute kernel) and DMA all-to-all, and optionally adds a parallel NIC executor ring (when ``NUM_QUEUE_PAIRS`` > 0).

**Key features:**

- **GFX or DMA mode:** Creates transfers for every (src GPU to dst GPU) pair on each rank. Optionally restricts to directly connected XGMI links (A2A_DIRECT=1).

- **Transfer modes:** Copy (1 src → 1 dst), read-only (1 src → null), write-only (null → 1 dst), or custom (numSrcs:numDsts).

- **NIC rings:** When ``NUM_QUEUE_PAIRS`` > 0, adds NIC-based ring transfers (GPU i → GPU (i+1)%N) using nearest-NIC RDMA.

- Prints a SRC x DST bandwidth matrix with row totals or column totals (configurable), aggregate bandwidth, and min/max/avg across ranks for multinode systems.

- Forces ``USE_SINGLE_STREAM=1`` for all-to-all.

- **On AMD hardware:** ``A2A_DIRECT=1`` uses ``hipExtGetLinkTypeAndHopCount`` to skip non-direct XGMI pairs.

- **Multinode:** Each rank must have the same number of GPUs. Differences in the NIC configuration across ranks produce a warning.

**Usage:**

.. code-block:: shell

    ./TransferBench a2a [numBytes]

Environment variables
----------------------

To modify the behavior of a2a preset, use the following environment variables:

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``A2A_DIRECT``
      - To use only directly connected XGMI links (hop count = 1). 0 = full all-to-all. This can be useful on older MI2XX hardware that doesn't feature full all-to-all XGMI connectivity, and running the standard all-to-all between all pairs of GPUs ends up using XGMI links more than once.
      - ``1``

    * - ``A2A_LOCAL``
      - To include local transfers (i→i). 0 = exclude, 1 = include.
      - ``0``

    * - ``A2A_MODE``
      - Transfer mode: 0=Copy, 1=Read-Only, 2=Write-Only, or numSrcs:numDsts for custom. Systems with multiple sources or destinations mimic the behavior of some collective algorithms such as RingReduce, which sometimes require reading from two local buffers, adding them together, then writing to a local output buffer and remote temp buffer.
      - ``0``

    * - ``GFX_UNROLL``
      - GFX kernel unroll factor. Overrides global default. See :ref:`gfx-unroll`.
      - ``2``

    * - ``MEM_TYPE``
      - GPU memory type: 0=default, 1=fine-grained, 2=uncached, 3=managed. See :ref:`mem-type`.
      - ``2``

    * - ``NUM_GPU_DEVICES``
      - Number of GPUs to use.
      - (detected)

    * - ``NUM_QUEUE_PAIRS``
      - Queue pairs per NIC transfer. 0 = no NIC rings.
      - ``0``

    * - ``NUM_RESULTS``
      - Shows top or bottom N results per cell for multinode. Default = 1 if numRanks > 1.
      - ``0`` or ``1``

    * - ``NUM_SUB_EXEC``
      - Sub-executors (CUs or WGPs) per transfer.
      - ``8``

    * - ``SHOW_DETAILS``
      - Shows full results per transfer.
      - ``0``

    * - ``USE_DMA_EXEC``
      - To use DMA Executor instead of GFX. Valid only for A2A_MODE=0 (copy).
      - ``0``

    * - ``USE_FINE_GRAIN``
      - To use MEM_TYPE.
      - (deprecated)

    * - ``USE_REMOTE_READ``
      - To use DST GPU as Executor (remote read) instead of SRC GPU (local read).
      - ``0``

Example output
---------------

.. tab-set::

    .. tab-item:: AMD Instinct™ MI300X

        .. image:: /data/a2a_MI300X.png
            :width: 100%
            :align: center

    .. tab-item:: AMD Instinct™ MI350X

        .. image:: /data/a2a_MI350X.png
            :width: 100%
            :align: center

The table in the output shows the transfer rate for each pair of GPUs, as measured using GPU timestamps.

- ``STotal``: Indicates the total send bandwidth as a sum of SRC GPU's bandwidth.

- ``RTotal``: Indicates the total receive bandwidth as a sum of DST GPU's bandwidth.

- ``Actual``: Reflects the actual time for the kernel to finish executing the slowest transfer. Because one GFX kernel is launched to handle all transfers to other GPUs, the kernel doesn't finish until the slowest transfer completes.

- ``CPU Timed``: Measures all the transfers.

.. note::

    To rule out any possibility of serialization, check if the CPU Timed bandwidth is close to the aggregate GPU Timed bandwidth.

    To avoid serialization when running with DMA Executor, increase the number of hardware queues available.

    As the following output shows, ``GPU_MAX_HW_QUEUES`` defaults to just 4 if not set:

    .. image:: /data/a2a_serialization.png
        :width: 100%
        :align: center

    Although TransferBench issues a warning ``[WARN] DMA 0 attempting n parallel transfers, however GPU_MAX_HW_QUEUES only set to 4``, the hardware queue insufficiency can also be noticed by the large discrepancy between CPU Timed aggregate bandwidth and GPU timed aggregated bandwidth.

.. _a2a_n:

All-to-all via nearest NIC preset (a2a_n)
==========================================

The a2a_n preset tests parallel transfers between all pairs of GPU devices using nearest NIC RDMA. Each transfer uses the NIC closest to the SRC GPU to send to the NIC closest to the DST GPU.

**Key features:**

- Creates transfers for every SRC GPU and DST GPU pair using the NIC closest to the SRC GPU to read, and the NIC closest to the DST GPU to write.

- Prints a SRC x DST bandwidth matrix with row totals, column totals, and aggregate bandwidth.

- Reports average and aggregate bandwidth (Tx-thread timed and CPU timed).

- Single-node only.

**Usage:**

.. code-block:: shell

    ./TransferBench a2a_n [numBytes]

Environment variables
----------------------

To modify the behavior of a2a_n preset, use the following environment variables:

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``MEM_TYPE``
      - GPU memory type: 0=default, 1=fine-grained, 2=uncached, 3=managed. See :ref:`mem-type`.
      - ``2``

    * - ``NUM_GPU_DEVICES``
      - Number of GPUs to use.
      - (detected)

    * - ``NUM_QUEUE_PAIRS``
      - Queue pairs per transfer.
      - ``1``

.. note::

    The a2a_n preset divides the available NIC bandwidth into the number of GPU peers.

.. _a2asweep:

All-to-all sweep preset (a2asweep)
===================================

The a2asweep preset performs a parameter sweep of GFX-based all-to-all transfers across different SubExecutor counts, unroll factors, and thread block sizes. It helps find optimal configurations for GPU all-to-all bandwidth on your hardware.

**Key features:**

- Sweeps ``BLOCKSIZES`` (thread block size).

- For each block size, sweeps ``NUM_SUB_EXECS`` (CU count) x ``UNROLLS`` (unroll factor).

- Sweep order: Outer loop over ``BLOCKSIZES``, then table of (``NUM_SUB_EXECS`` x ``UNROLLS``).

- By default, reports only the slowest GPU's bandwidth (min bandwidth) per CU-Unroll combination. To include the fastest GPU's bandwidth (max bandwidth) per config, set ``SHOW_MIN_ONLY`` = 0.

- Uses same transfer topology as a2a preset, such as direct links, A2A_MODE, and others.

**Restrictions:**

- Single-node only.

- Forces ``USE_SINGLE_STREAM=1``.

- ``USE_SPRAY`` is incompatible with multiple destination buffers (``numDsts`` > 1).

**Usage:**

.. code-block:: shell

    ./TransferBench a2asweep

To use custom sweep ranges:

.. code-block:: shell

    BLOCKSIZES=256,384 UNROLLS=2,4,8 NUM_SUB_EXECS=4,8,16 ./TransferBench a2asweep

Environment variables
----------------------

To modify the behavior of a2asweep preset, use the following environment variables:

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``A2A_DIRECT``
      - To use only directly-connected GPU pairs, set to ``1``. For full all-to-all, set to ``0``.
      - ``1``

    * - ``A2A_LOCAL``
      - To include local transfers, set to ``1``. To exclude, set to ``0``.
      - ``0``

    * - ``A2A_MODE``
      - Transfer mode: 0=Copy, 1=Read-Only, 2=Write-Only, or numSrcs:numDsts for custom.
      - ``0``

    * - ``BLOCKSIZES``
      - Comma-separated thread block sizes, such as 256, 384, or 512.
      - ``256``

    * - ``MEM_TYPE``
      - GPU memory type: 0=default, 1=fine-grained, 2=uncached, 3=managed. See :ref:`mem-type`.
      - ``2``

    * - ``NUM_GPU_DEVICES``
      - Number of GPUs in all-to-all group.
      - (all detected)

    * - ``NUM_SUB_EXECS``
      - Comma-separated SubExecutor (CU or WGP) counts to sweep.
      - ``4,8,12,16,24,32``

    * - ``SHOW_MIN_ONLY``
      - To show only the slowest GPU result, set to ``1``. To show the slowest and the fastest GPU results, set to ``0``.
      - ``1``

    * - ``UNROLLS``
      - Comma-separated unroll factors to sweep. See :ref:`gfx-unroll`.
      - ``1,2,3,4,6,8``

    * - ``USE_REMOTE_READ``
      - To use the Executor on DST, set to ``1``. To use the Executor on SRC, set to ``0``.
      - ``0``

    * - ``USE_SPRAY``
      - To configure each SubExecutor to target all GPUs, set to ``1``. To target only one GPU, set to ``0``. Invalid for multiple DST.
      - ``0``

    * - ``VERBOSE``
      - Shows detailed results per config.
      - ``0``

Example output
---------------

.. tab-set::

    .. tab-item:: AMD Instinct™ MI300X

        .. image:: /data/a2asweep_MI300X.png
            :width: 100%
            :align: center

    .. tab-item:: AMD Instinct™ MI350X

        .. image:: /data/a2asweep_MI350X.png
            :width: 100%
            :align: center

.. _nicrings:

NIC rings (nicrings)
=====================

The nicrings preset tests NIC rings created across identical NIC indices across ranks. It measures RDMA bandwidth in ring topologies where each rank sends to the next rank in the ring, using GPU or CPU memory closest to each NIC.

The following image shows the ring topology:

.. image:: /data/nicrings.png
    :width: 100%
    :align: center

**Key features:**

- Ring construction: Creates parallel RDMA rings across all ranks with one ring per GPU/CPU-to-NIC pair (memIndex-nicIndex), where that NIC is the closest to that memory.

- Topology of each ring: Rank 0->1->2->...->N-1->0.

- Can use GPU memory or CPU memory (NUMA nearest to NIC) as buffer.

- Supports RDMA read or write. To choose the rank for RDMA read or write in multirank systems, use ``USE_RDMA_READ``.

- Multinode supported with homogeneous ranks (same topology). Use ``TB_NIC_FILTER`` to limit NIC visibility if needed.

- Transfer direction: ``currRank`` sends to (``currRank`` + 1) % ``numRanks``.

- Executor placement: Executor is placed on the SRC rank for RDMA write and DST rank for RDMA read.

**Usage:**

.. code-block:: shell

    ./TransferBench nicrings

To use CPU memory:

.. code-block:: shell

  USE_CPU_MEM=1 ./TransferBench nicrings

To use RDMA read and see details:

.. code-block:: shell

  SHOW_DETAILS=1 USE_RDMA_READ=1 NUM_QUEUE_PAIRS=2 ./TransferBench nicrings

Environment variables
----------------------

To modify the behavior of nicrings preset, use the following environment variables:

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``MEM_TYPE``
      - Memory type index. See :ref:`mem-type`.
      - ``0``

    * - ``NUM_QUEUE_PAIRS``
      - Queue pairs per NIC transfer.
      - ``1``

    * - ``SHOW_DETAILS``
      - To see full transfer details, set to ``1``.
      - ``0``

    * - ``USE_CPU_MEM``
      - To use CPU memory closest to each NIC, set to ``1``. To use GPU memory, set to ``0``.
      - ``0``

    * - ``USE_RDMA_READ``
      - To use RDMA reads, set to ``1``. To use RDMA writes, set to ``0``. Applies when ``numRanks`` > 1.
      - ``0``

Example output
---------------

Here is an example output collected on four MI350X nodes with 8 NICs:

.. image:: /data/nicrings_MI350X.png
  :width: 100%
  :align: center

.. _nicp2p:

NIC peer-to-peer preset (nicp2p)
=================================

The nicp2p preset runs a multinode peer-to-peer RDMA transfer test between all NICs across all ranks. It measures bandwidth for every NIC-to-NIC pair using round-robin scheduling to avoid contention.

**Key features:**

- Tests all (``srcRank``, ``srcNic``) -> (``dstRank``, ``dstNic``) pairs.

- Device selection: Uses ``GetClosestDeviceToNic()`` to pick CPU NUMA or GPU closest to each NIC based on ``SRC_MEM_TYPE`` or ``DST_MEM_TYPE``, and ``USE_CPU_*`` flags.

- Allows using RDMA read instead of write through ``USE_REMOTE_READ``.

- Round-robin and combination schedule: Node pairs are scheduled in round-robin. Within each node pair, NIC pairs are tested in all combinations (controlled by ``NIC_PARALLEL_LEVEL``).

- Output: Full matrix or column format, including top 10 fastest and slowest connections.

- Progress report: Prints progress to stderr. For example, "Completed X/Y pairs in Zs, estimated remaining time Ws".

- Multinode supported with homogeneous ranks (same topology). Use ``TB_NIC_FILTER`` to limit NIC visibility if needed.

- NICs required: Exits with error if no NICs are detected.

**Usage:**

.. code-block:: shell

  ./TransferBench nicp2p

To use CPU memory and see output in column format:

.. code-block:: shell

  OUTPUT_FORMAT=0 USE_CPU_SRC_MEM=1 USE_CPU_DST_MEM=1 ./TransferBench nicp2p

Environment variables
----------------------

To modify the behavior of nicp2p preset, use the following environment variables:

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``NUM_QUEUE_PAIRS``
      - Queue pairs per transfer (displayed as ``NUM_NIC_SE``).
      - ``1``

    * - ``USE_REMOTE_READ``
      - To use DST GPU as Executor (remote read) instead of SRC GPU (local read).
      - ``0``

    * - ``OUTPUT_FORMAT``
      - To output full matrix, set to ``1``. For output in column format, set to ``0``. Column format is recommended when there are lots of NIC pairs.
      - ``1``

    * - ``USE_CPU_SRC_MEM``
      - To use CPU memory as SRC, set to ``1``. To use GPU memory as SRC, set to ``0``.
      - ``0``

    * - ``USE_CPU_DST_MEM``
      - To use CPU memory as DST, set to ``1``. To use GPU memory as DST, set to ``0``.
      - ``0``

    * - ``SRC_MEM_TYPE``
      - Source memory type index. See :ref:`mem-type`.
      - ``2``

    * - ``DST_MEM_TYPE``
      - Destination memory type index. See :ref:`mem-type`.
      - ``2``

    * - ``PARALLEL_NODE``
      - To execute node pairs in parallel, set to ``1``. For serial execution, set to ``0``. By default, nicp2p tries to run transfers between node pairs in parallel to reduce the overall runtime. For example, (Rank 0->Rank 1) + (Rank 2->Rank 3) are run in parallel instead of (Rank 0->Rank 1) followed by (Rank 2->Rank 3).
      - ``1``

    * - ``NIC_PARALLEL_LEVEL``
      - NIC-to-NIC pairs that run in parallel between a node pair. By default, between a pair of nodes, all available NICs are used in parallel. NICs aren't used more than once at a time. This option reduces the overall runtime, which can be disabled if it impacts the performance.
      - ``numNicsPerRank``

Example output
---------------

.. code-block:: shell

  [P2P Network Related]
  NUM_NIC_SE           =            1 : Using 1 queue pairs per Transfer
  USE_REMOTE_READ      =            0 : Using SRC as executor
  OUTPUT_FORMAT        =            1 : Printing results in full matrix format
  USE_CPU_SRC_MEM      =            0 : Source memory is GPU
  USE_CPU_DST_MEM      =            0 : Destination memory is GPU
  SRC_MEM_TYPE         =            2 : Using uncached GPU memory (0=default, 1=fine-grained, 2=uncached, 3=managed)
  DST_MEM_TYPE         =            2 : Using uncached GPU memory (0=default, 1=fine-grained, 2=uncached, 3=managed)
  PARALLEL_NODE        =            1 : Executing p2p node pairs in parallel: yes
  NIC_PARALLEL_LEVEL   =            8 : Between a pair of nodes, 8 pairs of NIC-NIC transfers executed in parallel

  Unidirectional copy peak bandwidth GB/s (NIC RDMA Using Nearest Device)
  Completed 8/256 pairs in  2.656s, estimated remaining time 82.326s.
  Completed 16/256 pairs in  5.537s, estimated remaining time 83.057s.
  Completed 24/256 pairs in  8.351s, estimated remaining time 80.731s.
  Completed 32/256 pairs in 11.251s, estimated remaining time 78.756s.
  Completed 40/256 pairs in 14.159s, estimated remaining time 76.460s.
  Completed 48/256 pairs in 16.688s, estimated remaining time 72.315s.
  Completed 56/256 pairs in 19.113s, estimated remaining time 68.261s.
  Completed 64/256 pairs in 21.748s, estimated remaining time 65.245s.
  Completed 72/256 pairs in 24.465s, estimated remaining time 62.521s.
  Completed 80/256 pairs in 27.377s, estimated remaining time 60.229s.
  Completed 88/256 pairs in 30.264s, estimated remaining time 57.777s.
  Completed 96/256 pairs in 32.851s, estimated remaining time 54.752s.
  Completed 104/256 pairs in 35.601s, estimated remaining time 52.033s.
  Completed 112/256 pairs in 38.404s, estimated remaining time 49.377s.
  Completed 120/256 pairs in 41.035s, estimated remaining time 46.507s.
  Completed 128/256 pairs in 43.756s, estimated remaining time 43.756s.
  Completed 144/256 pairs in 45.877s, estimated remaining time 35.682s.
  Completed 160/256 pairs in 47.736s, estimated remaining time 28.641s.
  Completed 176/256 pairs in 50.091s, estimated remaining time 22.769s.
  Completed 192/256 pairs in 51.892s, estimated remaining time 17.297s.
  Completed 208/256 pairs in 53.863s, estimated remaining time 12.430s.
  Completed 224/256 pairs in 55.850s, estimated remaining time  7.979s.
  Completed 240/256 pairs in 57.924s, estimated remaining time  3.862s.
  Completed 256/256 pairs in 60.043s, estimated remaining time  0.000s.
  ┌------------┬-------------------------┬---------------------------------------------------------------------------------------┬---------------------------------------------------------------------------------------┐
  │SRC+EXE\DST │                         │  Rank 00                                                                              │  Rank 01                                                                              │
  ├------------┼-------------------------┼---------------------------------------------------------------------------------------┼---------------------------------------------------------------------------------------┤
  │            │ NIC Device              │ bnxt_re0   bnxt_re1   bnxt_re2   bnxt_re3   bnxt_re4   bnxt_re5   bnxt_re6   bnxt_re7 │ bnxt_re0   bnxt_re1   bnxt_re2   bnxt_re3   bnxt_re4   bnxt_re5   bnxt_re6   bnxt_re7 │
  │            │              Mem Device │   GPU 00     GPU 01     GPU 02     GPU 03     GPU 04     GPU 05     GPU 06     GPU 07 │   GPU 00     GPU 01     GPU 02     GPU 03     GPU 04     GPU 05     GPU 06     GPU 07 │
  ├------------┼-------------------------┼---------------------------------------------------------------------------------------┼---------------------------------------------------------------------------------------┤
  │    Rank 00 │   bnxt_re0       GPU 00 │    31.36      31.31      31.31      31.31      31.31      31.31      31.31      31.30 │    31.32      31.31      31.31      31.30      31.30      31.31      31.31      31.31 │
  │            │   bnxt_re1       GPU 01 │    31.31      31.35      31.31      31.31      31.31      31.31      31.31      31.31 │    31.31      31.32      31.31      31.31      31.31      31.31      31.31      31.31 │
  │            │   bnxt_re2       GPU 02 │    31.31      31.32      31.36      31.31      31.31      31.31      31.30      31.31 │    31.30      31.30      31.32      31.31      31.31      31.30      31.30      31.31 │
  │            │   bnxt_re3       GPU 03 │    31.31      31.32      31.32      31.35      31.30      31.31      31.31      31.30 │    31.31      31.32      31.31      31.31      31.31      31.30      31.31      31.31 │
  │            │   bnxt_re4       GPU 04 │    31.31      31.32      31.31      31.32      31.35      31.31      31.31      31.30 │    31.31      31.31      31.31      31.31      31.32      31.31      31.31      31.30 │
  │            │   bnxt_re5       GPU 05 │    31.32      31.32      31.32      31.32      31.32      31.35      31.31      31.31 │    31.31      31.31      31.30      31.32      31.31      31.33      31.31      31.32 │
  │            │   bnxt_re6       GPU 06 │    31.31      31.31      31.32      31.32      31.32      31.32      31.36      31.31 │    31.31      31.31      31.31      31.31      31.31      31.31      31.33      31.31 │
  │            │   bnxt_re7       GPU 07 │    31.31      31.32      31.32      31.32      31.32      31.31      31.32      31.36 │    31.31      31.32      31.30      31.31      31.30      31.31      31.30      31.32 │
  ├------------┼-------------------------┼---------------------------------------------------------------------------------------┼---------------------------------------------------------------------------------------┤
  │    Rank 01 │   bnxt_re0       GPU 00 │    31.33      31.30      31.30      31.31      31.31      31.31      31.31      31.30 │    31.36      31.31      31.31      31.31      31.31      31.31      31.30      31.31 │
  │            │   bnxt_re1       GPU 01 │    31.32      31.32      31.31      31.30      31.31      31.31      31.31      31.31 │    31.32      31.36      31.31      31.31      31.30      31.30      31.30      31.30 │
  │            │   bnxt_re2       GPU 02 │    31.31      31.30      31.32      31.31      31.31      31.31      31.30      31.31 │    31.32      31.32      31.35      31.31      31.31      31.31      31.30      31.31 │
  │            │   bnxt_re3       GPU 03 │    31.31      31.31      31.31      31.32      31.30      31.31      31.30      31.31 │    31.31      31.32      31.32      31.36      31.31      31.31      31.31      31.30 │
  │            │   bnxt_re4       GPU 04 │    31.30      31.31      31.31      31.31      31.32      31.31      31.32      31.31 │    31.32      31.32      31.31      31.32      31.36      31.31      31.31      31.31 │
  │            │   bnxt_re5       GPU 05 │    31.30      31.31      31.31      31.31      31.30      31.32      31.31      31.31 │    31.31      31.32      31.32      31.32      31.32      31.36      31.31      31.31 │
  │            │   bnxt_re6       GPU 06 │    31.32      31.31      31.31      31.30      31.31      31.30      31.33      31.30 │    31.32      31.31      31.31      31.32      31.32      31.31      31.35      31.31 │
  │            │   bnxt_re7       GPU 07 │    31.31      31.31      31.31      31.31      31.31      31.31      31.31      31.32 │    31.31      31.31      31.32      31.32      31.31      31.32      31.32      31.35 │
  └------------┴-------------------------┴---------------------------------------------------------------------------------------┴---------------------------------------------------------------------------------------┘
  Summary of top 10 fastest/slowest connection
  ┌--------------------------┬--------------┬--------------┬--------------------------┬--------------┬--------------┐
  │ Fastest Bandwidth (GB/s) │          Src │          Dst │ Slowest Bandwidth (GB/s) │          Src │          Dst │
  ├--------------------------┼--------------┼--------------┼--------------------------┼--------------┼--------------┤
  │                    31.36 │ R00:bnxt_re0 │ R00:bnxt_re0 │                    31.30 │ R01:bnxt_re0 │ R00:bnxt_re1 │
  │                    31.36 │ R01:bnxt_re5 │ R01:bnxt_re5 │                    31.30 │ R00:bnxt_re4 │ R01:bnxt_re7 │
  │                    31.36 │ R00:bnxt_re7 │ R00:bnxt_re7 │                    31.30 │ R01:bnxt_re5 │ R00:bnxt_re4 │
  │                    31.36 │ R01:bnxt_re0 │ R01:bnxt_re0 │                    31.30 │ R00:bnxt_re3 │ R01:bnxt_re7 │
  │                    31.36 │ R00:bnxt_re2 │ R00:bnxt_re2 │                    31.30 │ R01:bnxt_re2 │ R00:bnxt_re1 │
  │                    31.36 │ R00:bnxt_re6 │ R00:bnxt_re6 │                    31.30 │ R01:bnxt_re0 │ R00:bnxt_re7 │
  │                    31.36 │ R01:bnxt_re1 │ R01:bnxt_re1 │                    31.30 │ R00:bnxt_re5 │ R01:bnxt_re2 │
  │                    31.36 │ R01:bnxt_re4 │ R01:bnxt_re4 │                    31.30 │ R01:bnxt_re1 │ R01:bnxt_re5 │
  │                    31.36 │ R01:bnxt_re3 │ R01:bnxt_re3 │                    31.30 │ R01:bnxt_re6 │ R01:bnxt_re7 │
  │                    31.35 │ R01:bnxt_re7 │ R01:bnxt_re7 │                    31.30 │ R01:bnxt_re2 │ R01:bnxt_re6 │
  └--------------------------┴--------------┴--------------┴--------------------------┴--------------┴--------------┘

.. _one2all:

One-to-all preset (one2all)
============================

The one2all preset tests all subsets of parallel transfers from one GPU to the others. It sweeps over varying numbers of DST peers (from ``SWEEP_MIN`` to ``SWEEP_MAX``), and for each count, tests every combination of DST GPUs from a single SRC or Executor GPU.

**Key features:**

- Requires at least two GPUs. Uses one GPU (``EXE_INDEX``) as SRC and Executor.

- Sweeps over all combinations of 1, 2, ..., N DST GPUs (excluding the SRC).

- Combination sweep: For each peer count ``p``, iterates over all bitmasks with exactly ``p`` bits set (excluding ``EXE_INDEX``).

- For each combination, runs parallel transfers and reports bandwidth per DST.

- Supports GFX or DMA executor. Each of SRC and DST can independently be GPU or Null, but not both Null simultaneously.

- Supports single node only: Multinode is not supported.

- Invalid configs are skipped in two cases:

  - ``exe`` = DMA and (``src`` = N or ``dst`` = N)
  - ``src`` = N and ``dst`` = N

- Output format: Each line shows bandwidth per DST GPU, ``p``, ``numSubExecs``, and transfer triplets.

**Usage:**

.. code-block:: shell

  ./TransferBench one2all

To run using GPU 2 as SRC and DST peers between 4 to 7:

.. code-block:: shell

  EXE_INDEX=2 SWEEP_MIN=4 SWEEP_MAX=7 ./TransferBench one2all

Environment variables
----------------------

To modify the behavior of one2all preset, use the following environment variables:

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``NUM_GPU_DEVICES``
      - Number of GPUs.
      - (all detected)

    * - ``NUM_GPU_SE``
      - Subexecutors (CUs) per transfer.
      - ``4``

    * - ``EXE_INDEX``
      - GPU index to use as Executor or SRC.
      - ``0``

    * - ``SWEEP_DIR``
      - Transfer direction.
      - ``0``

    * - ``SWEEP_SRC``
      - SRC memory types: G=GPU, N=Null.
      - ``G``

    * - ``SWEEP_DST``
      - DST memory types.
      - ``G``

    * - ``SWEEP_EXE``
      - Executor types: G=GFX, D=DMA.
      - ``G``

    * - ``SWEEP_MIN``
      - Minimum number of DST peers.
      - ``1``

    * - ``SWEEP_MAX``
      - Maximum number of DST peers.
      - ``numGpuDevices``

Example output
---------------

.. tab-set::

  .. tab-item:: AMD Instinct™ MI300X

    .. code-block:: shell

      [One-To-All Related]
      NUM_GPU_DEVICES      =            8 : Using 8 GPUs
      NUM_GPU_SE           =            4 : Using 4 subExecutors/CUs per Transfer
      EXE_INDEX            =            0 : Executing on GPU 0
      SWEEP_DIR            =            0 : Direction of transfer
      SWEEP_DST            =            G : DST memory types to sweep
      SWEEP_EXE            =            G : Executor type to use
      SWEEP_MAX            =            8 : Maximum number of peers
      SWEEP_MIN            =            1 : Minimum number of peers
      SWEEP_SRC            =            G : SRC memory types to sweep

      Executing (G0 -> G0 -> G*)
        GPU 1        GPU 2        GPU 3        GPU 4        GPU 5        GPU 6        GPU 7
      -------------------------------------------------------------------------------------------
          49.409                                                                                  1 4 (G0 G0 G1)
                      49.467                                                                     1 4 (G0 G0 G2)
                                    49.215                                                        1 4 (G0 G0 G3)
                                                47.526                                           1 4 (G0 G0 G4)
                                                              48.045                              1 4 (G0 G0 G5)
                                                                          48.278                 1 4 (G0 G0 G6)
                                                                                        48.132    1 4 (G0 G0 G7)
          48.954       35.346                                                                     2 4 (G0 G0 G1) (G0 G0 G2)
          48.851                    48.869                                                        2 4 (G0 G0 G1) (G0 G0 G3)
                      49.009       48.861                                                        2 4 (G0 G0 G2) (G0 G0 G3)
          48.962                                 47.599                                           2 4 (G0 G0 G1) (G0 G0 G4)
                      49.008                    47.486                                           2 4 (G0 G0 G2) (G0 G0 G4)
                                    35.706       47.563                                           2 4 (G0 G0 G3) (G0 G0 G4)
          48.833                                              31.660                              2 4 (G0 G0 G1) (G0 G0 G5)
                      49.002                                 35.160                              2 4 (G0 G0 G2) (G0 G0 G5)
                                    49.137                    47.565                              2 4 (G0 G0 G3) (G0 G0 G5)
                                                47.613       47.706                              2 4 (G0 G0 G4) (G0 G0 G5)
          48.972                                                           48.413                 2 4 (G0 G0 G1) (G0 G0 G6)
                      48.917                                              48.389                 2 4 (G0 G0 G2) (G0 G0 G6)
                                    37.319                                 48.397                 2 4 (G0 G0 G3) (G0 G0 G6)
                                                32.618                    48.334                 2 4 (G0 G0 G4) (G0 G0 G6)
                                                              47.749       48.497                 2 4 (G0 G0 G5) (G0 G0 G6)
          48.787                                                                        35.541    2 4 (G0 G0 G1) (G0 G0 G7)
                      48.824                                                           32.099    2 4 (G0 G0 G2) (G0 G0 G7)
                                    48.862                                              47.863    2 4 (G0 G0 G3) (G0 G0 G7)
                                                47.478                                 48.014    2 4 (G0 G0 G4) (G0 G0 G7)
                                                              47.705                    35.595    2 4 (G0 G0 G5) (G0 G0 G7)
                                                                          48.509       47.931    2 4 (G0 G0 G6) (G0 G0 G7)
          44.235       48.729       44.548                                                        3 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3)
          43.164       45.482                    43.238                                           3 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4)
          31.360                    48.819       31.280                                           3 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4)
                      31.624       48.941       31.406                                           3 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4)
          41.797       46.652                                 41.706                              3 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G5)
          41.739                    48.994                    41.575                              3 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G5)
                      42.676       48.992                    42.683                              3 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G5)
          42.621                                 47.369       42.536                              3 4 (G0 G0 G1) (G0 G0 G4) (G0 G0 G5)
                      43.504                    47.353       43.639                              3 4 (G0 G0 G2) (G0 G0 G4) (G0 G0 G5)
                                    31.263       47.357       31.202                              3 4 (G0 G0 G3) (G0 G0 G4) (G0 G0 G5)
          44.168       47.169                                              44.632                 3 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G6)
          30.692                    48.787                                 30.939                 3 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G6)
                      32.297       48.687                                 32.237                 3 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G6)
          28.916                                 47.483                    29.027                 3 4 (G0 G0 G1) (G0 G0 G4) (G0 G0 G6)
                      28.024                    47.429                    28.253                 3 4 (G0 G0 G2) (G0 G0 G4) (G0 G0 G6)
                                    27.484       47.547                    27.506                 3 4 (G0 G0 G3) (G0 G0 G4) (G0 G0 G6)
          43.660                                              40.609       44.131                 3 4 (G0 G0 G1) (G0 G0 G5) (G0 G0 G6)
                      44.196                                 46.915       44.520                 3 4 (G0 G0 G2) (G0 G0 G5) (G0 G0 G6)
                                    42.547                    47.627       43.041                 3 4 (G0 G0 G3) (G0 G0 G5) (G0 G0 G6)
                                                44.828       47.705       45.032                 3 4 (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
          46.291       44.552                                                           46.139    3 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G7)
          46.779                    48.784                                              46.969    3 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G7)
                      42.319       48.889                                              42.591    3 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G7)
          46.980                                 47.296                                 47.003    3 4 (G0 G0 G1) (G0 G0 G4) (G0 G0 G7)
                      44.806                    47.395                                 45.020    3 4 (G0 G0 G2) (G0 G0 G4) (G0 G0 G7)
                                    31.296       47.280                                 31.418    3 4 (G0 G0 G3) (G0 G0 G4) (G0 G0 G7)
          45.477                                              44.531                    45.229    3 4 (G0 G0 G1) (G0 G0 G5) (G0 G0 G7)
                      45.001                                 43.060                    44.962    3 4 (G0 G0 G2) (G0 G0 G5) (G0 G0 G7)
                                    47.083                    41.743                    46.937    3 4 (G0 G0 G3) (G0 G0 G5) (G0 G0 G7)
                                                42.876       45.829                    43.211    3 4 (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
          42.205                                                           48.237       42.679    3 4 (G0 G0 G1) (G0 G0 G6) (G0 G0 G7)
                      46.007                                              48.087       45.818    3 4 (G0 G0 G2) (G0 G0 G6) (G0 G0 G7)
                                    31.938                                 48.267       32.044    3 4 (G0 G0 G3) (G0 G0 G6) (G0 G0 G7)
                                                28.835                    48.077       28.934    3 4 (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
                                                              46.681       48.237       46.443    3 4 (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          40.538       39.734       40.637       39.989                                           4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4)
          43.540       35.372       43.132                    35.497                              4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G5)
          46.522       36.693                    46.656       36.883                              4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4) (G0 G0 G5)
          41.551                    35.359       41.382       35.482                              4 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5)
                      41.302       40.839       40.951       40.931                              4 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5)
          38.601       37.573       38.677                                 37.788                 4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G6)
          39.196       41.692                    39.371                    42.069                 4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4) (G0 G0 G6)
          39.194                    46.098       39.083                    45.956                 4 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4) (G0 G0 G6)
                      33.541       41.203       33.486                    41.436                 4 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G6)
          41.140       38.015                                 41.354       37.837                 4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G5) (G0 G0 G6)
          41.764                    42.981                    42.139       43.384                 4 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G5) (G0 G0 G6)
                      44.813       46.952                    45.157       47.063                 4 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G5) (G0 G0 G6)
          42.990                                 42.942       42.790       42.787                 4 4 (G0 G0 G1) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
                      42.439                    41.103       42.451       41.035                 4 4 (G0 G0 G2) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
                                    41.678       42.340       41.546       42.608                 4 4 (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
          46.897       43.268       46.988                                              43.206    4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G7)
          42.473       35.981                    42.221                                 35.803    4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4) (G0 G0 G7)
          39.066                    37.271       38.889                                 37.162    4 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4) (G0 G0 G7)
                      41.392       40.677       41.546                                 40.580    4 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G7)
          38.916       30.582                                 39.062                    30.730    4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G5) (G0 G0 G7)
          43.248                    39.370                    43.099                    39.565    4 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G5) (G0 G0 G7)
                      45.966       34.208                    46.186                    34.160    4 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G5) (G0 G0 G7)
          42.943                                 37.965       43.105                    37.827    4 4 (G0 G0 G1) (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
                      37.814                    29.784       37.870                    29.790    4 4 (G0 G0 G2) (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
                                    38.329       38.749       38.351                    38.800    4 4 (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
          44.992       32.694                                              44.743       32.608    4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G6) (G0 G0 G7)
          39.867                    39.650                                 39.837       39.575    4 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G6) (G0 G0 G7)
                      31.324       30.215                                 31.371       30.228    4 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G6) (G0 G0 G7)
          34.020                                 39.810                    33.860       39.709    4 4 (G0 G0 G1) (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
                      33.420                    33.132                    33.431       33.105    4 4 (G0 G0 G2) (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
                                    31.942       41.954                    32.008       41.790    4 4 (G0 G0 G3) (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
          37.573                                              31.076       37.701       31.144    4 4 (G0 G0 G1) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
                      38.455                                 36.476       38.483       36.316    4 4 (G0 G0 G2) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
                                    45.473                    38.297       45.467       38.204    4 4 (G0 G0 G3) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
                                                44.440       37.996       44.530       38.044    4 4 (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          37.237       44.266       37.207       44.146       37.286                              5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5)
          34.692       45.404       34.637       45.561                    34.513                 5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G6)
          35.046       32.117       34.965                    32.262       35.007                 5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G5) (G0 G0 G6)
          39.664       33.774                    39.592       33.895       39.598                 5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
          32.818                    32.518       32.747       32.515       32.774                 5 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
                      31.579       43.096       31.577       43.457       31.578                 5 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
          40.813       42.963       40.801       43.090                                 40.737    5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G7)
          40.565       34.567       40.630                    34.859                    40.559    5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G5) (G0 G0 G7)
          39.137       32.169                    39.183       32.270                    39.037    5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
          31.289                    34.060       31.225       34.050                    31.250    5 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
                      38.908       42.629       38.936       43.247                    38.947    5 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
          41.545       44.415       41.614                                 44.221       41.622    5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G6) (G0 G0 G7)
          34.760       37.380                    34.741                    37.467       34.541    5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
          28.091                    35.858       28.037                    35.823       28.072    5 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
                      28.942       37.485       28.963                    37.353       28.894    5 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
          32.473       36.354                                 32.466       36.272       32.430    5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          41.725                    37.835                    41.615       37.916       41.462    5 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
                      35.491       45.836                    35.415       45.785       35.436    5 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          44.632                                 38.803       44.496       38.664       44.305    5 4 (G0 G0 G1) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
                      39.944                    44.310       40.085       44.310       39.938    5 4 (G0 G0 G2) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
                                    29.816       36.004       29.770       35.960       29.717    5 4 (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          34.725       35.633       34.708       35.705       34.657       35.797                 6 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
          39.720       37.520       39.526       37.566       39.491                    37.550    6 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
          39.609       41.426       39.536       41.532                    39.521       41.447    6 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
          39.203       33.233       39.339                    33.162       39.220       33.234    6 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          35.246       34.889                    35.226       34.842       35.218       34.841    6 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          41.457                    37.283       41.567       37.332       41.352       37.204    6 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
                      33.003       37.075       33.068       36.900       32.971       36.937    6 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          38.626       41.000       38.632       41.087       38.518       40.911       38.775    7 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)

  .. tab-item:: AMD Instinct™ MI355X

    .. code-block:: shell

      [One-To-All Related]
      NUM_GPU_DEVICES      =            8 : Using 8 GPUs
      NUM_GPU_SE           =            4 : Using 4 subExecutors/CUs per Transfer
      EXE_INDEX            =            0 : Executing on GPU 0
      SWEEP_DIR            =            0 : Direction of transfer
      SWEEP_DST            =            G : DST memory types to sweep
      SWEEP_EXE            =            G : Executor type to use
      SWEEP_MAX            =            8 : Maximum number of peers
      SWEEP_MIN            =            1 : Minimum number of peers
      SWEEP_SRC            =            G : SRC memory types to sweep

      Executing (G0 -> G0 -> G*)
        GPU 1        GPU 2        GPU 3        GPU 4        GPU 5        GPU 6                                                                                                                                                                     GPU 7
      --------------------------------------------------------------------------------                                                                                                                                                             -----------
          57.060                                                                                  1 4 (G0 G0 G1)
                      56.969                                                                     1 4 (G0 G0 G2)
                                    49.018                                                        1 4 (G0 G0 G3)
                                                49.616                                           1 4 (G0 G0 G4)
                                                              56.926                              1 4 (G0 G0 G5)
                                                                          56.751                 1 4 (G0 G0 G6)
                                                                                        49.459    1 4 (G0 G0 G7)
          57.858       55.950                                                                     2 4 (G0 G0 G1) (G0 G0 G2)
          56.203                    56.584                                                        2 4 (G0 G0 G1) (G0 G0 G3)
                      56.249       55.990                                                        2 4 (G0 G0 G2) (G0 G0 G3)
          56.304                                 56.307                                           2 4 (G0 G0 G1) (G0 G0 G4)
                      55.829                    56.026                                           2 4 (G0 G0 G2) (G0 G0 G4)
                                    55.066       55.944                                           2 4 (G0 G0 G3) (G0 G0 G4)
          55.941                                              53.563                              2 4 (G0 G0 G1) (G0 G0 G5)
                      48.896                                 49.449                              2 4 (G0 G0 G2) (G0 G0 G5)
                                    50.291                    50.699                              2 4 (G0 G0 G3) (G0 G0 G5)
                                                49.792       49.264                              2 4 (G0 G0 G4) (G0 G0 G5)
          48.798                                                           49.999                 2 4 (G0 G0 G1) (G0 G0 G6)
                      55.917                                              53.447                 2 4 (G0 G0 G2) (G0 G0 G6)
                                    49.444                                 49.879                 2 4 (G0 G0 G3) (G0 G0 G6)
                                                50.038                    49.559                 2 4 (G0 G0 G4) (G0 G0 G6)
                                                              57.729       56.534                 2 4 (G0 G0 G5) (G0 G0 G6)
          56.182                                                                        55.834    2 4 (G0 G0 G1) (G0 G0 G7)
                      55.878                                                           55.928    2 4 (G0 G0 G2) (G0 G0 G7)
                                    56.481                                              57.752    2 4 (G0 G0 G3) (G0 G0 G7)
                                                49.900                                 49.185    2 4 (G0 G0 G4) (G0 G0 G7)
                                                              55.853                    56.308    2 4 (G0 G0 G5) (G0 G0 G7)
                                                                          56.321       55.775    2 4 (G0 G0 G6) (G0 G0 G7)
          52.080       50.746       51.941                                                        3 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3)
          54.335       54.254                    54.202                                           3 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4)
          49.266                    55.731       49.445                                           3 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4)
                      52.413       55.947       52.325                                           3 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4)
          39.503       54.296                                 39.712                              3 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G5)
          57.383                    56.119                    57.456                              3 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G5)
                      50.184       56.256                    50.205                              3 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G5)
          57.250                                 56.207       57.346                              3 4 (G0 G0 G1) (G0 G0 G4) (G0 G0 G5)
                      49.933                    56.055       49.519                              3 4 (G0 G0 G2) (G0 G0 G4) (G0 G0 G5)
                                    48.265       56.240       48.151                              3 4 (G0 G0 G3) (G0 G0 G4) (G0 G0 G5)
          47.040       50.109                                              47.149                 3 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G6)
          50.567                    56.220                                 50.564                 3 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G6)
                      56.907       56.313                                 56.986                 3 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G6)
          50.609                                 56.264                    50.417                 3 4 (G0 G0 G1) (G0 G0 G4) (G0 G0 G6)
                      56.975                    56.041                    56.826                 3 4 (G0 G0 G2) (G0 G0 G4) (G0 G0 G6)
                                    48.868       56.275                    48.590                 3 4 (G0 G0 G3) (G0 G0 G4) (G0 G0 G6)
          49.474                                              49.799       49.414                 3 4 (G0 G0 G1) (G0 G0 G5) (G0 G0 G6)
                      39.407                                 53.626       39.264                 3 4 (G0 G0 G2) (G0 G0 G5) (G0 G0 G6)
                                    52.668                    51.885       52.746                 3 4 (G0 G0 G3) (G0 G0 G5) (G0 G0 G6)
                                                54.683       50.035       54.503                 3 4 (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
          54.751       51.185                                                           54.714    3 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G7)
          49.464                    56.451                                              49.507    3 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G7)
                      50.542       56.494                                              50.419    3 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G7)
          47.802                                 53.791                                 47.561    3 4 (G0 G0 G1) (G0 G0 G4) (G0 G0 G7)
                      47.249                    52.755                                 47.091    3 4 (G0 G0 G2) (G0 G0 G4) (G0 G0 G7)
                                    41.682       55.054                                 41.609    3 4 (G0 G0 G3) (G0 G0 G4) (G0 G0 G7)
          53.857                                              50.240                    53.689    3 4 (G0 G0 G1) (G0 G0 G5) (G0 G0 G7)
                      46.694                                 49.802                    46.467    3 4 (G0 G0 G2) (G0 G0 G5) (G0 G0 G7)
                                    52.817                    49.695                    52.708    3 4 (G0 G0 G3) (G0 G0 G5) (G0 G0 G7)
                                                42.766       49.378                    42.681    3 4 (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
          47.020                                                           50.272       46.866    3 4 (G0 G0 G1) (G0 G0 G6) (G0 G0 G7)
                      51.293                                              50.344       51.281    3 4 (G0 G0 G2) (G0 G0 G6) (G0 G0 G7)
                                    52.745                                 50.363       52.573    3 4 (G0 G0 G3) (G0 G0 G6) (G0 G0 G7)
                                                43.464                    50.005       43.378    3 4 (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
                                                              52.110       53.252       52.204    3 4 (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          53.978       53.951       53.909       53.994                                           4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4)
          52.088       48.838       52.174                    48.706                              4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G5)
          54.746       51.347                    54.722       51.213                              4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4) (G0 G0 G5)
          53.295                    54.767       53.528       54.685                              4 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5)
                      50.468       48.308       50.462       47.927                              4 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5)
          50.893       46.216       50.966                                 46.051                 4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G6)
          52.775       43.437                    52.870                    43.390                 4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4) (G0 G0 G6)
          51.347                    47.597       51.299                    47.533                 4 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4) (G0 G0 G6)
                      54.851       54.193       54.852                    54.315                 4 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G6)
          52.597       53.273                                 52.389       53.026                 4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G5) (G0 G0 G6)
          49.185                    51.880                    49.343       51.712                 4 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G5) (G0 G0 G6)
                      50.603       56.058                    50.795       55.960                 4 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G5) (G0 G0 G6)
          49.493                                 53.818       49.462       53.614                 4 4 (G0 G0 G1) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
                      50.473                    52.841       50.388       52.713                 4 4 (G0 G0 G2) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
                                    55.233       53.448       54.880       53.259                 4 4 (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
          53.965       53.219       54.128                                              53.233    4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G7)
          48.949       50.712                    48.946                                 50.613    4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4) (G0 G0 G7)
          52.486                    47.821       52.730                                 47.730    4 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4) (G0 G0 G7)
                      51.232       49.069       51.309                                 48.869    4 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G7)
          49.876       51.404                                 49.772                    51.046    4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G5) (G0 G0 G7)
          57.132                    57.070                    56.963                    56.772    4 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G5) (G0 G0 G7)
                      49.970       57.176                    49.987                    56.920    4 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G5) (G0 G0 G7)
          57.333                                 49.658       57.264                    49.806    4 4 (G0 G0 G1) (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
                      50.165                    49.903       50.134                    49.860    4 4 (G0 G0 G2) (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
                                    52.488       51.273       52.639                    51.069    4 4 (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
          51.169       54.829                                              51.031       54.709    4 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G6) (G0 G0 G7)
          50.695                    57.240                                 50.471       56.931    4 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G6) (G0 G0 G7)
                      56.892       57.171                                 56.747       57.028    4 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G6) (G0 G0 G7)
          50.567                                 49.730                    50.262       49.642    4 4 (G0 G0 G1) (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
                      56.972                    49.999                    56.850       49.764    4 4 (G0 G0 G2) (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
                                    55.711       51.511                    55.656       51.235    4 4 (G0 G0 G3) (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
          51.660                                              54.717       51.631       54.697    4 4 (G0 G0 G1) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
                      48.339                                 50.612       48.182       50.660    4 4 (G0 G0 G2) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
                                    54.962                    54.106       54.762       53.969    4 4 (G0 G0 G3) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
                                                49.339       51.046       49.435       50.976    4 4 (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          44.784       52.269       44.716       52.113       44.546                              5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5)
          51.310       52.573       51.138       52.632                    51.216                 5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G6)
          53.244       47.458       53.128                    47.570       53.279                 5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G5) (G0 G0 G6)
          53.537       49.462                    53.431       49.427       53.541                 5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
          47.703                    56.413       47.796       56.427       47.780                 5 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
                      47.025       53.682       46.996       53.527       47.114                 5 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
          44.808       52.363       44.935       52.466                                 44.864    5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G7)
          53.774       44.200       53.745                    44.146                    53.889    5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G5) (G0 G0 G7)
          50.409       42.969                    50.554       42.820                    50.407    5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
          46.721                    55.426       46.727       55.217                    46.646    5 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
                      49.917       52.813       50.019       52.586                    49.718    5 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
          49.463       50.373       49.695                                 50.244       49.436    5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G6) (G0 G0 G7)
          49.394       50.794                    49.331                    50.565       49.373    5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
          47.873                    51.213       47.900                    51.305       47.921    5 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
                      47.109       51.965       47.182                    51.776       47.153    5 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
          50.039       54.672                                 50.159       54.760       50.229    5 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          46.918                    52.488                    47.028       52.327       47.033    5 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
                      49.807       52.877                    50.009       52.756       49.904    5 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          48.313                                 54.666       48.258       54.596       48.103    5 4 (G0 G0 G1) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
                      47.352                    52.476       47.680       52.375       47.412    5 4 (G0 G0 G2) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
                                    45.647       51.850       45.700       51.787       45.618    5 4 (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          53.797       53.041       53.715       53.185       53.728       53.055                 6 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6)
          50.819       49.056       50.912       49.257       50.800                    49.082    6 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G7)
          53.691       53.287       53.672       53.443                    53.601       53.312    6 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G6) (G0 G0 G7)
          51.184       51.978       51.156                    51.922       51.261       51.993    6 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          52.548       50.879                    52.511       51.038       52.776       50.962    6 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          52.010                    52.226       51.881       52.229       51.977       52.150    6 4 (G0 G0 G1) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
                      49.444       48.838       49.543       48.895       49.396       48.811    6 4 (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)
          48.520       53.242       48.504       53.057       48.517       53.075       48.642    7 4 (G0 G0 G1) (G0 G0 G2) (G0 G0 G3) (G0 G0 G4) (G0 G0 G5) (G0 G0 G6) (G0 G0 G7)

.. _p2p:

Peer-to-peer preset (p2p)
==========================

The p2p preset measures device memory bandwidth between all pairs of CPU NUMA nodes and GPUs. It tests unidirectional and bidirectional transfers for CPU-to-CPU, CPU-to-GPU, and GPU-to-GPU combinations.

**Key features:**

- Tests all SRC-to-DST pairs across CPUs and GPUs.

- Supports both unidirectional and bidirectional transfers (``P2P_MODE``).

- Uses GFX or DMA as GPU Executor (``USE_GPU_DMA``).

- Supports remote read (DST GPU as Executor) instead of source-side execution.

- Prints bandwidth matrix with row and column labels. Optionally shows min/max/stddev per iteration.


**Restrictions:**

- Single-node only.

- ``USE_FINE_GRAIN`` is deprecated: Returns error if ``USE_FINE_GRAIN`` is set. Use ``CPU_MEM_TYPE`` and ``GPU_MEM_TYPE`` instead.

- **NVIDIA platforms:** CPU executors can't access GPU memory; those pairs are skipped.

- Self-transfers skipped: CPU i-to-i and GPU i-to-i are skipped in bidirectional mode.

**Usage:**

.. code-block:: shell

  ./TransferBench p2p

For exclusively unidirectional transfer with DMA:

.. code-block:: shell

  P2P_MODE=1 USE_GPU_DMA=1 ./TransferBench p2p

Environment variables
----------------------

To modify the behavior of p2p preset, use the following environment variables:

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``CPU_MEM_TYPE``
      - CPU memory: 0=default, 1=coherent, 2=non-coherent, 3=uncached, 4=unpinned. See :ref:`mem-type`.
      - ``0``

    * - ``GPU_MEM_TYPE``
      - GPU memory: 0=default, 1=fine-grained, 2=uncached, 3=managed. See :ref:`mem-type`.
      - ``0``

    * - ``NUM_CPU_DEVICES``
      - Number of CPU NUMA nodes. To avoid using any pairs involving CPUs, set it to ``0``.
      - (all detected)

    * - ``NUM_CPU_SE``
      - CPU threads per CPU-executed transfer.
      - ``4``

    * - ``NUM_GPU_DEVICES``
      - Number of GPUs. This can be modified to reduce the number of GPUs to test.
      - (all detected)

    * - ``NUM_GPU_SE``
      - GPU CUs per transfer. Default value varies according to ``USE_GPU_DMA``.
      - (device max / GFX default)

    * - ``SHOW_ITERATIONS``
      - To show detailed min/max/stddev per iteration, set to ``1``.
      - ``0``

    * - ``P2P_MODE``
      - 1=Unidirectional only, 2=Bidirectional only, 0=both.
      - ``0``

    * - ``USE_GPU_DMA``
      - To use DMA for GPU Executor, set to ``1``. To use GFX, set to ``0``.
      - ``0``

    * - ``USE_REMOTE_READ``
      - To place the Executor on DST, set to ``1``. To place on SRC, set to ``0``.
      - ``0``

Example output
---------------

.. tab-set::

  .. tab-item:: AMD Instinct MI300X

    .. code-block:: shell

      [P2P Related]
      CPU_MEM_TYPE         =            0 : Using default CPU (0=default, 1=coherent, 2=non-coherent, 3=uncached, 4=unpinned)
      GPU_MEM_TYPE         =            0 : Using default GPU (0=default, 1=fine-grained, 2=uncached, 3=managed)
      NUM_CPU_DEVICES      =            2 : Using 2 CPUs
      NUM_CPU_SE           =            4 : Using 4 CPU threads per Transfer
      NUM_GPU_DEVICES      =            8 : Using 8 GPUs
      NUM_GPU_SE           =          304 : Using 304 GPU subexecutors/CUs per Transfer
      P2P_MODE             =            0 : Running Uni + Bi transfers
      USE_GPU_DMA          =            0 : Using GPU-GFX as GPU executor
      USE_REMOTE_READ      =            0 : Using SRC as executor
      Bytes Per Direction 268435456
      Unidirectional copy peak bandwidth GB/s [Local read / Remote write] (GPU-Executor: GFX)
      SRC+EXE\DST    CPU 00    CPU 01       GPU 00    GPU 01    GPU 02    GPU 03    GPU 04    GPU 05    GPU 06    GPU 07
        CPU 00  ->     37.62     38.04        39.44     34.00     33.12     35.53     31.90     29.73     28.11     31.00
        CPU 01  ->     37.84     37.69        29.92     29.85     31.19     29.63     38.99     38.41     38.32     39.56
        GPU 00  ->     55.36     55.25      1618.87     48.83     48.89     49.00     48.05     47.94     48.27     47.85
        GPU 01  ->     55.36     54.14        48.89   1860.47     48.95     48.95     47.91     48.04     48.49     48.32
        GPU 02  ->     55.35     55.26        48.83     49.01   1868.43     49.07     48.70     48.34     48.85     48.97
        GPU 03  ->     55.34     55.26        49.01     49.02     49.07   1877.42     48.51     48.17     48.85     49.04
        GPU 04  ->     55.30     55.38        47.95     48.26     48.85     48.61   1849.65     48.99     48.85     48.84
        GPU 05  ->     55.29     55.35        47.95     48.02     48.51     48.03     49.01   1853.87     49.15     49.01
        GPU 06  ->     55.32     55.34        48.31     48.62     48.88     48.94     48.99     48.83   1829.05     49.17
        GPU 07  ->     55.30     55.34        48.23     48.27     48.59     48.90     48.60     49.09     49.14   1841.42
                                CPU->CPU  CPU->GPU  GPU->CPU  GPU->GPU
      Averages (During UniDir):     37.94     33.67     55.25     48.65
      Bidirectional copy peak bandwidth GB/s [Local read / Remote write] (GPU-Executor: GFX)
          SRC\DST    CPU 00    CPU 01       GPU 00    GPU 01    GPU 02    GPU 03    GPU 04    GPU 05    GPU 06    GPU 07
        CPU 00  ->       N/A     33.59        33.54     36.84     33.35     35.02     29.55     31.33     31.30     28.09
        CPU 00 <-        N/A     39.94        54.81     54.73     54.51     54.48     29.25     28.84     28.13     30.44
        CPU 00 <->       N/A     73.52        88.35     91.57     87.86     89.51     58.80     60.17     59.43     58.53
        CPU 01  ->     36.21       N/A        31.09     28.54     31.93     31.76     38.02     38.74     37.19     36.09
        CPU 01 <-      33.60       N/A        28.85     28.27     27.93     28.54     54.85     54.80     54.68     54.70
        CPU 01 <->     69.81       N/A        59.94     56.81     59.86     60.30     92.87     93.54     91.86     90.78
        GPU 00  ->     54.77     29.18          N/A     46.15     46.10     46.55     46.16     46.05     46.31     45.95
        GPU 00 <-      34.70     30.98          N/A     46.21     46.40     46.65     46.12     46.00     46.25     45.98
        GPU 00 <->     89.47     60.15          N/A     92.36     92.50     93.20     92.27     92.05     92.56     91.93
        GPU 01  ->     54.77     29.18        46.19       N/A     46.08     46.54     46.17     46.05     46.33     46.14
        GPU 01 <-      32.11     30.59        46.11       N/A     46.64     46.42     46.16     46.09     46.51     46.20
        GPU 01 <->     86.89     59.77        92.30       N/A     92.73     92.97     92.32     92.14     92.84     92.33
        GPU 02  ->     54.76     29.56        46.40     46.63       N/A     46.62     46.49     46.16     46.41     46.09
        GPU 02 <-      32.05     27.70        46.07     46.05       N/A     46.24     46.18     46.26     46.12     46.27
        GPU 02 <->     86.81     57.25        92.47     92.68       N/A     92.86     92.67     92.42     92.53     92.37
        GPU 03  ->     54.73     30.33        46.62     46.44     46.23       N/A     46.15     46.34     46.25     46.47
        GPU 03 <-      33.13     29.77        46.50     46.52     46.61       N/A     46.17     46.22     46.23     46.46
        GPU 03 <->     87.86     60.10        93.13     92.96     92.84       N/A     92.32     92.56     92.48     92.93
        GPU 04  ->     29.91     54.85        46.18     46.20     46.21     46.17       N/A     46.56     46.23     46.50
        GPU 04 <-      30.60     34.45        46.27     46.37     46.58     46.17       N/A     46.49     46.25     46.44
        GPU 04 <->     60.52     89.30        92.45     92.57     92.78     92.34       N/A     93.05     92.49     92.93
        GPU 05  ->     30.58     54.76        45.99     46.04     46.24     46.32     46.51       N/A     46.38     46.15
        GPU 05 <-      26.98     35.95        46.00     46.01     46.18     46.38     46.56       N/A     46.26     46.20
        GPU 05 <->     57.55     90.70        91.99     92.05     92.43     92.69     93.07       N/A     92.63     92.36
        GPU 06  ->     30.22     54.65        46.34     46.40     46.13     46.24     46.26     46.33       N/A     46.43
        GPU 06 <-      27.72     35.78        46.37     46.35     46.35     46.28     46.25     46.37       N/A     46.30
        GPU 06 <->     57.94     90.44        92.72     92.75     92.48     92.52     92.51     92.70       N/A     92.73
        GPU 07  ->     30.55     54.66        46.03     46.15     46.35     46.38     46.39     46.17     46.35       N/A
        GPU 07 <-      27.28     36.17        46.05     46.11     46.12     46.45     46.48     46.15     46.41       N/A
        GPU 07 <->     57.83     90.83        92.08     92.26     92.47     92.83     92.87     92.32     92.76       N/A
                                CPU->CPU  CPU->GPU  GPU->CPU  GPU->GPU
      Averages (During  BiDir):     35.83     37.51     36.98     46.28

  .. tab-item:: AMD Instinct MI350X

    .. code-block:: shell

      [P2P Related]
      CPU_MEM_TYPE         =            0 : Using default CPU (0=default, 1=coherent, 2=non-coherent, 3=uncached, 4=unpinned)
      GPU_MEM_TYPE         =            0 : Using default GPU (0=default, 1=fine-grained, 2=uncached, 3=managed)
      NUM_CPU_DEVICES      =            2 : Using 2 CPUs
      NUM_CPU_SE           =            4 : Using 4 CPU threads per Transfer
      NUM_GPU_DEVICES      =            8 : Using 8 GPUs
      NUM_GPU_SE           =          256 : Using 256 GPU subexecutors/CUs per Transfer
      P2P_MODE             =            0 : Running Uni + Bi transfers
      USE_GPU_DMA          =            0 : Using GPU-GFX as GPU executor
      USE_REMOTE_READ      =            0 : Using SRC as executor
      Bytes Per Direction 268435456
      Unidirectional copy peak bandwidth GB/s [Local read / Remote write] (GPU-Executor: GFX)
      SRC+EXE\DST    CPU 00    CPU 01       GPU 00    GPU 01    GPU 02    GPU 03    GPU 04    GPU 05    GPU 06    GPU 07
        CPU 00  ->     83.89     93.99        42.90     42.89     42.94     42.93     42.90     42.88     41.76     42.81
        CPU 01  ->     91.09     83.25        42.77     42.84     42.27     42.88     42.79     42.91     42.83     42.79
        GPU 00  ->     53.18     53.14      2285.12     57.51     57.46     57.38     57.33     57.32     57.28     57.64
        GPU 01  ->     53.11     53.16        57.53   2280.83     57.36     57.30     57.32     57.33     57.48     57.44
        GPU 02  ->     53.11     53.13        57.45     57.29   2286.68     57.36     57.58     57.53     57.38     57.35
        GPU 03  ->     53.19     53.11        57.31     57.26     57.52   2281.59     57.52     57.47     57.33     57.38
        GPU 04  ->     53.11     53.12        57.33     57.27     57.57     57.53   2292.99     57.51     57.36     57.36
        GPU 05  ->     53.13     53.13        57.34     57.32     57.55     57.48     57.28   2276.23     57.42     57.50
        GPU 06  ->     53.18     53.19        57.28     57.47     57.39     57.35     57.54     57.40   2305.57     57.49
        GPU 07  ->     53.16     53.15        57.44     57.47     57.35     57.36     57.32     57.35     57.51   2289.74
                                CPU->CPU  CPU->GPU  GPU->CPU  GPU->GPU
      Averages (During UniDir):     92.54     42.76     53.14     57.41
      Bidirectional copy peak bandwidth GB/s [Local read / Remote write] (GPU-Executor: GFX)
          SRC\DST    CPU 00    CPU 01       GPU 00    GPU 01    GPU 02    GPU 03    GPU 04    GPU 05    GPU 06    GPU 07
        CPU 00  ->       N/A     79.71        42.40     42.40     42.39     42.51     42.05     41.14     42.22     42.45
        CPU 00 <-        N/A     80.90        52.72     52.61     52.74     52.69     52.69     52.68     52.61     52.64
        CPU 00 <->       N/A    160.62        95.11     95.01     95.13     95.20     94.75     93.82     94.83     95.09
        CPU 01  ->     80.77       N/A        42.27     42.39     42.50     42.49     42.50     42.47     42.43     42.46
        CPU 01 <-      79.50       N/A        52.68     52.60     52.69     52.66     52.68     52.65     52.68     52.68
        CPU 01 <->    160.27       N/A        94.95     94.99     95.19     95.15     95.17     95.11     95.11     95.14
        GPU 00  ->     52.72     52.61          N/A     54.77     54.78     54.61     54.66     54.58     54.51     54.85
        GPU 00 <-      42.48     42.34          N/A     54.77     54.72     54.52     54.58     54.53     54.57     54.72
        GPU 00 <->     95.20     94.95          N/A    109.54    109.51    109.13    109.23    109.11    109.08    109.57
        GPU 01  ->     52.68     52.69        54.75       N/A     54.66     54.51     54.54     54.57     54.74     54.70
        GPU 01 <-      42.43     42.40        54.84       N/A     54.46     54.55     54.45     54.61     54.82     54.79
        GPU 01 <->     95.11     95.09       109.59       N/A    109.12    109.06    108.99    109.18    109.56    109.50
        GPU 02  ->     52.72     52.59        54.80     54.52       N/A     54.62     54.87     54.86     54.64     54.53
        GPU 02 <-      42.48     42.36        54.80     54.62       N/A     54.71     54.79     54.75     54.59     54.56
        GPU 02 <->     95.20     94.94       109.60    109.15       N/A    109.33    109.66    109.61    109.23    109.09
        GPU 03  ->     52.61     52.59        54.43     54.52     54.64       N/A     54.80     54.82     54.61     54.59
        GPU 03 <-      42.49     42.38        54.63     54.53     54.63       N/A     54.79     54.73     54.47     54.49
        GPU 03 <->     95.09     94.97       109.06    109.05    109.28       N/A    109.59    109.56    109.08    109.08
        GPU 04  ->     52.69     52.59        54.56     54.50     54.74     54.76       N/A     54.75     54.57     54.64
        GPU 04 <-      41.98     42.47        54.66     54.53     54.82     54.81       N/A     54.56     54.74     54.53
        GPU 04 <->     94.67     95.06       109.22    109.03    109.56    109.57       N/A    109.31    109.31    109.17
        GPU 05  ->     52.71     52.58        54.54     54.56     54.78     54.71     54.55       N/A     54.59     54.73
        GPU 05 <-      42.33     42.36        54.59     54.58     54.85     54.83     54.74       N/A     54.50     54.68
        GPU 05 <->     95.04     94.94       109.13    109.14    109.64    109.55    109.29       N/A    109.09    109.41
        GPU 06  ->     52.64     52.70        54.56     54.82     54.63     54.53     54.61     54.59       N/A     54.82
        GPU 06 <-      42.37     42.52        54.53     54.83     54.66     54.56     54.60     54.59       N/A     54.75
        GPU 06 <->     95.02     95.22       109.10    109.65    109.28    109.09    109.21    109.18       N/A    109.57
        GPU 07  ->     52.70     52.66        54.70     54.84     54.58     54.53     54.50     54.68     54.83       N/A
        GPU 07 <-      42.16     42.45        54.88     54.72     54.55     54.63     54.61     54.73     54.73       N/A
        GPU 07 <->     94.85     95.11       109.58    109.56    109.12    109.16    109.11    109.41    109.56       N/A
                                CPU->CPU  CPU->GPU  GPU->CPU  GPU->GPU
      Averages (During  BiDir):     80.22     47.49     47.51     54.66

.. _scaling:

Scaling preset (scaling)
=========================

The scaling preset runs a scaling test from one GPU to all other devices (CPUs and GPUs). It varies the number of SubExecutors (CUs) from SWEEP_MIN to SWEEP_MAX and reports bandwidth for each target device. It helps find optimal CU count per transfer.

**Key feature:**

- Uses one GPU (``LOCAL_IDX``) as source.

- Runs one transfer per target at a time (one SRC to one DST per cell).

- Copies to each CPU NUMA node and every other GPU.

- For each CU count (``SWEEP_MIN`` to ``SWEEP_MAX``), runs one transfer per target and reports bandwidth.

- Prints a table: rows = CU count, columns = target device.

- Adds a ``Best`` row to the output showing peak bandwidth and optimal CU count per target.

**Restrictions:**

- Single-node only.

- ``USE_FINE_GRAIN`` is deprecated: Returns error if set. Use ``CPU_MEM_TYPE`` and ``GPU_MEM_TYPE`` instead.

**Usage:**

.. code-block:: shell

  ./TransferBench scaling

To run using GPU 2 as SRC with CU range between 4 and 64:

.. code-block:: shell

  LOCAL_IDX=2 SWEEP_MIN=4 SWEEP_MAX=64 ./TransferBench scaling

Environment variables
----------------------

To modify the behavior of scaling preset, use the following environment variables:

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``CPU_MEM_TYPE``
      - CPU memory type: 0=default, 1=coherent, 2=non-coherent, 3=uncached, 4=unpinned. See :ref:`mem-type`.
      - ``0``

    * - ``GPU_MEM_TYPE``
      - GPU memory type: 0=default, 1=fine-grained, 2=uncached, 3=managed. See :ref:`mem-type`.
      - ``0``

    * - ``LOCAL_IDX``
      - Index of the GPU performing copy to other GPUs.
      - ``0``

    * - ``NUM_CPU_DEVICES``
      - Number of CPU NUMA nodes.
      - (all detected)

    * - ``NUM_GPU_DEVICES``
      - Number of GPUs.
      - (all detected)

    * - ``SWEEP_MIN``
      - Minimum SubExecutors (CUs).
      - ``1``

    * - ``SWEEP_MAX``
      - Maximum SubExecutors.
      - ``32``

Example output
---------------

.. tab-set::

  .. tab-item:: AMD Instinct MI300X

    .. code-block:: shell

      [Scaling Related]
      CPU_MEM_TYPE         =            0 : Using default CPU (0=default, 1=coherent, 2=non-coherent, 3=uncached, 4=unpinned)
      GPU_MEM_TYPE         =            0 : Using default GPU (0=default, 1=fine-grained, 2=uncached, 3=managed)
      LOCAL_IDX            =            0 : Local GPU index
      NUM_CPU_DEVICES      =            2 : Using 2 CPUs
      NUM_GPU_DEVICES      =            8 : Using 8 GPUs
      SWEEP_MAX            =           32 : Max number of subExecutors to use
      SWEEP_MIN            =            1 : Min number of subExecutors to use
      GPU-GFX Scaling benchmark:
      ==========================
      - Copying 268435456 bytes from GPU 0 to other devices
      - All numbers reported as GB/sec
      NumCUs   CPU00        CPU01        GPU00        GPU01        GPU02        GPU03        GPU04        GPU05        GPU06        GPU07
        1     20.22        20.41        18.68        25.91        26.08        26.06        25.95        26.04        26.01        26.02
        2     37.37        37.03        36.88        48.65        48.33        49.24        47.91        47.72        48.48        47.13
        3     52.96        51.92        55.74        48.92        48.43        49.22        47.45        47.47        48.11        47.88
        4     56.38        53.18        73.05        49.41        49.19        49.34        47.84        47.79        48.46        48.05
        5     54.61        52.96        91.57        45.23        44.60        49.03        47.73        44.32        48.57        44.11
        6     54.61        53.78       109.70        48.98        48.82        49.13        47.84        47.46        48.38        48.19
        7     56.48        54.27       127.43        49.15        49.14        49.15        48.00        47.85        48.50        48.05
        8     56.60        54.71       142.35        49.14        49.37        49.31        47.86        47.98        48.39        48.13
        9     56.66        54.93       161.43        49.13        49.58        49.16        47.77        47.92        48.44        48.18
        10     56.83        55.31       178.08        49.17        49.33        49.07        47.79        47.99        48.33        48.23
        11     56.84        55.63       195.82        49.36        49.43        49.10        47.56        47.96        48.54        48.37
        12     57.12        55.83       210.39        49.50        48.97        49.43        47.97        47.73        48.63        48.27
        13     56.91        55.65       226.79        49.52        48.86        49.22        47.63        47.92        48.60        48.16
        14     57.10        55.83       238.49        49.26        49.42        49.13        48.08        48.18        48.44        48.18
        15     57.09        55.86       258.25        49.23        49.19        49.42        47.74        47.96        48.68        48.11
        16     57.11        55.98       271.55        49.62        49.25        49.54        47.84        47.75        48.39        47.93
        17     57.10        55.82       287.98        49.10        49.36        49.35        47.64        47.97        48.81        48.28
        18     57.10        55.81       306.06        49.33        49.14        49.34        47.81        47.99        48.47        48.05
        19     56.94        55.69       319.71        49.20        49.14        49.32        48.13        47.93        48.61        48.30
        20     57.14        55.88       334.89        49.35        49.25        49.22        48.19        47.97        48.62        48.24
        21     57.12        55.94       346.59        49.13        49.23        49.19        48.24        47.84        48.52        48.16
        22     57.13        56.01       362.42        49.34        49.39        49.09        47.95        48.00        48.53        48.20
        23     57.13        56.17       375.70        49.10        49.22        49.43        47.98        48.14        48.58        48.46
        24     57.14        56.23       388.97        49.25        49.24        49.52        47.72        48.06        48.67        48.31
        25     57.14        56.30       403.32        49.04        49.20        49.42        48.05        48.01        48.51        47.97
        26     57.14        56.17       417.88        49.57        49.59        49.57        47.89        48.04        48.79        48.34
        27     57.12        56.02       426.76        49.32        49.24        49.29        48.14        48.01        48.50        48.04
        28     57.13        56.05       444.58        49.31        49.37        49.12        48.00        47.96        48.44        47.99
        29     57.14        56.07       453.05        49.55        49.40        49.56        48.16        47.78        48.18        48.17
        30     57.14        56.12       462.74        49.11        49.27        49.33        47.97        48.20        48.63        48.26
        31     57.13        56.12       478.60        49.35        48.96        49.06        47.94        48.33        48.43        48.35
        32     57.15        56.35       493.17        49.23        49.55        49.33        47.77        48.28        48.56        48.22
      Best    57.15( 32)   56.35( 32)  493.17( 32)   49.62( 16)   49.59( 26)   49.57( 26)   48.24( 21)   48.33( 31)   48.81( 17)   48.46( 23)

  .. tab-item:: AMD Instinct MI350X

    .. code-block:: shell

      [Scaling Related]
      CPU_MEM_TYPE         =            0 : Using default CPU (0=default, 1=coherent, 2=non-coherent, 3=uncached, 4=unpinned)
      GPU_MEM_TYPE         =            0 : Using default GPU (0=default, 1=fine-grained, 2=uncached, 3=managed)
      LOCAL_IDX            =            0 : Local GPU index
      NUM_CPU_DEVICES      =            2 : Using 2 CPUs
      NUM_GPU_DEVICES      =            8 : Using 8 GPUs
      SWEEP_MAX            =           32 : Max number of subExecutors to use
      SWEEP_MIN            =            1 : Min number of subExecutors to use
      GPU-GFX Scaling benchmark:
      ==========================
      - Copying 268435456 bytes from GPU 0 to other devices
      - All numbers reported as GB/sec
      NumCUs   CPU00        CPU01        GPU00        GPU01        GPU02        GPU03        GPU04        GPU05        GPU06        GPU07
        1     26.51        26.30        15.81        26.48        26.57        26.44        25.68        26.39        26.58        26.04
        2     51.52        50.86        31.50        52.65        52.28        52.28        52.00        52.57        52.95        52.39
        3     42.83        43.01        46.13        53.32        57.39        55.81        49.08        57.41        49.65        48.31
        4     50.02        49.93        61.77        57.34        57.09        49.10        49.67        57.02        56.89        49.79
        5     53.58        53.58        77.07        55.85        57.78        57.22        50.69        57.72        54.80        50.27
        6     53.84        53.82        91.73        58.29        58.48        56.60        54.91        58.41        57.81        54.89
        7     53.56        53.63       106.60        57.98        57.24        57.18        55.86        56.97        57.79        55.87
        8     53.22        52.97       121.07        58.40        58.27        57.43        58.07        58.17        58.07        58.39
        9     54.22        54.22       135.97        58.37        57.88        57.54        58.10        57.72        58.21        58.26
        10     54.37        54.34       148.80        58.61        58.63        57.74        58.49        58.63        58.32        58.35
        11     54.62        54.58       163.28        57.83        58.55        58.26        58.17        58.53        57.94        58.38
        12     54.63        54.56       177.99        58.93        58.69        58.59        58.49        58.68        58.57        58.64
        13     54.66        54.69       191.79        58.55        58.51        58.59        58.24        58.50        58.42        58.33
        14     54.73        54.63       205.97        58.73        58.49        58.36        58.28        58.40        58.51        58.30
        15     54.73        54.64       221.70        58.65        58.55        58.41        58.44        58.61        58.56        58.43
        16     54.63        54.59       233.49        59.14        59.04        58.84        58.93        59.03        58.98        59.08
        17     54.75        54.76       247.85        58.78        58.56        58.43        58.55        58.61        58.62        58.48
        18     54.74        54.73       262.07        58.70        58.42        58.34        58.33        58.37        58.48        58.40
        19     54.77        54.73       274.98        58.57        58.42        58.38        58.37        58.55        58.40        58.33
        20     54.82        54.86       287.02        58.76        58.77        58.58        58.62        58.67        58.58        58.69
        21     54.79        54.76       301.35        58.62        58.48        58.38        58.40        58.45        58.47        58.38
        22     54.74        54.72       313.96        58.59        58.56        58.43        58.43        58.45        58.56        58.42
        23     54.79        54.78       328.28        58.55        58.53        58.41        58.38        58.48        58.41        58.34
        24     54.65        54.73       343.28        58.76        59.02        58.68        58.78        58.68        59.01        58.76
        25     54.72        54.78       354.62        58.57        58.50        58.38        58.42        58.39        58.50        58.41
        26     54.67        54.71       367.90        58.58        58.51        58.54        58.43        58.46        58.52        58.55
        27     54.74        54.73       377.03        58.52        58.41        58.26        58.31        58.45        58.39        58.36
        28     54.67        54.73       393.19        58.69        58.36        58.32        58.40        58.44        58.46        58.41
        29     54.72        54.71       402.84        58.50        58.31        58.26        58.33        58.36        58.48        58.35
        30     54.75        54.79       418.54        58.82        58.52        58.37        58.39        58.67        58.52        58.46
        31     54.79        54.75       429.11        58.65        58.33        58.33        58.55        58.35        58.41        58.41
        32     54.74        54.79       445.36        59.08        59.12        58.85        58.81        59.02        59.13        59.11
      Best    54.82( 20)   54.86( 20)  445.36( 32)   59.14( 16)   59.12( 32)   58.85( 32)   58.93( 16)   59.03( 16)   59.13( 32)   59.11( 32)

.. _schmoo:

Schmoo preset (schmoo)
=======================

The schmoo preset runs scaling tests for local and remote read, write, and copy operations between two GPUs. For each CU count (``SWEEP_MIN`` to ``SWEEP_MAX``), it measures six bandwidth values: Local Read, Local Write, Local Copy, Remote Read, Remote Write, and Remote Copy.

**Key features:**

- Minimum 2 GPUs: Requires at least two GPUs: ``LOCAL_IDX`` (local) and ``REMOTE_IDX`` (remote).

- Fixed topology: Always two GPUs (local and remote). No sweep over device count.

- For each CU count, runs the following six tests. Each test measures bandwidth for the corresponding operation pattern:

  - Local Read: Local GPU reads from local memory (SRC->G->null).

  - Local Write: Local GPU writes to local memory (null->G->DST).

  - Local Copy: Local GPU copies (local->local).

  - Remote Read: Local GPU reads from remote memory.

  - Remote Write: Local GPU writes to remote memory.

  - Remote Copy: Local GPU copies (local->remote).

- Outputs a table: rows = #CUs, columns = the 6 operation types.

- Supports single node only: Multinode is not supported.

**Usage:**

.. code-block:: shell

  ./TransferBench schmoo

To run using GPUs 0 and 3:

.. code-block:: shell

  LOCAL_IDX=0 REMOTE_IDX=3 SWEEP_MIN=4 SWEEP_MAX=32 ./TransferBench schmoo

To run using fine-grained memory:

.. code-block:: shell

  USE_FINE_GRAIN=1 ./TransferBench schmoo

Environment variables
----------------------

To modify the behavior of schmoo preset, use the following environment variables:

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``LOCAL_IDX``
      - Local GPU index.
      - ``0``

    * - ``REMOTE_IDX``
      - Remote GPU index.
      - ``1``

    * - ``SWEEP_MIN``
      - Minimum CUs.
      - ``1``

    * - ``SWEEP_MAX``
      - Maximum CUs.
      - ``32``

    * - ``USE_FINE_GRAIN``
      - To use fine-grained GPU memory, set to ``1``. For coarse-grained memory, set to ``0``.
      - ``0``

Example output
---------------

.. tab-set::

  .. tab-item:: AMD Instinct MI300X

    .. image:: /data/schmoo_MI300X.png
      :width: 100%
      :align: center

  .. tab-item:: AMD Instinct MI350X

    .. image:: /data/schmoo_MI350X.png
      :width: 100%
      :align: center

.. _sweep:

Sweep (sweep) and random sweep preset (rsweep)
===============================================

The sweep preset performs an ordered sweep through sets of transfers. It systematically tests combinations of (SRC, Executor, DST) with varying parallelism (from ``SWEEP_MIN`` simultaneous transfers up to ``SWEEP_MAX``) using lexicographic order (alphabetized by source-executor-destination triplet). The rsweep preset performs the same sweep in a random order.

.. note::

  This preset is primarily used for stress testing.

**Key features:**

- **Test set construction:** Builds all possible triplets (SRC, EXE, DST) from ``SWEEP_SRC``, ``SWEEP_EXE``, ``SWEEP_DST``, and device counts, as a Cartesian product (``srcList`` x ``exeList`` x ``dstList``) with filters such as XGMI hop count and CPU-on-GPU skip on NVIDIA.

- Optionally filters using XGMI hop count (``SWEEP_XGMI_MIN``, ``SWEEP_XGMI_MAX``).

- **Parallelism sweep:** Starts at ``SWEEP_MIN`` simultaneous transfers, exhausts all combinations at that count, then increments up to ``SWEEP_MAX`` (set to ``0`` for no limit).

- Ordered permutation: Uses ``std::prev_permutation`` to iterate through M-combinations of the possible transfer set in a deterministic order.

- Log format: Logs each test's transfers to ``SWEEP_FILE``. The ``SWEEP_FILE`` contains lines such as "# Test N" and "-M (src->exe->dst CUs bytes)...".

- Follows ``SWEEP_TEST_LIMIT`` and ``SWEEP_TIME_LIMIT``.

- Default executors: ``SWEEP_EXE`` = CDG includes CPU, DMA, and GFX for broad coverage.

- Single-node only.

.. note::

  On systems with many devices, set ``SWEEP_TEST_LIMIT`` or ``SWEEP_TIME_LIMIT`` to bound the runtime. Without these limits, the default sweep may never finish.

**Usage:**

.. code-block:: shell

  ./TransferBench sweep

To run with memory and Executor limited to GPU only, and XGMI:

.. code-block:: shell

  SWEEP_SRC=G SWEEP_DST=G SWEEP_EXE=G SWEEP_XGMI_MIN=1 SWEEP_MAX=16 ./TransferBench sweep

To limit the duration of run:

.. code-block:: shell

  SWEEP_TIME_LIMIT=3600 SWEEP_FILE=/tmp/mySweep.cfg ./TransferBench sweep

Environment variables
----------------------

To modify the behavior of sweep and rsweep preset, use the following environment variables:

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``CONTINUE_ON_ERROR``
      - To continue despite validation error, set to ``1``. To stop, set to ``0``.
      - ``0``

    * - ``NUM_CPU_DEVICES``
      - Number of CPU NUMA nodes.
      - (all detected)

    * - ``NUM_CPU_SE``
      - CPU threads per CPU-executed transfer.
      - ``4``

    * - ``NUM_GPU_DEVICES``
      - Number of GPUs.
      - (all detected)

    * - ``NUM_GPU_SE``
      - CUs per GPU-executed transfer.
      - ``4``

    * - ``SWEEP_SRC``
      - Source memory types: C=CPU, G=GPU, N=Null.
      - ``CG``

    * - ``SWEEP_DST``
      - Destination memory types.
      - ``CG``

    * - ``SWEEP_EXE``
      - Executor types: C=CPU, D=DMA, G=GFX.
      - ``CDG``

    * - ``SWEEP_FILE``
      - File where sweep configuration is saved.
      - ``/tmp/lastSweep.cfg``

    * - ``SWEEP_MIN``
      - Minimum simultaneous transfers.
      - ``1``

    * - ``SWEEP_MAX``
      - Maximum simultaneous transfers (0=no limit).
      - ``24``

    * - ``SWEEP_RAND_BYTES``
      - To use random transfer size, set to ``1``. For constant, set to ``0``.
      - ``0``

    * - ``SWEEP_SEED``
      - Random seed. Used for rsweep or ``SWEEP_RAND_BYTES``.
      - time(NULL)

    * - ``SWEEP_TEST_LIMIT``
      - Maximum number of tests allowed to run. ``0`` = no limit.
      - ``0``

    * - ``SWEEP_TIME_LIMIT``
      - Maximum allowed test duration (in seconds). ``0`` = no limit.
      - ``0``

    * - ``SWEEP_XGMI_MIN``
      - Minimum XGMI hops for transfers.
      - ``0``

    * - ``SWEEP_XGMI_MAX``
      - Maximum allowed XGMI hops. ``-1`` = no limit.
      - ``-1``

Example output
---------------

.. code-block:: shell

  [Sweep Related]
  CONTINUE_ON_ERROR    =            0 : Stop after first error
  NUM_CPU_DEVICES      =            2 : Using 2 CPUs
  NUM_CPU_SE           =            4 : Using 4 CPU threads per CPU executed Transfer
  NUM_GPU_DEVICES      =            8 : Using 8 GPUs
  NUM_GPU_SE           =            4 : Using 4 subExecutors/CUs per GPU executed Transfer
  SWEEP_DST            =           CG : Destination Memory Types to sweep
  SWEEP_EXE            =          CDG : Executor Types to sweep
  SWEEP_FILE           = /tmp/lastSweep.cfg : File to store the executing sweep configuration
  SWEEP_MAX            =           24 : Max simultaneous transfers (0 = no limit)
  SWEEP_MIN            =            1 : Min simultaenous transfers
  SWEEP_RAND_BYTES     =            0 : Using constant number of bytes per Transfer
  SWEEP_SEED           =   1773692223 : Random seed set to 1773692223
  SWEEP_SRC            =           CG : Source Memory Types to sweep
  SWEEP_TEST_LIMIT     =            0 : Max number of tests to run during sweep (0 = no limit)
  SWEEP_TIME_LIMIT     =            0 : Max number of seconds to run sweep for  (0 = no limit)
  SWEEP_XGMI_MAX       =           -1 : Max number of XGMI hops for Transfers  (-1 = no limit)
  SWEEP_XGMI_MIN       =            0 : Min number of XGMI hops for Transfers

  Sweep configuration saved to: /tmp/lastSweep.cfg
  Test 1:
  -------------------┬--------------┬------------┬-------------------┬--------------------
    Executor: CPU 00 │  30.660 GB/s │   8.755 ms │   268435456 bytes │  30.847 GB/s (sum)
  -------------------┼--------------┼------------┼-------------------┼--------------------
       Transfer 0    │  30.847 GB/s │   8.702 ms │   268435456 bytes │ C1 -> C0:4 -> G6
  -------------------┼--------------┼------------┼-------------------┼--------------------
    Executor: GPU 01 │  38.662 GB/s │   6.943 ms │   268435456 bytes │  38.669 GB/s (sum)
  -------------------┼--------------┼------------┼-------------------┼--------------------
       Transfer 8    │  38.669 GB/s │   6.942 ms │   268435456 bytes │ G2 -> G1:4 -> G1
  -------------------┼--------------┼------------┼-------------------┼--------------------
    Executor: GPU 02 │  61.598 GB/s │   4.358 ms │   268435456 bytes │  61.615 GB/s (sum)
  -------------------┼--------------┼------------┼-------------------┼--------------------
       Transfer 9    │  61.615 GB/s │   4.357 ms │   268435456 bytes │ G2 -> G2:4 -> G0
  -------------------┼--------------┼------------┼-------------------┼--------------------
    Executor: GPU 03 │  38.816 GB/s │   6.916 ms │   268435456 bytes │  38.826 GB/s (sum)
  -------------------┼--------------┼------------┼-------------------┼--------------------
       Transfer 10   │  38.826 GB/s │   6.914 ms │   268435456 bytes │ G2 -> G3:4 -> G7
  -------------------┼--------------┼------------┼-------------------┼--------------------
    Executor: GPU 06 │  44.298 GB/s │  12.120 ms │   536870912 bytes │  58.182 GB/s (sum)
  -------------------┼--------------┼------------┼-------------------┼--------------------
       Transfer 11   │  22.151 GB/s │  12.118 ms │   268435456 bytes │ G1 -> G6:4 -> C1
       Transfer 12   │  36.030 GB/s │   7.450 ms │   268435456 bytes │ G2 -> G6:4 -> G5
  -------------------┼--------------┼------------┼-------------------┼--------------------
    Executor: GPU 07 │  37.963 GB/s │   7.071 ms │   268435456 bytes │  37.969 GB/s (sum)
  -------------------┼--------------┼------------┼-------------------┼--------------------
       Transfer 13   │  37.969 GB/s │   7.070 ms │   268435456 bytes │ G4 -> G7:4 -> G6
  -------------------┼--------------┼------------┼-------------------┼--------------------
    Executor: DMA 01 │  43.428 GB/s │  12.362 ms │   536870912 bytes │  77.585 GB/s (sum)
  -------------------┼--------------┼------------┼-------------------┼--------------------
       Transfer 1    │  55.481 GB/s │   4.838 ms │   268435456 bytes │ C0 -> D1:4 -> G0
       Transfer 2    │  22.105 GB/s │  12.144 ms │   268435456 bytes │ G7 -> D1:4 -> C1
  -------------------┼--------------┼------------┼-------------------┼--------------------
    Executor: DMA 03 │  31.427 GB/s │   8.541 ms │   268435456 bytes │  32.353 GB/s (sum)
  -------------------┼--------------┼------------┼-------------------┼--------------------
       Transfer 3    │  32.353 GB/s │   8.297 ms │   268435456 bytes │ G4 -> D3:4 -> G6
  -------------------┼--------------┼------------┼-------------------┼--------------------
    Executor: DMA 04 │  22.214 GB/s │  12.084 ms │   268435456 bytes │  22.536 GB/s (sum)
  -------------------┼--------------┼------------┼-------------------┼--------------------
       Transfer 4    │  22.536 GB/s │  11.912 ms │   268435456 bytes │ C1 -> D4:4 -> G1
  -------------------┼--------------┼------------┼-------------------┼--------------------
    Executor: DMA 06 │  53.665 GB/s │  10.004 ms │   536870912 bytes │  72.749 GB/s (sum)
  -------------------┼--------------┼------------┼-------------------┼--------------------
       Transfer 5    │  27.768 GB/s │   9.667 ms │   268435456 bytes │ G2 -> D6:4 -> C0
       Transfer 6    │  44.981 GB/s │   5.968 ms │   268435456 bytes │ G3 -> D6:4 -> G2
  -------------------┼--------------┼------------┼-------------------┼--------------------
    Executor: DMA 07 │  57.440 GB/s │   4.673 ms │   268435456 bytes │  60.131 GB/s (sum)
  -------------------┼--------------┼------------┼-------------------┼--------------------
       Transfer 7    │  60.131 GB/s │   4.464 ms │   268435456 bytes │ G7 -> D7:4 -> G5
  -------------------┼--------------┼------------┼-------------------┼--------------------
     Aggregate (CPU) │ 295.108 GB/s │  12.735 ms │  3758096384 bytes │ Overhead 0.372 ms
  -------------------┴--------------┴------------┴-------------------┴--------------------

The exact format depends on ``OUTPUT_TO_CSV``. Typically shows test number, transfer count, bandwidth, and timing per test.

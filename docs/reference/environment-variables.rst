.. meta::
  :description: TransferBench is a utility to benchmark simultaneous transfers between user-specified devices (CPUs or GPUs)
  :keywords: TransferBench environment variables, TransferBench configuration, TransferBench client, TransferBench customization, TransferBench how to

.. _environment-variables:

====================================
TransferBench environment variables
====================================

TransferBench behavior can be customized using environment variables. This topic describes the environment variables that control the TransferBench client (frontend), backend library, and runtime.

.. note::

  Environment variables read by the frontend client apply to both configuration-file runs and preset runs. Variables prefixed with ``TB_`` are read by the backend library and are not part of the frontend client.

Frontend client environment variables
=======================================

The following environment variables are read by the TransferBench client and apply to configuration-file runs and presets.

General options
---------------

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``NUM_ITERATIONS``
      - Number of timed iterations per test. If negative, runs for that many seconds instead.
      - ``10``

    * - ``NUM_SUBITERATIONS``
      - Sub-iterations per iteration. Set to ``0`` for infinite sub-iterations.

        Within each iteration, the transfer is repeated ``NUM_SUBITERATIONS`` times. This can reduce the impact of kernel launch latencies, but might over-emphasize cache reuse.
      - ``1``

    * - ``NUM_WARMUPS``
      - Untimed warmup iterations per test.
      - ``3``

    * - ``SHOW_BORDERS``
      - Shows ASCII box-drawing characters in tables. Set to ``1`` to show, ``0`` to hide.
      - ``1``

    * - ``SHOW_ITERATIONS``
      - Shows per-iteration timing. Set to ``1`` to show, ``0`` to hide.
      - ``0``

    * - ``USE_INTERACTIVE``
      - Specifies whether to pause for user input before the transfer loop. Set to ``1`` to pause, ``0`` to skip.

        The first pause occurs after memory allocations are prepared and before any transfers are executed. The second pause occurs after transfers are executed and before transfers are validated. This is useful for profiling: start profiling after the first pause, then capture data before validation begins.
      - ``0``

    * - ``HIDE_ENV``
      - Hides the environment variable listing. Set to ``1`` to hide, ``0`` to show.
      - ``0``

    * - ``OUTPUT_TO_CSV``
      - Generates results in CSV format. Set to ``1`` for CSV, ``0`` for human-readable output.
      - ``0``

    * - ``SAMPLING_FACTOR``
      - Affects auto-generated N values when N is ``0``.
      - ``1``

.. _data-validation-var:

Data and validation options
----------------------------

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``ALWAYS_VALIDATE``
      - Specifies whether to validate after each iteration. Set to ``1`` to validate each iteration, ``0`` to validate once after all iterations.

        By default, validation is only done after all iterations, which can mask errors that occurred in all but the last iteration.
      - ``0``

    * - ``BLOCK_BYTES``
      - Granularity in bytes for dividing work across sub-executors.
      - ``256``

    * - ``BYTE_OFFSET``
      - Initial byte offset for allocations. Must be a multiple of 4.
      - ``0``

    * - ``FILL_COMPRESS``
      - Comma-separated percentages for 64-byte line fill across five bins: random, 1B0, 2B0, 4B0, and 32B0.

        This feature tests various compressible data patterns supported by XGMI. The integer values must sum to 100 and correspond to the following bins:

        .. list-table::
            :header-rows: 1

            * - Bin
              - Name
              - Description

            * - 0
              - Random
              - Random data

            * - 1
              - 1B0
              - The upper 1 byte of each 2 bytes in the 64-byte line is 0

            * - 2
              - 2B0
              - The upper 2 bytes of each 4 bytes in the 64-byte line are 0

            * - 3
              - 4B0
              - The upper 4 bytes of each 8 bytes in the 64-byte line are 0

            * - 4
              - 32B0
              - The upper 32 bytes of each 64-byte line are 0

      - —

    * - ``FILL_PATTERN``
      - Big-endian hex pattern for source data. Must have an even number of digits. Allows users to specify a particular data pattern.
      - —

    * - ``VALIDATE_DIRECT``
      - Specifies whether to validate the GPU destination directly. Set to ``1`` to validate directly, ``0`` to validate via a CPU staging buffer.

        On AMD hardware, the CPU can directly access GPU device memory, avoiding the need for a staging buffer. This feature is not supported on NVIDIA hardware.
      - ``0``

    * - ``VALIDATE_SOURCE``
      - Specifies whether to validate the source immediately after preparation. Set to ``1`` to validate, ``0`` to skip.

        This was introduced to help debug issues where the initial copy of source data to the GPU didn't meet expectations, due to a hardware DMA issue.
      - ``0``

.. _gfx-options:

GFX and kernel options
-----------------------

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``GFX_BLOCK_ORDER``
      - Block ordering when running in multitransfer single-stream mode. ``0`` = sequential, ``1`` = interleaved, ``2`` = random.

        This controls how threadblocks are assigned to transfers. For example, with 4 transfers (A, B, C, D) each using 3 CUs:

        .. code-block:: shell

          Threadblock : 00 01 02 03 04 05 06 07 08 09 10 11
          ====================================================
          0 = Sequential : A0 A1 A2 B0 B1 B2 C0 C1 C2 D0 D1 D2
          1 = Interleaved: A0 B0 C0 D0 A1 B1 C1 D1 A2 B2 C2 D2
          2 = Random      : C1 D2 B1 B0 A0 C0 D1 D0 A1 C2 A2 B2

        Use this setting to investigate how threadblock assignment to XCCs impacts performance.
      - ``0``

    * - ``GFX_BLOCK_SIZE``
      - Number of threads per threadblock. Must be a multiple of 64.
      - ``256``

    * - ``GFX_SE_TYPE``
      - Subexecutor granularity. ``0`` = threadblock, ``1`` = warp.

        By default, each subexecutor consists of one threadblock. Setting this to ``1`` makes each subexecutor consist of one warp instead. On some architectures such as AMD Instinct™ MI355X, this can impact performance, especially when used together with ``GFX_BLOCK_ORDER``.
      - ``0``

    * - ``GFX_TEMPORAL``
      - Controls how stores and loads are performed using non-temporal operations. ``0`` = none, ``1`` = loads only, ``2`` = stores only, ``3`` = both loads and stores.
      - ``0``

    * - ``GFX_UNROLL``
      - Unroll factor for the GFX kernel. Set to ``0`` for automatic selection. See :ref:`gfx-unroll`.
      - (architecture-dependent)

    * - ``GFX_SINGLE_TEAM``
      - Subexecutor memory access mode. ``1`` = subexecutors operate on the full array, ``0`` = subexecutors operate on disjoint subarrays.
      - ``1``

    * - ``GFX_WAVE_ORDER``
      - Stride ordering for GFX waves. ``0`` = UWC, ``1`` = UCW, ``2`` = WUC, ``3`` = WCU, ``4`` = CUW, ``5`` = CWU.
      - ``0``

    * - ``GFX_WORD_SIZE``
      - Packed data size in DWORDs. ``4`` = DWORD x 4, ``2`` = DWORD x 2, ``1`` = DWORD x 1.
      - ``4``

    * - ``USE_HIP_EVENTS``
      - Timing method for GFX and DMA transfers. ``1`` = use HIP events, ``0`` = use CPU wall time.
      - ``1``

    * - ``USE_SINGLE_STREAM``
      - Stream assignment. ``1`` = one stream per GPU, ``0`` = one stream per transfer.
      - ``1``

    * - ``CU_MASK``
      - CU mask for streams, specified as a comma-separated list of indices or ranges (for example, ``5,10-12,14``). AMD only.
      - —

    * - ``XCC_PREF_TABLE``
      - Preferred XCC per source-by-destination GPU pair, specified as a comma-separated list. Supported on AMD hardware only.
      - —

DMA options
-----------

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``USE_HSA_DMA``
      - DMA implementation. ``1`` = use ``hsa_amd_async_copy``, ``0`` = use ``hipMemcpy``.
      - ``0``

Variable subexecutor options
------------------------------

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``MIN_VAR_SUBEXEC``
      - Minimum number of subexecutors for variable subexecutor transfers.
      - ``1``

    * - ``MAX_VAR_SUBEXEC``
      - Maximum number of subexecutors. Set to ``0`` to use the device limit.
      - ``0``

NIC options
-----------

The following environment variables apply only when NIC support is enabled.

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``IB_GID_INDEX``
      - RoCE GID index. Set to ``-1`` for automatic selection.
      - ``-1``

    * - ``IB_PORT_NUMBER``
      - RDMA port number.
      - ``1``

    * - ``IP_ADDRESS_FAMILY``
      - IP address family. ``4`` = IPv4, ``6`` = IPv6.
      - ``4``

    * - ``NIC_CHUNK_BYTES``
      - Bytes per NIC RDMA chunk.
      - ``1073741824``

    * - ``NIC_RELAX_ORDER``
      - RDMA ordering. ``1`` = relaxed, ``0`` = strict.
      - ``1``

    * - ``ROCE_VERSION``
      - RoCE version.
      - ``2``

Backend and runtime options (client-read)
------------------------------------------

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description
      - Default value

    * - ``GPU_MAX_HW_QUEUES``
      - A HIP runtime environment variable that determines the maximum number of hardware queues each GPU has access to per process. When more than four GPU-executed transfers run simultaneously, they might serialize while waiting for available hardware queues. In this case, increase the value of this environment variable from the default ``4`` when ``USE_SINGLE_STREAM=0``. See :ref:`gpu-max-hw-queues`.
      - ``4``

Backend environment variables
==============================

The following environment variables are read by the TransferBench backend library or build system. They are not part of the frontend client and are prefixed with ``TB_``.

Socket-related variables
------------------------

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description

    * - ``TB_RANK``
      - The process rank (0-based). Used for socket-based multinode runs.

    * - ``TB_NUM_RANKS``
      - Total number of processes.

    * - ``TB_MASTER_ADDR``
      - IP address of rank 0, used by the socket communicator.

    * - ``TB_MASTER_PORT``
      - Port for rank coordination. Default: ``29500``.

    * - ``TB_SINGLE_LOG``
      - When set, only rank 0 produces output. Useful for multinode socket mode.

Backend variables
-----------------

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description

    * - ``TB_VERBOSE``
      - Backend verbosity level. For example, set to ``1`` for extra logging.

    * - ``TB_DUMP_CFG_FILE``
      - Path of the configuration file used to dump executed transfers.

        This dumps all executed transfers to a configuration file that can then be re-executed by TransferBench. This can be used to capture the transfers executed by a preset, facilitating any further modifications and customizations.

    * - ``TB_PAUSE``
      - Pause before execution. Useful for attaching a debugger.

        For example:

        .. code-block:: shell

          > TB_PAUSE=1 ./TransferBench
          # Pausing for debug attachment (PID: 2443974)

          > sudo gdb -p 2443974
          5741        while (pause);

          set pause=false
          continue

    * - ``TB_NIC_FILTER``
      - Regex pattern to limit visible NICs. Useful in preset scenarios that require homogeneous configurations.

        For example:

        .. code-block:: shell

          # Without filter:
          > mlx5_0, mlx5_1, mlx5_2, mlx5_3, mlx5_4, mlx5_5, mlx5_6, mlx5_7

          TB_NIC_FILTER="mlx5_1|mlx5_3"    > mlx5_1, mlx5_3
          TB_NIC_FILTER="mlx5_[1,4,5]"     > mlx5_1, mlx5_4, mlx5_5
          TB_NIC_FILTER="mlx5_[1-3,7]"     > mlx5_1, mlx5_2, mlx5_3, mlx5_7
          TB_NIC_FILTER="mlx5_.*"           > mlx5_0, mlx5_1, mlx5_2, mlx5_3, mlx5_4, mlx5_5, mlx5_6, mlx5_7

    * - ``TB_DUMP_LINES``
      - Number of 64-byte lines to dump when debugging ``FILL_COMPRESS``.

        For example:

        .. code-block:: shell

          TB_DUMP_LINES=10

          Input pattern 64B line statistics for bufferIdx 0:
          Total lines: 16384
          - 0: Random :  3276 ( 19.995%)
          - 1: 1B0    :  3277 ( 20.001%)
          - 2: 2B0    :  3277 ( 20.001%)
          - 3: 4B0    :  3277 ( 20.001%)
          - 4: 32B0   :  3277 ( 20.001%)

    * - ``TB_FORCE_SINGLE_POD``
      - Forces single pod mode, skipping AMD-SMI and NVML pod queries. This assumes that all GPUs are in the same pod and skips cluster membership API calls.

HSA runtime variables
---------------------

.. list-table::
    :header-rows: 1

    * - Environment variable
      - Description

    * - ``HSA_ENABLE_SDMA``
      - Enables SDMA when set to ``1``. To disable, set to ``0``.

        This is a HIP runtime environment variable. When SDMA is disabled, the DMA executor falls back to using blit kernels (GFX) internally.

.. meta::
  :description: Reference for TransferBench transfer definition syntax, including simple and advanced modes, memory and Executor letter codes, and wildcard syntax.
  :keywords: TransferBench transfer definition, TransferBench syntax, TransferBench memory types, TransferBench Executor types, TransferBench wildcards

.. _transfer-definition-syntax:

==========================
Transfer definition syntax
==========================

A transfer is a single operation where an Executor reads and adds values from source (SRC) memory, then writes the sum to destination (DST) memory. When a transfer has a single SRC and a single DST, it is a copy operation.

TransferBench supports two modes for defining transfers: :ref:`simple mode <simple-mode>` and :ref:`advanced mode <advanced-mode>`. Within either mode, a transfer entry can also be written as a :ref:`PingPong <pingpong-transfers>` to measure round-trip latency instead of bandwidth.

.. _simple-mode:

Simple mode
===========

Use simple mode when all transfers in a test share the same number of SubExecutors. The format is:

.. code-block:: shell

    #Transfers #SEs (srcMem1 Executor1 dstMem1) ... (srcMemL ExecutorL dstMemL)

The format uses the following fields:

- ``#Transfers``: A positive integer that specifies the number of parallel transfers.
- ``#SEs``: The number of SubExecutors (CUs, threads, or queue pairs) used by all transfers.
- ``(srcMem Executor dstMem)``: A triplet that describes one transfer.

The transfer size comes from the ``num_bytes`` command-line argument.

The following examples show valid simple mode definitions:

.. code-block:: shell

    1 4 (G0->G0->G1)                  # Uses 4 CUs on GPU 0 to copy from GPU 0 to GPU 1
    2 4 (G0 G0 G1) (G1 G1 G0)         # Two parallel transfers: GPU 0 to GPU 1 and GPU 1 to GPU 0
    1 4 (G0->G0->G0G1G2G3)            # Reads GPU 0 then writes to GPU 0, GPU 1, GPU 2, and GPU 3

.. _advanced-mode:

Advanced mode
=============

Use advanced mode when transfers in a test require different SubExecutor counts or sizes. The format is:

.. code-block:: shell

    -#Transfers (srcMem1 Executor1 dstMem1 #SEs1 Bytes1) ... (srcMemL ExecutorL dstMemL #SEsL BytesL)

The format uses the following fields:

- ``-#Transfers``: A negative integer that specifies the number of parallel transfers.
- ``(srcMem Executor dstMem #SEs Bytes)``: A quintuplet that describes one transfer.
- ``Bytes``: The per-transfer size. Set to ``0`` to use the command-line ``num_bytes`` value. You can suffix this value with ``K``, ``M``, or ``G``. A non-zero value overrides the command-line value for that transfer only.

The following example runs two transfers with different sizes and SubExecutor counts:

.. code-block:: shell

    -2 (G0 G0 G1 4 1M) (G1 G1 G0 2 2M)   # 1 MB GPU 0 to GPU 1 with 4 CUs; 2 MB GPU 1 to GPU 0 with 2 CUs

.. _pingpong-transfers:

PingPong (latency) transfers
=============================

Instead of a plain triplet or quintuplet, any transfer entry in :ref:`simple <simple-mode>` or :ref:`advanced <advanced-mode>` mode can be written as a PingPong. A PingPong joins a Ping triplet and a Pong triplet with ``+`` and an optional lap count. On each lap, the Ping executor writes a 1-byte flag to its DST and waits for the Pong executor to write one back to the Pong DST. The result is the GPU-timed round-trip latency per lap:

.. code-block:: shell

    (pingSrc PingExecutor pingDst)+N(pongSrc PongExecutor pongDst)

- ``N``: Number of laps (positive integer). A bare ``+`` means 1 lap.
- ``DST``: Required for both halves. Each half writes its flag to its own DST, and the partner polls it.
- ``SRC``: Optional (``N`` for none). If given, the flag value written on each lap is first read from SRC, which adds a real memory read to every lap.
- Each SRC and DST must be a single memory device. Wildcards are allowed: every expansion of the Ping half is paired with every expansion of the Pong half.

A PingPong counts as one transfer toward ``#Transfers``. It can be mixed with regular transfers in the same test, though doing so means aggregate CPU results are omitted for that test. It doesn't count toward executor bandwidth, and its bytes aren't reported. An executor that runs only PingPongs shows ``pingpong`` in place of a bandwidth sum.

Each half runs as a single threadblock, so ``#SEs``/``#CU`` doesn't apply:

- In simple mode, any ``#SEs`` value is silently discarded for PingPong transfers.
- In advanced mode, a PingPong must not include ``#SEs`` or ``Bytes``. Doing so is rejected with ``Pingpong %d must not specify #CU / #Bytes, use (SRC EXE DST)+N(SRC EXE DST)``.

Currently supported on GFX executors (``G``) only. Ping and Pong might run on different ranks in the same pod (requires pod communication support); cross-rank flags must be in GPU memory.

``NUM_SUBITERATIONS`` is ignored (forced to ``1``, with a warning) when every transfer in a test is a PingPong. The flag buffer and stall watchdog are controlled by ``PNP_FLAG_BUFFER``, ``PNP_STRIDE``, and ``PNP_SPIN_LIMIT``.

The following examples show valid PingPong definitions:

.. code-block:: shell

    1 1 (N->G0->F1)+1000(N->G1->F0)          # 1000 laps: GPU 0 writes a flag to GPU 1, GPU 1 answers into GPU 0
    -2 (G0 G0 G1 4 1M) (N G0 G0)+500(N G1 G1)  # Advanced mode: a 1 MB copy running alongside a 500-lap PingPong

In the results table, a PingPong is reported as a ``PingPong <idx>`` row showing latency per lap (in microseconds), total time across all laps (in milliseconds), the lap count, and ``pingSrc->PingExe->pingDst <+> pongSrc->PongExe->pongDst``. With ``SHOW_ITERATIONS=1``, per-iteration latency and a standard deviation follow. For ready-made latency matrices, see the :ref:`latency presets <latency-presets>`.

Memory and Executor letter codes
================================

The memory locations and Executors use a format consisting of letters indicating the memory or Executor type, and index. The following tables indicate what these letters stand for.

Memory location letters
-----------------------

Memory locations use the format ``[R<rank>]<MemType><Index>``, where ``MemType`` is a single letter and ``Index`` is the zero-based device index. If you omit the rank prefix ``R``, TransferBench uses the local rank.

The following table lists the supported memory type letters:

.. list-table::
    :header-rows: 1

    * - Letter
      - Memory type
      - Indexed by

    * - ``C``
      - Coarse-grained pinned host
      - CPU NUMA node, logical index (0 to #exposed - 1)

    * - ``P``
      - Pinned host (closest to GPU)
      - GPU index (0 to #GPUs - 1)

    * - ``B``
      - Coherent pinned host
      - NUMA node

    * - ``D``
      - Non-coherent pinned host
      - NUMA node

    * - ``K``
      - Uncached pinned host
      - NUMA node

    * - ``H``
      - Unpinned host
      - NUMA node

    * - ``G``
      - Coarse-grained global device
      - GPU (0 to #GPUs - 1)

    * - ``F``
      - Fine-grained device
      - GPU

    * - ``U``
      - Uncached device
      - GPU

    * - ``M``
      - Managed memory
      - GPU

    * - ``N``
      - Null (empty). Use this to denote read-only or write-only transfers.
      - Index ignored.

You can concatenate multiple memory locations for broadcast or reduce operations. For example, ``G0G1`` refers to GPU 0 and GPU 1.

Executor letters
----------------

Executors use the format ``[R<rank>]<ExeType><Index>[Slot][.<SubIndex>][SubSlot]``.

The following table lists the supported Executor type letters:

.. list-table::
    :header-rows: 1

    * - Letter
      - Executor type
      - Subexecutor
      - Indexed by

    * - ``C``
      - CPU
      - CPU thread
      - CPU NUMA node, logical index (0 to #exposed - 1)

    * - ``G``
      - GPU (GFX/kernel)
      - Threadblock/CU
      - GPU (0 to #GPUs - 1)

    * - ``D``
      - DMA (SDMA)
      - N/A (streams)
      - GPU

    * - ``I``
      - NIC RDMA
      - Queue pair
      - NIC index. Requires ``.SubIndex`` (for example, ``I0.2``).

    * - ``N``
      - Nearest NIC
      - Queue pair
      - GPU. Uses the closest NIC for SRC and DST. SubIndex is optional.

    * - ``B``
      - Batched DMA
      - Batch item
      - GPU (0 to #GPUs - 1). Requires a single SRC and at least one DST.

    * - ``T``
      - TDM
      - Threadblock/CU
      - GPU (0 to #GPUs - 1). Requires AMD gfx1250 or NVIDIA sm_90+.

The optional Executor fields have the following meanings:

- ``Slot`` (``A``, ``B``, ...): Selects which closest NIC to use, where ``A`` is the first and ``B`` is the second, and so on. Used for ``EXE_NIC_NEAREST``.
- ``SubIndex`` (after ``.``): Specifies the queue pair or sub-index for the NIC.
- ``SubSlot``: Specifies the alpha range for sub-slot selection.

.. note::

    CPU indices are logical. Since v1.70.0, TransferBench hides CPU NUMA nodes that have no cores, so ``C0``, ``C1``, and so on number only the exposed nodes and might not match the physical node IDs reported by ``numactl --hardware``. Set ``TB_SHOW_ALL_NUMA=1`` to expose every configured node and restore the original numbering.

Wildcards
=========

Wildcards expand a single configuration file line into multiple concrete transfers at runtime.

Numeric wildcards
-----------------

Numeric wildcards apply to ranks (``R``), device indices (for example, ``G0`` or ``C1``), and Executor indices.

The following table describes the supported numeric wildcard syntax:

.. list-table::
    :header-rows: 1

    * - Syntax
      - Meaning
      - Example

    * - ``*``
      - All values in range
      - ``R*G0``: GPU 0 on all ranks

    * - ``[n]``
      - Single value
      - ``R[2]G0``: GPU 0 on rank 2

    * - ``[n,m,...]``
      - Comma-separated list
      - ``G[0,2,4]``: GPUs 0, 2, and 4

    * - ``[start..end]``
      - Inclusive range
      - ``G[1..3]``: GPUs 1, 2, and 3

The following list shows additional examples:

- ``G*``: All GPUs (for example, G0, G1, G2, ...)
- ``G[0,2,4]``: GPUs 0, 2, and 4
- ``R[1..3]G0``: GPU 0 on ranks 1, 2, and 3

Alpha wildcards
---------------

Alpha wildcards apply to Executor slots and subslots (``A``, ``B``, ``C``, ...).

The following table describes the supported alpha wildcard syntax:

.. list-table::
    :header-rows: 1

    * - Syntax
      - Meaning
      - Example

    * - ``*``
      - Full wildcard, resolved at runtime
      - All available slots

    * - ``[A]`` or ``A``
      - Single letter
      - First closest NIC

    * - ``[A..C]``
      - Letter range
      - Slots A, B, and C

    * - ``[A,D,F]``
      - Comma-separated letters
      - Slots A, D, and F

Rank prefix (multinode)
-------------------------

The ``R`` prefix is optional and specifies the rank index for a memory location or Executor:

- ``R2G3``: GPU 3 on rank 2.
- ``G3`` (no ``R``): GPU 3 on the local rank.

The rank prefix accepts the same numeric wildcards as device indices, such as ``R*`` and ``R[0..1]``.

Nearest NIC wildcard
---------------------

For the ``N`` (Nearest NIC) Executor, you can omit the Executor index and sub-index. TransferBench resolves the correct NICs for the SRC and DST memory locations at runtime.

For example, the following definition:

.. code-block:: shell

  (R0G0 N R2G4)

This definition expands to:

.. code-block:: shell

  (R0G0 N0.4 R2G4)

Here, ``.4`` is the NIC sub-index resolved from the NIC closest to GPU 4 on rank 2.

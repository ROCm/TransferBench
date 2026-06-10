.. meta::
  :description: Frequently asked questions about TransferBench, covering common errors, warnings, and configuration issues including IOMMU, memory types, and XGMI.
  :keywords: TransferBench FAQ, TransferBench errors, TransferBench warnings, IOMMU, GPU_MAX_HW_QUEUES, GFX_UNROLL, validation, XGMI, UALoE, memory types

.. _faq:

==========================
Frequently asked questions
==========================

This topic answers common questions about TransferBench errors, warnings, features, environment variables, and presets.

Error and warning messages
===========================

This section describes common TransferBench error and warning messages and how to resolve them.

Unexpected mismatch at index
-----------------------------

TransferBench validates each transfer to ensure that data has been moved correctly. This
error indicates that the destination (DST) memory doesn't match the expected value.

For example:

.. code-block:: text

  [ERROR] Transfer 0: Unexpected mismatch at index 0 of destination 0 on rank 0: Expected 31.00000 Actual: 0.00000

In this example, the first element of the DST memory was expected to hold
``31.00000`` but actually contained ``0.00000``.

This error is generally not a TransferBench issue. It's usually a sign of a system
configuration problem.

Common causes include:

- Improperly configured IOMMU
- A ROCm runtime and driver version mismatch

IOMMU must be set to pass-through mode in the BIOS. To verify, check for ``iommu=pt``
in the kernel command line:

.. code-block:: shell

  # Check for iommu=pt in the output
  cat /proc/cmdline

  BOOT_IMAGE=/boot/vmlinuz-5.15.0-70-generic root=UUID=7489cc43-aaab-4b61-8c63-86a419728dea
  ro panic=0 nowatchdog msr.allow_writes=on nokaslr amdgpu.noretry=1 pci=realloc=off
  modprobe.blacklist=amdgpu intel_iommu=on iommu=pt numa_balancing=disable console=tty0
  console=ttyS0,115200n8

For IOMMU configuration guidance, see
`AMD Instinct MI300X system optimization <https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html>`_.

.. _gpu-max-hw-queues:

Attempting X parallel transfers, however GPU_MAX_HW_QUEUES only set to 4
-------------------------------------------------------------------------

The HIP runtime limits the number of independent hardware queues each GPU can use per
process. This limit is controlled by the ``GPU_MAX_HW_QUEUES`` environment variable. For
more information, see
`ROCm environment variables <https://rocm.docs.amd.com/en/latest/reference/env-variables.html#debug-variables>`_.

When the number of transfers requiring hardware queues exceeds the configured limit,
those transfers serialize instead of running in parallel. TransferBench detects this
condition and issues this warning.

This commonly occurs with DMA-executed transfers, because each DMA transfer requires one
hardware queue. It is frequently seen when running the :ref:`all-to-all preset <a2a>`.

To resolve this, set ``GPU_MAX_HW_QUEUES`` to a value greater than the number of
transfers. It is recommended to set at least one extra queue beyond the number of
transfers.

The following examples show the effect on an 8-GPU system running the all-to-all preset
with DMA execution enabled.

Without setting ``GPU_MAX_HW_QUEUES``:

.. code-block:: shell

  USE_DMA_EXEC=1 ./TransferBench a2a

  ...
  GPU-DMA All-To-All benchmark:
  ==============================
  [268435456 bytes per Transfer] [DMA:8] [1 Read(s) 1 Write(s)] [MemType:uncached GPU] [NIC QueuePairs:0] [#Ranks:1]

  Average bandwidth (GPU Timed): 60.952 GB/s
  Aggregate bandwidth (GPU Timed): 3413.290 GB/s
  Aggregate bandwidth (CPU Timed): 1338.252 GB/s
  [WARN] DMA 0 attempting 7 parallel transfers, however GPU_MAX_HW_QUEUES only set to 4
  [WARN] DMA 1 attempting 7 parallel transfers, however GPU_MAX_HW_QUEUES only set to 4
  [WARN] DMA 2 attempting 7 parallel transfers, however GPU_MAX_HW_QUEUES only set to 4
  [WARN] DMA 3 attempting 7 parallel transfers, however GPU_MAX_HW_QUEUES only set to 4
  [WARN] DMA 4 attempting 7 parallel transfers, however GPU_MAX_HW_QUEUES only set to 4
  [WARN] DMA 5 attempting 7 parallel transfers, however GPU_MAX_HW_QUEUES only set to 4
  [WARN] DMA 6 attempting 7 parallel transfers, however GPU_MAX_HW_QUEUES only set to 4
  [WARN] DMA 7 attempting 7 parallel transfers, however GPU_MAX_HW_QUEUES only set to 4

Setting ``GPU_MAX_HW_QUEUES=8``:

.. code-block:: shell

  GPU_MAX_HW_QUEUES=8 USE_DMA_EXEC=1 ./TransferBench a2a

  ...
  GPU-DMA All-To-All benchmark:
  ==============================
  [268435456 bytes per Transfer] [DMA:8] [1 Read(s) 1 Write(s)] [MemType:uncached GPU] [NIC QueuePairs:0] [#Ranks:1]

  Average bandwidth (GPU Timed): 60.091 GB/s
  Aggregate bandwidth (GPU Timed): 3365.111 GB/s
  Aggregate bandwidth (CPU Timed): 2222.415 GB/s

.. note::

  Individual transfer bandwidths are similar in both cases because each transfer is timed
  from when it starts. However, the CPU wall-clock time is nearly double in the
  ``GPU_MAX_HW_QUEUES=4`` case, because serialized transfers complete one after another
  instead of running in parallel.

Feature questions
==================

This section answers common questions about TransferBench features and behavior.

Can TransferBench target a specific UALoE station?
----------------------------------------------------

No. TransferBench has no direct control over which Unified Accelerator Link over Ethernet
(UALoE) station gets used, and doesn't have any knowledge of which station is selected.

Does TransferBench perform any validation?
-------------------------------------------

Yes. TransferBench initializes source data buffers with a pattern (which can be
user-specified), then checks that destination data buffers contain the expected result
after each transfer completes. For details, see :ref:`transferbench-data-validation`.

Does TransferBench alter underlying XGMI speeds when it runs?
--------------------------------------------------------------

No. TransferBench runs on the current hardware settings and doesn't modify them.

To query current XGMI settings on AMD Instinct machines, use ``amd-smi xgmi``:

.. code-block:: shell

  amd-smi xgmi

  LINK METRIC TABLE:
  bdf             bit_rate  max_bandwidth  link_type  GPU0     GPU1     GPU2     GPU3     GPU4     GPU5     GPU6     GPU7
  GPU0  0000:0c:00.0  38 Gb/s  608 Gb/s  XGMI
    Read   N/A       39.61 TB  15.40 TB  15.47 TB  5.349 TB  4.993 TB  5.078 TB  5.952 TB
    Write  N/A       41.96 TB  15.32 TB  15.00 TB  5.332 TB  4.859 TB  4.979 TB  5.448 TB

Environment variable questions
================================

This section answers common questions about TransferBench environment variables.

.. _gfx-unroll:

What is the GFX unroll factor?
--------------------------------

Specifying an unroll factor of X means that each GPU thread reads X pieces of source data
into registers, then writes those X pieces of data out to the destination, as shown in the following table:

.. raw:: html

   <style>
     .tb-unroll { border-collapse: collapse; }
     .tb-unroll td, .tb-unroll th { border: 1px solid #ccc; padding: 6px 14px; text-align: center; }
     .tb-unroll td { font-family: monospace; }
     .tb-unroll thead tr { background: var(--pst-color-primary, #f0f0f0); color: var(--pst-color-on-primary, #000); }
     .tb-unroll thead th { font-weight: bold; }
     .tb-unroll tbody tr:nth-child(odd) td:first-child  { background: var(--pst-color-surface, #f8f8f8); }
     .tb-unroll tbody tr:nth-child(even) td:first-child { background: var(--pst-color-on-background, #eaeaea); }
     .tb-unroll .r { background: #c8f0d8; }
     .tb-unroll .w { background: #f8c8c8; }
   </style>
   <table class="table table--middle-left tb-unroll">
     <thead>
       <tr>
         <th class="head"><p>Instruction order</p></th>
         <th class="head"><p>Unroll 1</p></th>
         <th class="head"><p>Unroll 2</p></th>
         <th class="head"><p>Unroll 4</p></th>
       </tr>
     </thead>
     <tbody>
       <tr><td>1</td><td class="r">READ [A]</td> <td class="r">READ [A]</td> <td class="r">READ [A]</td></tr>
       <tr><td>2</td><td class="w">WRITE [A]</td><td class="r">READ [B]</td> <td class="r">READ [B]</td></tr>
       <tr><td>3</td><td class="r">READ [B]</td> <td class="w">WRITE [A]</td><td class="r">READ [C]</td></tr>
       <tr><td>4</td><td class="w">WRITE [B]</td><td class="w">WRITE [B]</td><td class="r">READ [D]</td></tr>
       <tr><td>5</td><td class="r">READ [C]</td> <td class="r">READ [C]</td> <td class="w">WRITE [A]</td></tr>
       <tr><td>6</td><td class="w">WRITE [C]</td><td class="r">READ [D]</td> <td class="w">WRITE [B]</td></tr>
       <tr><td>7</td><td class="r">READ [D]</td> <td class="w">WRITE [C]</td><td class="w">WRITE [C]</td></tr>
       <tr><td>8</td><td class="w">WRITE [D]</td><td class="w">WRITE [D]</td><td class="w">WRITE [D]</td></tr>
     </tbody>
   </table>

Having more reads in flight can reduce write stalls. However, a higher unroll factor also
increases register pressure because more intermediate values must be held simultaneously.

The following example assumes four units of time before a read arrives or when the write can be issued. The example also assumes that the link hasn't reached the capacity.

.. raw:: html

   <style>.tb-timeline th { font-family: inherit; } .tb-timeline td { font-family: monospace; }</style>
   <table class="tb-timeline" style="border-collapse:collapse;text-align:center;">
     <colgroup>
       <col style="width:80px;">
       <col span="24" style="width:32px;">
     </colgroup>
     <tbody>
       <tr>
         <th style="text-align:left;padding:4px 8px;">Unroll 1</th>
         <td style="background:#c8f0d8;border:1px solid #ccc;">A</td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#c8f0d8;border:1px solid #ccc;">A</td>
         <td style="background:#f8c8c8;border:1px solid #ccc;">B</td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#f8c8c8;border:1px solid #ccc;">B</td>
         <td style="background:#c8f0d8;border:1px solid #ccc;">C</td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#f8c8c8;border:1px solid #ccc;">C</td>
         <td style="background:#c8f0d8;border:1px solid #ccc;">D</td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#f8c8c8;border:1px solid #ccc;">D</td>
         <td style="border:1px solid #ccc;"></td>
       </tr>
       <tr>
         <th style="text-align:left;padding:4px 8px;">Unroll 2</th>
         <td style="background:#c8f0d8;border:1px solid #ccc;">A</td>
         <td style="background:#c8f0d8;border:1px solid #ccc;">B</td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#f8c8c8;border:1px solid #ccc;">A</td>
         <td style="background:#f8c8c8;border:1px solid #ccc;">B</td>
         <td style="background:#c8f0d8;border:1px solid #ccc;">C</td>
         <td style="background:#c8f0d8;border:1px solid #ccc;">D</td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#aaa;border:1px solid #ccc;"></td>
         <td style="background:#f8c8c8;border:1px solid #ccc;">C</td>
         <td style="background:#f8c8c8;border:1px solid #ccc;">D</td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
       </tr>
       <tr>
         <th style="text-align:left;padding:4px 8px;">Unroll 4</th>
         <td style="background:#c8f0d8;border:1px solid #ccc;">A</td>
         <td style="background:#c8f0d8;border:1px solid #ccc;">B</td>
         <td style="background:#c8f0d8;border:1px solid #ccc;">C</td>
         <td style="background:#c8f0d8;border:1px solid #ccc;">D</td>
         <td style="background:#f8c8c8;border:1px solid #ccc;">A</td>
         <td style="background:#f8c8c8;border:1px solid #ccc;">B</td>
         <td style="background:#f8c8c8;border:1px solid #ccc;">C</td>
         <td style="background:#f8c8c8;border:1px solid #ccc;">D</td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
         <td style="border:1px solid #ccc;"></td>
       </tr>
     </tbody>
   </table>

The measured effect of unroll factor varies by transfer type. The following table shows
example bandwidth values (in GB/s):

.. list-table::
    :header-rows: 1

    * - ``GFX_UNROLL``
      - Local copy with 4 CUs (``1 4 G0->G0->G0``)
      - Remote 1 SubExecutor copy (``1 1 G0->G0->G1``)

    * - 1
      - 20.297
      - 20.297

    * - 2
      - 37.669
      - 36.599

    * - 3
      - 48.781
      - 48.439

    * - 4
      - 62.887
      - 59.407

    * - 5
      - 74.076
      - 44.100

    * - 6
      - 84.769
      - 59.386

    * - 7
      - 95.074
      - —

    * - 8
      - 101.101
      - —

For the remote copy case, performance doesn't scale monotonically beyond unroll 4 because
the link becomes the bottleneck rather than register occupancy.

To configure the unroll factor, see :ref:`GFX_UNROLL environment variable <gfx-options>`.

Preset questions
=================

This section answers common questions about TransferBench presets.

.. _mem-type:

What memory types do presets support?
---------------------------------------

Some TransferBench presets use the ``MEM_TYPE`` environment variable (or CPU- and
GPU-specific variants) to select the memory type used during the transfer. The following
table lists the supported memory types based on CPU or GPU:

.. list-table::
    :header-rows: 1

    * - Memory device
      - Memory type index
      - Description
      - Symbol
      - Allocation method

    * - CPU
      - 0
      - Default pinned host memory
      - ``C``
      - ``hipHostMalloc``

    * - CPU
      - 1
      - Coherent pinned host memory
      - ``B``
      - ``hipHostMalloc`` with ``hipHostMallocCoherent`` flag

    * - CPU
      - 2
      - Non-coherent pinned host memory
      - ``D``
      - ``hipHostMalloc`` with ``hipHostMallocNonCoherent`` flag

    * - CPU
      - 3
      - Uncached pinned host memory
      - ``K``
      - ``hipHostMalloc`` with ``hipHostMallocUncached`` flag

    * - CPU
      - 4
      - Unpinned host memory
      - ``H``
      - ``numa_alloc_onnode``

    * - GPU
      - 0
      - Default GPU memory
      - ``G``
      - ``hipMalloc``

    * - GPU
      - 1
      - Fine-grained GPU memory
      - ``F``
      - ``hipExtMallocWithFlags`` with ``hipDeviceMallocFinegrained``

    * - GPU
      - 2
      - Uncached GPU memory
      - ``U``
      - ``hipExtMallocWithFlags`` with ``hipDeviceMallocUncached``

    * - GPU
      - 3
      - Managed memory
      - ``M``
      - ``hipMallocManaged``

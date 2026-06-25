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

Common causes and resolutions:

- **Improperly configured IOMMU** — IOMMU must be set to pass-through mode in the BIOS. To verify, check for ``iommu=pt`` in the kernel command line:

  .. code-block:: shell

    # Check for iommu=pt in the output
    cat /proc/cmdline

    BOOT_IMAGE=/boot/vmlinuz-5.15.0-70-generic root=UUID=7489cc43-aaab-4b61-8c63-86a419728dea
    ro panic=0 nowatchdog msr.allow_writes=on nokaslr amdgpu.noretry=1 pci=realloc=off
    modprobe.blacklist=amdgpu intel_iommu=on iommu=pt numa_balancing=disable console=tty0
    console=ttyS0,115200n8

  For IOMMU configuration guidance, see
  `AMD Instinct MI300X system optimization <https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html>`_.

- **ROCm runtime and driver version mismatch** — Ensure the installed ROCm runtime and GPU driver versions are compatible. Refer to the ROCm compatibility matrix for supported combinations.

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
  ┌---------┬-----------------------------------------------------------------------┬-----------┬---------┐
  │ SRC\DST │ GPU 00  GPU 01  GPU 02  GPU 03  GPU 04  GPU 05  GPU 06  GPU 07      │  STotal   │  Actual │
  ├---------┼-----------------------------------------------------------------------┼-----------┼---------┤
  │ GPU 00  │  N/A    60.99   60.95   60.94   61.01   61.01   60.99   60.95       │  426.83   │  426.59 │
  │ GPU 01  │ 61.00    N/A    60.93   60.93   61.00   60.99   60.95   60.94       │  426.73   │  426.48 │
  │ GPU 02  │ 60.94   60.93    N/A    60.97   61.01   60.98   60.99   60.92       │  426.74   │  426.43 │
  │ GPU 03  │ 60.96   60.96   60.99    N/A    60.97   60.97   60.96   60.94       │  426.75   │  426.59 │
  │ GPU 04  │ 60.99   60.99   61.01   60.98    N/A    60.98   61.01   60.99       │  426.96   │  426.87 │
  │ GPU 05  │ 60.95   60.96   60.93   60.94   60.91    N/A    60.91   60.97       │  426.57   │  426.36 │
  │ GPU 06  │ 60.84   60.84   60.80   60.87   60.89   60.84    N/A    60.83       │  425.91   │  425.59 │
  │ GPU 07  │ 60.94   60.94   60.94   61.00   60.99   61.04   60.93    N/A        │  426.79   │  426.51 │
  ├---------┼-----------------------------------------------------------------------┼-----------┼---------┤
  │ RTotal  │426.62  426.61  426.55  426.64  426.78  426.82  426.72  426.54       │ 3413.29   │ 3411.43 │
  └---------┴-----------------------------------------------------------------------┼-----------┼---------┤
                                                                                    │CPU Timed: │ 1338.25 │
                                                                                    └-----------┴---------┘
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
  ┌---------┬-----------------------------------------------------------------------┬-----------┬---------┐
  │ SRC\DST │ GPU 00  GPU 01  GPU 02  GPU 03  GPU 04  GPU 05  GPU 06  GPU 07      │  STotal   │  Actual │
  ├---------┼-----------------------------------------------------------------------┼-----------┼---------┤
  │ GPU 00  │  N/A    60.52   60.38   60.41   60.96   60.98   60.93   60.97       │  425.15   │  422.69 │
  │ GPU 01  │ 60.46    N/A    60.47   60.49   60.94   60.95   60.95   60.99       │  425.24   │  423.20 │
  │ GPU 02  │ 57.91   58.13    N/A    58.28   58.77   58.80   58.88   58.89       │  409.66   │  405.36 │
  │ GPU 03  │ 57.96   58.18   58.32    N/A    59.15   58.37   56.43   56.54       │  404.95   │  395.03 │
  │ GPU 04  │ 60.32   60.43   60.48   60.97    N/A    60.96   61.02   60.99       │  425.17   │  422.25 │
  │ GPU 05  │ 60.42   60.37   60.37   60.94   60.96    N/A    60.98   61.02       │  425.06   │  422.59 │
  │ GPU 06  │ 60.40   60.27   60.37   60.96   61.00   60.95    N/A    60.96       │  424.91   │  421.90 │
  │ GPU 07  │ 60.38   60.35   60.37   60.94   60.96   61.01   60.97    N/A        │  424.97   │  422.45 │
  ├---------┼-----------------------------------------------------------------------┼-----------┼---------┤
  │ RTotal  │417.84  418.25  420.76  422.99  422.75  422.01  420.16  420.35       │ 3365.11   │ 3335.47 │
  └---------┴-----------------------------------------------------------------------┼-----------┼---------┤
                                                                                    │CPU Timed: │ 2222.41 │
                                                                                    └-----------┴---------┘
  Average bandwidth (GPU Timed): 60.091 GB/s
  Aggregate bandwidth (GPU Timed): 3365.111 GB/s
  Aggregate bandwidth (CPU Timed): 2222.415 GB/s

.. note::

  The per-cell bandwidth numbers in the SRC×DST matrix look similar in both cases because
  each individual transfer is timed from when it starts — not from when the full test begins.
  A serialized transfer still achieves full link speed; it just starts later than it would if
  a hardware queue were available. The effect of serialization is therefore visible in the
  CPU wall-clock time and the ``Actual`` column, not in the per-cell numbers. In the
  ``GPU_MAX_HW_QUEUES=4`` case the CPU-timed aggregate bandwidth is roughly half the
  GPU-timed aggregate, while in the ``GPU_MAX_HW_QUEUES=8`` case the two are much closer.

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
         bdf           bit_rate  max_bandwidth  link_type  GPU0      GPU1      GPU2      GPU3      GPU4      GPU5      GPU6      GPU7
  GPU0   0000:0c:00.0  38 Gb/s   608 Gb/s       XGMI
   Read                                                    N/A       39.61 TB  15.40 TB  15.47 TB  5.349 TB  4.993 TB  5.078 TB  5.952 TB
   Write                                                   N/A       41.96 TB  15.32 TB  15.00 TB  5.332 TB  4.859 TB  4.979 TB  5.448 TB
  GPU1   0000:22:00.0  38 Gb/s   608 Gb/s       XGMI
   Read                                                    41.69 TB  N/A       16.93 TB  16.61 TB  5.498 TB  5.850 TB  5.368 TB  5.261 TB
   Write                                                   39.38 TB  N/A       17.81 TB  16.70 TB  5.419 TB  5.842 TB  5.366 TB  5.371 TB
  GPU2   0000:38:00.0  38 Gb/s   608 Gb/s       XGMI
   Read                                                    15.28 TB  17.82 TB  N/A       16.96 TB  5.529 TB  5.402 TB  5.781 TB  5.194 TB
   Write                                                   15.40 TB  16.93 TB  N/A       17.81 TB  5.513 TB  5.475 TB  5.666 TB  5.263 TB
  GPU3   0000:5c:00.0  38 Gb/s   608 Gb/s       XGMI
   Read                                                    15.02 TB  16.66 TB  17.79 TB  N/A       5.791 TB  5.350 TB  5.131 TB  5.755 TB
   Write                                                   15.47 TB  16.61 TB  16.86 TB  N/A       6.251 TB  5.425 TB  5.244 TB  5.673 TB
  GPU4   0000:9f:00.0  38 Gb/s   608 Gb/s       XGMI
   Read                                                    5.346 TB  5.427 TB  5.524 TB  6.279 TB  N/A       5.797 TB  5.522 TB  5.454 TB
   Write                                                   5.349 TB  5.498 TB  5.531 TB  5.791 TB  N/A       6.243 TB  5.568 TB  5.336 TB
  GPU5   0000:af:00.0  38 Gb/s   608 Gb/s       XGMI
   Read                                                    4.861 TB  5.858 TB  5.471 TB  5.435 TB  6.233 TB  N/A       5.789 TB  5.727 TB
   Write                                                   4.995 TB  5.850 TB  5.402 TB  5.351 TB  5.771 TB  N/A       6.213 TB  5.711 TB
  GPU6   0000:bf:00.0  38 Gb/s   608 Gb/s       XGMI
   Read                                                    4.980 TB  5.348 TB  5.664 TB  5.249 TB  5.556 TB  6.208 TB  N/A       6.020 TB
   Write                                                   5.079 TB  5.368 TB  5.780 TB  5.134 TB  5.505 TB  5.791 TB  N/A       6.450 TB
  GPU7   0000:df:00.0  38 Gb/s   608 Gb/s       XGMI
   Read                                                    5.434 TB  5.372 TB  5.260 TB  5.651 TB  5.315 TB  5.650 TB  6.446 TB  N/A
   Write                                                   5.952 TB  5.261 TB  5.197 TB  5.755 TB  5.433 TB  5.686 TB  6.020 TB  N/A

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

For the local copy case, bandwidth scales steadily across all unroll values in the
preceding table because the GPU has sufficient compute and memory bandwidth to keep higher
unroll pipelines busy.

For the remote copy case, the preceding table shows that performance doesn't scale
monotonically beyond unroll 4. At that point the interconnect link — not register occupancy
— becomes the bottleneck. Once the link is saturated, issuing more reads ahead of time
provides no further benefit, and increased register pressure can hurt performance. The
remote copy column has no entries beyond unroll 6 because the link saturates before higher
unroll factors can show improvement.

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

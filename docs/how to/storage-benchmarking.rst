.. meta::
  :description: How to benchmark I/O between NVMe storage and GPU memory with TransferBench using the hipFile Executor, including setup, safety, and alignment.
  :keywords: TransferBench NVMe, TransferBench storage, TransferBench hipFile, AMD Infinity Storage, GPU storage benchmark, TB_STORAGE_PATHS

.. _storage-benchmarking:

==============================================
Benchmarking storage I/O using TransferBench
==============================================

TransferBench can benchmark reads from storage into GPU memory and writes from GPU memory to storage. Storage is typically a file on an NVMe-backed file system or a raw NVMe block device. This topic describes how to set up and run these tests, and what to check before you trust the results.

Storage is modeled as a memory type, ``S``, and the I/O is performed by the hipFile Executor:

.. list-table::
    :header-rows: 1

    * - Executor
      - Letter
      - How data is moved
      - Requires

    * - hipFile
      - ``F``
      - ``hipFileRead`` and ``hipFileWrite`` of the hipFile library. hipFile uses direct I/O between storage and GPU memory (its fastpath) when it can, and otherwise falls back to POSIX I/O through host memory (its compatibility path).
      - TransferBench built with hipFile support, and ``libhipfile.so.0`` at runtime

The hipFile Executor isn't supported on NVIDIA hardware.

Checking availability
=====================

The hipFile Executor is available when TransferBench was built with hipFile support, which happens automatically when the hipFile development package is installed at build time. See :ref:`source-build`.

Run TransferBench without arguments to see what is available:

.. code-block:: shell

  ./TransferBench

The version line lists ``hipFile`` among the supported features, and the topology output includes a storage section:

.. code-block:: shell

   Storage | Path (TB_STORAGE_PATHS)
  ---------+-------------------------
   S0      | /mnt/nvme0/tb.dat
   hipFile | Available (hipFile 0.5.0)

An available hipFile Executor doesn't guarantee that direct I/O (hipFile's fastpath) works for a particular GPU and storage device. Direct I/O also depends on the GPU kernel driver, on PCIe peer-to-peer DMA support in the kernel, and on the storage being a local NVMe device. The ``ais-check`` tool that ships with hipFile reports these requirements.

Selecting the storage
=====================

Set ``TB_STORAGE_PATHS`` to a comma-separated list of paths. Storage index ``N`` in a transfer (``SN``) refers to the N-th path, starting from 0:

.. code-block:: shell

  export TB_STORAGE_PATHS=/mnt/nvme0/tb.dat,/mnt/nvme1/tb.dat

Each path is one of the following:

- **A regular file.** TransferBench creates the file if needed and grows it to the size the test requires. The file isn't deleted afterwards. A file that already holds data is only read from, unless you allow it to be overwritten. See :ref:`storage-existing-files`. For direct I/O, hipFile requires the file to be on xfs or on ext4 mounted with ``data=ordered``.

  A path under ``/dev/`` that doesn't exist is rejected instead of being created, because it's most likely a mistyped device name: creating it would put a regular file in the ``devtmpfs`` file system, in host memory. Paths under ``/dev/shm/`` are the exception.

- **A raw block device**, such as ``/dev/nvme1n1``. TransferBench never creates or resizes a block device, and by default only reads from it. See :ref:`storage-block-devices`.

Each entry must be a distinct file or device. TransferBench rejects two entries that refer to the same one, and an empty entry ahead of a path (for example ``,/mnt/nvme1/tb.dat``), which would renumber the indices that follow it. With multiple ranks, each rank reads its own ``TB_STORAGE_PATHS``, and ranks running on the same host need distinct paths: a file is locked while a rank uses it, and a block device that is written to is opened exclusively.

.. warning::

  Use a file or device on the NVMe drive you want to measure. A path on another kind of storage, such as a ``tmpfs`` mount or a network file system, runs, but it measures that storage, and direct I/O isn't available there.

Running the nvme preset
=======================

The :ref:`nvme preset <nvme>` writes from each GPU to each storage path and then reads from each storage path into each GPU, validating the data of every transfer, using the hipFile Executor. The write and the read are separate transfers: TransferBench fills the storage with fresh source data before the read, so the read doesn't depend on what the write left behind. A block device or an existing file that TransferBench isn't allowed to write to is the exception. See :ref:`storage-existing-files` and :ref:`storage-block-devices`.

.. code-block:: shell

  TB_STORAGE_PATHS=/mnt/nvme0/tb.dat ./TransferBench nvme 1G

The preset prints the bandwidth and the I/O path of every transfer, and exits with a non-zero exit code if any transfer fails.

The file is left in place, and it now holds data. To run the preset on it again, delete the file first or allow it to be overwritten:

.. code-block:: shell

  TB_STORAGE_PATHS=/mnt/nvme0/tb.dat TB_STORAGE_ALLOW_OVERWRITE=1 ./TransferBench nvme 1G

Running custom storage transfers
================================

Storage transfers can be mixed with any other transfers in a configuration file or on the command line. See :ref:`storage-transfers` for the rules.

.. code-block:: shell

  # hipFile read from storage 0 into GPU 0 with 4 I/O threads
  ./TransferBench cmdline 1G "1 4 (S0->F0->G0)"

  # hipFile write from GPU 0 to storage 0 with 8 I/O threads
  ./TransferBench cmdline 1G "1 8 (G0->F0->S0)"

  # Four GPUs each reading from their own drive at the same time
  ./TransferBench cmdline 1G "4 4 (S0->F0->G0) (S1->F1->G1) (S2->F2->G2) (S3->F3->G3)"

The number of SubExecutors is the number of threads that issue blocking I/O calls for the transfer. Each thread handles a contiguous part of the data.

Several transfers in one test can use the same storage index. Each transfer gets its own region of the file or device, so they don't overwrite each other, and the file grows to the sum of the regions.

Set ``TB_VERBOSE=1`` to see which I/O path each hipFile Executor used:

.. code-block:: shell

  [INFO] Executor (R0F0) storage I/O path: compat

Checking the I/O path
=====================

The hipFile Executor reports one of the following I/O paths:

.. list-table::
    :header-rows: 1

    * - I/O path
      - Meaning

    * - ``fastpath``
      - Direct I/O between storage and GPU memory.

    * - ``compat``
      - hipFile's compatibility path: POSIX I/O through host memory, followed by a copy to or from the GPU.

    * - ``mixed``
      - Both paths were used during the test.

    * - ``unknown``
      - hipFile doesn't provide the statistics needed to tell: hipFile is older than 0.5.0, or its statistics are turned off with ``HIPFILE_STATS_LEVEL=0``.

A ``compat`` result measures something very different from a ``fastpath`` result, so always check the I/O path before comparing numbers. To control which path hipFile uses, set hipFile's own environment variables. TransferBench never sets them:

.. code-block:: shell

  # Fail instead of falling back, so that a result is guaranteed to be direct I/O
  HIPFILE_ALLOW_COMPAT_MODE=false ./TransferBench cmdline 1G "1 4 (S0->F0->G0)"

  # Measure the compatibility path only
  HIPFILE_FORCE_COMPAT_MODE=true ./TransferBench cmdline 1G "1 4 (S0->F0->G0)"

hipFile reads these variables once, when it is first used, so one process uses one setting for all its hipFile transfers. See :ref:`storage-env-vars` for the list.

.. note::

  If hipFile attempts direct I/O on a system where it can't work, the transfer fails with an error instead of falling back. Set ``HIPFILE_FORCE_COMPAT_MODE=true`` to run on such a system, for example on WSL2 with a hipFile version that doesn't detect it.

Alignment
=========

Direct I/O requires the file offset, the size, and the GPU buffer offset to be multiples of 4096 bytes. TransferBench therefore divides a storage transfer between its I/O threads in multiples of 4096 bytes, regardless of ``BLOCK_BYTES``.

A transfer size or ``BYTE_OFFSET`` that isn't a multiple of 4096 bytes is accepted with a warning, because hipFile uses its compatibility path for unaligned I/O. Expect a ``compat`` or ``mixed`` I/O path in that case.

A single I/O call moves at most ``0x7ffff000`` bytes. TransferBench issues as many calls as needed, so a transfer or a share of an I/O thread can be larger than that.

See :ref:`storage-timing` for what is timed.

.. _storage-existing-files:

Existing files
==============

TransferBench writes to storage even for a transfer that only reads from it, because it fills the storage source with known data first. A destination is also cleared before the test, so that data validation can't pass on what an earlier run left there. To keep this from destroying data, a regular file that already holds data when TransferBench first uses it is protected in the same way as a block device:

- The file is opened read-only and isn't grown. The test fails if the file is smaller than the test requires.

- A transfer that writes to it, such as ``(G0->F0->S0)``, is rejected before the file is opened.

- A transfer that reads from it, such as ``(S0->F0->G0)``, transfers the existing contents. Data validation is skipped for that transfer, and a warning is printed.

- Set ``TB_STORAGE_ALLOW_OVERWRITE=1`` to allow TransferBench to overwrite such files. Only the value ``1`` enables this.

A file that doesn't exist or is empty when TransferBench first uses it has nothing to lose. TransferBench creates it, and it stays writable for the rest of that run, so that later tests in the same run can reuse it. A file left behind by an earlier run holds data: delete it or set ``TB_STORAGE_ALLOW_OVERWRITE=1`` to use it again.

.. _storage-block-devices:

Block devices
=============

.. warning::

  Writing to a raw block device destroys its contents, including any file system and partition table on it. Double-check the device name before you allow it.

TransferBench protects block devices listed in ``TB_STORAGE_PATHS`` as follows:

- By default, block devices are opened read-only. A transfer that writes to one, such as ``(G0->F0->S0)``, is rejected before the device is opened.

- A transfer that reads from one, such as ``(S0->F0->G0)``, transfers the existing contents of the device. Because TransferBench can't fill the device with known data, data validation is skipped for that transfer, and a warning is printed.

- Set ``TB_STORAGE_ALLOW_BLOCK_WRITE=1`` to allow TransferBench to write to block devices. Reads are then validated too, because TransferBench fills the device region with source data first. Only the value ``1`` enables this.

- A block device that is written to is opened exclusively. The open fails if the device, or one of its partitions, is mounted or otherwise in use, for example as swap or as part of an LVM or RAID volume. It also fails if another rank on the same host has the device open for writing.

- A block device is never created, truncated, or resized. The test fails if the device is smaller than the test requires.

Reading or writing a block device usually requires root privileges or membership in the ``disk`` group.

Troubleshooting
===============

.. list-table::
    :header-rows: 1

    * - Message
      - Cause and action

    * - ``No storage paths are defined on rank 0``
      - ``TB_STORAGE_PATHS`` isn't set. Set it to a comma-separated list of files or block devices.

    * - ``hipFile executor is requested but is not available``
      - TransferBench was built without hipFile support. Install the hipFile development package (for example ``amdrocm-hipfile-dev10.2``) and rebuild. See :ref:`source-build`.

    * - ``error while loading shared libraries: libhipfile.so.0``
      - TransferBench was built with hipFile support and the hipFile runtime library isn't installed on this machine. Install the hipFile runtime package (for example ``amdrocm-hipfile10.2``), or add the directory that holds the library to ``LD_LIBRARY_PATH``.

    * - ``hipFile's fastpath may not be usable here``
      - hipFile attempted direct I/O and the HIP runtime rejected it. Check the system with ``ais-check``, or set ``HIPFILE_FORCE_COMPAT_MODE=true`` to use the compatibility path.

    * - ``Storage 0 (...) does not exist. Paths under /dev/ ... are not created``
      - The path is under ``/dev/`` and doesn't exist. Check the device name with ``lsblk``.

    * - ``Storage 0 (...) is a block device. Writing to it destroys its contents``
      - A transfer writes to a block device without ``TB_STORAGE_ALLOW_BLOCK_WRITE=1``. See :ref:`storage-block-devices`.

    * - ``Storage 0 (...) is an existing file that holds data. Writing to it destroys its contents``
      - A transfer writes to a file that already holds data without ``TB_STORAGE_ALLOW_OVERWRITE=1``. See :ref:`storage-existing-files`.

    * - ``Storage 1 (...) and storage 0 (...) refer to the same file``
      - Two ``TB_STORAGE_PATHS`` entries name the same file or device. Give each storage index its own.

    * - ``Storage 0 (...) is locked by another process``
      - Another TransferBench rank or run on this host is using the file. Give each rank its own path.

    * - ``TB_STORAGE_PATHS has an empty entry ahead of a path``
      - The list has an empty entry before a path, usually from an unset variable in a script. Remove it.

    * - ``Storage memory (S) can only be accessed by the hipFile (F) executor``
      - Storage memory was used with another Executor. See :ref:`storage-transfers`.

.. meta::
  :description: Explains how TransferBench validates transfer correctness by comparing destination memory against precomputed expected values derived from source buffers.
  :keywords: TransferBench data validation, TransferBench correctness, ValidateAllTransfers, PrepareReference, destination buffer, source buffer

.. _transferbench-data-validation:

==============================
TransferBench data validation
==============================

TransferBench validates the transfer results by comparing the destination (DST) memory to
precomputed expected values.

Overview
=========

For each transfer, the DST buffer must equal the element-wise sum of all SRC buffers, or zero if there are no sources. A transfer is correct if, for every element ``i``, the value matches the expected value given in the following table:

.. list-table::
    :header-rows: 1

    * - Number of sources
      - Expected value

    * - 0 sources
      - ``dst[i] == 0`` (or memset value)

    * - 1 source
      - ``dst[i] == src0[i]``

    * - N sources
      - ``dst[i] == src0[i] + src1[i] + ... + srcN-1[i]``

Source data preparation
=======================

Before any transfers run, TransferBench prepares the SRC and DST memories as discussed in the following sections:

Expected source pattern
-----------------------

Before any transfers run, TransferBench builds reference SRC buffers on the host using
``PrepareReference(cfg, cpuBuffer, bufferIdx)``.

The pattern used depends on the configuration:

.. list-table::
    :header-rows: 1

    * - Configuration
      - Behavior

    * - ``fillCompress`` (non-empty)
      - Mix of random floats with optional zeroing per 64-byte line:
        ``0`` = random, ``1`` = 1B0, ``2`` = 2B0, ``3`` = 4B0, ``4`` = 32B0.
        Percentages control the mix. For details, see
        :ref:`data-validation-var`.

    * - ``fillPattern`` (non-empty)
      - Repeats the given ``vector<float>`` over all SRC buffers.

    * - Default
      - Pseudo-random: ``PrepSrcValue(bufferIdx, i) = (((i % 383) * 517) % 383 + 31) * (bufferIdx + 1)``

        ``bufferIdx`` is the SRC index (0, 1, …) so each SRC buffer gets a different pattern.

Expected destination (``dstReference``)
----------------------------------------

The expected destination is computed once before the iteration loop:

.. code-block:: text

  dstReference[0] = memset to MEMSET_CHAR               # used when numSrcs == 0
  dstReference[1] = srcReference[0]                     # 1 source
  dstReference[2] = dstReference[1] + srcReference[1]   # 2 sources
  dstReference[k] = dstReference[k-1] + srcReference[k-1]  # k sources

``dstReference[numSrcs]`` is the expected result for a transfer with ``numSrcs`` sources.

Initializing source and destination memories
---------------------------------------------

For each transfer, the SRC memory on the rank that owns it is filled from the corresponding
``srcReference`` buffer via ``hipMemcpy`` (host-to-device or device-to-device as appropriate).
DST memory is zeroed (or memset) before transfers run.

How validation is timed
========================

The timing of validation is controlled by the ``alwaysValidate`` option. By default
(``alwaysValidate = 0``), validation runs once after all timed iterations complete,
minimizing overhead during benchmarking. When ``alwaysValidate = 1``, validation is
performed after every iteration; any detected error immediately stops the run.

.. list-table::
    :header-rows: 1

    * - Option
      - When
      - Behavior

    * - ``alwaysValidate = 0`` (default)
      - Once at the end of all iterations
      - ``ValidateAllTransfers`` called after the iteration loop.

    * - ``alwaysValidate = 1``
      - After every timed iteration
      - ``ValidateAllTransfers`` called inside the loop; any error stops the run.

How validation (``ValidateAllTransfers``) works
================================================

For each transfer and each DST, the following steps are performed:

1. **Rank check:** Only the rank that owns the destination performs validation.

2. **Get the actual output:**

   - **CPU destination** or ``validateDirect = 1``: Point directly at the destination memory.
   - **GPU destination** and ``validateDirect = 0``: Copy destination to a host ``outputBuffer``
     via ``hipMemcpy``, then compare against ``outputBuffer``.

3. **Comparison:** Performed using ``memcmp(output, expected, numBytes)``. On mismatch, the code finds the first differing index and returns an error with the index, expected value, and actual value.

4. **Expected values:** Calculated using ``expected = dstReference[t.srcs.size()].data()``. The precomputed sum for the number of sources.

Validation options
==================

The following options control when and how validation is performed. They can be set as
environment variables or in a configuration file.

.. list-table::
    :header-rows: 1

    * - Option
      - Environment variable
      - Description

    * - ``alwaysValidate``
      - ``ALWAYS_VALIDATE``
      - To validate after each iteration, set to ``1``. To validate once at the end, set to ``0``.

    * - ``validateDirect``
      - ``VALIDATE_DIRECT``
      - To compare GPU DST directly, set to ``1``. Supported on AMD hardware only and requires no host copy.
        To copy to host and compare, set to ``0``.

    * - ``validateSource``
      - ``VALIDATE_SOURCE``
      - To validate the SRC memory right after it's initialized, set to ``1`` (optional early check).

.. note::

  ``validateDirect`` is not supported on NVIDIA. The code falls back to copying to host.

.. meta::
  :description: How to run custom transfer tests using TransferBench by defining them in a configuration file or on the command line, including wildcard syntax and examples.
  :keywords: TransferBench usage, TransferBench how to, TransferBench configuration file, TransferBench custom transfers, TransferBench wildcards

.. _running-transferbench-customized:

========================================
Running custom tests using TransferBench
========================================

Define custom transfer tests in a configuration file and run them with TransferBench. This topic describes the configuration file format and how to run tests using a file or the command line.

.. seealso::

  :ref:`transfer-definition-syntax` — complete reference for simple and advanced mode syntax, memory and Executor letter codes, and wildcards.

Running TransferBench with a configuration file
================================================

To run TransferBench with a configuration file, use:

.. code-block:: shell

    ./TransferBench <config_file> [num_bytes]

The command accepts the following arguments:

- ``config_file``: Path to a configuration file that defines the transfers to run.

- ``num_bytes``: Number of bytes per transfer. You can suffix this value with ``K``, ``M``, or ``G`` (for example, ``128M``); the value must be a multiple of 4. This argument is optional.

.. note::

  If you set ``num_bytes`` to ``0``, TransferBench sweeps over transfer sizes from 1 KB to 512 MB in power-of-2 steps. Use ``SAMPLING_FACTOR`` to control how many sizes are sampled per power-of-2 range. For example, ``SAMPLING_FACTOR=4`` produces four evenly spaced sizes between each power of 2.

Alternative modes
------------------

In addition to configuration file mode, TransferBench supports the following alternative modes:

.. list-table::
    :header-rows: 1

    * - Mode
      - Usage
      - Description

    * - ``cmdline``
      - ``./TransferBench cmdline [num_bytes] <transfer_definitions...>``
      - Defines transfers on the command line instead of in a file.

    * - ``dryrun``
      - ``./TransferBench dryrun [num_bytes] <transfer_definitions...>``
      - Parses and prints the expanded transfers without executing them. Use this mode to validate wildcard expansion.

- Using ``cmdline`` mode:

  .. code-block:: shell

    ./TransferBench cmdline 1G "1 1 (G0->G0->G1)"

  This produces the same result as a configuration file containing:

  .. code-block:: shell

    1 1 (G0->G0->G1)

- Using ``dryrun`` mode:

  .. code-block:: shell

    ./TransferBench dryrun 1G "1 1 (G0->G0->G*)"

  The output lists all transfers that would be executed:

  .. code-block:: shell

    =============================================================================================================
    Transfers to be executed (dry-run):
    ================================================================================
    Transfer     0: (G0->G0->G0)
    Transfer     1: (G0->G0->G1)
    Transfer     2: (G0->G0->G2)
    Transfer     3: (G0->G0->G3)
    Transfer     4: (G0->G0->G4)
    Transfer     5: (G0->G0->G5)
    Transfer     6: (G0->G0->G6)
    Transfer     7: (G0->G0->G7)

Configuration file format
==========================

Each line in a configuration file defines one test, which is a set of transfers that run in parallel. The following rules apply:

- Blank lines are ignored.

- Lines starting with ``#`` are treated as comments and are ignored.

- Lines starting with ``##`` are echoed to the output. Use these for section headers.

- Round brackets ``()`` and arrows ``->`` are optional and ignored. You can include them for readability.

Example configuration file
============================

The following configuration file shows a range of transfer types using both simple and advanced modes:

.. code-block:: shell

  # Single GPU-executed transfer between GPUs 0 and 1 using 4 CUs
  1 4 (G0->G0->G1)

  # Single DMA transfer between GPUs 0 and 1
  1 1 (G0->D0->G1)

  # Advanced: 1 MB GPU 0 to GPU 1 with 4 CUs; 2 MB GPU 1 to GPU 0 with 8 CUs
  -2 (G0 G0 G1 4 1M) (G1 G1 G0 8 2M)

  # Memset by GPU 0 to its own memory (null source)
  1 32 (N0->G0->G0)

  # Read-only by CPU 0 (null destination)
  1 4 (C0->C0->N0)

  # Broadcast from GPU 0 to GPUs 0 and 1
  1 16 (G0->G0->G0G1)

  # Multi-rank: GPU 0 on rank 0 to GPU 1 on rank 1
  1 4 (R0G0->R0G0->R1G1)

For information on how to define transfers in the configuration file, including how to specify what memory to use, which Executor runs the transfer, and how many SubExecutors to use, see :ref:`transfer-definition-syntax`.

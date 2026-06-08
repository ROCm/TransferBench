.. meta::
  :description: TransferBench is a utility to benchmark simultaneous transfers between user-specified devices (CPUs or GPUs)
  :keywords: TransferBench workflow, RunTransfers, TransferBench internals, TransferBench architecture, TransferBench conceptual

.. _transferbench-workflow:

=======================
TransferBench workflow
=======================

Transfers enter the system either through presets or configuration (config) file, both of which ultimately call
``RunTransfers()``, which is the main utility function within TransferBench.

Entry points
============

.. list-table::
    :header-rows: 1

    * - Source
      - Flow

    * - Presets
      - The client selects a preset (for example, ``p2p``, ``nicp2p``, or ``sweep``).
        The preset builds a ``std::vector<Transfer>`` and calls
        ``TransferBench::RunTransfers(cfg, transfers, results)``.

    * - Config file or command line
      - The client parses transfers from a config file or command-line arguments via
        ``ParseTransfers()``, then calls ``RunTransfers()``.

RunTransfers workflow
=====================

``RunTransfers()`` is the main entry point into the backend TransferBench library:

.. code-block:: cpp

    /**
     * Run a set of Transfers
     *
     * @param[in] config     Configuration options
     * @param[in] transfers  Set of Transfers to execute
     * @param[out] results   Timing results
     * @returns true if and only if Transfers were run successfully without any fatal errors
     */
    bool RunTransfers(ConfigOptions const& config,
                      vector<Transfer> const& transfers,
                      TestResults& results);

As shown in the following diagram, the function executes four sequential phases:

.. image:: /data/workflow.png
  :width: 100%
  :align: center

1. **Initial validation:** Checks that inputs are consistent and valid.
2. **Prepare transfers:** Allocates resources and initializes memory.
3. **Iteration loop:** Runs the timed transfer iterations.
4. **Finalize:** Validates results and assembles output.

Initial validation
------------------

Here are the steps involved in the first phase:

1. **Check ConfigOptions:** Verify that the provided ``ConfigOptions`` are valid. ``ConfigOptions`` control how TransferBench runs (for example, GFX unroll factor and number of warmup iterations). When running in multinode mode, consistency across ranks is also checked.

2. **Check transfers:** Verify that the provided ``Transfers`` are properly specified. Checks include confirming that requested devices exist and that each transfer has an appropriate number of source (SRC) and destination (DST) endpoints. When running in multinode mode, consistency across ranks is also checked.

3. **Log transfers (optional):** If ``TB_DUMP_CFG_FILE`` is set, log the transfers to a config file that can be re-executed by TransferBench. This is useful for capturing the exact transfers run by a preset so they can be modified and replayed.

Prepare transfers
-----------------

Here are the steps involved in the second phase:

1. **Prepare executors:** Perform all executor-specific setup, such as creating HIP streams, allocating SRC and DST memory, and exchanging fabric handles for pod communication support. This step also divides the work across subexecutors.

2. **Initialize memory:** Initializes SRC memory buffers with data patterns and computes reference results used for later validation.

   .. note::

    For GPU SRC memory locations, data is copied onto the GPUs via DMA (``hipMemcpy``). If profiling, this copy appears as part of the profiling trace, which is why setting ``USE_INTERACTIVE``=``1`` is recommended when profiling.

3. **Optional pause:** When ``USE_INTERACTIVE``=``1``, TransferBench pauses for user input after all memory has been initialized. Virtual addresses are printed at this point, which is useful for attaching a profiler before any transfers execute.

Iteration loop
--------------

In the third phase, iteration loop runs for the number of iterations specified by ``NUM_ITERATIONS``.
Each iteration proceeds through the following steps:

1. **Barrier (pre):** Synchronizes all ranks before transfers begin, ensuring that every rank is ready before any rank starts executing transfers.

2. **Start CPU timing:** Starts a CPU timer on the current rank, capturing the total elapsed time across all transfers on this rank.

3. **Execute:** Spawns one CPU thread per executor. Each executor runs all the transfers it is assigned and is responsible for its own per-transfer timing.

4. **Barrier (post):** Waits for all executors across all ranks to finish before proceeding.

5. **Stop CPU timing:** Stops the CPU timer.

6. **Validate (optional):** If ``ALWAYS_VALIDATE``=``1``, performs a correctness check after each iteration to verify that destination memory matches the expected reference results.

   .. note::

    By default, validation runs only once after all iterations complete. Setting ``ALWAYS_VALIDATE``=``1`` validates after every iteration, which can help detect transient errors that would otherwise be masked by a passing final iteration.

Finalize
--------

Here are the steps involved in the last phase:

1. **Validate all transfers:** Checks all transfers to confirm that the DST memory matches the expected reference results computed during the Initialize Memory step.

2. **Prepare results:** Collects timing data from each executor and assembles the final ``TestResults`` output returned to the caller.

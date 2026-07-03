.. meta::
  :description: Instructions for installing TransferBench using the ROCm package manager on supported platforms.
  :keywords: Install TransferBench, TransferBench package manager, TransferBench apt, TransferBench ROCm package

.. _package-manager:

==================================================
Installing TransferBench from the package manager
==================================================

To install TransferBench from the package manager, first configure the ROCm repository (for instructions, see `ROCm installation <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_), then run:

.. code-block:: shell

  ## Install the transferbench-dev package
  sudo apt-get install transferbench-dev

This installs in ``/opt/{rocm-version}/bin/TransferBench``. To check, run:

.. code-block:: shell

  dpkg -L transferbench-dev

.. note::

  The pre-packaged installation includes only the default features. NIC Executor, MPI support, and pod support require a :ref:`source build <source-build>`.

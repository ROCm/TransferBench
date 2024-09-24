.. meta::
  :description: TransferBench is a utility to benchmark simultaneous transfers between user-specified devices (CPUs or GPUs)
  :keywords: Build TransferBench, Install TransferBench, API, ROCm, HIP

.. _install-transferbench:

---------------------------
Installing TransferBench
---------------------------

This document provides information required to install and build TransferBench.

Prerequisite
---------------

* Install ROCm stack on the system to obtain HIP runtime
* Install ``libnuma`` on the system
* Enable AMD IOMMU and set to passthrough for AMD Instinct cards

Build TransferBench
---------------------

To build TransferBench using Makefile, use:

.. code-block:: bash

            $ make

To build TransferBench using CMake, use:

.. code-block:: bash

  mkdir build
  cd build
  CXX=/opt/rocm/bin/hipcc cmake ..
  make

.. note::

  If ROCm is installed in a folder other than ``/opt/rocm/``, set ``ROCM_PATH`` appropriately.

Build documentation
-----------------------

To build documentation locally, use:

.. code-block:: bash

  cd docs
  pip3 install -r .sphinx/requirements.txt
  python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html

NVIDIA platform support
--------------------------

You can build TransferBench to run on NVIDIA platforms via :ref:`HIP <hip:index>` or native NVCC.

To build with HIP for NVIDIA, install a HIP-compatible CUDA version such as CUDA 11.5 and use:

.. code-block:: bash

  CUDA_PATH=<path_to_CUDA> HIP_PLATFORM=nvidia make`

To build with native NVCC, use:

.. code-block:: bash

  make

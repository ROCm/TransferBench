.. meta::
  :description: ROCm Validation Suite documentation 
  :keywords: ROCm Validation Suite, RVS, ROCm, documentation

-------------
Requirements
-------------

* ROCm stack installed on the system (HIP runtime)
* libnuma installed on system

--------------------------
Building TransferBench
--------------------------

To build TransferBench using Makefile:

.. code-block:: bash

    $ make

To build TransferBench using CMake:

..code-block:: bash

    $ mkdir build
    $ cd build
    $ CXX=/opt/rocm/bin/hipcc cmake ..
    $ make

If ROCm is installed in a folder other than `/opt/rocm/`, set ROCM_PATH appropriately

--------------------------
NVIDIA platform support
--------------------------

TransferBench may also be built to run on NVIDIA platforms via HIP, but requires a HIP-compatible CUDA version installed. For example, CUDA 11.5.

To build:

.. code-block:: bash
    
   CUDA_PATH=<path_to_CUDA> HIP_PLATFORM=nvidia make`


.. meta::
  :description: TransferBench documentation 
  :keywords: TransferBench, API, ROCm, HIP

---------------------------
TransferBench installation
---------------------------

The following software is required to install TransferBench:

* ROCm stack installed on the system (HIP runtime)
* `libnuma` installed on the system

--------------------------
Building TransferBench
--------------------------

To build TransferBench using Makefile, use the following instruction:

.. code-block:: bash

            $ make

To build TransferBench using CMake, use the following commands:

.. code-block:: bash

                $ mkdir build
    
                $ cd build
    
                $ CXX=/opt/rocm/bin/hipcc cmake ..
    
                $ make

.. Note:: 

If ROCm is installed in a folder other than `/opt/rocm/`, set `ROCM_PATH` appropriately.

--------------------------
NVIDIA platform support
--------------------------

TransferBench may also be built to run on NVIDIA platforms via HIP, but requires a HIP-compatible CUDA version installed. For example, CUDA 11.5.

To build on NVIDIA platforms, use the following instruction:

.. code-block:: bash
    
             CUDA_PATH=<path_to_CUDA> HIP_PLATFORM=nvidia make`


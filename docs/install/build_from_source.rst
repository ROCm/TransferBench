.. meta::
  :description: Instructions for building TransferBench from source using Makefile or CMake, including required and optional dependencies.
  :keywords: Build TransferBench, TransferBench source build, TransferBench Makefile, TransferBench CMake, TransferBench dependencies

.. _source-build:

===================================
Building TransferBench from source
===================================

To build TransferBench from source, install the following required dependencies first:

Required dependencies
======================

* `ROCm stack <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_ to obtain :doc:`HIP runtime <hip:index>`.

  - The installed HIP version might impact support for some features, such as amd-smi pod membership detection, or UALoE support.

* ``libnuma`` for allocating memory or spawning threads on correct NUMA nodes.

  - For Ubuntu/Debian:

    .. code-block:: shell

      sudo apt install libnuma-dev

  - For RHEL/CentOS:

    .. code-block:: shell

      sudo yum install numactl-devel

Optional dependencies
======================

Depending on your requirement, you can install these optional dependencies:

- ``libibverbs``: Required for enabling NIC Executor for RDMA transfers.

  - For Ubuntu/Debian:

    .. code-block:: shell

      sudo apt install rdma-core libibverbs-dev ibverbs-utils

  - For RHEL/CentOS:

    .. code-block:: shell

      sudo yum install rdma-core libibverbs libibverbs-devel

- MPI installation (any of the following)

  - ``OpenMPI``:

    - For Ubuntu/Debian:

      .. code-block:: shell

        sudo apt install openmpi-bin libopenmpi-dev

    - For RHEL/CentOS:

      .. code-block:: shell

        sudo yum install openmpi openmpi-devel

  - ``MPICH``:

    - For Ubuntu/Debian:

      .. code-block:: shell

        sudo apt-get install mpich libmpich-dev

    - For RHEL/CentOS:

      .. code-block:: shell

        sudo yum install mpich mpich-devel

You can build TransferBench from source using two methods: :ref:`Makefile <makefile>` and :ref:`CMake <cmake>`.

.. _makefile:

Method 1: Building from source using Makefile
==============================================

To build TransferBench from source using Makefile, run:

.. code-block:: shell

  git clone https://github.com/ROCm/TransferBench.git
  cd TransferBench
  make

.. note::

  By default, ``make`` targets the GPU architecture detected on the build machine (``GPU_TARGETS=native``). To target specific architectures, set ``GPU_TARGETS``. See :ref:`menv-var`.

.. _menv-var:

Makefile environment variables
-------------------------------

To modify the Makefile behavior, use the following environment variables:

.. raw:: html

  <div class="pst-scrollable-table-container">
    <table id="makefile-env-var" class="table">
        <thead>
            <tr>
                <th>Category</th>
                <th>Environment variable</th>
                <th>Description</th>
                <th>Default value</th>
            </tr>
        </thead>
        <colgroup>
            <col span="1">
            <col span="1">
        </colgroup>
        <tbody class="makefile-env-variables">
            <tr>
              <td rowspan="7"><b>Paths and compilers</b> - To customize which compiler to use or the library to link against.</td>
              <td><code>ROCM_PATH</code></td>
              <td>ROCm installation path for HIP compiler, includes, and libs.</td>
              <td><code>/opt/rocm</code></td>
            </tr>
            <tr>
              <td><code>CUDA_PATH</code></td>
              <td>CUDA installation path for NVCC when building <code>TransferBenchCuda</code>.</td>
              <td><code>/usr/local/cuda</code></td>
            </tr>
            <tr>
              <td><code>MPI_PATH</code></td>
              <td>MPI installation path (for <code>mpi.h</code> and MPI libraries).</td>
              <td><code>/usr/local/openmpi</code></td>
            </tr>
            <tr>
              <td><code>HIPCC</code></td>
              <td>HIP compiler. Falls back to <code>hipcc</code> if not found.</td>
              <td><code>$(ROCM_PATH)/bin/amdclang++</code></td>
            </tr>
            <tr>
              <td><code>NVCC</code></td>
              <td>NVIDIA CUDA compiler (for building <code>TransferBenchCuda</code>).</td>
              <td><code>$(CUDA_PATH)/bin/nvcc</code></td>
            </tr>
            <tr>
              <td><code>ROCM_DEVICE_LIB_PATH</code></td>
              <td>Path to <code>amdgcn</code> bitcode. Auto-detected from the ROCm layout.</td>
              <td><code>(auto)</code></td>
            </tr>
            <tr>
              <td><code>HIPCONFIG</code></td>
              <td>Path to <code>hipconfig</code>, which is used to query the HIP version (for pod communication support check).</td>
              <td><code>hipconfig</code></td>
            </tr>
            <tr>
              <td rowspan="5"><b>Feature flags</b> - To control enabling features that require compile-time support. By default, these are enabled under the right conditions.</td>
              <td><code>DISABLE_NIC_EXEC</code></td>
              <td>Disables NIC Executor support.</td>
              <td><code>0</code></td>
            </tr>
            <tr>
              <td><code>DISABLE_DMA_BUF</code></td>
              <td>Disables <code>DMA-BUF</code> for GPU Direct RDMA. Requires NIC Executor support.</td>
              <td><code>1</code></td>
            </tr>
            <tr>
              <td><code>DISABLE_MPI_COMM</code></td>
              <td>Disables MPI communication backend support for multinode TransferBench.</td>
              <td><code>0</code></td>
            </tr>
            <tr>
              <td><code>DISABLE_AMD_SMI</code></td>
              <td>Disables AMD-SMI pod membership checks.</td>
              <td><code>0</code></td>
            </tr>
            <tr>
              <td><code>DISABLE_POD_COMM</code></td>
              <td>Disables pod communication support (UALoE / MNNVL).</td>
              <td><code>0</code></td>
            </tr>
            <tr>
              <td rowspan="3"><b>Build options</b></td>
              <td><code>SINGLE_KERNEL</code></td>
              <td>To compile with a single GFX kernel (faster build, but fewer kernel variants), set to <code>1</code>. Used mostly for development and debug.</td>
              <td><code>0</code></td>
            </tr>
            <tr>
              <td><code>GPU_TARGETS</code></td>
              <td>Comma-separated GPU architecture targets such as gfx942, gfx950.</td>
              <td><code>native</code></td>
            </tr>
            <tr>
              <td><code>DEBUG</code></td>
              <td>To build in debug mode with debug symbols (-O0, -g), set to <code>1</code>. Runs otherwise in the release mode (-O3).</td>
              <td><code>0</code></td>
            </tr>
        </tbody>
    </table>
  </div>

.. _cmake:

Method 2: Building from source using CMake
============================================

To build TransferBench from source using CMake, run:

.. code-block:: shell

  git clone https://github.com/ROCm/TransferBench.git
  cd TransferBench
  mkdir build && cd build
  cmake ..
  make

CMake environment variables
----------------------------

To modify the CMake behavior, use the following environment variables:

.. raw:: html

  <div class="pst-scrollable-table-container">
    <table id="cmake-env-var" class="table">
        <thead>
            <tr>
                <th>Category</th>
                <th>Environment variable</th>
                <th>Description</th>
                <th>Default value</th>
            </tr>
        </thead>
        <colgroup>
            <col span="1">
            <col span="1">
        </colgroup>
        <tbody class="cmake-env-variables">
            <tr>
              <td rowspan="4"><b>Paths and compilers</b> - To customize which compiler to use or the library to link against.</td>
              <td><code>ROCM_PATH</code></td>
              <td>ROCm installation path.</td>
              <td><code>/opt/rocm</code></td>
            </tr>
            <tr>
              <td><code>CMAKE_TOOLCHAIN_FILE</code></td>
              <td>Toolchain file. Uses ROCM_PATH and CXX to select compiler.</td>
              <td><code>toolchain-linux.cmake</code></td>
            </tr>
            <tr>
              <td><code>CXX</code></td>
              <td>C++ compiler. If not set, <code>amdclang++</code> or <code>hipcc</code> is used.</td>
              <td> Taken from the toolchain</td>
            </tr>
            <tr>
              <td><code>MPI_PATH</code></td>
              <td>Path to MPI installation. Takes priority over <code>find_package(MPI)</code>.</td>
              <td> </td>
            </tr>
            <tr>
              <td rowspan="6"><b>Build options (ON/OFF)</b> - Pass -DVAR=value to set</td>
              <td><code>BUILD_LOCAL_GPU_TARGET_ONLY</code></td>
              <td>Builds only for the GPUs detected on the given machine using <code>rocm_agent_enumerator</code>.</td>
              <td><code>OFF</code></td>
            </tr>
            <tr>
              <td><code>ENABLE_NIC_EXEC</code></td>
              <td>Enables RDMA NIC Executor.</td>
              <td><code>OFF</code></td>
            </tr>
            <tr>
              <td><code>ENABLE_MPI_COMM</code></td>
              <td>Enables MPI communicator as backbone for multinode TransferBench.</td>
              <td><code>OFF</code></td>
            </tr>
            <tr>
              <td><code>ENABLE_DMA_BUF</code></td>
              <td>Enables DMA-BUF for GPU Direct RDMA (requires NIC).</td>
              <td><code>OFF</code></td>
            </tr>
            <tr>
              <td><code>ENABLE_AMD_SMI</code></td>
              <td>Enables AMD-SMI pod membership queries.</td>
              <td><code>OFF</code></td>
            </tr>
            <tr>
              <td><code>ENABLE_POD_COMM</code></td>
              <td>Enables pod communication (HIP >= 8.0).</td>
              <td><code>OFF</code></td>
            </tr>
            <tr>
              <td rowspan="2"><b>Build options</b></td>
              <td><code>SINGLE_KERNEL</code></td>
              <td>To compile with a single GFX kernel (faster build, but fewer kernel variants), set to <code>1</code>. Used mostly for development and debug.</td>
              <td><code>0</code></td>
            </tr>
            <tr>
              <td><code>DEBUG</code></td>
              <td>To build in debug mode with debug symbols (<code>-O0</code>, <code>-g</code>), set to <code>1</code>. Runs otherwise in release mode (<code>-O3</code>).</td>
              <td><code>0</code></td>
            </tr>
            <tr>
              <td rowspan="3"><b>CMake cache variables</b></td>
              <td><code>GPU_TARGETS</code></td>
              <td>Semicolon-separated GPU architectures. Overridden if <code>BUILD_LOCAL_GPU_TARGET_ONLY</code> is <code>ON</code></td>
              <td><code style="word-break: break-all;">gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201;gfx1250</code></td>
              </tr>
              <tr>
                <td><code>AMD_SMI_EXECUTABLE</code></td>
                <td>Path to <code>amd-smi</code> for AMD-SMI version check.</td>
                <td><code>amd-smi</code></td>
              </tr>
              <tr>
                <td><code>HIPCONFIG_EXECUTABLE</code></td>
                <td>Path to <code>hipconfig</code> for HIP version or pod check.</td>
                <td><code>hipconfig</code></td>
              </tr>
        </tbody>
    </table>
  </div>

.. note::

  CMake requires optional features to be explicitly enabled (all default to ``OFF``). Makefile enables features automatically when their dependencies are detected; use ``DISABLE_*`` flags to turn them off. To set cache variables, pass ``-DVAR=value`` to CMake.

**Example: building with MPI and NIC support**

.. code-block:: shell

  git clone https://github.com/ROCm/TransferBench.git
  cd TransferBench
  mkdir build && cd build
  cmake .. -DENABLE_NIC_EXEC=ON -DENABLE_MPI_COMM=ON
  make

Troubleshooting common build errors
====================================

Here are some commonly encountered build errors and their fixes:

- ``Could not find /opt/rocm/bin/amdclang++ or /opt/rocm/bin/hipcc. Check if the path is correct if you want to build TransferBench``

  Occurs if HIP isn't installed correctly. If it is installed in a different directory, specify it using ``ROCM_PATH``.

- ``Could not find standard C++ header 'cmath'``

  Normally occurs if the standard C++ headers aren't installed. Try installing ``g++-12`` or ``g++-14`` based on the OS version. For example, ``apt-get install g++-12``.

Building TransferBenchCuda
===========================

TransferBenchCuda is the NVIDIA build target. To build it on a system with NVIDIA CUDA installed, install the required dependencies first.

Required dependencies
----------------------

- CUDA: The installed CUDA version might impact support for some features such as MNNVL support.

- libnuma: Used for allocating memory or spawning threads on the right NUMA nodes. Here are the install instructions based on the OS:

  - Ubuntu/Debian:

    .. code-block:: shell

      sudo apt install libnuma-dev

  - RHEL/CentOS:

    .. code-block:: shell

      sudo yum install numactl-devel

Building TransferBenchCuda from source code
--------------------------------------------

To build TransferBenchCuda, run:

.. code-block:: shell

  git clone https://github.com/ROCm/TransferBench.git
  cd TransferBench
  make TransferBenchCuda

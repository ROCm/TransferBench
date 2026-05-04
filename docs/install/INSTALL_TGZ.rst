:orphan:

.. meta::
  :description: Install the relocatable TransferBench TGZ archive on any Linux distribution
  :keywords: TransferBench, TGZ, tarball, install, relocatable

.. _install-transferbench-tgz:

------------------------------------------------
Installing TransferBench from the TGZ archive
------------------------------------------------

The TransferBench TGZ archive (``amdrocm<MAJOR>-transferbench-*.tar.gz``) is a
relocatable install tree that works on any Linux distribution where a
compatible ROCm runtime is already present. Use it when you cannot or do not
want to install the DEB or RPM package — for example on a distribution
without a native ROCm package, or inside a non-root container.

The TGZ ships only the ``TransferBench`` binary and its supporting files. It
does **not** bundle ROCm; the host system must already provide the ROCm
runtime libraries (``hsa-rocr`` and the HIP runtime).

Pre-install: ROCm
-----------------

Install ROCm on the target system before extracting the TGZ. Follow the
official AMD documentation:

* `ROCm documentation <https://rocm.docs.amd.com/>`_
* `Linux install guide <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_

After installing, ``ROCM_PATH`` (typically ``/opt/rocm``) must be set
correctly and the ROCm libraries must be loadable by the dynamic linker.

Install runtime dependencies
----------------------------

The DEB and RPM packages declare these runtime dependencies; TGZ users must
install them manually on the target host.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Family
     - Required packages
   * - Debian / Ubuntu
     - ``numactl``, ``libnuma1``, plus the ROCm runtime (``hsa-rocr``)
   * - RHEL / Rocky / AlmaLinux
     - ``numactl``, plus the ROCm runtime (``hsa-rocr``)

Install commands:

.. code-block:: bash

  # Ubuntu / Debian
  sudo apt update && sudo apt install -y numactl libnuma1

  # RHEL / Rocky / AlmaLinux
  sudo dnf install -y numactl

The ROCm packages (``hsa-rocr`` and friends) come from the ROCm repo
configured in the pre-install step above.

Extract the TGZ
---------------

Extract the archive into ``/opt/rocm/extras-<MAJOR>``, where ``<MAJOR>`` is
the ROCm major version the package was built against (encoded in the package
name, for example ``amdrocm7-transferbench-*.tar.gz`` → major ``7``).

.. code-block:: bash

  # Example for ROCm major 7 — match your package
  sudo mkdir -p /opt/rocm/extras-7
  sudo tar -xzf amdrocm7-transferbench-*.tar.gz -C /opt/rocm/extras-7 --strip-components=1

The ``--strip-components=1`` option discards the top-level directory inside
the tarball so files land directly under ``/opt/rocm/extras-7/{bin,lib,...}``.

Configure ``PATH`` and ``LD_LIBRARY_PATH``
------------------------------------------

Point the shell at the extracted prefix and your ROCm install. Copy and paste
the block as one unit (replace paths with your real ``ROCM_PATH`` and major
version):

.. code-block:: bash

  export ROCM_PATH=/opt/rocm                          # or your real ROCm root
  export PATH=/opt/rocm/extras-7/bin:$ROCM_PATH/bin:$PATH
  export LD_LIBRARY_PATH=/opt/rocm/extras-7/lib:$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:$LD_LIBRARY_PATH

The ``TransferBench`` binary embeds an ``RPATH`` covering ``$ORIGIN``,
``$ORIGIN/../lib``, ``/opt/rocm/extras-<MAJOR>/lib``, ``/opt/rocm/lib``,
``/opt/rocm/lib/llvm/lib``, ``/opt/rocm/core-<MAJOR>/lib``, and
``/opt/rocm/core-<MAJOR>/lib/llvm/lib``. The ``LD_LIBRARY_PATH`` export above
is mainly defensive — useful if your ROCm tree lives somewhere non-standard
or if you want to override which copy of a library is loaded for
troubleshooting.

Verify the install
------------------

.. code-block:: bash

  TransferBench --help

You should see the TransferBench usage banner. If the binary fails to load a
shared library, inspect:

.. code-block:: bash

  ldd /opt/rocm/extras-7/bin/TransferBench
  readelf -d /opt/rocm/extras-7/bin/TransferBench | grep -E 'RPATH|RUNPATH'

Make a persistent shell setup
-----------------------------

To avoid re-exporting every shell, drop the variables into a profile script:

.. code-block:: bash

  sudo tee /etc/profile.d/transferbench.sh >/dev/null <<'EOF'
  export ROCM_PATH=/opt/rocm
  export PATH=/opt/rocm/extras-7/bin:$ROCM_PATH/bin:$PATH
  export LD_LIBRARY_PATH=/opt/rocm/extras-7/lib:$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:$LD_LIBRARY_PATH
  EOF
  sudo chmod 0644 /etc/profile.d/transferbench.sh

Log out and back in (or ``source /etc/profile.d/transferbench.sh``) for the
changes to apply.

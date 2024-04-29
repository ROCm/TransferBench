*******************************************
Welcome to TransferBench's documentation!
*******************************************
TransferBench is a simple utility capable of benchmarking simultaneous transfers between user-specified devices (CPUs/GPUs).
A Transfer is defined as a single operation where an executor reads and adds together values from source (SRC) memory locations, then writes the sum to destination (DST) memory locations. This simplifies to a simple copy operation when dealing with single SRC/DST.

The documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

       * :doc:`TransferBench installation <./install/install>`

  .. grid-item-card:: API library

    * :doc:`API library <../doxygen/html/files>`
    * :doc:`Functions <../doxygen/html/globals>`
    * :doc:`Data structures <../doxygen/html/annotated>`

  .. grid-item-card:: How to

    * :doc:`Use TransferBench <how to/use-transferbench>`


  .. grid-item-card:: Tutorials

    * :doc:`ConfigFile format <examples/configfile_format>`
 
To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.




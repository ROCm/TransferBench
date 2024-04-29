


Using TransferBench
---------------------
  
The user has control over the SRC and DST memory locations by indicating memory type followed by the device index. TransferBench supports coarse-grained pinned host memory, unpinned host memory, fine-grained host memory, coarse-grained global device memory, fine-grained global device memory, and null memory (for an empty transfer). In addition, the user can determine the size of the transfer (number of bytes to copy) for their tests.

The executor of the transfer can also be specified by the user. The options are CPU, kernel-based GPU, and SDMA-based GPU (DMA) executors. TransferBench also provides the option to choose the number of sub-executors. In case of a CPU executor this argument specifies the number of CPU threads, while for a GPU executor it defines the number of compute units (CU). If DMA is specified as the executor, the sub-executor argument determines the number of streams to be used.

For more examples, please refer to :ref:`Examples`

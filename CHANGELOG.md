# Changelog for TransferBench

Documentation for TransferBench is available at
[https://rocm.docs.amd.com/projects/TransferBench](https://rocm.docs.amd.com/projects/TransferBench).

## v1.50

### Added
- Adding new parallel copy preset benchmark (pcopy)
  - Usage: ./TransferBench pcopy <numBytes=64M> <#CUs=8> <srcGpu=0> <minGpus=1> <maxGpus=#GPU-1>
### Fixed
- Removed non-copies DMA Transfers (this had previously been using hipMemset)
- Fixed CPU executor when operating on null destination

## v1.49

### Fixes
* Enumerating previously missed DMA engines used only for CPU traffic in topology display

## v1.48

### Fixes
* Various fixes for TransferBenchCuda

### Additions
* Support for targeting specific DMA engines via executor subindex (e.g. D0.1)
* Printing warnings when exeuctors are overcommited

### Modifications
* USE_REMOTE_READ supported for rwrite preset benchmark

## v1.47

### Fixes
* Fixing CUDA support

## v1.46

### Fixes
* Fixing GFX_UNROLL set to 13 (past 8) on gfx906 cards

### Modifications
* GFX_SINGLE_TEAM=1 by default
* Adding field showing summation of individual Transfer bandwidths for Executors

## v1.45

### Additions
* Adding A2A_MODE to a2a preset (0 = copy, 1 = read-only, 2 = write-only)
* Adding GFX_UNROLL to modify GFX kernel's unroll factor
* Adding GFX_WAVE_ORDER to modify order in which wavefronts process data

### Modifications
* Rewrote the GFX reduction kernel to support new wave ordering

## v1.44

### Additions
* Adding rwrite preset to benchmark remote parallel writes
 * Usage: ./TransferBench rwrite <numBytes=64M> <#CUs=8> <srcGpu=0> <minGpus=1> <maxGpus=3>

## v1.43

### Changes
* Modifying a2a to show executor timing, as well as executor min/max bandwidth

## v1.42

### Fixes
* Fixing schmoo maxNumCus optional arg parsing
* Schmoo output modified to be easier to copy

## v1.41

### Additions
* Adding schmoo preset config benchmarks local/remote reads/writes/copies
  * Usage: ./TransferBench schmoo <numBytes=64M> <localIdx=0> <remoteIdx=1> <maxNumCUs=32>

### Fixes
* Fixing some misreported timings when running with non-fixed number of iterations

## v1.40

### Fixes
* Fixing XCC defaulting to 0 instead of random for preset configs, ignoring XCC_PREF_TABLE

## v1.39

### Additions
* (Experimental) Adding support for Executor sub-index
### Fixes
- Remove deprecated gcnArch code.  ROCm version must include support for hipDeviceMallocUncached

## v1.38

### Fixes
* Adding missing threadfence which could cause non-fine-grained Transfers to report higher speeds

## v1.37

### Changes
* USE_SINGLE_STREAM is enabled by default now.  (Disable via USE_SINGLE_STREAM=0)

### Fixes
* Fix unrecognized token error when XCC_PREF_TABLE is unspecified

## v1.36

### Additions

* (Experimental) Adding XCC filtering - combined with XCC_PREF_TABLE, this tries to select
  specific XCCs to use for specific (SRC->DST) Transfers

## v1.35

### Additions

* USE_FINE_GRAIN also applies to a2a preset

## v1.34

### Additions

* Set `GPU_KERNEL=3` as default for gfx942

## v1.33

### Additions

* Added the `ALWAYS_VALIDATE` environment variable to allow for validation after every iteration, instead
  of only once at the end of all iterations

## v1.32

### Changes

* Increased the line limit from 2048 to 32768

## v1.31

### Changes

* `SHOW_ITERATIONS` now shows XCC:CU instead of just CU ID
* `SHOW_ITERATIONS` is printed when `USE_SINGLE_STREAM`=1

## v1.30

### Additions

* `BLOCK_SIZE` has been added to control the threadblock size (must be a multiple of 64, up to 512)
* `BLOCK_ORDER` has been added to control how work is ordered for GFX-executors running
  `USE_SINGLE_STREAM`=1
  * 0 - Threadblocks for transfers are ordered sequentially (default)
  * 1 - Threadblocks for transfers are interleaved
  * 2 - Threadblocks for transfers are ordered randomly

## v1.29

### Additions

* A2A preset config now responds to `USE_REMOTE_READ`

### Fixes

* Race-condition during wall-clock initialization caused "inf" during single-stream runs
* CU numbering output after CU masking

### Changes

* The default number of warmups has been reverted to 3
* The default unroll factor for gfx940/941 has been set to 6

## v1.28

### Additions

* Added `A2A_DIRECT`, which only runs all-to-all on directly connected GPUs (now on by default)
* Added average statistics for P2P and A2A benchmarks
* Added `USE_FINE_GRAIN` for P2P benchmark
  * With older devices, P2P performance with default coarse-grain device memory stops timing as soon
    as a request is sent to data fabric, and not actually when it arrives remotely. This can artificially
    inflate bandwidth numbers, especially when sending small amounts of data.

### Changes

* Modified P2P output to help distinguish between CPU and GPU devices

### Fixes

* Fixed Makefile target to prevent unnecessary re-compilation

## v1.27

### Additions

* Added cmdline preset to allow specification of  simple tests on command line (e.g.,
  `./TransferBench cmdline 64M "1 4 G0->G0->G1"`)
* Adding the `HIDE_ENV` environment variable, which stops environment variable values from printing
* Adding the `CU_MASK` environment variable, which allows you to select the CUs to run on
* `CU_MASK` is specified in CU indices (0-#CUs-1), where ' - ' can be used to denote ranges of values
  (e.g., `CU_MASK`=3-8,16 requests that transfer be run only on CUs 3,4,5,6,7,8,16)
  * Note that this is somewhat experimental and may not work on all hardware
* `SHOW_ITERATIONS` now shows CU usage for that iteration (experimental)

### Changes

* Added extra comments on commonly missing includes with details on how to install them

### Fixes

* CUDA compilation works again (the `wall_clock64` CUDA alias was not defined)

## v1.26

### Additions

* Setting SHOW_ITERATIONS=1 provides additional information about per-iteration timing for file and
  P2P configs
  * For file configs, iterations are sorted from min to max bandwidth and displayed with standard
    deviation
  * For P2P, min/max/standard deviation is shown for each direction

### Changes

* P2P benchmark formatting now reports bidirectional bandwidth in each direction (as well as sum) for
  clarity

## v1.25

### Fixes

* Fixed a bug in the P2P bidirectional benchmark that used the incorrect number of `subExecutors` for
  CPU<->GPU tests

## v1.24

### Additions

* New All-To-All GPU benchmark accessed by preset "A2A"
* Added gfx941 wall clock frequency

## v1.23

### Additions

* New GPU subexec scaling benchmark accessed by preset "scaling"
  * Tests GPU-GFX copy performance based on # of CUs used

## v1.22

### Changes

* Switched the kernel timing function to `wall_clock64`

## v1.21

### Fixes

* Fixed a bug with `SAMPLING_FACTOR`

## v1.20

### Fixes

* `VALIDATE_DIRECT` can now be used with `USE_PREP_KERNEL`
* Switched to local GPU for validating GPU memory

## v1.19

### Additions

* `VALIDATE_DIRECT` now also applies to source memory array checking
* Added null memory pointer check prior to deallocation

## v1.18

### Additions

* Adding the ability to validate GPU destination memory directly without going through the CPU
  staging buffer (`VALIDATE_DIRECT`)
  * Note that this only works on AMD devices with large-bar access enabled, and may slow things down
    considerably

### Changes

* Refactored how environment variables are displayed
* Mismatch stops after the first detected error within an array instead of listing all mismatched
  elements

## v1.17

### Additions

* Allowed switch to GFX kernel for source array initialization (`USE_PREP_KERNEL`)
  * Note that `USE_PREP_KERNEL` can't be used with `FILL_PATTERN`
* Added the ability to compile with nvcc only (`TransferBenchCuda`)

### Changes

* The default pattern was set to [Element i = ((i * 517) modulo 383 + 31) * (srcBufferIdx + 1)]

### Fixes

* Added the `example.cfg` file

## v1.16

### Additions

* Additional src array validation during preparation
* Added a new environment variable (`CONTINUE_ON_ERROR`) to resume tests after a mis-match
  detection
* Initialized GPU memory to 0 during allocation

## v1.15

### Fixes

* Fixed a bug that prevented single transfers greater than 8 GB

### Changes

* Removed "check for latest ROCm" warning when allocating too much memory
* Off-source memory value is now printed when a mis-match is detected

## v1.14

### Additions

* Added documentation
* Added pthread linking in src/Makefile and CMakeLists.txt
* Added printing off the hex value of the floats for output and reference

## v1.13

### Additions

* Added support for cmake

### Changes

* Converted to the Pitchfork layout standard

## v1.12

### Additions

* Added support for TransferBench on NVIDIA platforms (via `HIP_PLATFORM`=nvidia)
  * Note that CPU executors on NVIDIA platform cannot access GPU memory (no large-bar access)

## v1.11

### Additions

* Added multi-input/multi-output (MIMO) support: transfers now can reduce (element-wise summation)
  multiple input memory arrays and write sums to multiple outputs
* Added GPU-DMA executor 'D', which uses `hipMemcpy` for SDMA copies
  * Previously, this was done using `USE_HIP_CALL`, but now GPU-GFX kernel can run in parallel with
    GPU-DMA, instead of applying to all GPU executors globally
  * GPU-DMA executor can only be used for single-input/single-output transfers
  * GPU-DMA executor can only be associated with one SubExecutor
* Added new "Null" memory type 'N', which represents empty memory. This allows for read-only or
  write-only transfers
* Added new `GPU_KERNEL` environment variable that allows switching between various GPU-GFX
  reduction kernels

### Optimizations

* Improved GPU-GFX kernel performance based on hardware architecture when running with
  fewer CUs

### Changes

* Updated the `example.cfg` file to cover new features
* Updated output to support MIMO
* Changed CU and CPU thread naming to SubExecutors for consistency
* Sweep Preset: default sweep preset executors now includes DMA
* P2P benchmarks:
  * Removed `p2p_rr`, `g2g` and `g2g_rr` (now only works via P2P)
    * Setting `NUM_CPU_DEVICES`=0 can only be used to benchmark GPU devices (like `g2g`)
    * The new `USE_REMOTE_READ` environment variable replaces `_rr` presets
  * New environment variable `USE_GPU_DMA`=1 replaces `USE_HIP_CALL`=1 for benchmarking with
    GPU-DMA Executor
  * Number of GPU SubExecutors for benchmark can be specified via `NUM_GPU_SE`
    * Defaults to all CUs for GPU-GFX, 1 for GPU-DMA
  * Number of CPU SubExecutors for benchmark can be specified via `NUM_CPU_SE`
* Psuedo-random input pattern has been slightly adjusted to have different patterns for each input
  array within same transfer

### Removals

* `USE_HIP_CALL`: use `GPU-DMA` executor 'D' or set `USE_GPU_DMA`=1 for P2P
  benchmark presets
  * Currently, a warning will be issued if `USE_HIP_CALL` is set to 1 and the program will stop
* `NUM_CPU_PER_TRANSFER`: the number of CPU SubExecutors will be whatever is specified for the
  transfer
* `USE_MEMSET`: this function can now be done via a transfer using the null memory type

## v1.10

### Fixes

* Fixed incorrect bandwidth calculation when using single stream mode and per-transfer data sizes

## v1.09

### Additions

* Printing off src/dst memory addresses during interactive mode

### Changes

* Switching to `numa_set_preferred` instead of `set_mempolicy`

## v1.08

### Changes

* Fixed handling of non-configured NUMA nodes
* Topology detection now shows actual NUMA node indices
* Fixed 'for' issue with `NUM_GPU_DEVICES`

## v1.07

### Fixes

* Fixed bug with allocations involving non-default CPU memory types

## v1.06

### Additions

* Unpinned CPU memory type ('U'), which may require `HSA_XNACK`=1 in order to access via
  GPU executors
* Added sweep configuration logging to `lastSweep.cfg`
* Ability to specify the number of CUs to use for sweep-based presets

### Changes

* Modified advanced configuration file format to accept bytes-per-transfer

### Fixes

* Fixed random sweep repeatability
* Fixed bug with CPU NUMA node memory allocation

## v1.05

### Additions

* Topology output now includes NUMA node information
* Support for NUMA nodes with no CPU cores (e.g., CXL memory)

### Removals

* The `SWEEP_SRC_IS_EXE` environment variable was removed

## v1.04

### Additions

* There are new environment variables for sweep based presets:
  * `SWEEP_XGMI_MIN`: The minumum number of XGMI hops for transfers
  * `SWEEP_XGMI_MAX`: The maximum number of XGMI hops for transfers
  * `SWEEP_SEED`: Uses a random seed
  * `SWEEP_RAND_BYTES`: Uses a random amount of bytes (up to pre-specified N) for each transfer

### Changes

* CSV output for sweep now includes an environment variables section followed by output
* CSV output no longer lists environment variable parameters in columns
* We changed the default number of warmup iterations from 3 to 1
* Split CSV output of link type to `ExeToSrcLinkType` and `ExeToDstLinkType`

## v1.03

### Additions

* There are new preset modes stress-test benchmarks: `sweep` and `randomsweep`
  * `sweep` iterates over all possible sets of transfers to test
  * `randomsweep` iterates over random sets of transfers
  * New sweep-only environment variables can modify `sweep`
    * `SWEEP_SRC`: String containing only "B","C","F", or "G" that defines possible source memory types
    * `SWEEP_EXE`: String containing only "C" or "G" that defines possible executors
    * `SWEEP_DST`: String containing only "B","C","F", or "G" that defines possible destination memory types
    * `SWEEP_SRC_IS_EXE`: Restrict the executor to be the same as the source, if non-zero
    * `SWEEP_MIN`: Minimum number of parallel transfers to test
    * `SWEEP_MAX`: Maximum number of parallel transfers to test
    * `SWEEP_COUNT`: Maximum number of tests to run
    * `SWEEP_TIME_LIMIT`: Maximum number of seconds to run tests
* New environment variables to restrict number of available devices to test on (primarily for sweep
  runs)
  * `NUM_CPU_DEVICES`: Number of CPU devices
  * `NUM_GPU_DEVICES`: Number of GPU devices

### Fixes

* Fixed timing display for CPU executors when using single-stream mode

## v1.02

### Additions

* Setting `NUM_ITERATIONS` to a negative number indicates a run of -`NUM_ITERATIONS` seconds per
  test

### Changes

* Copies are now referred to as 'transfers' instead of 'links'
* Reordered how environment variables are displayed (alphabetically now)

### Removals

* Combined timing is now always on for kernel-based GPU copies; the `COMBINED_TIMING`
  environment variable has been removed
* Single sync is no longer supported for facility variable iterations; the `USE_SINGLE_SYNC`
  environmental variable has been removed

## v1.01

### Additions

* Added the `USE_SINGLE_STREAM` feature
  * All Links that run on the same GPU device are run with a single kernel launch on a single stream
  * This doesn't work with `USE_HIP_CALL`, and it forces `USE_SINGLE_SYNC` to collect timings
  * Added the ability to request coherent or fine-grained host memory ('B')

### Changes

* Separated the TransferBench repository from the RCCL repository
* Peer-to-peer benchmark mode now works with `OUTPUT_TO_CSV`
* Toplogy display now works with `OUTPUT_TO_CSV`
* Moved the documentation about the config file into `example.cfg`

### Removals

* Removed config file generation
* Removed the 'show pointer address' (`SHOW_ADDR`) environment variable

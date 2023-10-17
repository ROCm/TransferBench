# Changelog for TransferBench

## v1.31
### Modified
- SHOW_ITERATIONS now show XCC:CU instead of just CU ID
- SHOW_ITERATIONS also printed when USE_SINGLE_STREAM=1

## v1.30
### Added
- BLOCK_SIZE added to control threadblock size (Must be multiple of 64, up to 512)
- BLOCK_ORDER added to control how work is ordered for GFX-executors running USE_SINGLE_STREAM=1
  - 0 - Threadblocks for Transfers are ordered sequentially (Default)
  - 1 - Threadblocks for Transfers are interleaved
  - 2 - Threadblocks for Transfers are ordered randomly

## v1.29
### Added
- a2a preset config now responds to USE_REMOTE_READ
### Fixed
- Race-condition during wall-clock initialization caused "inf" during single stream runs
- CU numbering output after CU masking
### Modified
- Default number of warmups reverted to 3
- Default unroll factor for gfx940/941 set to 6

## v1.28
### Added
- Added A2A_DIRECT which only executes all-to-all only directly connected GPUs (on by default now)
- Added average statistics for p2p and a2a benchmarks
- Added USE_FINE_GRAIN for p2p benchmark.
  - With older devices, p2p performance with default coarse grain device memory stops timing as soon as request sent to data fabric,
    not actually when it arrives remotely, which may artificially inflate bandwidth numbers, especially when sending small amounts of data
### Modified
- Modified P2P output to help distinguish between CPU / GPU devices
### Fixed
- Fixed Makefile target to prevent unnecessary re-compilation

## v1.27
### Added
- Adding cmdline preset to allow specify simple tests on command line
- E.g. ./TransferBench cmdline 64M "1 4 G0->G0->G1"
- Adding environment variable HIDE_ENV, which skips printing of environment variable values
- Adding environment variable CU_MASK, which allows selection of which CUs to execute on
- CU_MASK is specified in CU indices (0-#CUs-1), and '-' can be used to denote ranges of values
  - E.g.: CU_MASK=3-8,16 would request Transfer be executed only CUs 3,4,5,6,7,8,16
  - NOTE: This is somewhat experimental and may not work on all hardware
- SHOW_ITERATIONS now shows CU usage for that iteration (experimental)
### Modified
- Adding extra comments on commonly missing includes with details on how to install them
### Fixed
- CUDA compilation should work again (wall_clock64 CUDA alias was not defined)

## v1.26
### Added
- Setting SHOW_ITERATIONS=1 provides additional information about per-iteration timing for file and p2p configs
  - For file configs, iterations are sorted from min to max bandwidth and displayed with standard deviation
  - For p2p, min/max/standard deviation is shown for each direction.

### Changed
- P2P benchmark formatting changed.  Now reports bidirectional bandwidth in each direction (as well as sum) for clarity

## v1.25
### Fixed
- Fixed bug in P2P bidirectional benchmark using incorrect number of subExecutors for CPU<->GPU tests

## v1.24
### Added
- New All-To-All GPU benchmark accessed by preset "a2a"
- Adding gfx941 wall clock frequency

## v1.23
### Added
- New GPU subexec scaling benchmark accessed by preset "scaling"
  - Tests GPU-GFX copy performance based on # of CUs used

## v1.22
### Modified
- Switching kernel timing function to wall_clock64

## v1.21
### Fixed
- Fixed bug with SAMPLING_FACTOR

## v1.20
### Fixed
- VALIDATE_DIRECT can now be used with USE_PREP_KERNEL
- Switch to local GPU for validating GPU memory

## v1.19
### Added
- VALIDATE_DIRECT now also applies to source memory array checking
- Adding null memory pointer check prior to deallocation

## v1.18
### Added
- Adding ability to validate GPU destination memory directly without going through CPU staging buffer (VALIDATE_DIRECT)
  - NOTE: This will only work on AMD devices with large-bar access enable and may slow things down considerably
### Changed
- Refactored how environment variables are displayed
- Mismatch stops after first detected error within an array instead of list all mismatched elements

## v1.17
### Added
- Allow switch to GFX kernel for source array initialization (USE_PREP_KERNEL)
  - USE_PREP_KERNEL cannot be used with FILL_PATTERN
- Adding ability to compile with nvcc only (TransferBenchCuda)
### Changed
- Default pattern set to [Element i = ((i * 517) modulo 383 + 31) * (srcBufferIdx + 1)]
### Fixed
- Re-adding example.cfg file

## v1.16
### Added
- Additional src array validation during preparation
- Adding new env var CONTINUE_ON_ERROR to resume tests after mis-match detection
- Initializing GPU memory to 0 during allocation

## v1.15
### Fixed
- Fixed a bug that prevented single Transfers > 8GB
### Changed
- Removed "check for latest ROCm" warning when allocating too much memory
- Printing off source memory value as well when mis-match is detected

## v1.14
### Added
- Added documentation
- Added pthread linking in src/Makefile and CMakeLists.txt
- Added printing off the hex value of the floats for output and reference

## v1.13
### Added
- Added support for cmake

### Changed
- Converted to the Pitchfork layout standard

## v1.12
### Added
- Added support for TransferBench on NVIDIA platforms (via HIP_PLATFORM=nvidia)
  - CPU executors on NVIDIA platform cannot access GPU memory (no large-bar access)

## v1.11
### Added
- New multi-input / multi-output support (MIMO).  Transfers now can reduce (element-wise summation) multiple input memory arrays
  and write the sums to multiple outputs
- New GPU-DMA executor 'D' (uses hipMemcpy for SDMA copies).  Previously this was done using USE_HIP_CALL, but now this allows
  GPU-GFX kernel to run in parallel with GPU-DMA instead of applying to all GPU executors globally.
  - GPU-DMA executor can only be used for single-input/single-output Transfers
  - GPU-DMA executor can only be associated with one SubExecutor
- Added new "Null" memory type 'N', which represents empty memory. This allows for read-only or write-only Transfers
- Added new GPU_KERNEL environment variable that allows for switching between various GPU-GFX reduction kernels

### Optimized
- Slightly improved GPU-GFX kernel performance based on hardware architecture when running with fewer CUs

### Changed
- Updated the example.cfg file to cover the new features
- Updated output to support MIMO
- Changed CUs/CPUs threads naming to SubExecutors for consistency
- Sweep Preset:
  - Default sweep preset executors now includes DMA
- P2P Benchmarks:
  - Now only works via "p2p".  Removed "p2p_rr", "g2g" and "g2g_rr".
    - Setting NUM_CPU_DEVICES=0 can be used to only benchmark GPU devices (like "g2g")
    - New environment variable USE_REMOTE_READ replaces "_rr" presets
  - New environment variable USE_GPU_DMA=1 replaces USE_HIP_CALL=1 for benchmarking with GPU-DMA Executor
  - Number of GPU SubExecutors for benchmark can be specified via NUM_GPU_SE
    - Defaults to all CUs for GPU-GFX, 1 for GPU-DMA
  - Number of CPU SubExecutors for benchmark can be specified via NUM_CPU_SE
- Psuedo-random input pattern has been slightly adjusted to have different patterns for each input array within same Transfer

### Removed
- USE_HIP_CALL has been removed.  Use GPU-DMA executor 'D' or set USE_GPU_DMA=1 for P2P benchmark presets
  - Currently warning will be issued if USE_HIP_CALL is set to 1 and program will terminate
- Removed NUM_CPU_PER_TRANSFER - The number of CPU SubExecutors will be whatever is specified for the Transfer
- Removed USE_MEMSET environment variable.  This can now be done via a Transfer using the null memory type

## v1.10
### Fixed
- Fix incorrect bandwidth calculation when using single stream mode and per-Transfer data sizes

## v1.09
### Added
- Printing off src/dst memory addresses during interactive mode
### Changed
- Switching to numa_set_preferred instead of set_mempolicy

## v1.08
### Changed
- Fixing handling of non-configured NUMA nodes
- Topology detection now shows actual NUMA node indices
- Fix for issue with NUM_GPU_DEVICES

## v1.07
### Changed
- Fix bug with allocations involving non-default CPU memory types

## v1.06
### Added
- Added unpinned CPU memory type ('U').  May require HSA_XNACK=1 in order to access via GPU executors
- Adding logging of sweep configuration to lastSweep.cfg
- Adding ability to specify number of CUs to use for sweep-based presets
### Changed
- Fixing random sweep repeatibility
- Fixing bug with CPU NUMA node memory allocation
- Modified advanced configuration file format to accept bytes per Transfer

## v1.05
### Added
- Topology output now includes NUMA node information
- Support for NUMA nodes with no CPU cores (e.g. CXL memory)
### Removed
- SWEEP_SRC_IS_EXE environment variable

## v1.04
### Added
- New environment variables for sweep based presets
  - SWEEP_XGMI_MIN   - Min number of XGMI hops for Transfers
  - SWEEP_XGMI_MAX   - Max number of XGMI hops for Transfers
  - SWEEP_SEED       - Random seed being used
  - SWEEP_RAND_BYTES - Use random amount of bytes (up to pre-specified N) for each Transfer
### Changed
  - CSV output for sweep includes env vars section followed by output
  - CSV output no longer lists env var parameters in columns
  - Default number of warmup iterations changed from 3 to 1
  - Splitting CSV output of link type to ExeToSrcLinkType and ExeToDstLinkType

## v1.03
### Added
- New preset modes stress-test benchmarks "sweep" and "randomsweep"
  - sweep iterates over all possible sets of Transfers to test
  - randomsweep iterates over random sets of Transfers
  -  New sweep-only environment variables can modify sweep
     - SWEEP_SRC - String containing only "B","C","F", or "G", defining possible source memory types
     - SWEEP_EXE - String containing only "C", or "G", defining possible executors
     - SWEEP_DST - String containing only "B","C","F", or "G", defining possible destination memory types
     - SWEEP_SRC_IS_EXE - Restrict executor to be the same as the source if non-zero
     - SWEEP_MIN - Minimum number of parallel transfers to test
     - SWEEP_MAX - Maximum number of parallel transfers to test
     - SWEEP_COUNT - Maximum number of tests to run
     - SWEEP_TIME_LIMIT - Maximum number of seconds to run tests for
- New environment variable to restrict number of available GPUs to test on (primarily for sweep runs)
  - NUM_CPU_DEVICES - Number of CPU devices
  - NUM_GPU_DEVICES - Number of GPU devices
### Changed
- Fixed timing display for CPU-executors when using single stream mode

## v1.02
### Added
- Setting NUM_ITERATIONS to negative number indicates to run for -NUM_ITERATIONS seconds per Test
### Changed
- Copies are now refered to as Transfers instead of Links
- Re-ordering how env vars are displayed (alphabetically now)
### Removed
- Combined timing is now always on for kernel-based GPU copies. COMBINED_TIMING env var has been removed
- Use single sync is no longer supported to facility variable iterations. USE_SINGLE_SYNC env var has been removed

## v1.01
### Added
- Adding USE_SINGLE_STREAM feature
  - All Links that execute on the same GPU device are executed with a single kernel launch on a single stream
  - Does not work with USE_HIP_CALL and forces USE_SINGLE_SYNC to collect timings
  - Adding ability to request coherent / fine-grained host memory ('B')
### Changed
- Separating TransferBench from RCCL repo
- Peer-to-peer benchmark mode now works OUTPUT_TO_CSV
- Toplogy display now works with OUTPUT_TO_CSV
- Moving documentation about config file into example.cfg
### Removed
- Removed config file generation
- Removed show pointer address environment variable (SHOW_ADDR)

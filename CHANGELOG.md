# Changelog for TransferBench

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

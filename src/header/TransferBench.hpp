/*
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/// @cond
#pragma once
#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <ifaddrs.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <numa.h> // If not found, try installing libnuma-dev (e.g apt-get install libnuma-dev)
#include <numaif.h>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/utsname.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "IbvDynLoad.hpp"

#ifdef MPI_COMM_ENABLED
#include <mpi.h>
#endif

#if defined(__NVCC__)
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef NVML_ENABLED
#include <nvml.h>
#endif
#else
#include "hip/hip_ext.h"
#include "hip/hip_runtime.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#ifdef AMD_SMI_ENABLED
#include "amd_smi/amdsmi.h"
#endif

#endif

#include "tdmCopy.h"
/// @endcond

// Batched DMA executor is only supported with HIP >= 7.1 and CUDA 12.8
#if (defined(HIP_VERSION) && (HIP_VERSION >= 70100000)) || (defined(CUDA_VERSION) && (CUDA_VERSION >= 12080))
#define BMA_EXEC_ENABLED
#endif

namespace TransferBench
{
  using std::map;
  using std::pair;
  using std::set;
  using std::vector;

  constexpr char VERSION[] = "1.70";

  /**
   * Enumeration of supported Executor types
   *
   * @note The Executor is the device used to perform a Transfer
   */
  enum ExeType
  {
    EXE_CPU          = 0,                       ///<  CPU executor              (subExecutor = CPU thread)
    EXE_GPU_GFX      = 1,                       ///<  GPU kernel-based executor (subExecutor = threadblock/CU)
    EXE_GPU_DMA      = 2,                       ///<  GPU SDMA executor         (subExecutor = not supported)
    EXE_NIC          = 3,                       ///<  NIC RDMA executor         (subExecutor = queue pair)
    EXE_NIC_NEAREST  = 4,                       ///<  NIC RDMA nearest executor (subExecutor = queue pair)
    EXE_GPU_BDMA     = 5,                       ///<  GPU Batched SDMA executor (subExecutor = batch item)
    EXE_GPU_TDM      = 6,                       ///<  GPU TDM executor          (subExecutor = threadblock/CU)
  };
  char const ExeTypeStr[8] = "CGDINBT";
  inline bool IsCpuExeType(ExeType e){ return e == EXE_CPU; }
  inline bool IsGpuExeType(ExeType e){ return e == EXE_GPU_GFX || e == EXE_GPU_DMA || e == EXE_GPU_BDMA || e == EXE_GPU_TDM; }
  inline bool IsNicExeType(ExeType e){ return e == EXE_NIC || e == EXE_NIC_NEAREST; }

  /**
   * A ExeDevice defines a specific Executor
   */
  struct ExeDevice
  {
    ExeType exeType;                            ///< Executor type
    int32_t exeIndex;                           ///< Executor index
    int32_t exeRank = 0;                        ///< Executor rank
    int32_t exeSlot = 0;                        ///< Executor slot

    bool operator<(ExeDevice const& other) const {
      return ((exeRank  != other.exeRank)  ? (exeRank  < other.exeRank)  :
              (exeType  != other.exeType)  ? (exeType  < other.exeType)  :
              (exeIndex != other.exeIndex) ? (exeIndex < other.exeIndex) :
                                             (exeSlot  < other.exeSlot));
    }
  };

  /**
   * Enumeration of supported memory types
   *
   * @note These are possible types of memory to be used as sources/destinations
   */
  enum MemType
  {
    MEM_CPU             = 0,                    ///< Default pinned CPU memory     (via hipHostMalloc)
    MEM_CPU_CLOSEST     = 1,                    ///< Default pinned CPU memory     (indexed by closest GPU)
    MEM_CPU_COHERENT    = 2, MEM_CPU_FINE = 2,  ///< Coherent pinned CPU memory    (via hipHostMallocCoherent flag)
    MEM_CPU_NONCOHERENT = 3,                    ///< Noncoherent pinned CPU memory (via hipHostMallocNonCoherent flag)
    MEM_CPU_UNCACHED    = 4,                    ///< Uncached pinned CPU memory    (via hipHostMallocUncached flag)
    MEM_CPU_UNPINNED    = 5,                    ///< Unpinned CPU memory
    MEM_GPU             = 6,                    ///< Default GPU memory            (via hipMalloc)
    MEM_GPU_FINE        = 7,                    ///< Fine-grained GPU memory       (via hipDeviceMallocFinegrained flag)
    MEM_GPU_UNCACHED    = 8,                    ///< Uncached GPU memory           (via hipDeviceMallocUncached flag)
    MEM_MANAGED         = 9,                    ///< Managed memory
    MEM_NULL            = 10,                   ///< NULL memory - used for empty
  };
  char const MemTypeStr[12] = "CPBDKHGFUMN";
  inline bool IsCpuMemType(MemType m) { return (MEM_CPU <= m && m <= MEM_CPU_UNPINNED);}
  inline bool IsGpuMemType(MemType m) { return (MEM_GPU <= m && m <= MEM_MANAGED);}

  /**
   * Enumeration of supported GFX kernels
   */
  enum GfxKernelType
  {
    GFX_KERNEL_AUTO   = -1,                     ///< Automatically choose a kernel
    GFX_KERNEL_REDUCE =  0,                     ///< Default kernel that supports any multiple input/output buffers
    GFX_KERNEL_COPY   =  1,                     ///< Simpler kernel that supports copies only
    NUM_GFX_KERNELS   =  2                      ///< Number of GFX kernels currently supported
  };

  /**
   * Enumeration of supported TDM kernels
   */
  enum TdmKernelType
  {
    TDM_KERNEL_AUTO   = -1,                     ///< Automatically choose a kernel
    TDM_KERNEL_COPY   =  0,                     ///< Default kernel that copies a single input to a single output
    TDM_KERNEL_REDUCE =  1,                     ///< Kernel that supports multiple input/output buffers (sum-reduce)
    NUM_TDM_KERNELS   =  2                      ///< Number of TDM kernels currently supported
  };

  /**
   * A MemDevice indicates a memory type on a specific device
   */
  struct MemDevice
  {
    MemType memType;                            ///< Memory type
    int32_t memIndex;                           ///< Device index
    int32_t memRank = 0;                        ///< Rank index

    bool operator<(MemDevice const& other) const {
      return ((memType  != other.memType)  ? (memType  < other.memType) :
              (memIndex != other.memIndex) ? (memIndex < other.memIndex) :
                                             (memRank  < other.memRank));
    }
    bool operator==(MemDevice const& other) const {
      return (memType  == other.memType &&
              memIndex == other.memIndex &&
              memRank  == other.memRank);
    }
  };

  /**
   * A Transfer adds together data from zero or more sources then writes the sum to zero or more desintations
   */
  struct Transfer
  {
    size_t            numBytes    = 0;          ///< Number of bytes to Transfer
    vector<MemDevice> srcs        = {};         ///< List of source memory devices
    vector<MemDevice> dsts        = {};         ///< List of destination memory devices
    ExeDevice         exeDevice   = {};         ///< Executor to use
    int32_t           exeSubIndex = -1;         ///< Executor subindex
    int32_t           exeSubSlot  = 0;          ///< Executor subslot
    int               numSubExecs = 0;          ///< Number of subExecutors to use for this Transfer
  };

  /**
   * General options
   */
  struct GeneralOptions
  {
    int numIterations      = 10;                ///< Number of timed iterations to perform. If negative, run for -numIterations seconds instead
    int numSubIterations   = 1;                 ///< Number of sub-iterations per iteration
    int numWarmups         = 3;                 ///< Number of un-timed warmup iterations to perform
    int recordPerIteration = 0;                 ///< Record per-iteration timing information
    int useHipEvents       = 1;                 ///< Use HIP events for timing Executors that support it
    int useInteractive     = 0;                 ///< Pause for user-input before starting transfer loop
    int useMultiStream     = 0;                 ///< Split GFX/TDM Transfers into separate kernel launches in separate stream
  };

  /**
   * Data options
   */
  struct DataOptions
  {
    int           alwaysValidate   = 0;         ///< <0 = disable validation, 0 = validate once at end, >0 = validate after each iteration
    int           blockBytes       = 256;       ///< Each subexecutor works on a multiple of this many bytes
    int           byteOffset       = 0;         ///< Byte-offset for memory allocations
    vector<float> fillPattern      = {};        ///< Pattern of floats used to fill source data
    vector<int>   fillCompress     = {};        ///< Customized data patterns (overrides fillPattern if non-empty)
    int           validateDirect   = 0;         ///< Validate GPU results directly instead of copying to host
    int           validateSource   = 0;         ///< Validate src GPU memory immediately after preparation
  };

  /**
   * GFX Executor options
   */
  struct GfxOptions
  {
    int                 blockOrder     = 0;     ///< Determines how threadblocks are ordered (0=sequential, 1=interleaved, 2=random)
    int                 blockSize      = 256;   ///< Size of each threadblock (must be multiple of 64)
    vector<uint32_t>    cuMask         = {};    ///< Bit-vector representing the CU mask
    int                 gfxKernel      = 0;     ///< Kernel selector: -1=auto, 0=reduce, 1=copy-only
    vector<vector<int>> prefXccTable   = {};    ///< 2D table with preferred XCD to use for a specific [src][dst] GPU device
    int                 seType         = 0;     ///< SubExecutor granularity type (0=threadblock, 1=warp)
    int                 temporalMode   = 0;     ///< Non-temporal load/store mode 0=none, 1=load, 2=store, 3=both
    int                 unrollFactor   = 4;     ///< GFX-kernel unroll factor
    int                 useSingleTeam  = 0;     ///< Team all subExecutors across the data array
    int                 waveOrder      = 0;     ///< GFX-kernel wavefront ordering
    int                 wordSize       = 4;     ///< GFX-kernel packed data size (4=dwordx4, 2=dwordx2, 1=dwordx1)
  };

  /**
   * DMA Executor options
   */
  struct DmaOptions
  {
    int useHsaCopy   = 0;                       ///< Use HSA copy instead of HIP copy to perform DMA
  };

  /**
   * NIC Executor options
   */
  struct NicOptions
  {
    size_t      chunkBytes      = 1<<30;        ///< How much bytes to transfer at a time
    int         cqPollBatch     = 4;            ///< Maximum CQ entries polled per call
    int         ibGidIndex      = -1;           ///< GID Index for RoCE NICs (-1 is auto)
    uint8_t     ibPort          = 1;            ///< NIC port number to be used
    int         ipAddressFamily = 4;            ///< 4=IPv4, 6=IPv6 (used for auto GID detection)
    int         maxRecvWorkReq  = 16;           ///< Maximum number of recv work requests per queue pair
    int         maxSendWorkReq  = 1024;         ///< Maximum number of send work requests per queue pair
    int         queueSize       = 100;          ///< Completion queue size
    int         roceVersion     = 2;            ///< RoCE version (used for auto GID detection)
    int         useRelaxedOrder = 1;            ///< Use relaxed ordering
    uint8_t     serviceLevel    = 0;            ///< IB service level (sl) for InfiniBand QPs
    uint8_t     trafficClass    = 0;            ///< DSCP/traffic class byte for RoCE GRH
    int         useNuma         = 0;            ///< Switch to closest numa thread for execution
  };

  struct TdmOptions
  {
    int blockOrder = 0;                         ///< Determines how threadblocks are ordered (0=sequential, 1=interleaved, 2=random)
    int blockSize  = 256;                       ///< Size of each threadblock
    int ldsBytes   = 0;                         ///< Amount of __shared__ memory per threadblock to use as bounce buffer (0 = device max)
    int tdmKernel  = 0;                         ///< Kernel selector: -1=auto, 0=copy-only, 1=reduce
  };

  /**
   * Configuration options for performing Transfers
   */
  struct ConfigOptions
  {
    GeneralOptions general;                     ///< General options
    DataOptions    data;                        ///< Data options

    GfxOptions     gfx;                         ///< GFX executor options
    DmaOptions     dma;                         ///< DMA executor options
    NicOptions     nic;                         ///< NIC executor options
    TdmOptions     tdm;                         ///< TDM executor options
  };

  /**
   * Enumeration of possible error types
   */
  enum ErrType
  {
    ERR_NONE  = 0,                              ///< No errors
    ERR_WARN  = 1,                              ///< Warning - results may not be accurate
    ERR_FATAL = 2,                              ///< Fatal error - results are invalid
  };

  /**
   * Enumeration of GID priority
   *
   * @note These are the GID types ordered in priority from lowest (0) to highest
   */
  enum GidPriority
  {
    UNKNOWN           = -1,                      ///< Default
    ROCEV1_LINK_LOCAL = 0,                       ///< RoCEv1 Link-local
    ROCEV2_LINK_LOCAL = 1,                       ///< RoCEv2 Link-local fe80::/10
    ROCEV1_IPV6       = 2,                       ///< RoCEv1 IPv6
    ROCEV2_IPV6       = 3,                       ///< RoCEv2 IPv6
    ROCEV1_IPV4       = 4,                       ///< RoCEv1 IPv4-mapped IPv6
    ROCEV2_IPV4       = 5,                       ///< RoCEv2 IPv4-mapped IPv6 ::ffff:192.168.x.x
  };

  const char* GidPriorityStr[] = {
    "RoCEv1 Link-local",
    "RoCEv2 Link-local",
    "RoCEv1 IPv6",
    "RoCEv2 IPv6",
    "RoCEv1 IPv4-mapped IPv6",
    "RoCEv2 IPv4-mapped IPv6"
  };

  /**
   * Enumeration of possible communication mode types
   */
  enum CommType
  {
    COMM_NONE   = 0,                             ///< Single node only
    COMM_MPI    = 1,                             ///< MPI-based communication
    COMM_SOCKET = 2                              ///< Socket-based communication
  };

  /**
   * ErrResult consists of error type and error message
   */
  struct ErrResult
  {
    ErrType     errType;                        ///< Error type
    std::string errMsg;                         ///< Error details

    ErrResult() = default;
#if defined(__NVCC__)
    ErrResult(cudaError_t  err);
    ErrResult(CUresult     err);
#else
    ErrResult(hipError_t   err);
    ErrResult(hsa_status_t err);
#endif
    ErrResult(ErrType      err);
    ErrResult(ErrType      errType, const char* format, ...);
  };

  /**
   * Results for a single Executor
   */
  struct ExeResult
  {
    size_t      numBytes;                       ///< Total bytes transferred by this Executor
    double      avgDurationMsec;                ///< Averaged duration for all the Transfers for this Executor
    double      avgBandwidthGbPerSec;           ///< Average bandwidth for this Executor
    double      sumBandwidthGbPerSec;           ///< Naive sum of individual Transfer average bandwidths
    vector<int> transferIdx;                    ///< Indicies of Transfers this Executor executed
  };

  /**
   * Results for a single Transfer
   */
  struct TransferResult
  {
    size_t numBytes;                            ///< Number of bytes transferred by this Transfer
    double avgDurationMsec;                     ///< Duration for this Transfer, averaged over all timed iterations
    double avgBandwidthGbPerSec;                ///< Bandwidth for this Transfer based on averaged duration

    // Only filled in if recordPerIteration = 1
    vector<double> perIterMsec;                 ///< Duration for each individual iteration
    vector<set<pair<int,int>>> perIterCUs;      ///< GFX-Executor only. XCC:CU used per iteration

    ExeDevice exeDevice;                        ///< Tracks which executor performed this Transfer (e.g. for EXE_NIC_NEAREST)
    ExeDevice exeDstDevice;                     ///< Tracks actual destination executor (only valid for EXE_NIC/EXE_NIC_NEAREST)
  };

  /**
   * TestResults contain timing results for a set of Transfers as a group as well as per Executor and per Transfer
   * timing information
   */
  struct TestResults
  {
    int    numTimedIterations;                  ///< Number of iterations executed
    size_t totalBytesTransferred;               ///< Total bytes transferred per iteration
    double avgTotalDurationMsec;                ///< Wall-time (msec) to finish all Transfers (averaged across all timed iterations)
    double avgTotalBandwidthGbPerSec;           ///< Bandwidth based on all Transfers and average wall time
    double overheadMsec;                        ///< Difference between total wall time and slowest executor

    map<ExeDevice, ExeResult> exeResults;       ///< Per Executor results
    vector<TransferResult>    tfrResults;       ///< Per Transfer results
    vector<ErrResult>         errResults;       ///< List of any errors/warnings that occurred
  };

  /**
   * Run a set of Transfers
   *
   * @param[in]  config     Configuration options
   * @param[in]  transfers  Set of Transfers to execute
   * @param[out] results    Timing results
   * @returns true if and only if Transfers were run successfully without any fatal errors
   */
  bool RunTransfers(ConfigOptions    const& config,
                    vector<Transfer> const& transfers,
                    TestResults&            results);

  enum StrAttribute
  {
    ATR_SRC_PREP_DESCRIPTION                    ///< Description of how source memory is prepared
  };

  /**
   * Query attributes (string)
   *
   * @note This allows query of implementation details such as limits
   *
   * @param[in] attrtibute Attribute to query
   * @returns Value of the attribute
   */
  std::string GetStrAttribute(StrAttribute attribute);

  /**
   * Returns information about number of available Executors given an executor type
   *
   * @param[in] exeType         Executor type to query
   * @param[in] targetRank      Rank to query (-1 for local rank)
   * @returns Number of detected Executors of exeType
   */
  int GetNumExecutors(ExeType exeType, int targetRank = -1);

  /**
   * Returns information about number of available Executors given a memory type
   *
   * @param[in] memType         Memory type to query
   * @param[in] targetRank      Rank to query (-1 for local rank)
   * @returns Number of detected Executors for memType
   */
  int GetNumExecutors(MemType memType, int targetRank = -1);

  /**
   * Returns the number of possible Executor subindices
   *
   * @note For CPU, this is 0
   * @note For GFX, this refers to the number of XCDs
   * @note For DMA, this refers to the number of DMA engines
   *
   * @param[in] exeDevice       The specific Executor to query
   * @returns Number of detected executor subindices
   */
  int GetNumExecutorSubIndices(ExeDevice exeDevice);

  /**
   * Returns number of subExecutors for a given ExeDevice
   *
   * @param[in] exeDevice       The specific Executor to query
   * @returns Number of detected subExecutors for the given ExePair
   */
  int GetNumSubExecutors(ExeDevice exeDevice);

  /**
   * Returns the index of the NUMA node closest to the given GPU
   *
   * @param[in] gpuIndex        Index of the GPU to query
   * @param[in] targetRank      Rank to query (-1 for local rank)
   * @returns NUMA node index closest to GPU gpuIndex, or -1 if unable to detect
   */
  int GetClosestCpuNumaToGpu(int gpuIndex, int targetRank = -1);

  /**
   * Returns the index of the NUMA node closest to the given NIC
   *
   * @param[in] nicIndex        Index of the NIC to query
   * @param[in] targetRank      Rank to query (-1 for local rank)
   * @returns NUMA node index closest to the NIC nicIndex, or -1 if unable to detect
   */
  int GetClosestCpuNumaToNic(int nicIndex, int targetRank = -1);

  /**
   * Returns the index of a NIC closest to the given GPU
   *
   * @param[in] gpuIndex        Index of the GPU to query
   * @param[in] targetRank      Rank to query (-1 for local rank)
   * @note This function is applicable when the IBV/RDMA executor is available
   * @returns IB Verbs capable NIC index closest to GPU gpuIndex, or -1 if unable to detect
   */
  int GetClosestNicToGpu(int gpuIndex, int targetRank = -1);

  /**
   * Returns the indices of the NICs closest to the given CPU
   *
   * @param[out] nicIndices     Vector that will contain NIC indices closest to given CPU
   * @param[in]  cpuIndex       Index of the CPU to query
   * @param[in]  targetRank     Rank to query (-1 for local rank)
   * @note This function is applicable when the IBV/RDMA executor is available
   * @returns IB Verbs capable NIC indices closest to CPU cpuIndex, or empty if unable to detect
   */
  void GetClosestNicsToCpu(std::vector<int>& nicIndices, int cpuIndex, int targetRank = -1);

  /**
   * Returns the indices of the NICs closest to the given GPU
   *
   * @param[out] nicIndices     Vector that will contain NIC indices closest to given GPU
   * @param[in]  gpuIndex       Index of the GPU to query
   * @param[in]  targetRank     Rank to query (-1 for local rank)
   * @note This function is applicable when the IBV/RDMA executor is available
   * @returns IB Verbs capable NIC indices closest to GPU gpuIndex, or empty if unable to detect
   */
  void GetClosestNicsToGpu(std::vector<int>& nicIndices, int gpuIndex, int targetRank = -1);


  /**
   * Returns the index of a GPU closest to the given NIC
   *
   * @param[in] nicIndex        Index of the NIC to query
   * @param[in] targetRank      Rank to query (-1 for local rank)
   * @note This function is applicable when the IBV/RDMA executor is available
   * @returns GPU index closest to IB Verbs capable NIC index nicIndex, or -1 if unable to detect
   */
  int GetClosestGpuToNic(int nicIndex, int targetRank);

  /**
   * Returns the indices of the GPUs closest to the given NIC
   *
   * @param[out] gpuIndices     Vector that will contain GPU indices closest to given NIC
   * @param[in]  nicIndex        Index of the NIC to query
   * @param[in]  targetRank      Rank to query (-1 for local rank)
   * @note This function is applicable when the IBV/RDMA executor is available
   * @returns GPU indices closest to NIC nicIndex, or empty if unable to detect
   */
  void GetClosestGpusToNic(std::vector<int>& gpuIndices, int nicIndex, int targetRank = -1);

  /**
   * @returns 0-indexed rank for this process
   */
  int GetRank();

  /**
   * @returns The total numbers of ranks participating
   */
  int GetNumRanks();

  /**
   * @returns Gets the current communication mode
   */
  int GetCommMode();

  /**
   * @param[in] targetRank  Rank to query (-1 for local rank)
   * @returns Gets the hostname for the target rank
   **/
  std::string GetHostname(int targetRank = -1);

  /**
   * @param[in] targetRank Rank to query (-1 for local rank)
   * @returns Gets the unique pod identifier for the target rank based on its physical and virtual pod
   **/
  int64_t GetPodIdx(int targetRank = -1);

  /**
   * @param[in] targetRank  Remote rank to query
   * @param[in] sourceRank  Base rank to query (-1 for local rank)
   * @returns Whether source and target ranks belong to the same pod
   **/
  bool IsSamePod(int targetRank, int sourceRank = -1);

  /**
   * @param[in] exeDevice       The specific Executor to query
   * @returns Name of the executor
   */
  std::string GetExecutorName(ExeDevice exeDevice);

  /**
   *
   * @param[in] nicIndex        The NIC index to query
   * @param[in] targetRank Rank to query (-1 for local rank)
   * @returns Returns 1 if and only if NIC exists and has an active port
   */
  int NicIsActive(int nicIndex, int targetRank = -1);

  /**
   * Helper function to parse a line containing Transfers into a vector of Transfers
   *
   * @param[in]  str       String containing description of Transfers
   * @param[out] transfers List of Transfers described by 'str'
   * @returns Information about any error that may have occured
   */
  ErrResult ParseTransfers(std::string str,
                           std::vector<Transfer>& transfers);
}
//==========================================================================================
// End of TransferBench API
//==========================================================================================

// Redefinitions for CUDA compatibility
//==========================================================================================
#if defined(__NVCC__)

  // ROCm specific
  #define wall_clock64                                       clock64
  #define gcnArchName                                        name

  // Datatypes
  #define hipDeviceProp_t                                    cudaDeviceProp
  #define hipError_t                                         cudaError_t
  #define hipEvent_t                                         cudaEvent_t
  #define hipStream_t                                        cudaStream_t
  #define hipMemAllocationProp                               CUmemAllocationProp
  #define hipMemGenericAllocationHandle_t                    CUmemGenericAllocationHandle
  #define hipMemAccessDesc                                   CUmemAccessDesc
  #define hipMemFabricHandle_t                               CUmemFabricHandle

  // Enumerations
  #define hipDeviceAttributeClockRate                        cudaDevAttrClockRate
  #define hipDeviceAttributeMultiprocessorCount              cudaDevAttrMultiProcessorCount
  #define hipDeviceAttributeMaxSharedMemoryPerBlock          cudaDevAttrMaxSharedMemoryPerBlock
  #define hipDeviceAttributeWarpSize                         cudaDevAttrWarpSize
  #define hipErrorPeerAccessAlreadyEnabled                   cudaErrorPeerAccessAlreadyEnabled
  #define hipFuncCachePreferShared                           cudaFuncCachePreferShared
  #define hipMemcpyDefault                                   cudaMemcpyDefault
  #define hipMemcpyKind                                      cudaMemcpyKind
  #define hipMemcpyDeviceToHost                              cudaMemcpyDeviceToHost
  #define hipMemcpyHostToDevice                              cudaMemcpyHostToDevice
  #define hipSuccess                                         cudaSuccess
  #define hipMemLocationTypeDevice                           CU_MEM_LOCATION_TYPE_DEVICE
  #define hipMemAllocationTypePinned                         CU_MEM_ALLOCATION_TYPE_PINNED
  #define hipMemHandleTypeFabric                             CU_MEM_HANDLE_TYPE_FABRIC
  #define hipMemAllocationGranularityRecommended             CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
  #define hipMemAccessFlagsProtReadWrite                     CU_MEM_ACCESS_FLAGS_PROT_READWRITE

  // Functions
  #define hipDeviceCanAccessPeer                             cudaDeviceCanAccessPeer
  #define hipDeviceEnablePeerAccess                          cudaDeviceEnablePeerAccess
  #define hipDeviceGetAttribute                              cudaDeviceGetAttribute
  #define hipDeviceGetPCIBusId                               cudaDeviceGetPCIBusId
  #define hipDeviceSetCacheConfig                            cudaDeviceSetCacheConfig
  #define hipDeviceSynchronize                               cudaDeviceSynchronize
  #define hipEventCreate                                     cudaEventCreate
  #define hipEventDestroy                                    cudaEventDestroy
  #define hipEventElapsedTime                                cudaEventElapsedTime
  #define hipEventRecord                                     cudaEventRecord
  #define hipFree                                            cudaFree
  #define hipGetDeviceCount                                  cudaGetDeviceCount
  #define hipGetDeviceProperties                             cudaGetDeviceProperties
  #define hipGetErrorString                                  cudaGetErrorString
  #define hipGetLastError                                    cudaGetLastError
  #define hipHostFree                                        cudaFreeHost
  #define hipHostMalloc                                      cudaMallocHost
  #define hipMalloc                                          cudaMalloc
  #define hipMallocManaged                                   cudaMallocManaged
  #define hipMemcpy                                          cudaMemcpy
  #define hipMemcpyAsync                                     cudaMemcpyAsync
  #define hipMemcpyBatchAsync                                cudaMemcpyBatchAsync
  #define hipMemset                                          cudaMemset
  #define hipMemsetAsync                                     cudaMemsetAsync
  #define hipSetDevice                                       cudaSetDevice
  #define hipStreamCreate                                    cudaStreamCreate
  #define hipStreamDestroy                                   cudaStreamDestroy
  #define hipStreamSynchronize                               cudaStreamSynchronize
  // cu* driver API returns CUresult; cast to cudaError_t so callers can use a single error variable
#define hipMemGetAllocationGranularity(...)                  ((cudaError_t)cuMemGetAllocationGranularity(__VA_ARGS__))
  #define hipMemCreate(...)                                  ((cudaError_t)cuMemCreate(__VA_ARGS__))
  #define hipMemAddressReserve(...)                          ((cudaError_t)cuMemAddressReserve(__VA_ARGS__))
  #define hipMemMap(...)                                     ((cudaError_t)cuMemMap(__VA_ARGS__))
  #define hipMemSetAccess(...)                               ((cudaError_t)cuMemSetAccess(__VA_ARGS__))
  #define hipMemUnmap(...)                                   ((cudaError_t)cuMemUnmap(__VA_ARGS__))
  #define hipMemRelease(...)                                 ((cudaError_t)cuMemRelease(__VA_ARGS__))
  #define hipMemAddressFree(...)                             ((cudaError_t)cuMemAddressFree(__VA_ARGS__))
  #define hipMemExportToShareableHandle(...)                 ((cudaError_t)cuMemExportToShareableHandle(__VA_ARGS__))
  #define hipMemImportFromShareableHandle(...)               ((cudaError_t)cuMemImportFromShareableHandle(__VA_ARGS__))

  using gpu_device_ptr = CUdeviceptr;

  // Define float2 addition operator for NVIDIA platform
  __device__ inline float2& operator +=(float2& a, const float2& b)
  {
    a.x += b.x;
    a.y += b.y;
    return a;
  }

  // Define float4 addition operator for NVIDIA platform
  __device__ inline float4& operator +=(float4& a, const float4& b)
  {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
  }
#else
  using gpu_device_ptr = void*;
#endif

// Helper macro functions
//==========================================================================================

// Macro for collecting CU/SM GFX kernel is running on
#if defined(__GFX9__)
  #define GetHwId(hwId) asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s" (hwId))
#elif defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__)
  #define GetHwId(hwId) asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s" (hwId))
#elif defined(__NVCC__)
  #define GetHwId(hwId) asm("mov.u32 %0, %smid;" : "=r"(hwId))
#else
  #define GetHwId(hwId) hwId = 0
#endif

// Macro for collecting XCC GFX kernel is running on
#if defined(__gfx942__) || defined(__gfx950__)
#define GetXccId(val) asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s" (val))
#elif defined(__GFX12__)
#define GetXccId(val) \
  { asm volatile ("s_sendmsg_rtn_b32 %0, 0x87 \n" \
                  "s_wait_kmcnt 0"                \
                  : "=s" (val));                  \
    val = ((val >> 16) & 0xF);                    \
  }
#else
#define GetXccId(val) val = 0
#endif

// Error check macro (NOTE: This will return even for ERR_WARN)
#define ERR_CHECK(cmd)                                                       \
  do {                                                                       \
    ErrResult err = (cmd);                                                   \
    if (err.errType != ERR_NONE) {                                           \
      err.errMsg += std::string(" [") + __FILE__ + ":" +                     \
                    std::to_string(__LINE__) + " in " + __func__ + "]";      \
      return err;                                                            \
    }                                                                        \
  } while (0)

// Appends warn/fatal errors to a list, return false if fatal
#define ERR_APPEND(cmd, list)                                                \
  do {                                                                       \
    ErrResult err = (cmd);                                                   \
    if (err.errType != ERR_NONE) {                                           \
      err.errMsg += std::string(" [") + __FILE__ + ":" +                     \
                    std::to_string(__LINE__) + " in " + __func__ + "]";      \
      list.push_back(err);                                                   \
    }                                                                        \
    if (err.errType == ERR_FATAL)                                            \
      return false;               \
  } while (0)

// Helper macros for calling RDMA functions and reporting errors
#ifdef VERBS_DEBUG
#define IBV_CALL(__func__, ...)                                         \
  do {                                                                  \
    int error = __func__(__VA_ARGS__);                                  \
    if (error != 0) {                                                   \
      return {ERR_FATAL, "Encountered IbVerbs error (%d) at line (%d) " \
              "and function (%s)", (error), __LINE__, #__func__};       \
    }                                                                   \
  } while (0)

#define IBV_PTR_CALL(__ptr__, __func__, ...)                               \
  do {                                                                     \
    __ptr__ = __func__(__VA_ARGS__);                                       \
    if (__ptr__ == nullptr) {                                              \
      return {ERR_FATAL, "Encountered IbVerbs nullptr error at line (%d) " \
              "and function (%s)", __LINE__, #__func__};                   \
    }                                                                      \
  } while (0)
#else
#define IBV_CALL(__func__, ...)                                           \
  do {                                                                    \
    int error = __func__(__VA_ARGS__);                                    \
    if (error != 0) {                                                     \
      return {ERR_FATAL, "Encountered IbVerbs error (%d=%s) in func (%s)" \
              , error, strerror(errno), #__func__};                       \
    }                                                                     \
  } while (0)

#define IBV_PTR_CALL(__ptr__, __func__, ...)                                    \
  do {                                                                          \
    __ptr__ = __func__(__VA_ARGS__);                                            \
    if (__ptr__ == nullptr) {                                                   \
      return {ERR_FATAL, "Encountered IbVerbs nullptr error (%s) in func (%s) " \
              , strerror(errno), #__func__};                                    \
    }                                                                           \
  } while (0)
#endif

namespace TransferBench
{

/// @cond
// Helper functions ('hidden' in anonymous namespace)
//========================================================================================
namespace {

// Constants
//========================================================================================
  int   constexpr MAX_BLOCKSIZE  = 1024;               // Max threadblock size
  int   constexpr MAX_SRCS       = 8;                  // Max srcs per Transfer
  int   constexpr MAX_DSTS       = 8;                  // Max dsts per Transfer
  int   constexpr MEMSET_CHAR    = 75;                 // Value to memset (char)
  float constexpr MEMSET_VAL     = 13323083.0f;        // Value to memset (double)

  int GetWarpSize(std::vector<ErrResult>* errors = nullptr) {
    int warpSize = 0;
    hipError_t err = hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize, 0);
    if (err == hipSuccess) {
      return warpSize;
    }

    // Query failed, report error and fall back to compile-time default
    if (errors) {
      errors->push_back({ERR_WARN,
                        "Failed to query device warp size (hipDeviceGetAttribute error: %d). "
                        "Falling back to compile-time default", err});
    }
#if defined(__NVCC__)
    return 32;
#else
    return 64;
#endif
  }

  // Calculate grid Y dimension based on SE_TYPE
  int CalculateGridY(int seType, int blockSize, int numSubExecs) {
    // Warp-level: each subexecutor is a warp, pack warps into threadblocks
    if (seType == 1) {
      int warpsPerBlock = blockSize / GetWarpSize();
      return (numSubExecs + warpsPerBlock - 1) / warpsPerBlock;
    }

    // Default: Threadblock-level, each subexecutor is a threadblock
    return numSubExecs;
  }

// System singleton
//========================================================================================
/**
   * System singleton class used for multi-node capability / topology dectection
   *
   * This supports three possible communication modes - Socket-based, MPI-based, disabled
   *
   * - Will first attempt to use sockets when TB_NUM_RANKS is set (>= 2)
   * - Will then try MPI-based, if compiled with MPI support
   * - Drop back to single node functionality

   * - Configuration for socket-based communicator is read via environment variables
   *   - TB_NUM_RANKS:   Total number of processes (only variable required on rank 0; rank 0 logs how workers should connect)
   *   - TB_RANK:        Rank of this process (0-based); defaults to 0 if unset or empty
   *   - TB_MASTER_ADDR: Rank 0 address for workers to connect; optional on rank 0 (auto-detected IPv4 after listen)
   *   - TB_MASTER_IFACE: Optional interface name when auto-detecting rank-0 address (e.g. eth0)
   *   - TB_MASTER_PORT: Port for communication (default: 29500)
   */
  class System
  {
  public:
    static System& Get() {
      static System instance;
      return instance;
    }

    /**
     * @returns 0-indexed rank for this process
     */
    int GetRank() const { return rank; }

    /**
     * @returns The total numbers of ranks participating
     */
    int GetNumRanks() const { return numRanks; }

    /**
     * @returns The communication mode
     */
    int GetCommMode() const { return commMode; }

    bool& IsVerbose() { return verbose; }

    /**
     * Helper logging function that logs only on output ranks
     * - In MPI mode - Rank 0 only
     * - In socket mode - All ranks unless TB_SINGLE_LOG=1
     */
    void Log(const char* format, ...) const;

    /**
     * Helper function that logs Transfers being executed to a config file
     */
    void LogTransfers(std::vector<Transfer> const& transfers);

    // Communication functions
    /**
     * Barrier that all ranks must arrive at before proceeding
     */
    void Barrier() const;

    /**
     * Send data to a single destination rank
     * Requires a matching call to RecvData on destination rank
     * NOTE: For socket-based communicator, this must involve rank 0
     *
     * @param[in] dstRank       Rank to send to
     * @param[in] numBytes      Number of bytes to send
     * @param[in] sendData      Data to send
     */
    void SendData(int dstRank, size_t const numBytes, const void* sendData) const;

    /**
     * Recevive data from a single source rank
     * Requires a matching call to SendData on source rank
     * NOTE: For socket-based communicator, this must involve rank 0
     *
     * @param[in] srcRank       Rank to receive from
     * @param[in] numBytes      Number of bytes to receive
     * @param[in] recvData      Buffer to receive data into
     */
    void RecvData(int srcRank, size_t const numBytes, void* recvData) const;

    /**
     * Modifies provided input to true if any rank provides a true input
     *
     * @param[in] flag          Flag to compare across ranks
     * @returns   True if and only if any rank provided a flag with value of true
     */
    bool Any(bool const flag) const;

    /**
     * Broadcast data from root to all ranks
     * All ranks must participate in this call
     *
     * @param[in] root          Rank that sends data
     * @param[in] numBytes      Number of bytes to transfer
     * @param[in/out] data      Buffer to send from root / to receive into on other ranks
     */
    void Broadcast(int root, size_t const numBytes, void* data) const;

    /**
     * Collect errors across ranks
     * @param[in,out] errResults List of errors per rank
     */
    void AllGatherErrors(vector<ErrResult>& errResults) const;

    // Topology functions
    /**
     * Returns information about number of available Executors
     *
     * @param[in] exeType       Executor type to query
     * @param[in] targetRank    Rank to query.  (-1 for local rank)
     * @returns Number of detected Executors of exeType
     */
    int GetNumExecutors(ExeType exeType, int targetRank = -1) const;

    /**
     * Returns the number of possible Executor subindices
     *
     * @note For CPU, this is 0
     * @note For GFX, this refers to the number of XCDs
     * @note For DMA, this refers to the number of DMA engines
     *
     * @param[in] exeDevice     The specific Executor to query
     * @returns Number of detected executor subindices
     */
    int GetNumExecutorSubIndices(ExeDevice exeDevice) const;

    /**
     * Returns number of subExecutors for a given ExeDevice
     *
     * @param[in] exeDevice     The specific Executor to query
     * @returns Number of detected subExecutors for the given ExePair
     */
    int GetNumSubExecutors(ExeDevice exeDevice) const;

    /**
     * Returns the index of the NUMA node closest to the given GPU
     *
     * @param[in] gpuIndex      Index of the GPU to query
     * @param[in] targetRank    Rank to query (-1 for local rank)
     * @returns NUMA node index closest to GPU gpuIndex, or -1 if unable to detect
     */
    int GetClosestCpuNumaToGpu(int gpuIndex, int targetRank = -1) const;

    /**
     * Returns the index of the NUMA node closest to the given NIC
     *
     * @param[in] nicIndex      Index of the NIC to query
     * @param[in] targetRank    Rank to query (-1 for local rank)
     * @returns NUMA node index closest to the NIC nicIndex, or -1 if unable to detect
     */
    int GetClosestCpuNumaToNic(int nicIndex, int targetRank = -1) const;

    /**
     * Returns the indices of the NICs closest to the given GPU
     *
     * @param[out] nicIndices     Vector that will contain NIC indices closest to given GPU
     * @param[in] gpuIndex        Index of the GPU to query
     * @param[in] targetRank      Rank to query (-1 for local rank)
     * @note This function is applicable when the IBV/RDMA executor is available
     * @returns IB Verbs capable NIC indices closest to GPU gpuIndex, or empty if unable to detect
     */
    void GetClosestNicsToGpu(std::vector<int>& nicIndices, int gpuIndex, int targetRank = -1) const;

    /**
     * Returns the indices of the GPUs closest to the given NIC
     *
     * @param[out] gpuIndices     Vector that will contain GPU indices closest to given NIC
     * @param[in] nicIndex        Index of the NIC to query
     * @param[in] targetRank      Rank to query (-1 for local rank)
     * @note This function is applicable when the IBV/RDMA executor is available
     * @returns GPU indices closest to NIC nicIndex, or empty if unable to detect
     */
    void GetClosestGpusToNic(std::vector<int>& gpuIndices, int nicIndex, int targetRank = -1) const;

    std::string GetHostname(int targetRank) const;
    int64_t  GetPodIdx(int targetRank) const;
    bool IsSamePod(int targetRank, int sourceRank) const;
    std::string GetExecutorName(ExeDevice exeDevice) const;
    int NicIsActive(int nicIndex, int targetRank) const;

#if !defined(__NVCC__)
    ErrResult GetHsaAgent(ExeDevice const& exeDevice, hsa_agent_t& agent) const;
    ErrResult GetHsaAgent(MemDevice const& memDevice, hsa_agent_t& agent) const;
#endif

    template <typename T>
    void BroadcastVector(int root, vector<T>& data) const;
    void BroadcastString(int root, std::string& string) const;
    void BroadcastExeResult(int root, ExeResult& exeResult) const;
    void BroadcastTfrResult(int root, TransferResult& tfrResult) const;


  private:
    System();
    ~System();
    System(System const&)            = delete;
    System(System&&)                 = delete;
    System& operator=(System const&) = delete;
    System& operator=(System&&)      = delete;

    int rank;
    int numRanks;
    bool verbose = false;
    bool rankDoesOutput = true;
    FILE* dumpCfgFile = nullptr;

#if !defined(__NVCC__)
    std::vector<hsa_agent_t> cpuAgents;
    std::vector<hsa_agent_t> gpuAgents;
#endif

    int commMode;                             ///< Communication mode

#ifdef MPI_COMM_ENABLED
    bool mpiInit = false;                     ///< Whether or not MPI_Init was called
    MPI_Comm comm;                            ///< MPI communicator
#endif

    // Socket related
    std::string      masterAddr;              ///< Rank 0 master address
    int              masterPort;              ///< Rank 0 master port
    std::vector<int> sockets;                 ///< Master list of sockets
    int              listenSocket;            ///< Master listener socket

    // Topology related
    struct RankTopology
    {
      char    hostname[33];
      char    ppodId[16];
      int64_t vpodId;

      std::map<ExeType,            int>         numExecutors;
      std::map<pair<ExeType, int>, int>         numExecutorSubIndices;
      std::map<pair<ExeType, int>, int>         numSubExecutors;
      std::map<int,                int>         closestCpuNumaToGpu;
      std::map<int,                int>         closestCpuNumaToNic;
      std::map<int,                int>         nicIsActive;
      std::map<int,                vector<int>> closestNicsToGpu;
      std::map<int,                vector<int>> closestGpusToNic;
      std::map<pair<ExeType, int>, std::string> executorName;
    };

    std::vector<RankTopology> rankInfo;       ///< Topology of each rank

    void SetupSocketCommunicator();
    void SetupMpiCommunicator();
    void CollectPodMembership(char* ppodId, int64_t& vpodId);
    void GetRankTopology(RankTopology& topo);
    void CollectTopology();
    std::string GetCpuName() const;

    template <typename KeyType, typename ValType>
    void SendMap(int peerRank, std::map<KeyType, std::vector<ValType>> const& mapToSend) const;
    template <typename KeyType, typename ValType>
    void SendMap(int peerRank, std::map<KeyType, ValType> const& mapToSend) const;
    template <typename KeyType>
    void SendMap(int peerRank, std::map<KeyType, std::string> const& mapToSend) const;

    template <typename KeyType, typename ValType>
    void RecvMap(int peerRank, std::map<KeyType, std::vector<ValType>>& mapToRecv) const;
    template <typename KeyType, typename ValType>
    void RecvMap(int peerRank, std::map<KeyType, ValType>& mapToRecv) const;
    template <typename KeyType>
    void RecvMap(int peerRank, std::map<KeyType, std::string>& mapToRecv) const;

    void SendRankTopo(int peerRank, RankTopology const& topo) const;
    void RecvRankTopo(int peerRank, RankTopology& topo) const;
  };

// Parsing-related functions
//========================================================================================
  static ErrResult CharToMemType(char const c, MemType& memType)
  {
    char const* val = strchr(MemTypeStr, toupper(c));
    if (val) {
      memType = (MemType)(val - MemTypeStr);
      return ERR_NONE;
    }
    return {ERR_FATAL, "Unexpected memory type (%c)", c};
  }

  static ErrResult CharToExeType(char const c, ExeType& exeType)
  {
    char const* val = strchr(ExeTypeStr, toupper(c));
    if (val) {
      exeType = (ExeType)(val - ExeTypeStr);
      return ERR_NONE;
    }
    return {ERR_FATAL, "Unexpected executor type (%c)", c};
  }

  struct WildcardMemDevice
  {
    MemType memType;
    vector<int> memRanks;
    vector<int> memIndices;
  };

  struct WildcardExeDevice
  {
    ExeType exeType;
    std::vector<int> exeRanks;
    std::vector<int> exeIndices;
    std::vector<int> exeSlots;
    std::vector<int> exeSubIndices;
    std::vector<int> exeSubSlots;
  };

  struct WildcardTransfer
  {
    std::vector<WildcardMemDevice> mem[2]; // 0 = SRCs, 1 = DSTs
    WildcardExeDevice exe;
  };

  static char const* ParseRange(char const* start, int fullCount, std::vector<int>& range)
  {
    range.clear();

    char const* ptr = start;
    if (!ptr) return 0;

    // Full wildcard
    if (*ptr == '*') {
      if (fullCount >= 0) {
        for (int i = 0; i < fullCount; i++)
        range.push_back(i);
      } else {
        range.push_back(fullCount);
      }
      return ++ptr;
    }

    // Ranged wildcard
    if (*ptr == '[') {
      std::string rangeStr(++ptr);
      size_t endPos = rangeStr.find(']');
      if (endPos == std::string::npos) return 0;
      rangeStr.erase(endPos);
      ptr += endPos+1;

      std::set<int> values;
      char* token = strtok(rangeStr.data(), ",");
      while (token) {
        int start, end;
        if (sscanf(token, "%d..%d", &start, &end) == 2) {
          if (start < 0 || end < 0 || end <= start) return 0;
          for (int i = start; i <= end; i++)
            values.insert(i);
        } else if (sscanf(token, "%d", &start) == 1) {
          values.insert(start);
        } else {
          return 0;
        }
        token = strtok(NULL, ",");
      }
      if (values.empty()) return 0;
      for (auto v : values) range.push_back(v);
      return ptr;
    }

    // Single number
    char* endPtr;
    int val = strtol(ptr, &endPtr, 10);
    if (endPtr == ptr) return 0;
    else range.push_back(val);
    return endPtr;
  }

  static char const* ParseAlphaRange(char const* start, std::vector<int>& range)
  {
    range.clear();

    char const* ptr = start;
    if (!ptr) return 0;

    // Full wildcard
    if (*ptr == '*') {
      range.push_back(-1);
      return ++ptr;
    }

    // Ranged wildcard
    if (*ptr == '[') {
      std::string rangeStr(++ptr);
      size_t endPos = rangeStr.find(']');
      if (endPos == std::string::npos) return 0;
      rangeStr.erase(endPos);
      ptr += endPos+1;

      std::set<int> values;
      char* token = strtok(rangeStr.data(), ",");
      while (token) {
        char start, end;
        if (sscanf(token, "%c..%c", &start, &end) == 2 && isalpha(toupper(start)) && isalpha(toupper(end))) {
          int realStart = toupper(start) - 'A';
          int realEnd   = toupper(end)   - 'A';
          if (realStart < 0 || realEnd < 0) return 0;
          for (int i = realStart; i <= realEnd; i++)
            values.insert(i);
        } else if (sscanf(token, "%c", &start) == 1 && isalpha(toupper(start))) {
          int realStart = toupper(start) - 'A';
          values.insert(realStart);
        } else {
          return 0;
        }
        token = strtok(NULL, ",");
      }
      for (auto v : values) range.push_back(v);
      return ptr;
    }

    // Single character
    if (isalpha(toupper(*ptr))) {
      range.push_back(toupper(*ptr)-'A');
      ++ptr;
    }
    return ptr;
  }

  static ErrResult ParseMemType(std::string const& token,
                                std::vector<WildcardMemDevice>& memDevices)
  {
    memDevices.clear();

    char const* ptr = token.c_str();
    while (*ptr) {
      WildcardMemDevice w;

      // Parse memory rank if it exists
      if (*ptr == 'R' || *ptr == 'r') {
        ptr++; // Skip 'R'
        ptr = ParseRange(ptr, GetNumRanks(), w.memRanks);
        if (!ptr) return {ERR_FATAL, "Unable to parse rank index in memory token %s", token.c_str()};
      } else {
        // Otherwise will be replaced by "local" wildcard
        w.memRanks.clear();
      }

      // Parse memory type
      ErrResult err = CharToMemType(*ptr, w.memType);
      if (err.errType != ERR_NONE) {
        return {err.errType, "Error parsing token [%s]: %s\n", token.c_str(), err.errMsg.c_str()};
      }
      ptr++; // Skip memory type

      // Parse memory index
      if (w.memType != MEM_NULL) {
        ptr = ParseRange(ptr, -1, w.memIndices);
        if (!ptr) return {ERR_FATAL, "Unable to parse device index in memory token %s", token.c_str()};
        memDevices.push_back(w);
      } else {
        break;
      }
    }
    return ERR_NONE;
  }

  static ErrResult ParseExeType(std::string const& token,
                                WildcardExeDevice& exeDevice)
  {
    char const* ptr = token.c_str();

    // Check for rank prefix
    if (*ptr == 'R' || *ptr == 'r') {
      ptr++; // Skip 'R'
      ptr = ParseRange(ptr, GetNumRanks(), exeDevice.exeRanks);
      if (!ptr) return {ERR_FATAL, "Unable to parse rank index in executor token %s", token.c_str()};
    } else {
      exeDevice.exeRanks.clear();
    }

    // Parse executor type
    ERR_CHECK(CharToExeType(*ptr, exeDevice.exeType));
    ptr++; // Skip executor type char

    // Parse executor index
    // This is optional for EXE_NIC_NEAREST as long as nothing further is specified
    char const* endPtr = ParseRange(ptr, -1, exeDevice.exeIndices);
    if (!endPtr) {
      if (exeDevice.exeType == EXE_NIC_NEAREST && *endPtr == 0) {
        if (exeDevice.exeRanks.size() != 0) {
          return {ERR_FATAL, "Wildcard NIC executor may not be specified with rank in executor token %s", token.c_str()};
        }
        exeDevice.exeIndices.clear();
        return ERR_NONE;
      } else {
        return {ERR_FATAL, "Unable to parse device index in executor token %s", token.c_str()};
      }
    } else {
      ptr = endPtr;
    }

    // Parse (optional) executor slot
    ptr = ParseAlphaRange(ptr, exeDevice.exeSlots);
    if (!ptr) return {ERR_FATAL, "Unable to parse executor slot in executor token %s", token.c_str()};

    // Check for subindex after device
    if (*ptr == '.') {
      ptr++; // Skip '.'
      ptr = ParseRange(ptr, -2, exeDevice.exeSubIndices);
      if (!ptr) return {ERR_FATAL, "Unable to parse subindex in executor token %s", token.c_str()};
    }

    // Ensure that EXE_NIC has non-empty subindex
    if (exeDevice.exeType == EXE_NIC && exeDevice.exeSubIndices.size() == 0) {
      return {ERR_FATAL, "NIC executor requires specification of a subindex in executor token %s", token.c_str()};
    }

    // Parse (optional) executor subslot
    ptr = ParseAlphaRange(ptr, exeDevice.exeSubSlots);
    if (!ptr) return {ERR_FATAL, "Unable to parse subslot in executor token %s", token.c_str()};

    return ERR_NONE;
  }

// Memory-related functions
//========================================================================================
  // Enable peer access between two GPUs
  static ErrResult EnablePeerAccess(int const deviceId, int const peerDeviceId)
  {
    int canAccess;
    ERR_CHECK(hipDeviceCanAccessPeer(&canAccess, deviceId, peerDeviceId));
    if (!canAccess)
      return {ERR_FATAL, "Peer access is unavailable between GPU devices %d to %d."
                         "For AMD hardware, check IOMMU configuration", peerDeviceId, deviceId};

    ERR_CHECK(hipSetDevice(deviceId));
    hipError_t error = hipDeviceEnablePeerAccess(peerDeviceId, 0);
    if (error != hipSuccess && error != hipErrorPeerAccessAlreadyEnabled) {
      return {ERR_FATAL,
              "Unable to enable peer to peer access from %d to %d (%s)",
              deviceId, peerDeviceId, hipGetErrorString(error)};
    }
    return ERR_NONE;
  }

  // Check that CPU memory array of numBytes has been allocated on targetId NUMA node
  static ErrResult CheckPages(char* array, size_t numBytes, int targetId)
  {
    size_t const pageSize = getpagesize();
    size_t const numPages = (numBytes + pageSize - 1) / pageSize;

    std::vector<void *> pages(numPages);
    std::vector<int> status(numPages);

    pages[0] = array;
    for (int i = 1; i < numPages; i++) {
      pages[i] = (char*)pages[i-1] + pageSize;
    }

    long const retCode = move_pages(0, numPages, pages.data(), NULL, status.data(), 0);
    if (retCode)
      return {ERR_FATAL,
              "Unable to collect page table information for allocated memory. "
              "Ensure NUMA library is installed properly"};

    size_t mistakeCount = 0;
    for (size_t i = 0; i < numPages; i++) {
      if (status[i] < 0)
        return {ERR_FATAL,
                "Unexpected page status (%d) for page %llu", status[i], i};
      if (status[i] != targetId) mistakeCount++;
    }
    if (mistakeCount > 0) {
      return {ERR_FATAL,
              "%lu out of %lu pages for memory allocation were not on NUMA node %d."
              " This could be due to hardware memory issues, or the use of numa-rebalancing daemons such as numad",
              mistakeCount, numPages, targetId};
    }
    return ERR_NONE;
  }

#ifdef POD_COMM_ENABLED
  static ErrResult GetMemAllocationProp(MemDevice const& memDevice, hipMemAllocationProp& prop)
  {

    switch (memDevice.memType) {
    case MEM_CPU: case MEM_CPU_CLOSEST: case MEM_GPU:
      prop.type = hipMemAllocationTypePinned; break;
    case MEM_CPU_UNCACHED: case MEM_GPU_UNCACHED:
#if defined (__NVCC__)
      return {ERR_FATAL, "Uncached memory type unsupported in CUDA"};
#else
      prop.type = hipMemAllocationTypeUncached; break;
#endif
    default:
      return {ERR_FATAL, "Unsupported memory type for pod communication"};
    }

    prop.requestedHandleTypes = hipMemHandleTypeFabric;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = memDevice.memIndex;
    return ERR_NONE;
  }
#endif

  // Returns a human-readable label for a MemType (e.g. "CPU_PINNED", "GPU", "GPU_FINE")
  static char const* GetMemTypeName(MemType m)
  {
    switch (m) {
    case MEM_CPU:             return "CPU_PINNED";
    case MEM_CPU_CLOSEST:     return "CPU_PINNED_CLOSEST";
    case MEM_CPU_COHERENT:    return "CPU_COHERENT";
    case MEM_CPU_NONCOHERENT: return "CPU_NONCOHERENT";
    case MEM_CPU_UNCACHED:    return "CPU_UNCACHED";
    case MEM_CPU_UNPINNED:    return "CPU_UNPINNED";
    case MEM_GPU:             return "GPU";
    case MEM_GPU_FINE:        return "GPU_FINE";
    case MEM_GPU_UNCACHED:    return "GPU_UNCACHED";
    case MEM_MANAGED:         return "GPU_MANAGED";
    case MEM_NULL:            return "NULL";
    default:                  return "UNKNOWN";
    }
  }

  // Returns BDF string for a GPU index, or "" on failure
  static std::string GetGpuBdf(int gpuIndex)
  {
    char buf[64] = "";
    if (hipDeviceGetPCIBusId(buf, sizeof(buf), gpuIndex) != hipSuccess) return "";
    return std::string(buf);
  }

  // Returns closest CPU NUMA node for a MemDevice as a string (or "?" if unknown)
  static std::string GetMemDeviceNuma(MemDevice const& md)
  {
    if (IsCpuMemType(md.memType)) {
      int numaIdx = (md.memType == MEM_CPU_CLOSEST)
                    ? GetClosestCpuNumaToGpu(md.memIndex, md.memRank)
                    : md.memIndex;
      return (numaIdx >= 0) ? std::to_string(numaIdx) : "?";
    }
    if (IsGpuMemType(md.memType)) {
      int numaIdx = GetClosestCpuNumaToGpu(md.memIndex, md.memRank);
      return (numaIdx >= 0) ? std::to_string(numaIdx) : "?";
    }
    return "?";
  }

  // Returns whether a memory allocation can be directly read by host code.
  static ErrResult GetMemHostAccessibility(MemDevice const& memDevice, bool& hostAccessible)
  {
    hostAccessible = IsCpuMemType(memDevice.memType) || memDevice.memType == MEM_MANAGED;
    if (hostAccessible || !IsGpuMemType(memDevice.memType)) return ERR_NONE;

#if defined(__NVCC__)
    return ERR_NONE;
#else
    int isLargeBar = 0;
    ERR_CHECK(hipDeviceGetAttribute(&isLargeBar, hipDeviceAttributeIsLargeBar, memDevice.memIndex));
    hostAccessible = (isLargeBar != 0);
    return ERR_NONE;
#endif
  }

  // Allocate memory
  static ErrResult AllocateMemory(MemDevice memDevice, size_t numBytes, void** memPtr,
                                  size_t* actualBytes = NULL,
                                  hipMemGenericAllocationHandle_t* memHandle = NULL)
  {
    if (numBytes == 0) {
      return {ERR_FATAL, "Unable to allocate 0 bytes"};
    }
    *memPtr = nullptr;

    MemType const& memType = memDevice.memType;
    int deviceIdx = memDevice.memIndex;
    if (memType == MEM_CPU_CLOSEST) {
      deviceIdx = GetClosestCpuNumaToGpu(memDevice.memIndex);
    }

    if (IsCpuMemType(memType)) {
      // Set NUMA policy prior to call to hipHostMalloc
      numa_set_preferred(deviceIdx);
    } else if (IsGpuMemType(memType)) {
      // Switch to the appropriate GPU
      // IMP: if the remapping above changes, remember to modify this!
      ERR_CHECK(hipSetDevice(deviceIdx));
    }

    // If memHandle is provided, allocate sharable memory
    if (memHandle != NULL) {
#ifdef POD_COMM_ENABLED
      hipMemAllocationProp prop = {};
      ERR_CHECK(GetMemAllocationProp(memDevice, prop));

      // Determine recommended allocation granularity
      size_t granularity;
      ERR_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
                                               hipMemAllocationGranularityRecommended));
      size_t roundedUpBytes = (numBytes + granularity - 1) / granularity * granularity;
      if (actualBytes != NULL) *actualBytes = roundedUpBytes;

      // Create memory allocation described by properties and size
      ERR_CHECK(hipMemCreate(memHandle, roundedUpBytes, &prop, 0));

      // Reserve a virtual address range for the memory allocation
      ERR_CHECK(hipMemAddressReserve((gpu_device_ptr*)memPtr, roundedUpBytes, 0, 0, 0));

      // Map the allocation handle to the reserved address range
      ERR_CHECK(hipMemMap((gpu_device_ptr)*memPtr, roundedUpBytes, 0, *memHandle, 0));

      // Specify memory access descriptor to enable local read/write
      hipMemAccessDesc desc;
      desc.location.type = hipMemLocationTypeDevice;
      desc.location.id = memDevice.memIndex;
      desc.flags = hipMemAccessFlagsProtReadWrite;

      // Set access flags for virtual address range
      ERR_CHECK(hipMemSetAccess((gpu_device_ptr)*memPtr, roundedUpBytes, &desc, 1));

      // Clear the memory
      if (IsCpuMemType(memType)) {
        memset(*memPtr, 0, roundedUpBytes);
        // Check that the allocated pages are actually on the correct NUMA node
        ERR_CHECK(CheckPages((char*)*memPtr, roundedUpBytes, deviceIdx));
        numa_set_preferred(-1);
      } else if (IsGpuMemType(memType)) {
        ERR_CHECK(hipMemset(*memPtr, 0, numBytes));
        ERR_CHECK(hipDeviceSynchronize());
      }
      return ERR_NONE;
#else
      if (IsCpuMemType(memType)) numa_set_preferred(-1);
      return {ERR_FATAL, "Unable to allocate sharable memory if not compiled with pod communication support"};
#endif
    } else {
      if (actualBytes != NULL) *actualBytes = numBytes;
    }

    if (IsCpuMemType(memType)) {

      // Allocate host-pinned memory (should respect NUMA mem policy)
      int flags = 0;
#if !defined (__NVCC__)
      flags |= hipHostMallocNumaUser;
#endif
      if (memType == MEM_CPU || memType == MEM_CPU_CLOSEST) {
        ERR_CHECK(hipHostMalloc((void **)memPtr, numBytes, flags));
      } else if (memType == MEM_CPU_COHERENT) {
#if defined (__NVCC__)
        return {ERR_FATAL, "Coherent pinned-CPU memory not supported on NVIDIA platform"};
#else
        ERR_CHECK(hipHostMalloc((void **)memPtr, numBytes, flags | hipHostMallocCoherent));
#endif
      } else if (memType == MEM_CPU_NONCOHERENT) {
#if defined (__NVCC__)
        return {ERR_FATAL, "Non-coherent pinned-CPU memory not supported on NVIDIA platform"};
#else
        ERR_CHECK(hipHostMalloc((void **)memPtr, numBytes, flags | hipHostMallocNonCoherent));
#endif
      } else if (memType == MEM_CPU_UNCACHED) {
#if defined (__NVCC__)
        return {ERR_FATAL, "Coherent CPU memory not supported on NVIDIA platform"};
#else
#if HIP_VERSION_MAJOR >= 7
        ERR_CHECK(hipHostMalloc((void **)memPtr, numBytes, flags | hipHostMallocUncached));
#else
        return {ERR_FATAL, "Uncached pinned-CPU memory requires ROCm 7.0"};
#endif
#endif
      } else if (memType == MEM_CPU_UNPINNED) {
        *memPtr = numa_alloc_onnode(numBytes, deviceIdx);
      }

      // Check that the allocated pages are actually on the correct NUMA node
      memset(*memPtr, 0, numBytes);
      ERR_CHECK(CheckPages((char*)*memPtr, numBytes, deviceIdx));

      // Reset to default numa mem policy
      numa_set_preferred(-1);
    } else if (IsGpuMemType(memType)) {

      if (memType == MEM_GPU) {
        // Allocate GPU memory on appropriate device
        ERR_CHECK(hipMalloc((void**)memPtr, numBytes));
      } else if (memType == MEM_GPU_FINE) {
#if defined (__NVCC__)
        return {ERR_FATAL, "Fine-grained GPU memory not supported on NVIDIA platform"};
#else
        int flag = hipDeviceMallocFinegrained;
        ERR_CHECK(hipExtMallocWithFlags((void**)memPtr, numBytes, flag));
#endif
      } else if (memType == MEM_GPU_UNCACHED) {
#if defined (__NVCC__)
        return {ERR_FATAL, "Uncached GPU memory not supported on NVIDIA platform"};
#else
        int flag = hipDeviceMallocUncached;
        ERR_CHECK(hipExtMallocWithFlags((void**)memPtr, numBytes, flag));
#endif
      } else if (memType == MEM_MANAGED) {
        ERR_CHECK(hipMallocManaged((void**)memPtr, numBytes));
      }

      // Clear the memory
      ERR_CHECK(hipMemset(*memPtr, 0, numBytes));
      ERR_CHECK(hipDeviceSynchronize());
    } else {
      return {ERR_FATAL, "Unsupported memory type (%d)", memType};
    }
    return ERR_NONE;
  }

  // Deallocate memory
  static ErrResult DeallocateMemory(MemType memType, void *memPtr, size_t const bytes,
                                    hipMemGenericAllocationHandle_t* memHandle = nullptr)
  {
    // Avoid deallocating nullptr
    if (memPtr == nullptr)
      return {ERR_FATAL, "Attempted to free null pointer for %lu bytes", bytes};

    if (memHandle == nullptr || *memHandle == NULL) {
      switch (memType) {
      case MEM_CPU: case MEM_CPU_CLOSEST: case MEM_CPU_COHERENT: case MEM_CPU_NONCOHERENT: case MEM_CPU_UNCACHED:
      {
        ERR_CHECK(hipHostFree(memPtr));
        break;
      }
      case MEM_CPU_UNPINNED:
      {
        numa_free(memPtr, bytes);
        break;
      }
      case MEM_GPU : case MEM_GPU_FINE: case MEM_GPU_UNCACHED: case MEM_MANAGED:
      {
        ERR_CHECK(hipFree(memPtr));
        break;
      }
      default:
        return {ERR_FATAL, "Attempting to deallocate unrecognized memory type (%d)", memType};
      }
    } else {
#ifdef POD_COMM_ENABLED
      // Unmap the backing memory of the given virtual address
      ERR_CHECK(hipMemUnmap((gpu_device_ptr)memPtr, bytes));
      // Release the backing memory via its handle
      ERR_CHECK(hipMemRelease(*memHandle));
      // Free virtual address range reservation
      ERR_CHECK(hipMemAddressFree((gpu_device_ptr)memPtr, bytes));
#else
      return {ERR_FATAL, "Unable to deallocate sharable memory if not compiled with pod communication support"};
#endif
    }
    return ERR_NONE;
  }

#if defined(__NVCC__)
  static bool CheckDmabufSupport()
  {
    return false;
  }
  static ErrResult ExportDmabuf(void* gpuPtr, size_t numBytes, int& dmabufFd, uint64_t& dmabufOffset)
  {
    return {ERR_FATAL, "DMA-BUF export not yet supported on NVIDIA platform"};
  }
#else
  hsa_status_t (*pfn_hsa_amd_portable_export_dmabuf)(const void*, size_t, int*, uint64_t*);
  // Check kernel configuration for required DMA-BUF support
  // Returns true if kernel supports CONFIG_DMABUF_MOVE_NOTIFY and CONFIG_PCI_P2PDMA
  static bool CheckDmabufSupport()
  {
    static int support = -1;  // -1: not checked, 0: disabled, 1: enabled

    if (support != -1) {
      return support;
    }
    // Check hsa_amd_portable_export_dmabuf and ibv_reg_dmabuf_mr symbols are available
    // rocr and hsa_ext_amd header is always mandatory, so no need to check for them
    pfn_hsa_amd_portable_export_dmabuf =
        (hsa_status_t (*)(const void*, size_t, int*, uint64_t*))dlsym(RTLD_DEFAULT, "hsa_amd_portable_export_dmabuf");
    if (pfn_hsa_amd_portable_export_dmabuf == nullptr || !IsIbvDmabufPresent()) {
      support = 0;
      return support;
    }

    struct utsname utsname;
    FILE* fp = NULL;
    char kernel_opt1[28] = "CONFIG_DMABUF_MOVE_NOTIFY=y";
    char kernel_opt2[20] = "CONFIG_PCI_P2PDMA=y";
    char kernel_conf_file[128];
    char buf[256];
    int found_opt1 = 0;
    int found_opt2 = 0;
    int dmaBufSupport = 0;  // Start as disabled, enable only when both options found

    // Check for kernel name exists
    if (uname(&utsname) == -1) {
      if (System::Get().IsVerbose()) {
        printf("[WARN] Could not get kernel name\n");
      }
      support = 0;
      return 0;
    }

    // Format and check kernel conf file location
    const char* possiblePaths[] = {
      "/proc/config.gz",
      "/boot/config-%s",
      "/usr/src/linux-%s/.config",
      "/usr/src/linux/.config",
      "/usr/lib/modules/%s/config",
      "/usr/lib/ostree-boot/config-%s",
      "/usr/lib/kernel/config-%s",
      "/usr/src/linux-headers-%s/.config",
      "/lib/modules/%s/build/.config",
    };

    // Check if zcat is available in the system
    int has_zcat = (system("which zcat > /dev/null 2>&1") == 0);

    for (const auto& path : possiblePaths) {
      // Reset flags for each file
      found_opt1 = 0;
      found_opt2 = 0;

      snprintf(kernel_conf_file, sizeof(kernel_conf_file), path, utsname.release);

      // Special handling for /proc/config.gz
      if (strstr(path, "/proc/config.gz") != NULL) {
        // Skip .gz files if zcat is not available
        if (!has_zcat) {
          if (System::Get().IsVerbose()) {
            printf("[WARN] zcat not available, skipping %s\n", kernel_conf_file);
          }
          continue;
        }
        fp = popen("zcat /proc/config.gz 2>/dev/null", "r");
      } else {
        fp = fopen(kernel_conf_file, "r");
      }

      if (fp != NULL) {
        // Look for kernel_opt1 and kernel_opt2 in the conf file
        while (fgets(buf, sizeof(buf), fp) != NULL) {
          if (strstr(buf, kernel_opt1) != NULL) {
            found_opt1 = 1;
            if (System::Get().IsVerbose()) {
              printf("[INFO] %s in %s\n", kernel_opt1, kernel_conf_file);
            }
          }
          if (strstr(buf, kernel_opt2) != NULL) {
            found_opt2 = 1;
            if (System::Get().IsVerbose()) {
              printf("[INFO] %s in %s\n", kernel_opt2, kernel_conf_file);
            }
          }
        }

        // Close file handle
        if (strstr(path, "/proc/config.gz") != NULL) {
          pclose(fp);
        } else {
          fclose(fp);
        }

        // Check if both options were found
        if (found_opt1 && found_opt2) {
          dmaBufSupport = 1;
          if (System::Get().IsVerbose()) {
            printf("[INFO] DMA_BUF Support Enabled\n");
          }
          break;  // Found both options, exit loop
        } else {
          // File found but missing required options, continue to next file
          if (System::Get().IsVerbose()) {
            printf("[WARN] CONFIG_DMABUF_MOVE_NOTIFY and/or CONFIG_PCI_P2PDMA not found in %s, trying next location\n", kernel_conf_file);
          }
        }
      }
    }

    // If DMA-BUF support not enabled, log warning
    if (!dmaBufSupport) {
      if (System::Get().IsVerbose()) {
        printf("[WARN] DMA_BUF_SUPPORT Failed: kernel config not found or missing required options\n");
        printf("[WARN] Falling back to standard ibv_reg_mr\n");
      }
    }

    support = dmaBufSupport;
    return support;
  }

  // Export GPU memory as DMA-BUF for RDMA operations
  static ErrResult ExportDmabuf(void* gpuPtr, size_t numBytes, int& dmabufFd, uint64_t& dmabufOffset)
  {
    // Round down to host page alignment (4KB typically)
    const uint64_t HOST_PAGE_SIZE = 1ULL << 12;
    void* alignedPtr = (void*)((uint64_t)gpuPtr & ~(HOST_PAGE_SIZE - 1));
    uint64_t offset = (uint64_t)gpuPtr - (uint64_t)alignedPtr;

    // Adjust size to account for alignment
    size_t alignedSize = numBytes + offset;
    alignedSize = (alignedSize + HOST_PAGE_SIZE - 1) & ~(HOST_PAGE_SIZE - 1);

    // Export the aligned GPU buffer as DMA-BUF
    uint64_t exportOffset = 0;
    hsa_status_t status = pfn_hsa_amd_portable_export_dmabuf(alignedPtr, alignedSize, &dmabufFd, &exportOffset);

    if (status != HSA_STATUS_SUCCESS) {
      return {ERR_FATAL, "Failed to export DMA-BUF: hsa_amd_portable_export_dmabuf returned %d", status};
    }

    // The offset is the sum of export offset and alignment offset
    dmabufOffset = exportOffset + offset;

    return ERR_NONE;
  }
#endif

// Setup validation-related functions
//========================================================================================
  // This function resolves executors that may be indexed by "nearest"
  static ErrResult GetActualExecutor(ExeDevice     const& origExeDevice,
                                     ExeDevice&           actualExeDevice,
                                     int                  rankOverride = -1)
  {
    // By default, nothing needs to change
    actualExeDevice = origExeDevice;

    // Check that executor rank is valid
    int exeRank = (rankOverride == -1 ? origExeDevice.exeRank : rankOverride);
    if (exeRank < 0 || exeRank >= GetNumRanks())
      return {ERR_FATAL, "Rank index must be between 0 and %d (instead of %d)", GetNumRanks() - 1, exeRank};

    // When using NIC_NEAREST, remap to the closest NIC to the GPU
    if (origExeDevice.exeType == EXE_NIC_NEAREST) {
      actualExeDevice.exeType  = EXE_NIC;
      actualExeDevice.exeRank  = exeRank;
      std::vector<int> nicIndices;
      GetClosestNicsToGpu(nicIndices, origExeDevice.exeIndex, exeRank);
      if (origExeDevice.exeSlot < 0 || origExeDevice.exeSlot >= nicIndices.size()) {
        return {ERR_FATAL, "Rank %d GPU %d closest NIC slot %d is invalid (%lu slots detected)",
          exeRank, origExeDevice.exeIndex, origExeDevice.exeSlot, nicIndices.size()};
      }
      actualExeDevice.exeIndex = nicIndices[actualExeDevice.exeSlot];
      actualExeDevice.exeSlot = 0;
    }
    return ERR_NONE;
  }

  // Validate that MemDevice exists
  static ErrResult CheckMemDevice(MemDevice const& memDevice)
  {
    if (memDevice.memType == MEM_NULL)
      return ERR_NONE;

    if (memDevice.memRank < 0 || memDevice.memRank >= GetNumRanks()) {
      return {ERR_FATAL,
              "Rank index must be between 0 and %d (instead of %d)", GetNumRanks() - 1, memDevice.memRank};
    }

    if (IsCpuMemType(memDevice.memType) && memDevice.memType != MEM_CPU_CLOSEST) {
      int numCpus = GetNumExecutors(EXE_CPU, memDevice.memRank);
      if (memDevice.memIndex < 0 || memDevice.memIndex >= numCpus)
        return {ERR_FATAL,
                "CPU index must be between 0 and %d (instead of %d) on rank %d", numCpus - 1, memDevice.memIndex, memDevice.memRank};

      if (GetRank() == memDevice.memRank && !numa_bitmask_isbitset(numa_get_mems_allowed(), memDevice.memIndex)) {
        return {ERR_FATAL, "CPU %d on rank %d cannot allocate memory due to process memory policy/cpuset",
          memDevice.memIndex, memDevice.memRank};
      }
      return ERR_NONE;
    }

    if (IsGpuMemType(memDevice.memType) || memDevice.memType == MEM_CPU_CLOSEST) {
      int numGpus = GetNumExecutors(EXE_GPU_GFX, memDevice.memRank);
      if (memDevice.memIndex < 0 || memDevice.memIndex >= numGpus)
        return {ERR_FATAL,
                "GPU index must be between 0 and %d (instead of %d) on rank %d", numGpus - 1, memDevice.memIndex, memDevice.memRank};
      if (memDevice.memType == MEM_CPU_CLOSEST) {
        int actualNumaIdx = GetClosestCpuNumaToGpu(memDevice.memIndex, memDevice.memRank);
        if (actualNumaIdx == -1) {
          return {ERR_FATAL, "Unable to determine closest NUMA node for GPU %d on rank %d", memDevice.memIndex, memDevice.memRank};
        }
        if (GetRank() == memDevice.memRank && !numa_bitmask_isbitset(numa_get_mems_allowed(), actualNumaIdx))
          return {ERR_FATAL, "CPU %d on rank %d cannot allocate memory due to process memory policy/cpuset",
            memDevice.memIndex, memDevice.memRank};
      }
      return ERR_NONE;
    }
    return {ERR_FATAL, "Unsupported memory type (%d)", memDevice.memType};
  }

  static void CheckMultiNodeConfigConsistency(ConfigOptions const& cfg,
                                              std::vector<ErrResult>& errors)
  {
    if (GetCommMode() == COMM_NONE) return;
    if (System::Get().IsVerbose()) {
      System::Get().Log("[INFO] Rank %d checking config consistency\n", GetRank());
    }

    // To check consistency, compare against rank 0
    int root = 0;

    #define ADD_ERROR(STR) errors.push_back({ERR_FATAL, STR " must be consistent across all ranks"})

    // Compare general options
    {
      GeneralOptions general = cfg.general;
      System::Get().Broadcast(root, sizeof(general), &general);
      if (general.numIterations      != cfg.general.numIterations)      ADD_ERROR("cfg.general.numIterations");
      if (general.numSubIterations   != cfg.general.numSubIterations)   ADD_ERROR("cfg.general.numSubIterations");
      if (general.numWarmups         != cfg.general.numWarmups)         ADD_ERROR("cfg.general.numWarmups");
      if (general.recordPerIteration != cfg.general.recordPerIteration) ADD_ERROR("cfg.general.recordPerIteration");
      if (general.useHipEvents       != cfg.general.useHipEvents)       ADD_ERROR("cfg.general.useHipEvents");
      if (general.useInteractive     != cfg.general.useInteractive)     ADD_ERROR("cfg.general.useInteractive");
      if (general.useMultiStream     != cfg.general.useMultiStream)     ADD_ERROR("cfg.general.useMultiStream");
    }

    // Compare data options
    {
      DataOptions data = cfg.data;
      // Null out vector members before sizeof-broadcast: vectors carry heap pointers that are
      // invalid on other ranks; freeing a remote pointer on scope exit causes a segfault
      // These fields are permitted to differ across ranks and are not compared below
      decltype(data.fillPattern)().swap(data.fillPattern);
      decltype(data.fillCompress)().swap(data.fillCompress);
      System::Get().Broadcast(root, sizeof(data), &data);

      if (data.alwaysValidate != cfg.data.alwaysValidate) ADD_ERROR("cfg.data.alwaysValidate");
      if (data.blockBytes != cfg.data.blockBytes) ADD_ERROR("cfg.data.blockBytes");
      if (data.byteOffset != cfg.data.byteOffset) ADD_ERROR("cfg.data.byteOffset");

      size_t fillPatternSize = cfg.data.fillPattern.size();
      System::Get().Broadcast(root, sizeof(fillPatternSize), &fillPatternSize);
      if (fillPatternSize != cfg.data.fillPattern.size()) {
        ADD_ERROR("cfg.data.fillPattern");
      } else if (fillPatternSize > 0) {
        auto fillPatternTemp = cfg.data.fillPattern;
        System::Get().BroadcastVector(0, fillPatternTemp);
        for (size_t i = 0; i < fillPatternSize; i++) {
          if (fillPatternTemp[i] != cfg.data.fillPattern[i]) {
            ADD_ERROR("cfg.data.fillPattern");
            break;
          }
        }
      }

      size_t fillCompressSize = cfg.data.fillCompress.size();
      System::Get().Broadcast(root, sizeof(fillCompressSize), &fillCompressSize);
      if (fillCompressSize != cfg.data.fillCompress.size()) {
        ADD_ERROR("cfg.data.fillCompress");
      } else if (fillCompressSize > 0) {
        auto fillCompressTemp = cfg.data.fillCompress;
        System::Get().BroadcastVector(0, fillCompressTemp);
        for (size_t i = 0; i < fillCompressSize; i++) {
          if (fillCompressTemp[i] != cfg.data.fillCompress[i]) {
            ADD_ERROR("cfg.data.fillCompress");
            break;
          }
        }
      }

      // data.validateDirect is permitted to be different across ranks
      // data.validateSource is permitted to be different across ranks
    }

    // Compare GFX Executor options
    {
      GfxOptions gfx = cfg.gfx;
      // Null out vector members before sizeof broadcast
      decltype(gfx.cuMask)().swap(gfx.cuMask);
      decltype(gfx.prefXccTable)().swap(gfx.prefXccTable);
      System::Get().Broadcast(root, sizeof(gfx), &gfx);
      if (gfx.blockOrder     != cfg.gfx.blockOrder)     ADD_ERROR("cfg.gfx.blockOrder");
      if (gfx.blockSize      != cfg.gfx.blockSize)      ADD_ERROR("cfg.gfx.blockSize");
      // gfx.cuMask       is permitted to be different across ranks
      if (gfx.gfxKernel      != cfg.gfx.gfxKernel)      ADD_ERROR("cfg.gfx.gfxKernel");
      // gfx.perfXccTable is permitted to be different across ranks
      if (gfx.seType         != cfg.gfx.seType)         ADD_ERROR("cfg.gfx.seType");
      if (gfx.temporalMode   != cfg.gfx.temporalMode)   ADD_ERROR("cfg.gfx.temporalMode");
      if (gfx.unrollFactor   != cfg.gfx.unrollFactor)   ADD_ERROR("cfg.gfx.unrollFactor)");
      if (gfx.useSingleTeam  != cfg.gfx.useSingleTeam)  ADD_ERROR("cfg.gfx.useSingleTeam");
      if (gfx.waveOrder      != cfg.gfx.waveOrder)      ADD_ERROR("cfg.gfx.waveOrder");
      if (gfx.wordSize       != cfg.gfx.wordSize)       ADD_ERROR("cfg.gfx.wordSize");
    }

    // Compare DMA Executor options
    {
      DmaOptions dma = cfg.dma;
      System::Get().Broadcast(root, sizeof(dma), &dma);
      if (dma.useHsaCopy   != cfg.dma.useHsaCopy)   ADD_ERROR("cfg.dma.useHsaCopy");
    }

    // Compare NIC options
    {
      NicOptions nic = cfg.nic;
      System::Get().Broadcast(root, sizeof(nic), &nic);
      if (nic.chunkBytes      != cfg.nic.chunkBytes)      ADD_ERROR("cfg.nic.chunkBytes");
      if (nic.cqPollBatch     != cfg.nic.cqPollBatch)     ADD_ERROR("cfg.nic.cqPollBatch");
      // nic.ibGidIndex  is permitted to be different across ranks
      // nic.ibPort      is permitted to be different across ranks
      if (nic.ipAddressFamily != cfg.nic.ipAddressFamily) ADD_ERROR("cfg.nic.ipAddressFamily");
      if (nic.maxRecvWorkReq  != cfg.nic.maxRecvWorkReq)  ADD_ERROR("cfg.nic.maxRecvWorkReq");
      if (nic.maxSendWorkReq  != cfg.nic.maxSendWorkReq)  ADD_ERROR("cfg.nic.maxSendWorkReq");
      // nic.queueSize   is permitted to be different across ranks
      if (nic.roceVersion     != cfg.nic.roceVersion)     ADD_ERROR("cfg.nic.roceVersion");
      if (nic.serviceLevel    != cfg.nic.serviceLevel)    ADD_ERROR("cfg.nic.serviceLevel");
      if (nic.trafficClass    != cfg.nic.trafficClass)    ADD_ERROR("cfg.nic.trafficClass");
      if (nic.useRelaxedOrder != cfg.nic.useRelaxedOrder) ADD_ERROR("cfg.nic.useRelaxedOrder");
      if (nic.useNuma         != cfg.nic.useNuma)         ADD_ERROR("cfg.nic.useNuma");
    }

    // Compare TDM options
    {
      TdmOptions tdm = cfg.tdm;
      System::Get().Broadcast(root, sizeof(tdm), &tdm);
      if (tdm.blockOrder     != cfg.tdm.blockOrder)     ADD_ERROR("cfg.tdm.blockOrder");
      if (tdm.blockSize      != cfg.tdm.blockSize)      ADD_ERROR("cfg.tdm.blockSize");
      if (tdm.tdmKernel      != cfg.tdm.tdmKernel)      ADD_ERROR("cfg.tdm.tdmKernel");
      if (tdm.ldsBytes       != cfg.tdm.ldsBytes)       ADD_ERROR("cfg.tdm.ldsBytes");
    }
    #undef ADD_ERROR
  }

  // Forward declaration
  int GetGpuKernelUnrollIdx(int unroll);

  // Validate configuration options - return trues if and only if an fatal error is detected
  static bool ConfigOptionsHaveErrors(ConfigOptions const&    cfg,
                                      std::vector<ErrResult>& errors)
  {
    // Check general options
    if (cfg.general.numWarmups < 0)
      errors.push_back({ERR_FATAL, "[general.numWarmups] must be a non-negative number"});

    // Check that config options are consistent (where necessary) across all ranks
    CheckMultiNodeConfigConsistency(cfg, errors);

    // Check data options
    if (cfg.data.blockBytes == 0 || cfg.data.blockBytes % 4)
      errors.push_back({ERR_FATAL, "[data.blockBytes] must be positive multiple of %lu", sizeof(float)});
    if (cfg.data.byteOffset < 0 || cfg.data.byteOffset % sizeof(float))
      errors.push_back({ERR_FATAL, "[data.byteOffset] must be positive multiple of %lu", sizeof(float)});
    if (cfg.data.fillCompress.size() > 0 && cfg.data.fillPattern.size() > 0)
      errors.push_back({ERR_WARN, "[data.fillCompress] will override [data.fillPattern] when both are specified"});
    if (cfg.data.fillCompress.size() > 0) {
      int sum = 0;
      for (int bin : cfg.data.fillCompress)
        sum += bin;
      if (sum != 100) {
        errors.push_back({ERR_FATAL, "[data.fillCompress] values must add up to 100"});
      }
    }
    if (cfg.data.fillCompress.size() > 5) {
      errors.push_back({ERR_FATAL, "[data.fillCompress] may only have up to 5 values"});
    }

    // Check GFX options
    if (cfg.gfx.blockOrder < 0 || cfg.gfx.blockOrder > 2)
      errors.push_back({ERR_FATAL,
          "[gfx.blockOrder] must be 0 for sequential, 1 for interleaved, or 2 for random"});

    if (cfg.general.useMultiStream && cfg.gfx.blockOrder > 0)
      errors.push_back({ERR_WARN, "[gfx.blockOrder] will be ignored when running in multi-stream mode"});

    if (cfg.gfx.blockSize < 0 || cfg.gfx.blockSize % 64 || cfg.gfx.blockSize > MAX_BLOCKSIZE)
      errors.push_back({ERR_FATAL,
                        "[gfx.blockSize] must be positive multiple of 64 less than or equal to %d",
                        MAX_BLOCKSIZE});

    if (cfg.gfx.temporalMode < 0 || cfg.gfx.temporalMode > 3)
      errors.push_back({ERR_FATAL,
                        "[gfx.temporalMode] must be non-negative and less than or equal to 3"});

#if defined(__NVCC__)
    if (cfg.gfx.temporalMode > 0)
      errors.push_back({ERR_FATAL,
          "[gfx.temporalMode] is not supported on NVIDIA hardware"});
#endif

    if (GetGpuKernelUnrollIdx(cfg.gfx.unrollFactor) == -1) {
      errors.push_back({ERR_FATAL,
          "[gfx.unrollFactor] unroll factor of %d is unsupported", cfg.gfx.unrollFactor});
    }
    if (cfg.gfx.waveOrder < 0 || cfg.gfx.waveOrder >= 6)
      errors.push_back({ERR_FATAL,
                        "[gfx.waveOrder] must be non-negative and less than 6"});

    if (!(cfg.gfx.wordSize == 1 || cfg.gfx.wordSize == 2 || cfg.gfx.wordSize == 4))
      errors.push_back({ERR_FATAL, "[gfx.wordSize] must be either 1, 2 or 4"});

    if (cfg.gfx.gfxKernel < -1 || cfg.gfx.gfxKernel >= NUM_GFX_KERNELS)
      errors.push_back(
        {ERR_FATAL, "[gfx.gfxKernel] must be -1 for auto, or less than %d", NUM_GFX_KERNELS});

    int numGpus = GetNumExecutors(EXE_GPU_GFX);
    int numXccs = GetNumExecutorSubIndices({EXE_GPU_GFX, 0});
    vector<vector<int>> const& table = cfg.gfx.prefXccTable;

    if (!table.empty()) {
      if (table.size() != numGpus) {
        errors.push_back({ERR_FATAL, "[gfx.prefXccTable] must be have size %dx%d", numGpus, numGpus});
      } else {
        for (int i = 0; i < table.size(); i++) {
          if (table[i].size() != numGpus) {
            errors.push_back({ERR_FATAL, "[gfx.prefXccTable] must be have size %dx%d", numGpus, numGpus});
            break;
          } else {
            for (auto x : table[i]) {
              if (x < 0 || x >= numXccs) {
                errors.push_back({ERR_FATAL, "[gfx.prefXccTable] must contain values between 0 and %d",
                    numXccs - 1});
                break;
              }
            }
          }
        }
      }
    }

    // Check TDM options (TDM-capable hardware: gfx1250 on AMD, sm_90+ on NVIDIA)
    if (cfg.tdm.blockSize <= 0 || cfg.tdm.blockSize % 32 || cfg.tdm.blockSize > MAX_BLOCKSIZE)
      errors.push_back({ERR_FATAL,
                        "[tdm.blockSize] must be a positive multiple of 32 less than or equal to %d",
                        MAX_BLOCKSIZE});

    if (cfg.tdm.tdmKernel < -1 || cfg.tdm.tdmKernel >= NUM_TDM_KERNELS)
      errors.push_back(
        {ERR_FATAL, "[tdm.tdmKernel] must be -1 for auto, or less than %d", NUM_TDM_KERNELS});

    if (cfg.tdm.ldsBytes < 0)
      errors.push_back({ERR_FATAL, "[tdm.ldsBytes] must be positive or 0"});
    else {
      int const numGpus = GetNumExecutors(EXE_GPU_TDM);
      for (int i = 0; i < numGpus; i++) {
        int deviceMax = 0;
        if (hipDeviceGetAttribute(&deviceMax, hipDeviceAttributeMaxSharedMemoryPerBlock, i) == hipSuccess) {
          if (cfg.tdm.ldsBytes > deviceMax) {
            errors.push_back({ERR_FATAL,
                "[tdm.ldsBytes] (%d) exceeds device max shared memory per block (%d)",
                cfg.tdm.ldsBytes, deviceMax});
          } else if (deviceMax == 0) {
            errors.push_back({ERR_FATAL,
                "TDM Executor requires shared memory but GPU %d reports none available\n", i});
          }
        } else {
          errors.push_back({ERR_FATAL,
              "Unable to query max amount of shared memory per block on GPU %%d\n", i});
        }
      }
    }

    // Check NIC options
    if (IsIbvSymbolsReady()) {
      if (cfg.nic.chunkBytes == 0 || (cfg.nic.chunkBytes % 4 != 0)) {
        errors.push_back({ERR_FATAL, "[nic.chunkBytes] must be a non-negative multiple of 4"});
      }
      if (cfg.nic.cqPollBatch <= 0) {
        errors.push_back({ERR_FATAL, "[nic.cqPollBatch] must be positive"});
      }
    }

    // NVIDIA specific
#if defined(__NVCC__)
    if (cfg.data.validateDirect)
      errors.push_back({ERR_FATAL, "[data.validateDirect] is not supported on NVIDIA hardware"});
#else
    // AMD specific
    // Check for largeBar enablement on GPUs
    for (int i = 0; i < numGpus; i++) {
      int isLargeBar = 0;
      hipError_t err = hipDeviceGetAttribute(&isLargeBar, hipDeviceAttributeIsLargeBar, i);
      if (err != hipSuccess) {
        errors.push_back({ERR_FATAL, "Unable to query if GPU %d has largeBAR enabled", i});
      } else if (!isLargeBar) {
        errors.push_back({ERR_WARN,
                          "Large BAR is not enabled for GPU %d in BIOS. "
                          "Large BAR is required to enable multi-gpu data access", i});
      }
    }
#endif

    // Check for fatal errors
    for (auto const& err : errors)
      if (err.errType == ERR_FATAL) return true;
    return false;
  }

  static void CheckMultiNodeTransferConsistency(std::vector<Transfer> const& transfers,
                                                std::vector<ErrResult>& errors)
  {
    if (GetCommMode() == COMM_NONE) return;

    if (System::Get().IsVerbose()) {
      System::Get().Log("[INFO] Rank %d checking transfers consistency\n", GetRank());
    }

    // To check consistency, compare against rank 0
    int root = 0;

    #define ADD_ERROR(STR)         \
    do {                          \
      isInconsistent = true;                                            \
      if (System::Get().IsVerbose())                                    \
        errors.push_back({ERR_FATAL, STR " must be the same for Transfer %d on all ranks", i}); \
    } while(0)

    size_t numTransfers = transfers.size();
    System::Get().Broadcast(root, sizeof(numTransfers), &numTransfers);
    if (numTransfers != transfers.size()) {
      errors.push_back({ERR_FATAL, "The number of Transfers to run must be consistent across ranks"});
    }

    bool isInconsistent = false;
    for (size_t i = 0; i < numTransfers; i++) {
      Transfer t = transfers[i];

      System::Get().Broadcast(root, sizeof(t.numBytes), &t.numBytes);
      System::Get().BroadcastVector(root, t.srcs);
      System::Get().BroadcastVector(root, t.dsts);
      System::Get().Broadcast(root, sizeof(t.exeDevice),   &t.exeDevice);
      System::Get().Broadcast(root, sizeof(t.exeSubIndex), &t.exeSubIndex);
      System::Get().Broadcast(root, sizeof(t.exeSubSlot),  &t.exeSubSlot);
      System::Get().Broadcast(root, sizeof(t.numSubExecs), &t.numSubExecs);

      if (t.numBytes    != transfers[i].numBytes)    ADD_ERROR("numBytes");
      if (t.srcs        != transfers[i].srcs)        ADD_ERROR("Source memory locations");
      if (t.dsts        != transfers[i].dsts)        ADD_ERROR("Destination memory locations");
      if (t.exeDevice < transfers[i].exeDevice ||
          transfers[i].exeDevice < t.exeDevice)      ADD_ERROR("Executor device");
      if (t.exeSubIndex != transfers[i].exeSubIndex) ADD_ERROR("Executor subindex");
      if (t.exeSubSlot  != transfers[i].exeSubSlot)  ADD_ERROR("Executor dst slot");
      if (t.numSubExecs != transfers[i].numSubExecs) ADD_ERROR("Num SubExecutors");
    }

    if (isInconsistent && !System::Get().IsVerbose()) {
      errors.push_back({ERR_FATAL, "Transfers to execute must be identical across all ranks"});
    }

    #undef ADD_ERROR
  }

  // Returns true if the given Transfer requires pod communication
  static bool IsPodTransfer(Transfer const& t)
  {
    if (IsCpuExeType(t.exeDevice.exeType) || IsGpuExeType(t.exeDevice.exeType)) {
      for (auto const& src : t.srcs)
        if (src.memRank != t.exeDevice.exeRank) return true;
      for (auto const& dst : t.dsts)
        if (dst.memRank != t.exeDevice.exeRank) return true;
    }
    return false;
  }

  // Validate Transfers to execute - returns true if and only if fatal error detected
  static bool TransfersHaveErrors(ConfigOptions         const& cfg,
                                  std::vector<Transfer> const& transfers,
                                  std::vector<ErrResult>&      errors)
  {
    std::set<ExeDevice>      executors;
    std::map<ExeDevice, int> transferCount;
    std::map<ExeDevice, int> useSubIndexCount;
    std::map<ExeDevice, int> totalSubExecs;

    // Check that the set of requested transfers is consistent across all ranks
    CheckMultiNodeTransferConsistency(transfers, errors);

    // Per-Transfer checks
    bool hasFatalError = false;
    for (size_t i = 0; i < transfers.size(); i++) {
      Transfer const& t = transfers[i];

      if (t.numBytes == 0) {
        errors.push_back({ERR_FATAL, "Transfer %d: Cannot perform 0-byte transfers", i});
        break;
      }

      if (t.numBytes % 4) {
        errors.push_back({ERR_FATAL, "Transfer %d: numBytes (%lu) must be a multiple of 4\n", i, t.numBytes});
        break;
      }

      // Each subexecutor is assigned a multiple of cfg.data.blockBytes, however this may
      // mean that some subexecutors might not have any work assigned to them if the amount to
      // transfer is small

      if (t.exeDevice.exeType != EXE_GPU_DMA) { // GPU DMA-Executor ignores subexecutors
        size_t const N               = t.numBytes / sizeof(float);
        int    const targetMultiple  = cfg.data.blockBytes / sizeof(float);
        int    const maxSubExecToUse = std::min((size_t)(N + targetMultiple - 1) / targetMultiple,
                                                (size_t)t.numSubExecs);

        if (maxSubExecToUse < t.numSubExecs)
          errors.push_back({ERR_WARN,
                            "Transfer %d data size is too small - will only use %d of %d subexecutors due to blockBytes of %d",
                            i, maxSubExecToUse, t.numSubExecs, cfg.data.blockBytes});
      }

      // Check sources and destinations
      if (t.srcs.empty() && t.dsts.empty()) {
        errors.push_back({ERR_FATAL, "Transfer %d: Must have at least one source or destination", i});
        break;
      }

      for (int j = 0; j < t.srcs.size(); j++) {
        ErrResult err = CheckMemDevice(t.srcs[j]);
        if (err.errType != ERR_NONE) {
          errors.push_back({ERR_FATAL, "Transfer %d: SRC %d: %s", i, j, err.errMsg.c_str()});
          hasFatalError = true;
          break;
        }
      }
      if (hasFatalError) break;

      for (int j = 0; j < t.dsts.size(); j++) {
        ErrResult err = CheckMemDevice(t.dsts[j]);
        if (err.errType != ERR_NONE) {
          errors.push_back({ERR_FATAL, "Transfer %d: DST %d: %s", i, j, err.errMsg.c_str()});
          hasFatalError = true;
          break;
        }
      }
      if (hasFatalError) break;

      // Check executor rank
      if (t.exeDevice.exeRank < 0 || t.exeDevice.exeRank >= GetNumRanks()) {
        errors.push_back({ERR_FATAL,
            "Rank index for executor must be between 0 and %d (instead of %d)", GetNumRanks() - 1, t.exeDevice.exeRank});
        break;
      }

      executors.insert(t.exeDevice);
      transferCount[t.exeDevice]++;
      int numExecutors = GetNumExecutors(t.exeDevice.exeType, t.exeDevice.exeRank);

      switch (t.exeDevice.exeType) {
      case EXE_CPU:
        if (t.exeDevice.exeIndex < 0 || t.exeDevice.exeIndex >= numExecutors) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: CPU index must be between 0 and %d (instead of %d) for rank %d",
                            i, numExecutors - 1, t.exeDevice.exeIndex, t.exeDevice.exeRank});
          hasFatalError = true;
        }
        break;
      case EXE_GPU_TDM:
        // The copy TDM kernel only supports a single SRC/DST; the reduce kernel supports
        // any number of inputs/outputs. When auto-selecting, allow the reduce cardinalities.
        if (cfg.tdm.tdmKernel == TDM_KERNEL_COPY && (t.srcs.size() != 1 || t.dsts.size() != 1)) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: GPU TDM copy kernel currently requires exactly 1 SRC and 1 DST", i});
          hasFatalError = true;
          break;
        }
        // The multi-SRC/DST sum-reduce kernel is only implemented on the AMD TDM
        // backend; disable it on the NVIDIA platform. This covers both explicitly
        // requesting the reduce kernel and auto-selecting it via reduce cardinalities.
        if (TDM_PLATFORM_NV &&
            (cfg.tdm.tdmKernel == TDM_KERNEL_REDUCE ||
             (cfg.tdm.tdmKernel == TDM_KERNEL_AUTO && (t.srcs.size() != 1 || t.dsts.size() != 1)))) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: GPU TDM reduce kernel (multi-SRC/DST) is not supported on the NVIDIA platform", i});
          hasFatalError = true;
          break;
        }
        if (t.exeDevice.exeIndex < 0 || t.exeDevice.exeIndex >= numExecutors) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: GPU TDM kernel: device index must be between 0 and %d (instead of %d) for rank %d",
                            i, numExecutors - 1, t.exeDevice.exeIndex, t.exeDevice.exeRank});
          hasFatalError = true;
          break;
        }
        if (t.exeSubIndex != -1) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: GPU TDM executor does not support subindices", i});
          hasFatalError = true;
          break;
        }
        if (!tdm::IsTdmCopySupported(t.exeDevice.exeIndex)) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: GPU TDM kernel requires TDM-capable hardware (gfx1250 or NVIDIA sm_90+), but GPU %d is not supported",
                            i, t.exeDevice.exeIndex});
          hasFatalError = true;
        }
        break;
      case EXE_GPU_GFX:
        if (t.exeDevice.exeIndex < 0 || t.exeDevice.exeIndex >= numExecutors) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: GFX index must be between 0 and %d (instead of %d) for rank %d",
                            i, numExecutors - 1, t.exeDevice.exeIndex, t.exeDevice.exeRank});
          hasFatalError = true;
          break;
        } else {
          if (t.exeSubIndex != -1) {
#if defined(__NVCC__)
            errors.push_back({ERR_FATAL,
                              "Transfer %d: GFX executor subindex not supported on NVIDIA hardware", i});
            hasFatalError = true;
#else
            useSubIndexCount[t.exeDevice]++;
            int numSubIndices = GetNumExecutorSubIndices(t.exeDevice);
            if (t.exeSubIndex >= numSubIndices) {
              errors.push_back({ERR_FATAL,
                  "Transfer %d: GFX subIndex (XCC) must be between 0 and %d for rank %d", i, numSubIndices - 1, t.exeDevice.exeRank});
              hasFatalError = true;
            }
#endif
          }
        }
        break;
      case EXE_GPU_DMA:
        if (t.srcs.size() != 1) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: DMA executor must have exactly 1 source", i});
          hasFatalError = true;
          break;
        }
        if (t.dsts.size() < 1) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: DMA executor must have at least 1 destination", i});
          hasFatalError = true;
          break;
        }

        if (t.exeDevice.exeIndex < 0 || t.exeDevice.exeIndex >= numExecutors) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: DMA index must be between 0 and %d (instead of %d) for rank %d",
                            i, numExecutors - 1, t.exeDevice.exeIndex, t.exeDevice.exeRank});
          hasFatalError = true;
          break;
        }

        if (t.exeSubIndex != -1) {
#if defined(__NVCC__)
          errors.push_back({ERR_FATAL,
                            "Transfer %d: DMA executor subindex not supported on NVIDIA hardware", i});
          hasFatalError = true;
          break;
#else
          useSubIndexCount[t.exeDevice]++;
          int numSubIndices = GetNumExecutorSubIndices(t.exeDevice);
          if (t.exeSubIndex >= numSubIndices) {
            errors.push_back({ERR_FATAL,
                              "Transfer %d: DMA subIndex (engine) must be between 0 and %d",
                              i, numSubIndices - 1});
            hasFatalError = true;
            break;
          }

          // Check that engine Id exists between agents
          hsa_agent_t srcAgent, dstAgent;
          ErrResult err;
          err = System::Get().GetHsaAgent(t.srcs[0], srcAgent);
          if (err.errType != ERR_NONE) {
            errors.push_back(err);
            if (err.errType == ERR_FATAL) {
              hasFatalError = true;
              break;
            }

          }

          int numDsts = (int)t.dsts.size();
          for (int dstIdx = 0; dstIdx < numDsts; dstIdx++) {
            err = System::Get().GetHsaAgent(t.dsts[dstIdx], dstAgent);
            if (err.errType != ERR_NONE) {
              errors.push_back(err);
              if (err.errType == ERR_FATAL) {
                hasFatalError = true;
                break;
              }
            }

            // Skip check of engine Id mask for self copies
            if (srcAgent.handle != dstAgent.handle) {
              uint32_t engineIdMask = 0;
              err = hsa_amd_memory_copy_engine_status(dstAgent, srcAgent, &engineIdMask);
              if (err.errType != ERR_NONE) {
                errors.push_back(err);
                if (err.errType == ERR_FATAL) {
                  hasFatalError = true;
                  break;
                }
              }
              hsa_amd_sdma_engine_id_t sdmaEngineId = (hsa_amd_sdma_engine_id_t)(1U << t.exeSubIndex);
              if (!(sdmaEngineId & engineIdMask)) {
                errors.push_back({ERR_FATAL,
                    "Transfer %d: DMA %d.%d does not exist or cannot copy between src/dst",
                    i, t.exeDevice.exeIndex, t.exeSubIndex});
                hasFatalError = true;
                break;
              }
            }
          }
          if (hasFatalError) break;
#endif
        }

        if (!IsGpuMemType(t.srcs[0].memType) && !IsGpuMemType(t.dsts[0].memType)) {
          errors.push_back({ERR_WARN,
              "Transfer %d: No GPU memory for source or destination.  Copy might not execute on DMA %d",
              i, t.exeDevice.exeIndex});
        } else {
          // Currently HIP will use src agent if source memory is GPU, otherwise dst agent
          if (IsGpuMemType(t.srcs[0].memType)) {
            if (t.srcs[0].memIndex != t.exeDevice.exeIndex) {
              errors.push_back({ERR_WARN,
                  "Transfer %d: DMA executor may automatically switch to using the source memory device (%d) not (%d)",
                  i, t.srcs[0].memIndex, t.exeDevice.exeIndex});
            }
          } else if (t.dsts[0].memIndex != t.exeDevice.exeIndex) {
            errors.push_back({ERR_WARN,
                "Transfer %d: DMA executor may automatically switch to using the destination memory device (%d) not (%d)",
                i, t.dsts[0].memIndex, t.exeDevice.exeIndex});
          }
        }
        break;
      case EXE_GPU_BDMA:
#ifdef BMA_EXEC_ENABLED
        if (t.srcs.size() != 1) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: BMA executor must have exactly 1 source", i});
          hasFatalError = true;
          break;
        }
        if (t.dsts.size() < 1) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: BMA executor must have at least 1 destination", i});
          hasFatalError = true;
          break;
        }

        if (t.exeDevice.exeIndex < 0 || t.exeDevice.exeIndex >= numExecutors) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: BMA index must be between 0 and %d (instead of %d) for rank %d",
                            i, numExecutors - 1, t.exeDevice.exeIndex, t.exeDevice.exeRank});
          hasFatalError = true;
          break;
        }

        if (t.exeSubIndex != -1) {
          errors.push_back({ERR_FATAL,
              "Transfer %d: BMA executor does not support executor subindices (SDMA engine selection)", i});
          hasFatalError = true;
          break;
        }

        if (!IsGpuMemType(t.srcs[0].memType) && !IsGpuMemType(t.dsts[0].memType)) {
          errors.push_back({ERR_WARN,
              "Transfer %d: No GPU memory for source or destination.  Copy might not execute on BMA %d",
              i, t.exeDevice.exeIndex});
        } else {
          if (IsGpuMemType(t.srcs[0].memType)) {
            if (t.srcs[0].memIndex != t.exeDevice.exeIndex) {
              errors.push_back({ERR_WARN,
                  "Transfer %d: BMA executor may use the source memory device (%d) not (%d)",
                  i, t.srcs[0].memIndex, t.exeDevice.exeIndex});
            }
          } else if (t.dsts[0].memIndex != t.exeDevice.exeIndex) {
            errors.push_back({ERR_WARN,
                "Transfer %d: BMA executor may use the destination memory device (%d) not (%d)",
                i, t.dsts[0].memIndex, t.exeDevice.exeIndex});
          }
        }
        break;
#else
        errors.push_back({ERR_FATAL,
            "Transfer %d: BMA executor requires ROCm 7.1 or newer (AMD HIP with hipMemcpyBatchAsync)", i});
        hasFatalError = true;
        break;
#endif
      case EXE_NIC: case EXE_NIC_NEAREST:
      if (IsIbvSymbolsReady())
      {
        // NIC Executors can only execute a copy operation
        if (t.srcs.size() != 1 || t.dsts.size() != 1) {
          errors.push_back({ERR_FATAL, "Transfer %d: NIC executor requires single SRC and single DST", i});
          hasFatalError = true;
          break;
        }

        // NIC executor cannot do remote read + remote write - either src or dst must be local
        int srcExeRank = t.exeDevice.exeRank;
        int srcMemRank = t.srcs[0].memRank;
        int dstMemRank = t.dsts[0].memRank;
        int dstExeRank = (srcExeRank == srcMemRank ? dstMemRank : srcMemRank);
        if (srcMemRank != srcExeRank && dstMemRank != srcExeRank) {
          errors.push_back({ERR_FATAL,
              "Transfer %d: NIC executor rank (%d) must be same as SRC memory rank (%d) or DST memory rank (%d)", i, srcExeRank, srcMemRank, dstMemRank});
          hasFatalError = true;
          break;
        }

        // The SRC NIC executor is the one that initiates either a (remote read/local write) or (local read/remote write) copy operation
        ExeDevice srcExeDevice;
        ErrResult errSrc = GetActualExecutor(t.exeDevice, srcExeDevice);
        if (errSrc.errType != ERR_NONE) errors.push_back(errSrc);

        // Check that the SRC NIC exists and is active
        if (srcExeDevice.exeIndex < 0 || srcExeDevice.exeIndex >= GetNumExecutors(EXE_NIC, srcExeRank)) {
          errors.push_back({ERR_FATAL, "Transfer %d: Rank %d SRC NIC executor indexes an out-of-range NIC (%d).  Detected %d NICs",
              i, srcExeRank, srcExeDevice.exeIndex, GetNumExecutors(EXE_NIC, srcExeRank)});
          hasFatalError = true;
          break;
        } else if (!NicIsActive(srcExeDevice.exeIndex, srcExeDevice.exeRank)) {
          errors.push_back({ERR_FATAL, "Transfer %d: Rank %d SRC NIC executor %d is not active", i, srcExeDevice.exeRank, srcExeDevice.exeIndex});
          hasFatalError = true;
          break;
        }

        // The DST NIC executor facilitates the copy but issues no commands
        ExeDevice dstOrgDevice = {t.exeDevice.exeType, t.exeSubIndex, dstExeRank, t.exeSubSlot};
        ExeDevice dstExeDevice;
        ErrResult errDst = GetActualExecutor(dstOrgDevice, dstExeDevice);

        // Check that the DST NIC exists and is active
        if (dstExeDevice.exeIndex < 0 || dstExeDevice.exeIndex >= GetNumExecutors(EXE_NIC, dstExeRank)) {
          errors.push_back({ERR_FATAL, "Transfer %d: Rank %d DST NIC executor indexes an out-of-range NIC (%d).  Detected %d NICs",
              i, dstExeRank, dstExeDevice.exeIndex, GetNumExecutors(EXE_NIC, dstExeRank)});
          hasFatalError = true;
          break;
        } else if (!NicIsActive(dstExeDevice.exeIndex, dstExeDevice.exeRank)) {
          errors.push_back({ERR_FATAL, "Transfer %d: Rank %d DST NIC executor %d is not active", i, dstExeDevice.exeRank, dstExeDevice.exeIndex});
          hasFatalError = true;
          break;
        }
      } else {
        errors.push_back({ERR_FATAL, "Transfer %d: NIC executor is requested but is not available.", i});
        hasFatalError = true;
      }
      break;
      }

      // Skip further tests if fatal error detected
      if (hasFatalError) break;

      // Check for multi-node support
      if (IsPodTransfer(t)) {
#ifndef POD_COMM_ENABLED
        errors.push_back({ERR_FATAL,
            "Transfer %d: Cross-rank GPU memory access requires pod communication support (HIP 8.0+)", i});
        hasFatalError = true;
        break;
#endif
        // In order to support pod communication, the participanting ranks need to be members of the same pod
        int exeRank = t.exeDevice.exeRank;
        bool samePod = true;

        for (auto const& src : t.srcs) {
          if (!(samePod = IsSamePod(src.memRank, exeRank)))
            break;
        }
        if (samePod) {
          for (auto const& dst : t.dsts) {
            if (!(samePod = IsSamePod(dst.memRank, exeRank)))
              break;
          }
        }

        if (!samePod || IsCpuExeType(t.exeDevice.exeType)) {
          errors.push_back({ERR_FATAL, "Transfer %d: Executor on rank %d can not access memory across ranks\n",
              i, t.exeDevice.exeRank});
          break;
        }

        // Pod (cross-rank) transfers with a GPU executor are exchanged via CUDA/HIP fabric handles,
        // which, for current version, only support device backed allocations.
        // Reject host memory allocations up front.
        if (IsGpuExeType(t.exeDevice.exeType)) {
          bool hasRemoteCpuMem = false;
          MemDevice offender = {};
          char const* role = nullptr;
          for (auto const& src : t.srcs) {
            if (src.memRank != t.exeDevice.exeRank && IsCpuMemType(src.memType)) {
              hasRemoteCpuMem = true; offender = src; role = "SRC"; break;
            }
          }
          if (!hasRemoteCpuMem) {
            for (auto const& dst : t.dsts) {
              if (dst.memRank != t.exeDevice.exeRank && IsCpuMemType(dst.memType)) {
                hasRemoteCpuMem = true; offender = dst; role = "DST"; break;
              }
            }
          }
          if (hasRemoteCpuMem) {
            errors.push_back({ERR_FATAL,
                "Transfer %d: Cross-rank GPU executor (R%d%c%d) cannot access remote host memory "
                "(%s on rank %d is %s).  Fabric-handle sharing only supports GPU memory for 1.67; use a NIC "
                "executor (e.g. R%dN..) for cross-rank transfers involving host memory.",
                i, t.exeDevice.exeRank, ExeTypeStr[t.exeDevice.exeType], t.exeDevice.exeIndex,
                role, offender.memRank, GetMemTypeName(offender.memType), t.exeDevice.exeRank});
            hasFatalError = true;
            break;
          }
        }
      }

      // Check subexecutors
      if (t.numSubExecs <= 0)
        errors.push_back({ERR_FATAL, "Transfer %d: # of subexecutors must be positive", i});
      else
        totalSubExecs[t.exeDevice] += t.numSubExecs;

    }

    int gpuMaxHwQueues = 4;
    if (getenv("GPU_MAX_HW_QUEUES"))
      gpuMaxHwQueues = atoi(getenv("GPU_MAX_HW_QUEUES"));

    // Aggregate checks
    for (auto const& exeDevice : executors) {
      switch (exeDevice.exeType) {
      case EXE_CPU:
      {
        // Check total number of subexecutors requested
        int numCpuSubExec = GetNumSubExecutors(exeDevice);
        if (totalSubExecs[exeDevice] > numCpuSubExec)
          errors.push_back({ERR_WARN,
                            "CPU %d requests %d total cores however only %d available. "
                            "Serialization will occur",
                            exeDevice.exeIndex, totalSubExecs[exeDevice], numCpuSubExec});
        break;
      }
      case EXE_GPU_GFX:
      {
        // Check total number of subexecutors requested
        int numGpuSubExec = GetNumSubExecutors(exeDevice);
        // For warp-level dispatch, multiply by warps per threadblock
        if (cfg.gfx.seType == 1) {
          int warpsPerBlock = cfg.gfx.blockSize / GetWarpSize(&errors);
          numGpuSubExec *= warpsPerBlock;
        }
        if (totalSubExecs[exeDevice] > numGpuSubExec)
          errors.push_back({ERR_WARN,
                            "GPU %d requests %d total %s however only %d available. "
                            "Serialization will occur",
                            exeDevice.exeIndex, totalSubExecs[exeDevice],
                            cfg.gfx.seType == 0 ? "CUs" : "warps", numGpuSubExec});
        // Check that if executor subindices are used, all Transfers specify executor subindices
        if (useSubIndexCount[exeDevice] > 0 && useSubIndexCount[exeDevice] != transferCount[exeDevice]) {
          errors.push_back({ERR_FATAL,
                            "GPU %d specifies XCC on only %d of %d Transfers. "
                            "Must either specific none or all",
                            exeDevice.exeIndex, useSubIndexCount[exeDevice], transferCount[exeDevice]});
          break;
        }

        if (cfg.general.useMultiStream && transferCount[exeDevice] > gpuMaxHwQueues) {
          errors.push_back({ERR_WARN,
                            "GPU %d attempting %d parallel transfers, however GPU_MAX_HW_QUEUES only set to %d",
                            exeDevice.exeIndex, transferCount[exeDevice], gpuMaxHwQueues});
        }
        break;
      }
      case EXE_GPU_TDM:
      {
        int numGpuSubExec = GetNumSubExecutors(exeDevice);
        if (totalSubExecs[exeDevice] > numGpuSubExec)
          errors.push_back({ERR_WARN,
                            "TDM %d requests %d total subexecutors however only %d available. "
                            "Serialization will occur",
                            exeDevice.exeIndex, totalSubExecs[exeDevice], numGpuSubExec});

        if (cfg.general.useMultiStream && transferCount[exeDevice] > gpuMaxHwQueues) {
          errors.push_back({ERR_WARN,
              "TDM %d attempting %d parallel transfers, however GPU_MAX_HW_QUEUES only set to %d",
              exeDevice.exeIndex, transferCount[exeDevice], gpuMaxHwQueues});
        }
        break;
      }
      case EXE_GPU_DMA:
      {
        // Check that if executor subindices are used, all Transfers specify executor subindices
        if (useSubIndexCount[exeDevice] > 0 && useSubIndexCount[exeDevice] != transferCount[exeDevice]) {
          errors.push_back({ERR_FATAL,
                            "DMA %d specifies engine on only %d of %d Transfers. "
                            "Must either specific none or all",
                            exeDevice.exeIndex, useSubIndexCount[exeDevice], transferCount[exeDevice]});
          break;
        }
        if (transferCount[exeDevice] > gpuMaxHwQueues) {
          errors.push_back({ERR_WARN,
                           "DMA %d attempting %d parallel transfers, however GPU_MAX_HW_QUEUES only set to %d",
                           exeDevice.exeIndex, transferCount[exeDevice], gpuMaxHwQueues});
        }

        char* enableSdma = getenv("HSA_ENABLE_SDMA");
        if (enableSdma && !strcmp(enableSdma, "0"))
          errors.push_back({ERR_WARN,
                            "DMA functionality disabled due to environment variable HSA_ENABLE_SDMA=0. "
                            "DMA %d copies will fallback to blit (GFX) kernels", exeDevice.exeIndex});
        break;
      }
      case EXE_GPU_BDMA:
      {
        if (transferCount[exeDevice] > gpuMaxHwQueues) {
          errors.push_back({ERR_WARN,
                           "BMA %d attempting %d parallel transfers, however GPU_MAX_HW_QUEUES only set to %d",
                           exeDevice.exeIndex, transferCount[exeDevice], gpuMaxHwQueues});
        }
        break;
      }
      default:
        break;
      }
    }

    // Check for fatal errors
    for (auto const& err : errors)
      if (err.errType == ERR_FATAL) return true;
    return false;
  }

// Internal data structures
//========================================================================================

  // Parameters for each SubExecutor
  struct SubExecParam
  {
    // Inputs
    size_t                     N;                 ///< Number of floats this subExecutor works on
    int                        numSrcs;           ///< Number of source arrays
    int                        numDsts;           ///< Number of destination arrays
    float*                     src[MAX_SRCS];     ///< Source array pointers
    float*                     dst[MAX_DSTS];     ///< Destination array pointers
    int32_t                    preferredXccId;    ///< XCC ID to execute on (GFX only)

    // Prepared
    int                        teamSize;          ///< Index of this sub executor amongst team
    int                        teamIdx;           ///< Size of team this sub executor is part of

    // Outputs
    int64_t                    startCycle;        ///< Start timestamp for in-kernel timing (GPU-GFX executor)
    int64_t                    stopCycle;         ///< Stop  timestamp for in-kernel timing (GPU-GFX executor)
    uint32_t                   hwId;              ///< Hardware ID
    uint32_t                   xccId;             ///< XCC ID
  };

  // Internal resources allocated per Transfer
  typedef hipMemGenericAllocationHandle_t memHandle_t;
  struct TransferResources
  {
    int                        transferIdx;       ///< The associated Transfer
    size_t                     numBytes;          ///< Number of bytes to Transfer
    vector<float*>             srcMem;            ///< Source memory
    vector<float*>             dstMem;            ///< Destination memory
    vector<size_t>             srcActualBytes;    ///< Actual amount of src memory allocated (after padding)
    vector<size_t>             dstActualBytes;    ///< Actual amount of dst memory allocated (after padding)
    vector<memHandle_t>        srcMemHandle;      ///< Memory handles for source memory
    vector<memHandle_t>        dstMemHandle;      ///< Memory handles for destination memory
    vector<SubExecParam>       subExecParamCpu;   ///< Defines subarrays for each subexecutor
    vector<int>                subExecIdx;        ///< Indices into subExecParamGpu
    int                        numaNode;          ///< NUMA node to use for this Transfer

    // For GFX executor
    SubExecParam*              subExecParamGpuPtr;

    // For targeted-SDMA
#if !defined(__NVCC__)
    vector<hsa_agent_t>        dstAgent;          ///< DMA destination memory agents
    hsa_agent_t                srcAgent;          ///< DMA source memory agent
    hsa_signal_t               signal;            ///< HSA signal for completion
    hsa_amd_sdma_engine_id_t   sdmaEngineId;      ///< DMA engine ID
#endif

    // For IBV executor
    int                        srcNicIndex;       ///< SRC NIC index
    int                        dstNicIndex;       ///< DST NIC index
    ibv_context*               srcContext;        ///< Device context for SRC NIC
    ibv_context*               dstContext;        ///< Device context for DST NIC
    ibv_pd*                    srcProtect;        ///< Protection domain for SRC NIC
    ibv_pd*                    dstProtect;        ///< Protection domain for DST NIC
    ibv_cq*                    srcCompQueue;      ///< Completion queue for SRC NIC
    ibv_cq*                    dstCompQueue;      ///< Completion queue for DST NIC
    ibv_port_attr              srcPortAttr;       ///< Port attributes for SRC NIC
    ibv_port_attr              dstPortAttr;       ///< Port attributes for DST NIC
    ibv_gid                    srcGid;            ///< GID handle for SRC NIC
    ibv_gid                    dstGid;            ///< GID handle for DST NIC
    vector<ibv_qp*>            srcQueuePairs;     ///< Queue pairs for SRC NIC
    vector<ibv_qp*>            dstQueuePairs;     ///< Queue pairs for DST NIC
    ibv_mr*                    srcMemRegion;      ///< Memory region for SRC
    ibv_mr*                    dstMemRegion;      ///< Memory region for DST
    int                        srcDmabufFd;       ///< DMA-BUF file descriptor for SRC (if using dmabuf)
    int                        dstDmabufFd;       ///< DMA-BUF file descriptor for DST (if using dmabuf)
    uint64_t                   srcDmabufOffset;   ///< Offset within SRC DMA-BUF
    uint64_t                   dstDmabufOffset;   ///< Offset within DST DMA-BUF
    uint32_t                   qpCount;           ///< Number of QPs to be used for transferring data
    bool                       srcIsExeNic;       ///< Whether SRC or DST NIC initiates traffic
    vector<vector<ibv_sge>>    sgePerQueuePair;   ///< Scatter-gather elements per queue pair
    vector<vector<ibv_send_wr>>sendWorkRequests;  ///< Send work requests per queue pair

    // For BMA executor
#ifdef BMA_EXEC_ENABLED
    vector<void*>              batchDsts;         ///< Destination pointers (per batch item)
    vector<void*>              batchSrcs;         ///< Source pointers (per batch item)
    vector<size_t>             batchBytes;        ///< Bytes to copy (per batch item)
#endif

    // Counters
    double                     totalDurationMsec; ///< Total duration for all iterations for this Transfer
    vector<double>             perIterMsec;       ///< Duration for each individual iteration
    vector<set<pair<int,int>>> perIterCUs;        ///< GFX-Executor only. XCC:CU used per iteration
  };

  // Internal resources allocated per Executor
  struct ExeInfo
  {
    size_t                     totalBytes;        ///< Total bytes this executor transfers
    double                     totalDurationMsec; ///< Total duration for all iterations for this Executor
    int                        totalSubExecs;     ///< Total number of subExecutors to use
    bool                       useSubIndices;     ///< Use subexecutor indicies
    int                        numSubIndices;     ///< Number of subindices this ExeDevice has
    vector<SubExecParam>       subExecParamCpu;   ///< Subexecutor parameters for this executor
    vector<TransferResources>  resources;         ///< Per-Transfer resources

    // For GPU-Executors
    SubExecParam*              subExecParamGpu;   ///< GPU copy of subExecutor parameters
    bool                       subExecParamHostAccessible; ///< Host can directly read subExecParamGpu
    vector<hipStream_t>        streams;           ///< HIP streams to launch on
    vector<hipEvent_t>         startEvents;       ///< HIP start timing event
    vector<hipEvent_t>         stopEvents;        ///< HIP stop timing event
    int                        wallClockRate;     ///< (GFX-only) Device wall clock rate
    int                        gfxKernelToUse;    ///< (GFX-only) Which GFX kernel to use

    // For TDM-Executors
    uint32_t                   ldsBytesActual;    ///< Actual number of LDS bytes to use as buffer
    int                        tdmKernelToUse;    ///< (TDM-only) Which TDM kernel to use
  };

  // Structure to track PCIe topology
  struct PCIeNode
  {
    std::string        address;                   ///< PCIe address for this PCIe node
    std::string        description;               ///< Description for this PCIe node
    std::set<PCIeNode> children;                  ///< Children PCIe nodes

    // Default constructor
    PCIeNode() : address(""), description("") {}

    // Constructor
    PCIeNode(std::string const& addr) : address(addr) {}

    // Constructor
    PCIeNode(std::string const& addr, std::string const& desc)
      :address(addr), description(desc) {}

    // Comparison operator for std::set
    bool operator<(PCIeNode const& other) const {
      return address < other.address;
    }
  };

  // Structure to track information about IBV devices
  struct IbvDevice
  {
    ibv_device* devicePtr;
    std::string name;
    std::string busId;
    bool        hasActivePort;
    int         numaNode;
    int         gidIndex;
    std::string gidDescriptor;
    bool        isRoce;
  };

// Function to collect information about IBV devices
//========================================================================================

  static bool IsConfiguredGid(union ibv_gid const& gid)
  {
    const struct in6_addr *a = (struct in6_addr *) gid.raw;
    int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
    if (((a->s6_addr32[0] | trailer) == 0UL) ||
        ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
      return false;
    }
    return true;
  }

  static bool LinkLocalGid(union ibv_gid const& gid)
  {
    const struct in6_addr *a = (struct in6_addr *) gid.raw;
    if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL) {
      return true;
    }
    return false;
  }

  static ErrResult GetRoceVersionNumber(struct ibv_context* const& context,
                                        int const&  portNum,
                                        int const&  gidIndex,
                                        int&        version)
  {
    char const* deviceName = ibv_get_device_name(context->device);
    char gidRoceVerStr[16]      = {};
    char roceTypePath[PATH_MAX] = {};
    sprintf(roceTypePath, "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d",
            deviceName, portNum, gidIndex);

    int fd = open(roceTypePath, O_RDONLY);
    if (fd == -1)
      return {ERR_FATAL, "Failed while opening RoCE file path (%s)", roceTypePath};

    int ret = read(fd, gidRoceVerStr, 15);
    close(fd);

    if (ret == -1)
      return {ERR_FATAL, "Failed while reading RoCE version"};

    if (strlen(gidRoceVerStr)) {
      if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0
          || strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0) {
        version = 1;
      }
      else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0) {
        version = 2;
      }
    }
    return ERR_NONE;
  }

  static bool IsIPv4MappedIPv6(const union ibv_gid &gid)
  {
    // look for ::ffff:x.x.x.x format
    // From Broadcom documentation
    // https://techdocs.broadcom.com/us/en/storage-and-ethernet-connectivity/ethernet-nic-controllers/bcm957xxx/adapters/frequently-asked-questions1.html
    // "The IPv4 address is really an IPv4 address mapped into the IPv6 address space.
    // This can be identified by 80 “0” bits, followed by 16 “1” bits (“FFFF” in hexadecimal)
    // followed by the original 32-bit IPv4 address."
    return (gid.global.subnet_prefix == 0    &&
            gid.raw[8]               == 0    &&
            gid.raw[9]               == 0    &&
            gid.raw[10]              == 0xff &&
            gid.raw[11]              == 0xff);
  }

  static ErrResult GetGidIndex(struct ibv_context*          context,
                               int const&                   gidTblLen,
                               int const&                   portNum,
                               std::pair<int, std::string>& gidInfo)
  {
    if(gidInfo.first >= 0) return ERR_NONE; // honor user choice
    union ibv_gid gid;

    GidPriority highestPriority = GidPriority::UNKNOWN;
    int gidIndex = -1;

    for (int i = 0; i < gidTblLen; ++i) {
      IBV_CALL(ibv_query_gid, context, portNum, i, &gid);
      if (!IsConfiguredGid(gid)) continue;
      int gidCurrRoceVersion;
      if(GetRoceVersionNumber(context, portNum, i, gidCurrRoceVersion).errType != ERR_NONE) continue;
      GidPriority currPriority;
      if (IsIPv4MappedIPv6(gid)) {
        currPriority = (gidCurrRoceVersion == 2) ? GidPriority::ROCEV2_IPV4 : GidPriority::ROCEV1_IPV4;
      } else if (!LinkLocalGid(gid)) {
        currPriority = (gidCurrRoceVersion == 2) ? GidPriority::ROCEV2_IPV6 : GidPriority::ROCEV1_IPV6;
      } else {
        currPriority = (gidCurrRoceVersion == 2) ? GidPriority::ROCEV2_LINK_LOCAL : GidPriority::ROCEV1_LINK_LOCAL;
      }
      if(currPriority > highestPriority) {
        highestPriority = currPriority;
        gidIndex = i;
      }
    }

    if (highestPriority == GidPriority::UNKNOWN) {
      gidInfo.first = -1;
      return {ERR_FATAL, "Failed to auto-detect a valid GID index. Try setting it manually through IB_GID_INDEX"};
    }
    gidInfo.first = gidIndex;
    gidInfo.second = GidPriorityStr[highestPriority];
    return ERR_NONE;
  }

  static vector<IbvDevice>& GetIbvDeviceList()
  {
    static bool isInitialized = false;
    static vector<IbvDevice> ibvDeviceList = {};

    // Build list on first use
    if (IsIbvSymbolsReady() && !isInitialized) {

      // Query the number of IBV devices
      int numIbvDevices = 0;
      ibv_device** deviceList = ibv_get_device_list(&numIbvDevices);

      // Check for TB_NIC_FILTER
      // By default, accept all NIC names
      std::string nicFilterPattern = getenv("TB_NIC_FILTER") ? getenv("TB_NIC_FILTER") : ".*";

      if (deviceList && numIbvDevices > 0) {
        // Loop over each device to collect information
        for (int i = 0; i < numIbvDevices; i++) {

          // Filter by name
          if (!std::regex_match(deviceList[i]->name, std::regex(nicFilterPattern))) continue;

          IbvDevice ibvDevice;
          ibvDevice.devicePtr = deviceList[i];
          ibvDevice.name = deviceList[i]->name;
          ibvDevice.hasActivePort = false;
          {
            struct ibv_context *context = ibv_open_device(ibvDevice.devicePtr);
            if (context) {
              struct ibv_device_attr deviceAttr;
              if (!ibv_query_device(context, &deviceAttr)) {
                int activePort;
                ibvDevice.gidIndex = -1;
                for (int port = 1; port <= deviceAttr.phys_port_cnt; ++port) {
                  struct ibv_port_attr portAttr;
                  if (ibv_query_port(context, port, &portAttr)) continue;
                  if (portAttr.state == IBV_PORT_ACTIVE) {
                    activePort = port;
                    ibvDevice.hasActivePort = true;
                    if(portAttr.link_layer == IBV_LINK_LAYER_ETHERNET) {
                      ibvDevice.isRoce = true;
                      std::pair<int, std::string> gidInfo (-1, "");
                      auto res = GetGidIndex(context, portAttr.gid_tbl_len, activePort, gidInfo);
                      if (res.errType == ERR_NONE) {
                        ibvDevice.gidIndex = gidInfo.first;
                        ibvDevice.gidDescriptor = gidInfo.second;
                      }
                    }
                    break;
                  }
                }
              }
              ibv_close_device(context);
            }
          }
          ibvDevice.busId = "";
          {
            std::string device_path = std::string(ibvDevice.devicePtr->dev_path) + "/device";
            if (std::filesystem::exists(device_path)) {
              std::string pciPath = std::filesystem::canonical(device_path).string();
              std::size_t pos = pciPath.find_last_of('/');
              if (pos != std::string::npos) {
                ibvDevice.busId = pciPath.substr(pos + 1);
              }
            }
          }

          // Get nearest numa node for this device
          ibvDevice.numaNode = -1;
          std::filesystem::path devicePath = "/sys/bus/pci/devices/" + ibvDevice.busId + "/numa_node";
          if (std::filesystem::exists(devicePath)) {
            std::string canonicalPath = std::filesystem::canonical(devicePath).string();
            std::ifstream file(canonicalPath);
            if (file.is_open()) {
              std::string numaNodeStr;
              std::getline(file, numaNodeStr);
              int numaNodeVal;
              if (sscanf(numaNodeStr.c_str(), "%d", &numaNodeVal) == 1)
                ibvDevice.numaNode = numaNodeVal;
              file.close();
            }
          }
          ibvDeviceList.push_back(ibvDevice);
        }
      }
      ibv_free_device_list(deviceList);
      isInitialized = true;
    }
    return ibvDeviceList;
  }

// PCIe-related functions
//========================================================================================

  // Prints off PCIe tree
  static inline void PrintPCIeTree(PCIeNode    const& node,
                                   std::string const& prefix = "",
                                   bool               isLast = true)
  {
    if (!node.address.empty()) {
      System::Get().Log("%s%s%s", prefix.c_str(), (isLast ? "└── " : "├── "), node.address.c_str());
      if (!node.description.empty()) {
        System::Get().Log("(%s)", node.description.c_str());
      }
      System::Get().Log("\n");
    }
    auto const& children = node.children;
    for (auto it = children.begin(); it != children.end(); ++it) {
      PrintPCIeTree(*it, prefix + (isLast ? "    " : "│   "), std::next(it) == children.end());
    }
  }

  // Inserts nodes along pcieAddress down a tree starting from root
  static ErrResult InsertPCIePathToTree(std::string const& pcieAddress,
                                        std::string const& description,
                                        PCIeNode&          root)
  {
    std::string lowerAddress = pcieAddress;
    std::transform(lowerAddress.begin(), lowerAddress.end(), lowerAddress.begin(), ::tolower);

    std::filesystem::path devicePath = "/sys/bus/pci/devices/" + lowerAddress;
    if (!std::filesystem::exists(devicePath)) {
      return {ERR_FATAL, "Device path %s does not exist", devicePath.c_str()};
    }

    std::string canonicalPath = std::filesystem::canonical(devicePath).string();
    std::istringstream iss(canonicalPath);
    std::string token;

    PCIeNode* currNode = &root;
    while (std::getline(iss, token, '/')) {
      auto it = (currNode->children.insert(PCIeNode(token))).first;
      currNode = const_cast<PCIeNode*>(&(*it));
    }
    currNode->description = description;

    return ERR_NONE;
  }

  // Returns root node for PCIe tree.  Constructed on first use
  static PCIeNode* GetPCIeTreeRoot()
  {
    static bool isInitialized = false;
    static PCIeNode pcieRoot;

    // Build PCIe tree on first use
    if (!isInitialized) {
      // Add NICs to the tree
      auto const& ibvDeviceList = GetIbvDeviceList();
      for (IbvDevice const& ibvDevice : ibvDeviceList) {
        if (!ibvDevice.hasActivePort || ibvDevice.busId == "") continue;
        InsertPCIePathToTree(ibvDevice.busId, ibvDevice.name, pcieRoot);
      }

      // Add GPUs to the tree
      int numGpus = 0;
      if (hipGetDeviceCount(&numGpus) != hipSuccess) numGpus = 0;
      for (int i = 0; i < numGpus; ++i) {
        char hipPciBusId[64];
        if (hipDeviceGetPCIBusId(hipPciBusId, sizeof(hipPciBusId), i) == hipSuccess) {
          InsertPCIePathToTree(hipPciBusId, "GPU " + std::to_string(i), pcieRoot);
        }
      }
#ifdef VERBS_DEBUG
      PrintPCIeTree(pcieRoot);
#endif
      isInitialized = true;
    }
    return &pcieRoot;
  }

  // Finds the lowest common ancestor in PCIe tree between two nodes
  static PCIeNode const* GetLcaBetweenNodes(PCIeNode    const* root,
                                            std::string const& node1Address,
                                            std::string const& node2Address)
  {
    if (!root || root->address == node1Address || root->address == node2Address)
      return root;

    PCIeNode const* lcaFound1 = nullptr;
    PCIeNode const* lcaFound2 = nullptr;

    // Recursively iterate over children
    for (auto const& child : root->children) {
      PCIeNode const* lca = GetLcaBetweenNodes(&child, node1Address, node2Address);
      if (!lca) continue;
      if (!lcaFound1) {
        // First time found
        lcaFound1 = lca;
      } else {
        // Second time found
        lcaFound2 = lca;
        break;
      }
    }

    // If two children were found, then current node is the lowest common ancestor
    return (lcaFound1 && lcaFound2) ? root : lcaFound1;
  }

  // Gets the depth of an node in the PCIe tree
  static int GetLcaDepth(std::string const&     targetBusID,
                         PCIeNode const* const& node,
                         int                    depth = 0)
  {
    if (!node) return -1;
    if (targetBusID == node->address) return depth;

    for (auto const& child : node->children) {
      int distance = GetLcaDepth(targetBusID, &child, depth + 1);
      if (distance != -1)
        return distance;
    }
    return -1;
  }

  // Function to extract the bus number from a PCIe address (domain:bus:device.function)
  static int ExtractBusNumber(std::string const& pcieAddress)
  {
    int domain, bus, device, function;
    char delimiter;

    std::istringstream iss(pcieAddress);
    iss >> std::hex >> domain >> delimiter >> bus >> delimiter >> device >> delimiter >> function;
    if (iss.fail()) {
#ifdef VERBS_DEBUG
      System::Get().Log("Invalid PCIe address format: %s\n", pcieAddress.c_str());
#endif
      return -1;
    }
    return bus;
  }

  // Function to compute the distance between two bus IDs
  static int GetBusIdDistance(std::string const& pcieAddress1,
                              std::string const& pcieAddress2)
  {
    int bus1 = ExtractBusNumber(pcieAddress1);
    int bus2 = ExtractBusNumber(pcieAddress2);
    return (bus1 < 0 || bus2 < 0) ? -1 : std::abs(bus1 - bus2);
  }

  // Given a target busID and a set of candidate devices, returns a set of indices
  // that is "closest" to the target
  static std::set<int> GetNearestDevicesInTree(std::string              const& targetBusId,
                                               std::vector<std::string> const& candidateBusIdList)
  {
    int maxDepth = -1;
    int minDistance = std::numeric_limits<int>::max();
    std::set<int> matches = {};

    // Loop over the candidates to find the ones with the lowest common ancestor (LCA)
    for (int i = 0; i < candidateBusIdList.size(); i++) {
      std::string const& candidateBusId = candidateBusIdList[i];
      if (candidateBusId == "") continue;
      PCIeNode const* lca = GetLcaBetweenNodes(GetPCIeTreeRoot(), targetBusId, candidateBusId);
      if (!lca) continue;

      int depth = GetLcaDepth(lca->address, GetPCIeTreeRoot());
      int currDistance = GetBusIdDistance(targetBusId, candidateBusId);

      // When more than one LCA match is found, choose the one with smallest busId difference
      // NOTE: currDistance could be -1, which signals problem with parsing, however still
      //       remains a valid "closest" candidate, so is included
      if (depth > maxDepth || (depth == maxDepth && depth >= 0 && currDistance < minDistance)) {
        maxDepth = depth;
        matches.clear();
        matches.insert(i);
        minDistance = currDistance;
      } else if (depth == maxDepth && depth >= 0 && currDistance == minDistance) {
        matches.insert(i);
      }
    }
    return matches;
  }

// IB Verbs-related functions
//========================================================================================

  // Create a queue pair
  static ErrResult CreateQueuePair(ConfigOptions const& cfg,
                                   struct ibv_pd*       pd,
                                   struct ibv_cq*       cq,
                                   struct ibv_qp*&      qp)
  {
    // Set queue pair attributes
    struct ibv_qp_init_attr attr = {};
    attr.qp_type          = IBV_QPT_RC;                  // Set type to reliable connection
    attr.send_cq          = cq;                          // Send completion queue
    attr.recv_cq          = cq;                          // Recv completion queue
    attr.cap.max_send_wr  = cfg.nic.maxSendWorkReq;      // Max send work requests
    attr.cap.max_recv_wr  = cfg.nic.maxRecvWorkReq;      // Max recv work requests
    attr.cap.max_send_sge = 1;                           // Max send scatter-gather entries
    attr.cap.max_recv_sge = 1;                           // Max recv scatter-gather entries

    qp = ibv_create_qp(pd, &attr);
    if (qp == NULL)
      return {ERR_FATAL, "Error while creating QP"};

    return ERR_NONE;
  }

  // Initialize a queue pair
  static ErrResult InitQueuePair(struct ibv_qp* qp,
                                 uint8_t        port,
                                 unsigned       flags)
  {
    struct ibv_qp_attr attr = {};                        // Clear all attributes
    attr.qp_state        = IBV_QPS_INIT;                 // Set the QP state to INIT
    attr.pkey_index      = 0;                            // Set the partition key index to 0
    attr.port_num        = port;                         // Set the port number to the defined IB_PORT
    attr.qp_access_flags = flags;                        // Set the QP access flags to the provided flags

    int ret = ibv_modify_qp(qp, &attr,
                            IBV_QP_STATE      |          // Modify the QP state
                            IBV_QP_PKEY_INDEX |          // Modify the partition key index
                            IBV_QP_PORT       |          // Modify the port number
                            IBV_QP_ACCESS_FLAGS);        // Modify the access flags

    if (ret != 0)
      return {ERR_FATAL, "Error during QP Init. IB Verbs Error code: %d", ret};

    return ERR_NONE;
  }

  // Structure used to exchange connection information
  struct __attribute__((packed)) ConnInfo
  {
    uint16_t lid;     // Local  routing id
    ibv_gid  gid;     // Global routing id (RoCE)
    int      gidIdx;  // Global routing id index (RoCE)
    uint32_t qpn;     // Queue pair number
    uint32_t rkey;    // Remote memory access key
    uint64_t vaddr;   // Remote virtual address of the memory region
  };

  // Transition QueuePair to Ready to Receive State
  static ErrResult TransitionQpToRtr(ibv_qp*         qp,
                                     ConnInfo const& connInfo,
                                     uint8_t  const& port,
                                     bool     const& isRoCE,
                                     ibv_mtu  const& mtu,
                                     uint8_t  const& trafficClass,
                                     uint8_t  const& serviceLevel)
  {
    // Prepare QP attributes
    struct ibv_qp_attr attr = {};
    attr.qp_state           = IBV_QPS_RTR;
    attr.path_mtu           = mtu;
    attr.rq_psn             = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer      = 12;
    if (isRoCE) {
      attr.ah_attr.is_global                     = 1;
      attr.ah_attr.grh.dgid.global.subnet_prefix = connInfo.gid.global.subnet_prefix;
      attr.ah_attr.grh.dgid.global.interface_id  = connInfo.gid.global.interface_id;
      attr.ah_attr.grh.flow_label                = 0;
      attr.ah_attr.grh.sgid_index                = connInfo.gidIdx;
      attr.ah_attr.grh.hop_limit                 = 255;
      attr.ah_attr.grh.traffic_class             = trafficClass;
    } else {
      attr.ah_attr.is_global = 0;
      attr.ah_attr.dlid      = connInfo.lid;
    }
    attr.ah_attr.sl            = serviceLevel;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num      = port;
    attr.dest_qp_num           = connInfo.qpn;

    // Modify the QP
    int ret = ibv_modify_qp(qp, &attr,
                            IBV_QP_STATE              |
                            IBV_QP_AV                 |
                            IBV_QP_PATH_MTU           |
                            IBV_QP_DEST_QPN           |
                            IBV_QP_RQ_PSN             |
                            IBV_QP_MAX_DEST_RD_ATOMIC |
                            IBV_QP_MIN_RNR_TIMER);
    if (ret != 0)
      return {ERR_FATAL, "Error during QP RTR. IB Verbs Error code: %d", ret};

    return ERR_NONE;
  }

  // Transition QueuePair to Ready to Send state
  static ErrResult TransitionQpToRts(struct ibv_qp *qp)
  {
    struct ibv_qp_attr attr = {};
    attr.qp_state           = IBV_QPS_RTS;
    attr.sq_psn             = 0;
    attr.timeout            = 14;
    attr.retry_cnt          = 7;
    attr.rnr_retry          = 7;
    attr.max_rd_atomic      = 1;

    int ret = ibv_modify_qp(qp, &attr,
                            IBV_QP_STATE     |
                            IBV_QP_TIMEOUT   |
                            IBV_QP_RETRY_CNT |
                            IBV_QP_RNR_RETRY |
                            IBV_QP_SQ_PSN    |
                            IBV_QP_MAX_QP_RD_ATOMIC);
    if (ret != 0)
      return {ERR_FATAL, "Error during QP RTS. IB Verbs Error code: %d", ret};

    return ERR_NONE;
  }

  static ErrResult PrepareNicTransferResources(ConfigOptions const& cfg,
                                               ExeDevice     const& nicExeDevice,
                                               Transfer      const& t,
                                               TransferResources&   rss)

  {
    // The NIC executor is the one that initiates either a (remote read/local write) or (local read/remote write) copy operation
    // The NON executor is the NIC executor that facilitates the copy but issues no commands
    // TransferResources will be mostly prepared only on the ranks that are involved in this transfer, although all ranks pass
    // through this code
    int const srcMemRank = t.srcs[0].memRank;
    int const dstMemRank = t.dsts[0].memRank;
    int const nicExeRank = nicExeDevice.exeRank;
    int const nonExeRank = (nicExeRank == srcMemRank ? dstMemRank : srcMemRank);
    rss.srcIsExeNic = (srcMemRank == nicExeRank);

    // Figure out non Executor (Accounts for possible remap due to use of EXE_NIC_NEAREST)
    ExeDevice nonOrgDevice = {t.exeDevice.exeType, t.exeSubIndex, nonExeRank, t.exeSubSlot};
    ExeDevice nonExeDevice;
    ERR_CHECK(GetActualExecutor(nonOrgDevice, nonExeDevice));

    // All ranks track which NIC was used and number of queue pairs used
    rss.srcNicIndex = (nicExeRank == srcMemRank ? nicExeDevice.exeIndex : nonExeDevice.exeIndex);
    rss.dstNicIndex = (nicExeRank == srcMemRank ? nonExeDevice.exeIndex : nicExeDevice.exeIndex);
    rss.qpCount     = t.numSubExecs;

    // Initialize DMA-BUF fields
    rss.srcDmabufFd = -1;
    rss.dstDmabufFd = -1;
    rss.srcDmabufOffset = 0;
    rss.dstDmabufOffset = 0;

    // Print DMA-BUF support status (only once per rank)
    if (System::Get().IsVerbose()) {
      static bool dmabufStatusPrinted = false;
      if (!dmabufStatusPrinted) {
        dmabufStatusPrinted = true;
        printf("[INFO] Rank %d DMA-BUF support: ", GetRank());
        bool kernelSupport = CheckDmabufSupport();
        if (kernelSupport) {
          printf("ENABLED\n");
        } else {
          printf("DISABLED (kernel config or export symbol missing, using standard ibv_reg_mr)\n");
        }
      }
    }

    // Establish memory access flags
    unsigned int rdmaAccessFlags = (IBV_ACCESS_LOCAL_WRITE    |
                                    IBV_ACCESS_REMOTE_READ    |
                                    IBV_ACCESS_REMOTE_WRITE   |
                                    IBV_ACCESS_REMOTE_ATOMIC);

    unsigned int rdmaMemRegFlags = rdmaAccessFlags;
    if (cfg.nic.useRelaxedOrder) rdmaMemRegFlags |= IBV_ACCESS_RELAXED_ORDERING;

    int const port = cfg.nic.ibPort;

    // Prepare NIC on SRC mem rank
    int srcGidIndex = cfg.nic.ibGidIndex;
    bool srcIsRoCE = false;
    if (GetRank() == srcMemRank) {
      // Switch to closest CPU NUMA domain
      int numaNode = GetIbvDeviceList()[rss.srcNicIndex].numaNode;
      if (numaNode != -1)
        numa_run_on_node(numaNode);
      // Open SRC NIC context
      IBV_PTR_CALL(rss.srcContext, ibv_open_device, GetIbvDeviceList()[rss.srcNicIndex].devicePtr);
      // Open SRC protection domain
      IBV_PTR_CALL(rss.srcProtect, ibv_alloc_pd, rss.srcContext);

      // Export DMA-BUF for SRC memory if it's GPU memory
      if (CheckDmabufSupport() && !t.srcs.empty() && IsGpuMemType(t.srcs[0].memType)) {
        ERR_CHECK(ExportDmabuf(rss.srcMem[0], rss.numBytes, rss.srcDmabufFd, rss.srcDmabufOffset));
        if (System::Get().IsVerbose()) {
          printf("[INFO] Rank %d exported SRC GPU memory as DMA-BUF (fd=%d, offset=%lu)\n",
                 GetRank(), rss.srcDmabufFd, rss.srcDmabufOffset);
        }
      }

      // Register SRC memory region
      if (rss.srcDmabufFd >= 0) {
        IBV_PTR_CALL(rss.srcMemRegion, ibv_reg_dmabuf_mr, rss.srcProtect, rss.srcDmabufOffset,
                     rss.numBytes, (uint64_t)rss.srcMem[0], rss.srcDmabufFd, rdmaMemRegFlags);
        if (System::Get().IsVerbose()) {
          printf("[INFO] Rank %d registered SRC memory using ibv_reg_dmabuf_mr\n", GetRank());
        }
      } else {
        IBV_PTR_CALL(rss.srcMemRegion, ibv_reg_mr, rss.srcProtect, rss.srcMem[0], rss.numBytes, rdmaMemRegFlags);
        if (System::Get().IsVerbose()) {
          printf("[INFO] Rank %d registered SRC memory using ibv_reg_mr (standard path)\n", GetRank());
        }
      }
      // Create SRC completion queues
      // Ensure CQ size is at least as large as the number of queue pairs to avoid overflow
      int srcCQSize = std::max(cfg.nic.queueSize, static_cast<int>(rss.qpCount));
      IBV_PTR_CALL(rss.srcCompQueue, ibv_create_cq, rss.srcContext, srcCQSize, NULL, NULL, 0);
      // Get SRC port attributes
      IBV_CALL(ibv_query_port, rss.srcContext, port, &rss.srcPortAttr);
      // Check for RDMA over Converged Ethernet (RoCE) and update GID index appropriately
      srcIsRoCE = (rss.srcPortAttr.link_layer == IBV_LINK_LAYER_ETHERNET);
      if (srcIsRoCE) {
        // Try to auto-detect the GID index
        std::pair<int, std::string> srcGidInfo (srcGidIndex, "");
        ERR_CHECK(GetGidIndex(rss.srcContext, rss.srcPortAttr.gid_tbl_len, port, srcGidInfo));
        srcGidIndex = srcGidInfo.first;
        IBV_CALL(ibv_query_gid, rss.srcContext, port, srcGidIndex, &rss.srcGid);
      }

      // Prepare queue pairs and send elements
      rss.srcQueuePairs.resize(rss.qpCount);
      for (int i = 0; i < rss.qpCount; i++) {
       // Create SRC queue pair
        ERR_CHECK(CreateQueuePair(cfg, rss.srcProtect, rss.srcCompQueue, rss.srcQueuePairs[i]));
        // Initialize SRC queue pairs
        ERR_CHECK(InitQueuePair(rss.srcQueuePairs[i], port, rdmaAccessFlags));
      }
    }

    // Prepare NIC on DST mem rank
    int dstGidIndex = cfg.nic.ibGidIndex;
    bool dstIsRoCE = false;
    if (GetRank() == dstMemRank) {
      // Switch to closest CPU NUMA domain
      int numaNode = GetIbvDeviceList()[rss.dstNicIndex].numaNode;
      if (numaNode != -1)
        numa_run_on_node(numaNode);
      // Open DST NIC contexts
      IBV_PTR_CALL(rss.dstContext, ibv_open_device, GetIbvDeviceList()[rss.dstNicIndex].devicePtr);
      // Open DST protection domain
      IBV_PTR_CALL(rss.dstProtect, ibv_alloc_pd, rss.dstContext);

      // Export DMA-BUF for DST memory if it's GPU memory
      if (CheckDmabufSupport() && !t.dsts.empty() && IsGpuMemType(t.dsts[0].memType)) {
        ERR_CHECK(ExportDmabuf(rss.dstMem[0], rss.numBytes, rss.dstDmabufFd, rss.dstDmabufOffset));
        if (System::Get().IsVerbose()) {
          printf("[INFO] Rank %d exported DST GPU memory as DMA-BUF (fd=%d, offset=%lu)\n",
                 GetRank(), rss.dstDmabufFd, rss.dstDmabufOffset);
        }
      }

      // Register DST memory region
      if (rss.dstDmabufFd >= 0) {
        IBV_PTR_CALL(rss.dstMemRegion, ibv_reg_dmabuf_mr, rss.dstProtect, rss.dstDmabufOffset,
                     rss.numBytes, (uint64_t)rss.dstMem[0], rss.dstDmabufFd, rdmaMemRegFlags);
        if (System::Get().IsVerbose()) {
          printf("[INFO] Rank %d registered DST memory using ibv_reg_dmabuf_mr\n", GetRank());
        }
      } else {
        IBV_PTR_CALL(rss.dstMemRegion, ibv_reg_mr, rss.dstProtect, rss.dstMem[0], rss.numBytes, rdmaMemRegFlags);
        if (System::Get().IsVerbose()) {
          printf("[INFO] Rank %d registered DST memory using ibv_reg_mr (standard path)\n", GetRank());
        }
      }
      // Create DST completion queues
      // Ensure CQ size is at least as large as the number of queue pairs to avoid overflow
      int dstCQSize = std::max(cfg.nic.queueSize,static_cast<int>(rss.qpCount));
      IBV_PTR_CALL(rss.dstCompQueue, ibv_create_cq, rss.dstContext, dstCQSize, NULL, NULL, 0);
      // Get DST port attributes
      IBV_CALL(ibv_query_port, rss.dstContext, port, &rss.dstPortAttr);
      // Check for RDMA over Converged Ethernet (RoCE) and update GID index appropriately
      dstIsRoCE = (rss.dstPortAttr.link_layer == IBV_LINK_LAYER_ETHERNET);
      if (dstIsRoCE) {
        // Try to auto-detect the GID index
        std::pair<int, std::string> dstGidInfo (dstGidIndex, "");
        ERR_CHECK(GetGidIndex(rss.dstContext, rss.dstPortAttr.gid_tbl_len, port, dstGidInfo));
        dstGidIndex = dstGidInfo.first;
        IBV_CALL(ibv_query_gid, rss.dstContext, port, dstGidIndex, &rss.dstGid);
      }
      // Prepare queue pairs
      rss.dstQueuePairs.resize(rss.qpCount);
      for (int i = 0; i < rss.qpCount; i++) {
        // Create DST queue pair
        ERR_CHECK(CreateQueuePair(cfg, rss.dstProtect, rss.dstCompQueue, rss.dstQueuePairs[i]));
        // Initialize SRC/DST queue pairs
        ERR_CHECK(InitQueuePair(rss.dstQueuePairs[i], port, rdmaAccessFlags));
      }
    }

    // Executor rank prepares send elements and work requests
    if (GetRank() == nicExeRank) {
      rss.sgePerQueuePair.resize(rss.qpCount);
      rss.sendWorkRequests.resize(rss.qpCount);
    }

    // Broadcast SRC/DST port link_layer so that all ranks know it so that they can be compared
    System::Get().Broadcast(srcMemRank, sizeof(rss.srcPortAttr.link_layer), &rss.srcPortAttr.link_layer);
    System::Get().Broadcast(dstMemRank, sizeof(rss.dstPortAttr.link_layer), &rss.dstPortAttr.link_layer);
    if (rss.srcPortAttr.link_layer != rss.dstPortAttr.link_layer) {
      return {ERR_FATAL, "SRC NIC (%d) [Rank %d] and DST NIC (%d) [Rank %d] do not have the same link layer [%d vs %d]",
        rss.srcNicIndex, srcMemRank, rss.dstNicIndex, dstMemRank, rss.srcPortAttr.link_layer, rss.dstPortAttr.link_layer};
    }

    ConnInfo dstConnInfo = {};
    ConnInfo srcConnInfo = {};
    for (int i = 0; i < rss.qpCount; i++) {
      // Prepare and exchange SRC connection information
      if (GetRank() == srcMemRank) {
        srcConnInfo.lid    = rss.srcPortAttr.lid;
        srcConnInfo.gid    = rss.srcGid;
        srcConnInfo.gidIdx = srcGidIndex;
        srcConnInfo.qpn    = rss.srcQueuePairs[i]->qp_num;
        srcConnInfo.rkey   = rss.srcMemRegion->rkey;
        srcConnInfo.vaddr  = (uint64_t)rss.subExecParamCpu[i].src[0];
      }
      System::Get().Broadcast(srcMemRank, sizeof(srcConnInfo), &srcConnInfo);

      // Prepare and exchange DST connection information
      if (GetRank() == dstMemRank) {
        dstConnInfo.lid    = rss.dstPortAttr.lid;
        dstConnInfo.gid    = rss.dstGid;
        dstConnInfo.gidIdx = dstGidIndex;
        dstConnInfo.qpn    = rss.dstQueuePairs[i]->qp_num;
        dstConnInfo.rkey   = rss.dstMemRegion->rkey;
        dstConnInfo.vaddr  = (uint64_t)rss.subExecParamCpu[i].dst[0];
      }
      System::Get().Broadcast(dstMemRank, sizeof(dstConnInfo), &dstConnInfo);

      // Move queue pairs to ready-to-receive (RTR), using exchanged connection info
      // Then move them to read-to-send (RTS)
      // Broadcast each rank's result so all ranks fail together rather than
      // hanging on the next iteration's Broadcast when qpCount > 1.
      struct QpTransitionResult { ErrType errType; bool rtrFailed; };
      static_assert(std::is_trivially_copyable<QpTransitionResult>::value, "QpTransitionResult must be trivially copyable for MPI broadcast");
      QpTransitionResult srcQpResult = {ERR_NONE, false};
      if (GetRank() == srcMemRank) {
        ErrResult err = TransitionQpToRtr(rss.srcQueuePairs[i], dstConnInfo, port, srcIsRoCE, rss.srcPortAttr.active_mtu, cfg.nic.trafficClass, cfg.nic.serviceLevel);
        srcQpResult.rtrFailed = (err.errType != ERR_NONE);
        if (err.errType == ERR_NONE) {
          err = TransitionQpToRts(rss.srcQueuePairs[i]);
        }
        srcQpResult.errType = err.errType;
      }
      System::Get().Broadcast(srcMemRank, sizeof(srcQpResult), &srcQpResult);
      if (srcQpResult.errType != ERR_NONE) {
        return {ERR_FATAL, "SRC rank %d failed to transition QP %d to %s",
                srcMemRank, i, srcQpResult.rtrFailed ? "RTR" : "RTS"};
      }

      QpTransitionResult dstQpResult = {ERR_NONE, false};
      if (GetRank() == dstMemRank) {
        ErrResult err = TransitionQpToRtr(rss.dstQueuePairs[i], srcConnInfo, port, dstIsRoCE, rss.dstPortAttr.active_mtu, cfg.nic.trafficClass, cfg.nic.serviceLevel);
        dstQpResult.rtrFailed = (err.errType != ERR_NONE);
        if (err.errType == ERR_NONE) {
          err = TransitionQpToRts(rss.dstQueuePairs[i]);
        }
        dstQpResult.errType = err.errType;
      }
      System::Get().Broadcast(dstMemRank, sizeof(dstQpResult), &dstQpResult);
      if (dstQpResult.errType != ERR_NONE) {
        return {ERR_FATAL, "DST rank %d failed to transition QP %d to %s",
                dstMemRank, i, dstQpResult.rtrFailed ? "RTR" : "RTS"};
      }

      // Prepare scatter-gather element / work request for this queue pair in advance
      if (GetRank() == nicExeRank) {
        // Process the data to transfer in chunks (of cfg.nic.chunkBytes)
        size_t       remaining = rss.subExecParamCpu[i].N * sizeof(float);
        size_t const numChunks = (remaining + cfg.nic.chunkBytes - 1) / cfg.nic.chunkBytes;
        uint8_t*     local     = (nicExeRank == srcMemRank ? (uint8_t*)rss.subExecParamCpu[i].src[0]
                                                           : (uint8_t*)rss.subExecParamCpu[i].dst[0]);
        auto const   opcode    = (nicExeRank == srcMemRank ? IBV_WR_RDMA_WRITE             : IBV_WR_RDMA_READ);
        uint64_t     remote    = (nicExeRank == srcMemRank ? dstConnInfo.vaddr             : srcConnInfo.vaddr);
        auto const   lkey      = (nicExeRank == srcMemRank ? rss.srcMemRegion->lkey        : rss.dstMemRegion->lkey);
        auto const   rkey      = (nicExeRank == srcMemRank ? dstConnInfo.rkey              : srcConnInfo.rkey);
        if (System::Get().IsVerbose()) {
          System::Get().Log("[INFO] Transfer %d SubExec %d executed by rank %d NIC %d is %s with %lu chunks\n",
                            rss.transferIdx, i, nicExeRank, nicExeDevice.exeIndex,
                            (opcode == IBV_WR_RDMA_WRITE ? "remote write" : "remote read"),
                            numChunks);
        }
        rss.sgePerQueuePair[i].resize(numChunks, {});
        rss.sendWorkRequests[i].resize(numChunks, {});

        for (size_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
          bool   isLastChunk    = (chunkIdx == numChunks - 1);
          size_t currChunkBytes = isLastChunk ? remaining : cfg.nic.chunkBytes;

          // Prepare scatter gather element
          ibv_sge& sg = rss.sgePerQueuePair[i][chunkIdx];
          sg.length = currChunkBytes;
          sg.addr   = (uintptr_t)local;
          sg.lkey   = lkey;

          // Prepare work request
          ibv_send_wr& wr = rss.sendWorkRequests[i][chunkIdx];
          wr.wr_id               = i;
          wr.sg_list             = &rss.sgePerQueuePair[i][chunkIdx];
          wr.num_sge             = 1;
          wr.send_flags          = isLastChunk ? IBV_SEND_SIGNALED : 0;  // Only last chunk is signalled
          wr.opcode              = opcode;
          wr.wr.rdma.remote_addr = remote;
          wr.wr.rdma.rkey        = rkey;

          if (System::Get().IsVerbose()) {
            System::Get().Log("[INFO] Transfer %d SubExec %d chunk %lu local %p remote %p of size %lu\n",
                              rss.transferIdx, i, chunkIdx, (void*)local, (void*)remote, currChunkBytes);
          }

          // Increment locations
          remaining -= currChunkBytes;
          local     += currChunkBytes;
          remote    += currChunkBytes;
        }
      }
    }
    return ERR_NONE;
  }

  static ErrResult TeardownNicTransferResources(TransferResources& rss, Transfer const& t)
  {
    bool isSrcRank = (GetRank() == t.srcs[0].memRank);
    bool isDstRank = (GetRank() == t.dsts[0].memRank);

    // Deregister memory regions
    if (isSrcRank) IBV_CALL(ibv_dereg_mr, rss.srcMemRegion);
    if (isDstRank) IBV_CALL(ibv_dereg_mr, rss.dstMemRegion);

    // Close DMA-BUF file descriptors
    if (CheckDmabufSupport()) {
      if (isSrcRank && rss.srcDmabufFd >= 0) {
        close(rss.srcDmabufFd);
        rss.srcDmabufFd = -1;
      }
      if (isDstRank && rss.dstDmabufFd >= 0) {
        close(rss.dstDmabufFd);
        rss.dstDmabufFd = -1;
      }
    }

    // Destroy queue pairs
    if (isSrcRank) {
      for (auto srcQueuePair : rss.srcQueuePairs)
        IBV_CALL(ibv_destroy_qp, srcQueuePair);
      rss.srcQueuePairs.clear();
    }
    if (isDstRank) {
      for (auto dstQueuePair : rss.dstQueuePairs)
        IBV_CALL(ibv_destroy_qp, dstQueuePair);
      rss.dstQueuePairs.clear();
    }

    // Destroy completion queues
    if (isSrcRank) IBV_CALL(ibv_destroy_cq, rss.srcCompQueue);
    if (isDstRank) IBV_CALL(ibv_destroy_cq, rss.dstCompQueue);

    // Deallocate protection domains
    if (isSrcRank) IBV_CALL(ibv_dealloc_pd, rss.srcProtect);
    if (isDstRank) IBV_CALL(ibv_dealloc_pd, rss.dstProtect);

    // Destroy context
    if (isSrcRank) IBV_CALL(ibv_close_device, rss.srcContext);
    if (isDstRank) IBV_CALL(ibv_close_device, rss.dstContext);

    return ERR_NONE;
  }

// Data validation-related functions
//========================================================================================

  // Pseudo-random formula for each element in array
  static __host__ float PrepSrcValue(int srcBufferIdx, size_t idx)
  {
    return (((idx % 383) * 517) % 383 + 31) * (srcBufferIdx + 1);
  }

  // Fills a pre-sized buffer with the pattern, based on which src index buffer
  // Note: Can also generate expected dst buffer
  static void PrepareReference(ConfigOptions const& cfg, std::vector<float>& cpuBuffer, int bufferIdx)
  {
    size_t N = cpuBuffer.size();

    if (!cfg.data.fillCompress.empty()) {
      // 0 -> Random
      // 1 ->  1B0 - The upper  1 byte  of each aligned 2 bytes is 0
      // 2 ->  2B0 - The upper  2 bytes of each aligned 4 bytes are 0
      // 3 ->  4B0 - The upper  4 bytes of each aligned 8 bytes are 0
      // 4 -> 32B0 - The upper 32 bytes of each 64-byte line are 0

      // Fill buffer with random floats
      std::mt19937 gen;
      gen.seed(bufferIdx * 425);
      std::uniform_real_distribution<float> dist(-100000.0f, +100000.0f);
      for (size_t i = 0; i < N; i++) {
        cpuBuffer[i] = dist(gen);
      }

      // Figure out distribution for lines based on the percentages given
      size_t numLines = N / 16;
      size_t leftover = numLines;
      std::vector<size_t> lineCounts(5, 0);
      std::set<std::pair<double, int>> remainder;

      // Assign rounded down values first
      std::vector<int> percentages = cfg.data.fillCompress;
      while (percentages.size() < 5) percentages.push_back(0);
      for (int i = 0; i < percentages.size(); i++){
        lineCounts[i] = (size_t)(numLines * (percentages[i] / 100.0));
        leftover -= lineCounts[i];
        remainder.insert(std::make_pair(numLines * (percentages[i] / 100.0) - lineCounts[i], i));
      }

      // Assign leftovers based on largest remainder
      while (leftover != 0) {
        auto last = *remainder.rbegin();
        lineCounts[last.second]++;
        remainder.erase(last);
        leftover--;
      }

      // Randomly decide which lines get assigned to which types
      std::vector<int> lineTypes(numLines, 0);
      int offset = lineCounts[0];
      for (int i = 1; i < 5; i++) {
        for (int j = 0; j < lineCounts[i]; j++)
          lineTypes[offset++] = i;
      }
      std::shuffle(lineTypes.begin(), lineTypes.end(), gen);

      // Apply zero-ing
      int dumpLines = getenv("TB_DUMP_LINES") ? atoi(getenv("TB_DUMP_LINES")) : 0;

      if (dumpLines) {
        System::Get().Log("Input pattern 64B line statistics for bufferIdx %d:\n", bufferIdx);
        System::Get().Log("Total lines: %lu\n", numLines);
        System::Get().Log("- 0: Random : %8lu (%8.3f%%)\n", lineCounts[0], 100.0 * lineCounts[0] / (1.0 * numLines));
        System::Get().Log("- 1: 1B0    : %8lu (%8.3f%%)\n", lineCounts[1], 100.0 * lineCounts[1] / (1.0 * numLines));
        System::Get().Log("- 2: 2B0    : %8lu (%8.3f%%)\n", lineCounts[2], 100.0 * lineCounts[2] / (1.0 * numLines));
        System::Get().Log("- 3: 4B0    : %8lu (%8.3f%%)\n", lineCounts[3], 100.0 * lineCounts[3] / (1.0 * numLines));
        System::Get().Log("- 4: 32B0   : %8lu (%8.3f%%)\n", lineCounts[4], 100.0 * lineCounts[4] / (1.0 * numLines));
      }

      for (int line = 0; line < numLines; line++) {
        unsigned char* linePtr = (unsigned char*)&cpuBuffer[line * 16];

        switch (lineTypes[line]) {
        case 1: // 1B0
          for (int i = 0; i < 32; i++)
            linePtr[2*i+1] = 0;
          break;
        case 2: // 2B0
          for (int i = 0; i < 16; i++) {
            linePtr[4*i+2] = 0;
            linePtr[4*i+3] = 0;
          }
          break;
        case 3: // 4B0
          for (int i = 0; i < 8; i++) {
            linePtr[8*i+4] = 0;
            linePtr[8*i+5] = 0;
            linePtr[8*i+6] = 0;
            linePtr[8*i+7] = 0;
          }
          break;
        case 4: // 32B0
          for (int i = 32; i < 64; i++)
            linePtr[i] = 0;
          break;
        }

        if (line < dumpLines) {
          System::Get().Log("Line %02d [%d]: ", line, lineTypes[line]);
          for (int j = 63; j >= 0; j--){
            System::Get().Log("%02x ", linePtr[j]);
            if (j % 16 == 0) System::Get().Log(" ");
          }
          System::Get().Log("\n");
        }
      }
    } else {
      // Use fill pattern if specified
      size_t patternLen = cfg.data.fillPattern.size();
      if (patternLen > 0) {
        size_t copies   = N / patternLen;
        size_t leftOver = N % patternLen;
        float* cpuBufferPtr = cpuBuffer.data();
        for (int i = 0; i < copies; i++) {
          memcpy(cpuBufferPtr, cfg.data.fillPattern.data(), patternLen * sizeof(float));
          cpuBufferPtr += patternLen;
        }
        if (leftOver)
          memcpy(cpuBufferPtr, cfg.data.fillPattern.data(), leftOver * sizeof(float));
      } else {
        // Fall back to pseudo-random
        for (size_t i = 0; i < N; ++i)
          cpuBuffer[i] = PrepSrcValue(bufferIdx, i);
      }
    }
  }

  // Checks that destination buffers match expected values
  static ErrResult ValidateAllTransfers(ConfigOptions              const& cfg,
                                        vector<Transfer>           const& transfers,
                                        vector<TransferResources*> const& transferResources,
                                        vector<vector<float>>      const& dstReference,
                                        vector<float>&                    outputBuffer)
  {
    float* output;
    bool const verbose  = System::Get().IsVerbose();
    size_t initOffset   = cfg.data.byteOffset / sizeof(float);
    ErrResult firstErr  = ERR_NONE;

    for (auto rss : transferResources) {
      int transferIdx = rss->transferIdx;
      Transfer const& t = transfers[transferIdx];
      size_t N = t.numBytes / sizeof(float);

      float const* expected = dstReference[t.srcs.size()].data();
      for (int dstIdx = 0; dstIdx < (int)rss->dstMem.size(); dstIdx++) {
        // Validation is only done on the rank the destination memory is on
        if (t.dsts[dstIdx].memRank != GetRank()) continue;
        if (IsCpuMemType(t.dsts[dstIdx].memType) || cfg.data.validateDirect) {
          output = (rss->dstMem[dstIdx]) + initOffset;
          if (verbose) {
            System::Get().Log("[INFO] Validation: transfer %d DST[%d] direct read from %s%d  %zu bytes  ptr=%p\n",
                              transferIdx, dstIdx,
                              GetMemTypeName(t.dsts[dstIdx].memType), t.dsts[dstIdx].memIndex,
                              t.numBytes, output);
          }
        } else {
          ERR_CHECK(hipSetDevice(t.dsts[dstIdx].memIndex));
          if (verbose) {
            System::Get().Log("[INFO] Validation memcpy: transfer %d DST[%d] %s%d->host  %zu bytes  src=%p dst=%p\n",
                              transferIdx, dstIdx,
                              GetMemTypeName(t.dsts[dstIdx].memType), t.dsts[dstIdx].memIndex,
                              t.numBytes,
                              (rss->dstMem[dstIdx]) + initOffset,
                              outputBuffer.data());
          }
          ERR_CHECK(hipMemcpy(outputBuffer.data(), (rss->dstMem[dstIdx]) + initOffset, t.numBytes, hipMemcpyDefault));
          ERR_CHECK(hipDeviceSynchronize());
          output = outputBuffer.data();
        }

        ErrResult dstErr = ERR_NONE;
        if (memcmp(output, expected, t.numBytes)) {
          // Difference found - find first error
          for (size_t i = 0; i < N; i++) {
            if (output[i] != expected[i]) {
              dstErr = {ERR_FATAL, "Transfer %d: Unexpected mismatch at index %lu of destination %d on rank %d: Expected %10.5f Actual: %10.5f",
                transferIdx, i, dstIdx, t.dsts[dstIdx].memRank, expected[i], output[i]};
              break;
            }
          }
          if (dstErr.errType == ERR_NONE)
            // memcmp found a difference but float != didn't (e.g. +0.0f vs -0.0f bit pattern)
            dstErr = {ERR_FATAL, "Transfer %d: Unexpected output mismatch for destination %d", transferIdx, dstIdx};
        }

        if (verbose)
          System::Get().Log("[INFO] Validation: transfer %d DST[%d] %s%d — %s\n",
                            transferIdx, dstIdx,
                            GetMemTypeName(t.dsts[dstIdx].memType), t.dsts[dstIdx].memIndex,
                            dstErr.errType == ERR_NONE ? "PASS" : "FAIL");

        if (dstErr.errType != ERR_NONE) {
          if (!verbose) return dstErr;
          if (firstErr.errType == ERR_NONE) firstErr = dstErr;
        }
      }
    }
    return firstErr;
  }

  // Determine eligibility requirements for a particular GFX kernel
  static bool CanUseGfxKernel(int const                gpuKernelIdx,
                              ConfigOptions const&     cfg,
                              vector<Transfer> const&  transfers,
                              ExeInfo const&           exeInfo)
  {
    // GpuReduceKernel always works
    if (gpuKernelIdx == GFX_KERNEL_REDUCE) return true;

    // CopyKernel works if all Transfers have at most one SRC / one DST with no warp subexecutors
    if (gpuKernelIdx == GFX_KERNEL_COPY) {
      if (cfg.gfx.seType != 0) return false;
      if (exeInfo.resources.empty()) return false;
      for (auto const& rss : exeInfo.resources) {
        Transfer const& t = transfers[rss.transferIdx];
        if (t.srcs.size() > 1 || t.dsts.size() > 1) return false;
        if (cfg.gfx.useSingleTeam && t.numSubExecs > 1) return false;
      }
      return true;
    }

    return false;
  }

  static ErrResult SelectGfxKernel(ConfigOptions const& cfg, vector<Transfer> const& transfers, ExeInfo& exeInfo)
  {
    // Decide on which GFX kernel to use
    // Auto-select - prefer copyKernel if eligible
    if (cfg.gfx.gfxKernel == GFX_KERNEL_AUTO) {
      exeInfo.gfxKernelToUse = CanUseGfxKernel(GFX_KERNEL_COPY, cfg, transfers, exeInfo) ? 1 : 0;
    } else {
      exeInfo.gfxKernelToUse = cfg.gfx.gfxKernel;
    }

    // Warn if using forcing copy kernel, but allow kernel to continue
    if (cfg.gfx.gfxKernel == GFX_KERNEL_COPY && !CanUseGfxKernel(GFX_KERNEL_COPY, cfg, transfers, exeInfo)) {
      return {ERR_WARN,
        "GFX copy kernel forced even though deemed incompatible for current set of Transfers / config"};
    }
    return ERR_NONE;
  }

  static bool CanUseTdmKernel(int const                tdmKernelIdx,
                              ConfigOptions const&     cfg,
                              vector<Transfer> const&  transfers,
                              ExeInfo const&           exeInfo)
  {
    // Reduce kernel supports any number of inputs/outputs
    if (tdmKernelIdx == TDM_KERNEL_REDUCE) return true;

    // Copy kernel works if all Transfers have exactly one SRC / one DST
    if (tdmKernelIdx == TDM_KERNEL_COPY) {
      if (exeInfo.resources.empty()) return false;
      for (auto const& rss : exeInfo.resources) {
        Transfer const& t = transfers[rss.transferIdx];
        if (t.srcs.size() != 1 || t.dsts.size() != 1) return false;
      }
      return true;
    }

    return false;
  }

  static ErrResult SelectTdmKernel(ConfigOptions const& cfg, vector<Transfer> const& transfers, ExeInfo& exeInfo)
  {
    // Decide on which TDM kernel to use
    // Auto-select - prefer copy kernel if eligible, otherwise fall back to reduce
    if (cfg.tdm.tdmKernel == TDM_KERNEL_AUTO) {
      exeInfo.tdmKernelToUse = CanUseTdmKernel(TDM_KERNEL_COPY, cfg, transfers, exeInfo)
                             ? TDM_KERNEL_COPY : TDM_KERNEL_REDUCE;
    } else {
      exeInfo.tdmKernelToUse = cfg.tdm.tdmKernel;
    }

    // Warn if forcing copy kernel even though incompatible, but allow kernel to continue
    if (cfg.tdm.tdmKernel == TDM_KERNEL_COPY && !CanUseTdmKernel(TDM_KERNEL_COPY, cfg, transfers, exeInfo)) {
      return {ERR_WARN,
        "TDM copy kernel forced even though deemed incompatible for current set of Transfers / config"};
    }
    return ERR_NONE;
  }

// Preparation-related functions
//========================================================================================

  // Prepares input parameters for each subexecutor
  // Determines how sub-executors will split up the work
  // Initializes counters
  static ErrResult PrepareSubExecParams(ConfigOptions const& cfg,
                                        Transfer      const& transfer,
                                        TransferResources&   rss)
  {
    // Each subExecutor needs to know src/dst pointers and how many elements to transfer
    // Figure out the sub-array each subExecutor works on for this Transfer
    // - Partition N as evenly as possible, but try to keep subarray sizes as multiples of data.blockBytes
    //   except the very last one, for alignment reasons
    size_t const N              = transfer.numBytes   / sizeof(float);
    int    const initOffset     = cfg.data.byteOffset / sizeof(float);
    int    const targetMultiple = cfg.data.blockBytes / sizeof(float);

    // In some cases, there may not be enough data for all subExecutors
    int const maxSubExecToUse = std::min((size_t)(N + targetMultiple - 1) / targetMultiple,
                                         (size_t)transfer.numSubExecs);

    vector<SubExecParam>& subExecParam = rss.subExecParamCpu;
    subExecParam.clear();
    subExecParam.resize(transfer.numSubExecs);

    size_t assigned = 0;
    for (int i = 0; i < transfer.numSubExecs; ++i) {
      SubExecParam& p  = subExecParam[i];
      p.numSrcs        = rss.srcMem.size();
      p.numDsts        = rss.dstMem.size();
      p.startCycle     = 0;
      p.stopCycle      = 0;
      p.hwId           = 0;
      p.xccId          = 0;

      // In single team mode, subexecutors stripe across the entire array
      if (cfg.gfx.useSingleTeam && transfer.exeDevice.exeType == EXE_GPU_GFX) {
        p.N        = N;
        p.teamSize = transfer.numSubExecs;
        p.teamIdx  = i;
        for (int iSrc = 0; iSrc < p.numSrcs; ++iSrc) p.src[iSrc] = rss.srcMem[iSrc] + initOffset;
        for (int iDst = 0; iDst < p.numDsts; ++iDst) p.dst[iDst] = rss.dstMem[iDst] + initOffset;
      } else {
        // Otherwise, each subexecutor works on separate subarrays
        int    const subExecLeft = std::max(0, maxSubExecToUse - i);
        size_t const leftover    = N - assigned;
        size_t const roundedN    = (leftover + targetMultiple - 1) / targetMultiple;

        p.N        = subExecLeft ? std::min(leftover, ((roundedN / subExecLeft) * targetMultiple)) : 0;
        p.teamSize = 1;
        p.teamIdx  = 0;
        for (int iSrc = 0; iSrc < p.numSrcs; ++iSrc) p.src[iSrc] = rss.srcMem[iSrc] + initOffset + assigned;
        for (int iDst = 0; iDst < p.numDsts; ++iDst) p.dst[iDst] = rss.dstMem[iDst] + initOffset + assigned;
        assigned += p.N;
      }

      p.preferredXccId = transfer.exeSubIndex;
      // Override if XCC table has been specified
      vector<vector<int>> const& table = cfg.gfx.prefXccTable;
      if (transfer.exeDevice.exeType == EXE_GPU_GFX && transfer.exeSubIndex == -1 && !table.empty() &&
          transfer.dsts.size() == 1 && IsGpuMemType(transfer.dsts[0].memType)) {
        if (table.size() <= transfer.exeDevice.exeIndex ||
            table[transfer.exeDevice.exeIndex].size() <= transfer.dsts[0].memIndex) {
          return {ERR_FATAL, "[gfx.xccPrefTable] is too small"};
        }
        p.preferredXccId = table[transfer.exeDevice.exeIndex][transfer.dsts[0].memIndex];
        if (p.preferredXccId < 0 || p.preferredXccId >= GetNumExecutorSubIndices(transfer.exeDevice)) {
          return {ERR_FATAL, "[gfx.xccPrefTable] defines out-of-bound XCC index %d", p.preferredXccId};
        }
      }
    }

#ifdef BMA_EXEC_ENABLED
    // Prepare src/dst pointers for batched DMA executor
    rss.batchDsts.clear();
    rss.batchSrcs.clear();
    rss.batchBytes.clear();
    if (transfer.exeDevice.exeType == EXE_GPU_BDMA) {
      for (int i = 0; i < transfer.numSubExecs; ++i) {
        for (int j = 0; j < (int)rss.dstMem.size(); j++) {
          rss.batchSrcs.push_back(subExecParam[i].src[0]);
          rss.batchDsts.push_back(subExecParam[i].dst[j]);
          rss.batchBytes.push_back(subExecParam[i].N * sizeof(float));
        }
      }
    }
#endif

    // Clear counters
    rss.totalDurationMsec = 0.0;

    return ERR_NONE;
  }

  static ErrResult ExchangeMemory(MemDevice const& memDevice, ExeDevice const& exeDevice, size_t* pActualBytes,
                                  float** memPtr, hipMemGenericAllocationHandle_t* memHandle)
  {
    // Pass this pointer to all ranks (Used for pointer arithmetic, not defererenced on non-local ranks)
    // NOTE: This will be overwritten on executor rank if pod communication is required
    System::Get().Broadcast(memDevice.memRank, sizeof(*memPtr), memPtr);

    // Broadcast actualBytes from owning rank so importing rank gets the correct (rounded-up) size
    System::Get().Broadcast(memDevice.memRank, sizeof(*pActualBytes), pActualBytes);

    // If pod communication is required, export/import fabric handle
    if (memDevice.memRank != exeDevice.exeRank && IsGpuExeType(exeDevice.exeType)) {
#ifdef POD_COMM_ENABLED
      // mem rank exports to shareable fabric handle; broadcast handle + status so all
      // ranks fail together instead of hanging on the next collective if export fails
      hipMemFabricHandle_t fabricHandle = {};
      hipError_t exportErr = hipSuccess;
      const char* exportStep = "hipSetDevice";
      if (memDevice.memRank == GetRank()) {
        exportErr = hipSetDevice(memDevice.memIndex);
        if (exportErr == hipSuccess) {
          exportStep = "hipMemExportToShareableHandle";
          exportErr = hipMemExportToShareableHandle(&fabricHandle, *memHandle, hipMemHandleTypeFabric, 0);
        }
      }

      System::Get().Broadcast(memDevice.memRank, sizeof(hipMemFabricHandle_t), &fabricHandle);
      System::Get().Broadcast(memDevice.memRank, sizeof(hipError_t), &exportErr);
      if (exportErr != hipSuccess) {
        return {ERR_FATAL, "HIP Error in %s during fabric handle export: %s", exportStep, hipGetErrorString(exportErr)};
      }

      // exe rank imports the fabric handle; broadcast result so all ranks fail together
      hipError_t importErr = hipSuccess;
      const char* importStep = "hipSetDevice";
      if (exeDevice.exeRank == GetRank()) {
        importErr = hipSetDevice(exeDevice.exeIndex);
        if (importErr == hipSuccess) {
          importStep = "hipMemImportFromShareableHandle";
          importErr = hipMemImportFromShareableHandle(memHandle, (void*)&fabricHandle, hipMemHandleTypeFabric);
        }
        if (importErr == hipSuccess) {
          importStep = "hipMemAddressReserve";
          importErr = hipMemAddressReserve((gpu_device_ptr*)memPtr, *pActualBytes, 0, 0, 0);
        }
        if (importErr == hipSuccess) {
          importStep = "hipMemMap";
          importErr = hipMemMap((gpu_device_ptr)*memPtr, *pActualBytes, 0, *memHandle, 0);
        }
        if (importErr == hipSuccess) {
          importStep = "hipMemSetAccess";
          hipMemAccessDesc desc;
          desc.location = {hipMemLocationTypeDevice, exeDevice.exeIndex};
          desc.flags = hipMemAccessFlagsProtReadWrite;
          importErr = hipMemSetAccess((gpu_device_ptr)*memPtr, *pActualBytes, &desc, 1);
        }
      }
      System::Get().Broadcast(exeDevice.exeRank, sizeof(hipError_t), &importErr);
      if (importErr != hipSuccess) {
        return {ERR_FATAL, "HIP Error in %s during fabric handle import: %s", importStep, hipGetErrorString(importErr)};
      }
#else
      return {ERR_FATAL, "Unable to export/import fabric handle without compiling with pod communication support"};
#endif
    }
    return ERR_NONE;
  }

  // Prepare each executor
  // Allocates memory for src/dst, prepares subexecutors, executor-specific data structures
  static ErrResult PrepareExecutor(ConfigOptions    const& cfg,
                                   vector<Transfer> const& transfers,
                                   ExeDevice        const& exeDevice,
                                   ExeInfo&                exeInfo)
  {
    exeInfo.totalDurationMsec = 0.0;
    int const localRank = GetRank();
    bool const verbose  = System::Get().IsVerbose();
    // Executor BDF and NUMA are constant across all transfers — compute once
    std::string exeBdf = IsGpuExeType(exeDevice.exeType) ? GetGpuBdf(exeDevice.exeIndex) : "";
    int exeNuma = IsGpuExeType(exeDevice.exeType)
                  ? GetClosestCpuNumaToGpu(exeDevice.exeIndex, exeDevice.exeRank)
                  : IsNicExeType(exeDevice.exeType)
                    ? GetClosestCpuNumaToNic(exeDevice.exeIndex, exeDevice.exeRank)
                    : exeDevice.exeIndex;
    if (verbose) {
      System::Get().Log("[INFO] Rank %d preparing executor (R%d%c%d)\n",
                        localRank, exeDevice.exeRank, ExeTypeStr[exeDevice.exeType], exeDevice.exeIndex);
    }

    // Loop over each transfer this executor is involved in
    for (auto& rss : exeInfo.resources) {
      Transfer const& t = transfers[rss.transferIdx];
      rss.numBytes = t.numBytes;

      if (verbose) {
        System::Get().Log("[INFO] Rank %d preparing transfer %d (%lu SRC %lu DST) %zu bytes\n",
                          localRank, rss.transferIdx, t.srcs.size(), t.dsts.size(), t.numBytes);
        System::Get().Log("[INFO]   EXE: R%d%c%d NUMA %d%s%s\n",
                          exeDevice.exeRank, ExeTypeStr[exeDevice.exeType], exeDevice.exeIndex,
                          exeNuma,
                          exeBdf.empty() ? "" : " BDF ",
                          exeBdf.c_str());
      }

      // Allocate source memory
      rss.srcMem.resize(t.srcs.size());
      rss.srcActualBytes.resize(t.srcs.size());
      rss.srcMemHandle.resize(t.srcs.size(), NULL);
      for (int iSrc = 0; iSrc < t.srcs.size(); ++iSrc) {
        MemDevice const& srcMemDevice = t.srcs[iSrc];

        // Ensure executing GPU can access source memory
        // This only applies to memory being accessed by a local GPU executor
        if (IsGpuExeType(exeDevice.exeType)    &&
            IsGpuMemType(srcMemDevice.memType) &&
            srcMemDevice.memRank == localRank  &&
            exeDevice.exeRank    == localRank  &&
            srcMemDevice.memIndex != exeDevice.exeIndex) {
          if (verbose) {
            System::Get().Log("[INFO]   Enabling peer access: GPU %d -> GPU %d\n",
                              exeDevice.exeIndex, srcMemDevice.memIndex);
          }
          ERR_CHECK(EnablePeerAccess(exeDevice.exeIndex, srcMemDevice.memIndex));
        }

        // Allocate source memory (on the correct rank)
        bool requiresFabricHandle = (srcMemDevice.memRank != exeDevice.exeRank) && IsGpuExeType(exeDevice.exeType);
        if (srcMemDevice.memRank == localRank) {
          if (verbose) {
            std::string bdf = IsGpuMemType(srcMemDevice.memType) ? GetGpuBdf(srcMemDevice.memIndex) : "";
            System::Get().Log("[INFO]   SRC[%d]: %s idx=%d NUMA=%s%s%s (Rank %d) %zu bytes%s\n",
                              iSrc, GetMemTypeName(srcMemDevice.memType), srcMemDevice.memIndex,
                              GetMemDeviceNuma(srcMemDevice).c_str(),
                              bdf.empty() ? "" : " BDF ", bdf.c_str(),
                              srcMemDevice.memRank, t.numBytes + cfg.data.byteOffset,
                              requiresFabricHandle ? " [fabric-exportable]" : "");
          }
          ERR_CHECK(AllocateMemory(srcMemDevice, t.numBytes + cfg.data.byteOffset, (void**)&rss.srcMem[iSrc],
                                   &rss.srcActualBytes[iSrc], requiresFabricHandle ? &rss.srcMemHandle[iSrc] : nullptr));
          if (verbose) {
            System::Get().Log("[INFO]   SRC[%d]: allocated at %p\n", iSrc, rss.srcMem[iSrc]);
          }
        }

        // Exchange memory pointer across ranks
        ERR_CHECK(ExchangeMemory(srcMemDevice, exeDevice, &rss.srcActualBytes[iSrc],
                                 &rss.srcMem[iSrc], &rss.srcMemHandle[iSrc]));
      }

      // Allocate destination memory
      rss.dstMem.resize(t.dsts.size());
      rss.dstActualBytes.resize(t.dsts.size());
      rss.dstMemHandle.resize(t.dsts.size(), NULL);
      for (int iDst = 0; iDst < t.dsts.size(); ++iDst) {
        MemDevice const& dstMemDevice = t.dsts[iDst];

        // Ensure executing GPU can access destination memory
        if (IsGpuExeType(exeDevice.exeType)    &&
            IsGpuMemType(dstMemDevice.memType) &&
            dstMemDevice.memRank == localRank  &&
            exeDevice.exeRank    == localRank  &&
            dstMemDevice.memIndex != exeDevice.exeIndex) {
          if (verbose) {
            System::Get().Log("[INFO]   Enabling peer access: GPU %d -> GPU %d\n",
                              exeDevice.exeIndex, dstMemDevice.memIndex);
          }
          ERR_CHECK(EnablePeerAccess(exeDevice.exeIndex, dstMemDevice.memIndex));
        }

        // Allocate destination memory (on the correct rank)
        bool requiresFabricHandle = (dstMemDevice.memRank != exeDevice.exeRank) && IsGpuExeType(exeDevice.exeType);
        if (dstMemDevice.memRank == localRank) {
          if (verbose) {
            std::string bdf = IsGpuMemType(dstMemDevice.memType) ? GetGpuBdf(dstMemDevice.memIndex) : "";
            System::Get().Log("[INFO]   DST[%d]: %s idx=%d NUMA=%s%s%s (Rank %d) %zu bytes%s\n",
                              iDst, GetMemTypeName(dstMemDevice.memType), dstMemDevice.memIndex,
                              GetMemDeviceNuma(dstMemDevice).c_str(),
                              bdf.empty() ? "" : " BDF ", bdf.c_str(),
                              dstMemDevice.memRank, t.numBytes + cfg.data.byteOffset,
                              requiresFabricHandle ? " [fabric-exportable]" : "");
          }
          ERR_CHECK(AllocateMemory(dstMemDevice, t.numBytes + cfg.data.byteOffset, (void**)&rss.dstMem[iDst],
                                   &rss.dstActualBytes[iDst], requiresFabricHandle ? &rss.dstMemHandle[iDst] : NULL));
          if (verbose) {
            System::Get().Log("[INFO]   DST[%d]: allocated at %p\n", iDst, rss.dstMem[iDst]);
          }
        }

        // Exchange memory pointer across ranks
        ERR_CHECK(ExchangeMemory(dstMemDevice, exeDevice, &rss.dstActualBytes[iDst],
                                 &rss.dstMem[iDst], &rss.dstMemHandle[iDst]));
      }

      // Prepare HSA DMA copy specific resources
      if (exeDevice.exeType == EXE_GPU_DMA && (t.exeSubIndex != -1 || cfg.dma.useHsaCopy) && exeDevice.exeRank == localRank) {
#if !defined(__NVCC__)
        // Collect HSA agent information
        hsa_amd_pointer_info_t info;
        info.size = sizeof(info);
        int numDst = (int)rss.dstMem.size();
        rss.dstAgent.resize(numDst);
        for (int dstIdx = 0; dstIdx < numDst; dstIdx++) {
          ERR_CHECK(hsa_amd_pointer_info(rss.dstMem[dstIdx], &info, NULL, NULL, NULL));
          rss.dstAgent[dstIdx] = info.agentOwner;
        }

        ERR_CHECK(hsa_amd_pointer_info(rss.srcMem[0], &info, NULL, NULL, NULL));
        rss.srcAgent = info.agentOwner;

        // Create HSA completion signal
        ERR_CHECK(hsa_signal_create(1, 0, NULL, &rss.signal));

        if (t.exeSubIndex != -1)
          rss.sdmaEngineId = (hsa_amd_sdma_engine_id_t)(1U << t.exeSubIndex);
#endif
      }

      // Prepare subexecutor parameters (on all ranks)
      ERR_CHECK(PrepareSubExecParams(cfg, t, rss));
    }

    // Prepare additional requirements for GPU-based executors
    if (IsGpuExeType(exeDevice.exeType) && exeDevice.exeRank == localRank) {
      ERR_CHECK(hipSetDevice(exeDevice.exeIndex));

      // Determine how many streams to use
      int const numStreamsToUse = (exeDevice.exeType == EXE_GPU_DMA || exeDevice.exeType == EXE_GPU_BDMA ||
                                   (cfg.general.useMultiStream && (exeDevice.exeType == EXE_GPU_GFX ||
                                                                   exeDevice.exeType == EXE_GPU_TDM)))
                                  ? exeInfo.resources.size() : 1;
      exeInfo.streams.resize(numStreamsToUse);

      // Create streams
      for (int i = 0; i < numStreamsToUse; ++i) {
        if (cfg.gfx.cuMask.size()) {
#if !defined(__NVCC__)
          ERR_CHECK(hipExtStreamCreateWithCUMask(&exeInfo.streams[i], cfg.gfx.cuMask.size(),
                                                 cfg.gfx.cuMask.data()));
#else
          return {ERR_FATAL, "CU Masking in not supported on NVIDIA hardware"};
#endif
        } else {
          ERR_CHECK(hipStreamCreate(&exeInfo.streams[i]));
        }
      }

      if (cfg.general.useHipEvents) {
        exeInfo.startEvents.resize(numStreamsToUse);
        exeInfo.stopEvents.resize(numStreamsToUse);
        for (int i = 0; i < numStreamsToUse; ++i) {
          ERR_CHECK(hipEventCreate(&exeInfo.startEvents[i]));
          ERR_CHECK(hipEventCreate(&exeInfo.stopEvents[i]));
        }
      }

      // Determine how much shared memory to use for TDS
      if (exeDevice.exeType == EXE_GPU_TDM) {
        if (cfg.tdm.ldsBytes == 0) {
          int ldsMaxBytes;
          ERR_CHECK(hipDeviceGetAttribute(&ldsMaxBytes,
                                          hipDeviceAttributeMaxSharedMemoryPerBlock, exeDevice.exeIndex));
          exeInfo.ldsBytesActual = static_cast<uint32_t>(ldsMaxBytes);
        } else {
          exeInfo.ldsBytesActual = cfg.tdm.ldsBytes;
        }
      }
    }

    // Prepare for GPU GFX / TDM executor (both consume SubExecParam from GPU memory)
    if ((exeDevice.exeType == EXE_GPU_GFX || exeDevice.exeType == EXE_GPU_TDM) &&
        exeDevice.exeRank == localRank) {
      // Allocate one contiguous chunk of GPU memory for threadblock parameters
      // This allows support for executing one transfer per stream, or all transfers in a single stream
#if !defined(__NVCC__)
      MemType memType = MEM_GPU;      // AMD hardware can directly access GPU memory from host
#else
      MemType memType = MEM_MANAGED;  // NVIDIA hardware requires managed memory to access from host
#endif
      ERR_CHECK(AllocateMemory({memType, exeDevice.exeIndex}, exeInfo.totalSubExecs * sizeof(SubExecParam),
                               (void**)&exeInfo.subExecParamGpu));
      ERR_CHECK(GetMemHostAccessibility({memType, exeDevice.exeIndex}, exeInfo.subExecParamHostAccessible));

      // Create subexecutor parameter array for entire executor
      exeInfo.subExecParamCpu.clear();
      exeInfo.numSubIndices = GetNumExecutorSubIndices(exeDevice);
#if defined(__NVCC__)
      exeInfo.wallClockRate = 1000000;
#else
      ERR_CHECK(hipDeviceGetAttribute(&exeInfo.wallClockRate, hipDeviceAttributeWallClockRate,
                                      exeDevice.exeIndex));
#endif
      int transferOffset = 0;
      if (cfg.general.useMultiStream || cfg.gfx.blockOrder == 0) {
        // Threadblocks are ordered sequentially one transfer at a time
        for (auto& rss : exeInfo.resources) {
          rss.subExecParamGpuPtr = exeInfo.subExecParamGpu + transferOffset;
          for (auto p : rss.subExecParamCpu) {
            rss.subExecIdx.push_back(exeInfo.subExecParamCpu.size());
            exeInfo.subExecParamCpu.push_back(p);
            transferOffset++;
          }
        }
      } else if (cfg.gfx.blockOrder == 1) {
        // Interleave threadblocks of different Transfers
        for (int subExecIdx = 0; exeInfo.subExecParamCpu.size() < exeInfo.totalSubExecs; ++subExecIdx) {
          for (auto& rss : exeInfo.resources) {
            Transfer const& t = transfers[rss.transferIdx];
            if (subExecIdx < t.numSubExecs) {
              rss.subExecIdx.push_back(exeInfo.subExecParamCpu.size());
              exeInfo.subExecParamCpu.push_back(rss.subExecParamCpu[subExecIdx]);
            }
          }
        }
      } else if (cfg.gfx.blockOrder == 2) {
        // Build randomized threadblock list
        std::vector<std::pair<int,int>> indices;
        for (int i = 0; i < exeInfo.resources.size(); i++) {
          auto const& rss = exeInfo.resources[i];
          Transfer const& t = transfers[rss.transferIdx];
          for (int j = 0; j < t.numSubExecs; j++)
            indices.push_back(std::make_pair(i,j));
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);

        // Build randomized threadblock list
        for (auto p : indices) {
          auto& rss = exeInfo.resources[p.first];
          rss.subExecIdx.push_back(exeInfo.subExecParamCpu.size());
          exeInfo.subExecParamCpu.push_back(rss.subExecParamCpu[p.second]);
        }
      }

      // Copy sub executor parameters to GPU
      ERR_CHECK(hipSetDevice(exeDevice.exeIndex));
      if (verbose) {
        System::Get().Log("[INFO] SubExecParam upload: GPU%d  %zu bytes (%zu params)  host=%p dev=%p\n",
                          exeDevice.exeIndex,
                          exeInfo.totalSubExecs * sizeof(SubExecParam),
                          exeInfo.totalSubExecs,
                          exeInfo.subExecParamCpu.data(),
                          exeInfo.subExecParamGpu);
      }
      ERR_CHECK(hipMemcpy(exeInfo.subExecParamGpu,
                          exeInfo.subExecParamCpu.data(),
                          exeInfo.totalSubExecs * sizeof(SubExecParam),
                          hipMemcpyHostToDevice));
      ERR_CHECK(hipDeviceSynchronize());
    }

    // Prepare for NIC-based executors
    if (IsNicExeType(exeDevice.exeType)) {
      if (IsIbvSymbolsReady()) {
        for (auto& rss : exeInfo.resources) {
          Transfer const& t = transfers[rss.transferIdx];
          ERR_CHECK(PrepareNicTransferResources(cfg, exeDevice, t, rss));
        }
      } else {
        return {ERR_FATAL, "RDMA executor is not supported"};
      }
    }

    // Check that GPU wallclock rate is non-zero (GFX and TDM both use it for in-kernel cycle timing)
    if ((exeDevice.exeType == EXE_GPU_GFX || exeDevice.exeType == EXE_GPU_TDM) &&
        exeInfo.wallClockRate == 0 && exeDevice.exeRank == localRank) {
      if (getenv("TB_WALLCLOCK_RATE")) {
        exeInfo.wallClockRate = atoi(getenv("TB_WALLCLOCK_RATE"));
        return {ERR_WARN,
          "GPU %d wallclock rate query returned 0 unexpectedly.  Setting to %d instead as specified by TB_WALLCLOCK_RATE",
          exeDevice.exeIndex, exeInfo.wallClockRate};
      } else {
        exeInfo.wallClockRate = 100000;
        /*
        return {ERR_WARN,
          "GPU %d wallclock rate query returned 0 unexpectedly.  Setting to %d instead.  Use TB_WALLCLOCK_RATE to customize",
          exeDevice.exeIndex, exeInfo.wallClockRate};
        */
      }
    }

    return ERR_NONE;
  }

// Teardown-related functions
//========================================================================================

  // Clean up all resources
  static ErrResult TeardownExecutor(ConfigOptions    const& cfg,
                                    ExeDevice        const& exeDevice,
                                    vector<Transfer> const& transfers,
                                    ExeInfo&                exeInfo)
  {
    int const localRank = GetRank();
    bool const verbose  = System::Get().IsVerbose();

    // Loop over each transfer this executor is involved in
    for (auto& rss : exeInfo.resources) {
      Transfer const& t = transfers[rss.transferIdx];

      if (verbose) {
        System::Get().Log("[INFO] Rank %d tearing down transfer %d\n", localRank, rss.transferIdx);
      }

      // Deallocate source memory
      for (int iSrc = 0; iSrc < t.srcs.size(); ++iSrc) {
        if (t.srcs[iSrc].memRank == localRank) {
          if (verbose) {
            System::Get().Log("[INFO]   Free SRC[%d]: %s idx=%d %p (%zu bytes)\n",
                              iSrc, GetMemTypeName(t.srcs[iSrc].memType),
                              t.srcs[iSrc].memIndex, rss.srcMem[iSrc], rss.srcActualBytes[iSrc]);
          }
          ERR_CHECK(DeallocateMemory(t.srcs[iSrc].memType, rss.srcMem[iSrc],
                                     rss.srcActualBytes[iSrc],
                                     &rss.srcMemHandle[iSrc]));
        } else if (exeDevice.exeRank == localRank && rss.srcMemHandle[iSrc] != 0) {
          if (verbose) {
            System::Get().Log("[INFO]   Unmap remote SRC[%d]: %p (%zu bytes) from Rank %d\n",
                              iSrc, rss.srcMem[iSrc], rss.srcActualBytes[iSrc], t.srcs[iSrc].memRank);
          }
#ifdef POD_COMM_ENABLED
          ERR_CHECK(hipMemUnmap((gpu_device_ptr)rss.srcMem[iSrc], rss.srcActualBytes[iSrc]));
          ERR_CHECK(hipMemRelease(rss.srcMemHandle[iSrc]));
          ERR_CHECK(hipMemAddressFree((gpu_device_ptr)rss.srcMem[iSrc], rss.srcActualBytes[iSrc]));
#endif
        }
      }

      // Deallocate destination memory
      for (int iDst = 0; iDst < t.dsts.size(); ++iDst) {
        if (t.dsts[iDst].memRank == localRank) {
          if (verbose) {
            System::Get().Log("[INFO]   Free DST[%d]: %s idx=%d %p (%zu bytes)\n",
                              iDst, GetMemTypeName(t.dsts[iDst].memType),
                              t.dsts[iDst].memIndex, rss.dstMem[iDst], rss.dstActualBytes[iDst]);
          }
          ERR_CHECK(DeallocateMemory(t.dsts[iDst].memType, rss.dstMem[iDst],
                                     rss.dstActualBytes[iDst],
                                     &rss.dstMemHandle[iDst]));
        } else if (exeDevice.exeRank == localRank && rss.dstMemHandle[iDst] != 0) {
          if (verbose) {
            System::Get().Log("[INFO]   Unmap remote DST[%d]: %p (%zu bytes) from Rank %d\n",
                              iDst, rss.dstMem[iDst], rss.dstActualBytes[iDst], t.dsts[iDst].memRank);
          }
#ifdef POD_COMM_ENABLED
          ERR_CHECK(hipMemUnmap((gpu_device_ptr)rss.dstMem[iDst], rss.dstActualBytes[iDst]));
          ERR_CHECK(hipMemRelease(rss.dstMemHandle[iDst]));
          ERR_CHECK(hipMemAddressFree((gpu_device_ptr)rss.dstMem[iDst], rss.dstActualBytes[iDst]));
#endif
        }
      }

      // Destroy HSA signal for DMA executor
#if !defined(__NVCC__)
      if (exeDevice.exeType == EXE_GPU_DMA && (t.exeSubIndex != -1 || cfg.dma.useHsaCopy) && exeDevice.exeRank == localRank) {
        ERR_CHECK(hsa_signal_destroy(rss.signal));
      }
#endif

      // Destroy NIC related resources
      if (IsIbvSymbolsReady() && IsNicExeType(exeDevice.exeType)) {
        ERR_CHECK(TeardownNicTransferResources(rss, t));
      }
    }

    // Teardown additional requirements for GPU-based executors
    if (IsGpuExeType(exeDevice.exeType) && exeDevice.exeRank == localRank) {
      for (auto stream : exeInfo.streams)
        ERR_CHECK(hipStreamDestroy(stream));
      for (auto event : exeInfo.startEvents)
        ERR_CHECK(hipEventDestroy(event));
      for (auto event : exeInfo.stopEvents)
        ERR_CHECK(hipEventDestroy(event));
    }

    if ((exeDevice.exeType == EXE_GPU_GFX || exeDevice.exeType == EXE_GPU_TDM) &&
        exeDevice.exeRank == localRank) {
#if !defined(__NVCC__)
      MemType memType = MEM_GPU;
#else
      MemType memType = MEM_MANAGED;
#endif
      ERR_CHECK(DeallocateMemory(memType, exeInfo.subExecParamGpu, exeInfo.totalSubExecs * sizeof(SubExecParam)));
    }

    return ERR_NONE;
  }

// CPU Executor-related functions
//========================================================================================

  // Kernel for CPU execution (run by a single subexecutor)
  static void CpuReduceKernel(SubExecParam const& p, int numSubIterations)
  {
    if (p.N == 0) return;

    int subIteration = 0;
    do {
      int const& numSrcs = p.numSrcs;
      int const& numDsts = p.numDsts;

      if (numSrcs == 0) {
        for (int i = 0; i < numDsts; ++i) {
          memset(p.dst[i], MEMSET_CHAR, p.N * sizeof(float));
          //for (int j = 0; j < p.N; j++) p.dst[i][j] = MEMSET_VAL;
        }
      } else if (numSrcs == 1) {
        float const* __restrict__ src = p.src[0];
        if (numDsts == 0) {
          float sum = 0.0;
          for (int j = 0; j < p.N; j++)
            sum += p.src[0][j];

          // Add a dummy check to ensure the read is not optimized out
          if (sum != sum) {
            System::Get().Log("[ERROR] Nan detected\n");
          }
        } else {
          for (int i = 0; i < numDsts; ++i)
            memcpy(p.dst[i], src, p.N * sizeof(float));
        }
      } else {
        float sum = 0.0f;
        for (int j = 0; j < p.N; j++) {
          sum = p.src[0][j];
          for (int i = 1; i < numSrcs; i++) sum += p.src[i][j];
          for (int i = 0; i < numDsts; i++) p.dst[i][j] = sum;
        }
      }
    } while (++subIteration != numSubIterations);
  }

  // Execution of a single CPU Transfers
  static void ExecuteCpuTransfer(int           const  iteration,
                                 ConfigOptions const& cfg,
                                 int           const  exeIndex,
                                 TransferResources&   rss)
  {
    auto cpuStart = std::chrono::high_resolution_clock::now();
    vector<std::thread> childThreads;

    for (auto const& subExecParam : rss.subExecParamCpu)
      childThreads.emplace_back(std::thread(CpuReduceKernel, std::cref(subExecParam), cfg.general.numSubIterations));

    for (auto& subExecThread : childThreads)
      subExecThread.join();
    childThreads.clear();

    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double deltaMsec = (std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0) / cfg.general.numSubIterations;

    if (iteration >= 0) {
      rss.totalDurationMsec += deltaMsec;
      if (cfg.general.recordPerIteration)
        rss.perIterMsec.push_back(deltaMsec);
    }
  }

  // Execution of a single CPU executor
  static ErrResult RunCpuExecutor(int           const  iteration,
                                  ConfigOptions const& cfg,
                                  int           const  exeIndex,
                                  ExeInfo&             exeInfo)
  {
    numa_run_on_node(exeIndex);
    auto cpuStart = std::chrono::high_resolution_clock::now();

    vector<std::thread> asyncTransfers;
    for (auto& rss : exeInfo.resources) {
      asyncTransfers.emplace_back(std::thread(ExecuteCpuTransfer,
                                              iteration,
                                              std::cref(cfg),
                                              exeIndex,
                                              std::ref(rss)));
    }
    for (auto& asyncTransfer : asyncTransfers)
      asyncTransfer.join();

    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double deltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0 / cfg.general.numSubIterations;

    if (iteration >= 0)
      exeInfo.totalDurationMsec += deltaMsec;
    return ERR_NONE;
  }

  // Execution of a single NIC Transfer
  static ErrResult ExecuteNicTransfer(int           const  iteration,
                                      ConfigOptions const& cfg,
                                      int           const  exeIndex,
                                      TransferResources&   rss)
  {
    // Loop over each of the queue pairs and post work request
    ibv_send_wr* badWorkReq;
    for (int qpIndex = 0; qpIndex < rss.qpCount; qpIndex++) {
      size_t numChunks = rss.sendWorkRequests[qpIndex].size();
      for (size_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
        int error = ibv_post_send(rss.srcIsExeNic ? rss.srcQueuePairs[qpIndex] : rss.dstQueuePairs[qpIndex],
                                  &rss.sendWorkRequests[qpIndex][chunkIdx], &badWorkReq);
        if (error)
          return {ERR_FATAL, "Transfer %d: Error when calling ibv_post_send for QP %d chunk %lu of %lu (Error code %d = %s)\n",
            rss.transferIdx, qpIndex, chunkIdx, numChunks, error, strerror(error)};
      }
    }
    return ERR_NONE;
  }

  // Execution of a single NIC executor
  static ErrResult RunNicExecutor(int           const  iteration,
                                  ConfigOptions const& cfg,
                                  int           const  exeIndex,
                                  ExeInfo&             exeInfo)
  {
    // Switch to the closest NUMA node to this NIC
    if (cfg.nic.useNuma) {
      int numaNode = GetIbvDeviceList()[exeIndex].numaNode;
      if (numaNode != -1)
        numa_run_on_node(numaNode);
    }

    auto transferCount = exeInfo.resources.size();
    std::vector<double> totalTimeMsec(transferCount, 0.0);

    int subIterations = 0;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    std::vector<std::chrono::high_resolution_clock::time_point> transferTimers(transferCount);

    do {
      std::vector<uint32_t> receivedQPs(transferCount, 0);
      // post the sends
      for (auto i = 0; i < transferCount; i++) {
        transferTimers[i] = std::chrono::high_resolution_clock::now();
        ERR_CHECK(ExecuteNicTransfer(iteration, cfg, exeIndex, exeInfo.resources[i]));
      }
      // poll for completions
      size_t completedTransfers = 0;
      int pollBatch = std::max(1, cfg.nic.cqPollBatch);
      std::vector<ibv_wc> wc((size_t)pollBatch);
      ibv_wc* wc_array = wc.data();
      while (completedTransfers < transferCount) {
        for (auto i = 0; i < transferCount; i++) {
          if(receivedQPs[i] < exeInfo.resources[i].qpCount) {
            auto& rss = exeInfo.resources[i];
            // Poll the completion queue until all queue pairs are complete
            // The order of completion doesn't matter because this completion queue is dedicated to this Transfer
            // Use batch polling to drain multiple completions at once for better efficiency
            int nc = ibv_poll_cq(rss.srcIsExeNic ? rss.srcCompQueue : rss.dstCompQueue, pollBatch, wc_array);
            if (nc > 0) {
              // Process all completions in the batch
              for (int j = 0; j < nc; j++) {
                if (wc_array[j].status != IBV_WC_SUCCESS) {
                  return {ERR_FATAL, "Transfer %d: Received unsuccessful work completion [status code %d]", rss.transferIdx, wc_array[j].status};
                }
                receivedQPs[i]++;
              }
            } else if (nc < 0) {
              return {ERR_FATAL, "Transfer %d: Received negative work completion", rss.transferIdx};
            }
            if(receivedQPs[i] == rss.qpCount) {
              auto cpuDelta = std::chrono::high_resolution_clock::now() - transferTimers[i];
              double deltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0;
              if (iteration >= 0) {
                totalTimeMsec[i] += deltaMsec;
              }
              completedTransfers++;
            }
          }
        }
      }
    } while(++subIterations < cfg.general.numSubIterations);

    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double deltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0 / cfg.general.numSubIterations;

    if (iteration >= 0) {
      exeInfo.totalDurationMsec += deltaMsec;
      for (int i = 0; i < transferCount; i++) {
        auto& rss = exeInfo.resources[i];
        double transferTimeMsec = totalTimeMsec[i] / cfg.general.numSubIterations;
        rss.totalDurationMsec += transferTimeMsec;
        if (cfg.general.recordPerIteration)
          rss.perIterMsec.push_back(transferTimeMsec);
      }
    }
    return ERR_NONE;
  }

// GFX Executor-related functions
//========================================================================================

  // Converts register value to a CU/SM index
  static uint32_t GetId(uint32_t hwId)
  {
#if defined(__NVCC_)
    return hwId;
#else
    // Based on instinct-mi200-cdna2-instruction-set-architecture.pdf
    int const shId = (hwId >> 12) &  1;
    int const cuId = (hwId >>  8) & 15;
    int const seId = (hwId >> 13) &  3;
    return (shId << 5) + (cuId << 2) + seId;
#endif
  }

  // Device level timestamp function
  __device__ int64_t GetTimestamp()
  {
#if defined(__NVCC__)
    int64_t result;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(result));
    return result;
#else
    return wall_clock64();
#endif
  }

  // Helper function for memset
  template <typename T> __device__ __forceinline__ T      MemsetVal();
  template <>           __device__ __forceinline__ float  MemsetVal(){ return MEMSET_VAL; };
  template <>           __device__ __forceinline__ float2 MemsetVal(){ return make_float2(MEMSET_VAL,
                                                                                          MEMSET_VAL); };
  template <>           __device__ __forceinline__ float4 MemsetVal(){ return make_float4(MEMSET_VAL,
                                                                                          MEMSET_VAL,
                                                                                          MEMSET_VAL,
                                                                                          MEMSET_VAL); }


  // Helper function for temporal/non-temporal reads / writes
  #define TEMPORAL_NONE  0
  #define TEMPORAL_LOAD  1
  #define TEMPORAL_STORE 2
  #define TEMPORAL_BOTH  3

  template <int TEMPORAL_MODE>
  __device__ __forceinline__ void Load(float const* src, float& dst) {
    if (TEMPORAL_MODE & TEMPORAL_LOAD) {
#if !defined(__NVCC__)
      dst = __builtin_nontemporal_load(src);

#endif
    } else {
      dst = *src;
    }
  }

  template <int TEMPORAL_MODE>
  __device__ __forceinline__ void Load(float2 const* src, float2& dst) {
    if (TEMPORAL_MODE & TEMPORAL_LOAD) {
#if !defined(__NVCC__)
      dst.x = __builtin_nontemporal_load(&(src->x));
      dst.y = __builtin_nontemporal_load(&(src->y));
#endif
    } else {
      dst = *src;
    }
  }

  template <int TEMPORAL_MODE>
  __device__ __forceinline__ void Load(float4 const* src, float4& dst) {
    if (TEMPORAL_MODE & TEMPORAL_LOAD) {
#if !defined(__NVCC__)
      dst.x = __builtin_nontemporal_load(&(src->x));
      dst.y = __builtin_nontemporal_load(&(src->y));
      dst.z = __builtin_nontemporal_load(&(src->z));
      dst.w = __builtin_nontemporal_load(&(src->w));
#endif
    } else {
      dst = *src;
    }
  }

  template <int TEMPORAL_MODE>
  __device__ __forceinline__ void Store(float const& src, float* dst) {
    if (TEMPORAL_MODE & TEMPORAL_STORE) {
#if !defined(__NVCC__)
      __builtin_nontemporal_store(src, dst);
#endif
    } else {
      *dst = src;
    }
  }

  template <int TEMPORAL_MODE>
  __device__ __forceinline__ void Store(float2 const& src, float2* dst) {
    if (TEMPORAL_MODE & TEMPORAL_STORE) {
#if !defined(__NVCC__)
      __builtin_nontemporal_store(src.x, &(dst->x));
      __builtin_nontemporal_store(src.y, &(dst->y));
#endif
    } else {
      *dst = src;
    }
  }

  template <int TEMPORAL_MODE>
  __device__ __forceinline__ void Store(float4 const& src, float4* dst) {
    if (TEMPORAL_MODE & TEMPORAL_STORE) {
#if !defined(__NVCC__)
      __builtin_nontemporal_store(src.x, &(dst->x));
      __builtin_nontemporal_store(src.y, &(dst->y));
      __builtin_nontemporal_store(src.z, &(dst->z));
      __builtin_nontemporal_store(src.w, &(dst->w));
#endif
    } else {
      *dst = src;
    }
  }

  // Simplified Kernel for GFX execution for copies only
  template <typename PACKED_FLOAT, int LAUNCH_BOUND, int UNROLL, int TEMPORAL_MODE>
  __global__ void __launch_bounds__(LAUNCH_BOUND)
    GpuCopyKernel(SubExecParam* params, int seType, int waveOrder, int numSubIterations)
  {
    int64_t startCycle;
    // For warp-level, each warp's first thread records timing; for threadblock-level, only first thread of block
    bool shouldRecordTiming = (seType == 1) ? (threadIdx.x % warpSize == 0) : (threadIdx.x == 0);
    if (shouldRecordTiming) startCycle = GetTimestamp();

    // seType: 0=threadblock, 1=warp
    int subExecIdx;
    if (seType == 0) {
      // Threadblock-level: each threadblock is a subexecutor
      subExecIdx = blockIdx.y;
    } else {
      // Warp-level: each warp is a subexecutor
      int warpIdx       = threadIdx.x / warpSize;
      int warpsPerBlock = blockDim.x  / warpSize;
      subExecIdx = blockIdx.y * warpsPerBlock + warpIdx;
    }

    SubExecParam& p = params[subExecIdx];

    // For warp-level dispatch, inactive warps should return early
    if (seType == 1 && p.N == 0) return;

    // Filter by XCC
#if !defined(__NVCC__)
    int32_t xccId;
    GetXccId(xccId);
    if (p.preferredXccId != -1 && xccId != p.preferredXccId) return;
#endif

    // Collect data information
    bool hasSrc = p.numSrcs > 0;
    bool hasDst = p.numDsts > 0;
    PACKED_FLOAT const* __restrict__ srcFloatPacked = (PACKED_FLOAT const*)p.src[0];
    PACKED_FLOAT*       __restrict__ dstFloatPacked = (PACKED_FLOAT*)p.dst[0];

    // Operate on wavefront granularity
    int32_t const nTeams  = p.teamSize;             // Number of threadblocks working together on this subarray
    int32_t const teamIdx = p.teamIdx;              // Index of this threadblock within the team
    int32_t nWaves, waveIdx;
    if (seType == 0) {
      // Threadblock-level: all wavefronts in block work together
      nWaves  = blockDim.x  / warpSize;              // Number of wavefronts within this threadblock
      waveIdx = threadIdx.x / warpSize;              // Index of this wavefront within the threadblock
    } else {
      // Warp-level: each warp works independently
      nWaves  = 1;
      waveIdx = 0;
    }
    int32_t const tIdx = threadIdx.x % warpSize;     // Thread index within wavefront

    size_t  const numPackedFloat = p.N / (sizeof(PACKED_FLOAT)/sizeof(float));

    int32_t teamStride, waveStride, unrlStride, teamStride2, waveStride2;
    switch (waveOrder) {
    case 0: /* U,W,C */ unrlStride = 1; waveStride = UNROLL; teamStride = UNROLL * nWaves;  teamStride2 = nWaves; waveStride2 = 1     ; break;
    case 1: /* U,C,W */ unrlStride = 1; teamStride = UNROLL; waveStride = UNROLL * nTeams;  teamStride2 = 1;      waveStride2 = nTeams; break;
    case 2: /* W,U,C */ waveStride = 1; unrlStride = nWaves; teamStride = nWaves * UNROLL;  teamStride2 = nWaves; waveStride2 = 1     ; break;
    case 3: /* W,C,U */ waveStride = 1; teamStride = nWaves; unrlStride = nWaves * nTeams;  teamStride2 = nWaves; waveStride2 = 1     ; break;
    case 4: /* C,U,W */ teamStride = 1; unrlStride = nTeams; waveStride = nTeams * UNROLL;  teamStride2 = 1;      waveStride2 = nTeams; break;
    case 5: /* C,W,U */ teamStride = 1; waveStride = nTeams; unrlStride = nTeams * nWaves;  teamStride2 = 1;      waveStride2 = nTeams; break;
    }

    int subIterations = 0;
    while (1) {
      // First loop: Each wavefront in the team works on UNROLL PACKED_FLOAT per thread
      size_t const loop1Stride = nTeams * nWaves * UNROLL * warpSize;
      size_t const loop1Limit  = numPackedFloat / loop1Stride * loop1Stride;
      {
        PACKED_FLOAT val[UNROLL];
        if (!hasSrc) {
          #pragma unroll
          for (int u = 0; u < UNROLL; u++)
            val[u] = MemsetVal<PACKED_FLOAT>();
        }

        for (size_t idx = (teamIdx * teamStride + waveIdx * waveStride) * warpSize + tIdx; idx < loop1Limit; idx += loop1Stride) {
          // Read sources into memory and accumulate in registers
          if (hasSrc) {
            #pragma unroll
            for (int u = 0; u < UNROLL; u++)
              Load<TEMPORAL_MODE>(&srcFloatPacked[idx + u * unrlStride * warpSize], val[u]);
          }

          // Write accumulation to all outputs
          if (hasDst) {
            #pragma unroll
            for (int u = 0; u < UNROLL; u++)
              Store<TEMPORAL_MODE>(val[u], &dstFloatPacked[idx + u * unrlStride * warpSize]);
          }
        }
      }

      // Second loop: Deal with remaining PACKED_FLOAT
      {
        if (loop1Limit < numPackedFloat) {
          PACKED_FLOAT val;
          if (!hasSrc) val = MemsetVal<PACKED_FLOAT>();

          size_t const loop2Stride = nTeams * nWaves * warpSize;
          for (size_t idx = loop1Limit + (teamIdx * teamStride2 + waveIdx * waveStride2) * warpSize + tIdx;
               idx < numPackedFloat; idx += loop2Stride) {
            if (hasSrc) {
              Load<TEMPORAL_MODE>(&srcFloatPacked[idx], val);
            }
            if (hasDst) {
              Store<TEMPORAL_MODE>(val, &dstFloatPacked[idx]);
            }
          }
        }
      }

      // Third loop; Deal with remaining floats
      {
        if (numPackedFloat * (sizeof(PACKED_FLOAT)/sizeof(float)) < p.N) {
          float val;
          if (!hasSrc) val = MemsetVal<float>();

          size_t const loop3Stride = nTeams * nWaves * warpSize;
          for (size_t idx = numPackedFloat * (sizeof(PACKED_FLOAT)/sizeof(float)) + (teamIdx * teamStride2 + waveIdx * waveStride2) * warpSize + tIdx; idx < p.N; idx += loop3Stride) {
            if (hasSrc) {
              Load<TEMPORAL_MODE>(&p.src[0][idx], val);
            }

            if (hasDst) {
              Store<TEMPORAL_MODE>(val, &p.dst[0][idx]);
            }
          }
        }
      }

      // Wait for all threads to finish this subiteration
      if (seType == 1) {
        // For warp-level, sync within warp only
#if defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
        __builtin_amdgcn_wave_barrier();
#else
        __syncwarp();
#endif
      } else {
        // For threadblock-level, sync all threads
        __syncthreads();
      }

      // Allows for numSubiterations == 0 to run infinitely
      if (++subIterations == numSubIterations) break;
    }

    if (shouldRecordTiming) {
      // Previous sync ensures data from all threads in this subexecutor has landed in cache
      // so only one thread needs to do system level threadfence.
      __threadfence_system();
      p.stopCycle  = GetTimestamp();
      p.startCycle = startCycle;
      GetHwId(p.hwId);
      GetXccId(p.xccId);
    }
  }

  // Kernel for GFX execution
  template <typename PACKED_FLOAT, int LAUNCH_BOUND, int UNROLL, int TEMPORAL_MODE>
  __global__ void __launch_bounds__(LAUNCH_BOUND)
    GpuReduceKernel(SubExecParam* params, int seType, int waveOrder, int numSubIterations)
  {
    int64_t startCycle;
    // For warp-level, each warp's first thread records timing; for threadblock-level, only first thread of block
    bool shouldRecordTiming = (seType == 1) ? (threadIdx.x % warpSize == 0) : (threadIdx.x == 0);
    if (shouldRecordTiming) startCycle = GetTimestamp();

    // seType: 0=threadblock, 1=warp
    int subExecIdx;
    if (seType == 0) {
      // Threadblock-level: each threadblock is a subexecutor
      subExecIdx = blockIdx.y;
    } else {
      // Warp-level: each warp is a subexecutor
      int warpIdx       = threadIdx.x / warpSize;
      int warpsPerBlock = blockDim.x  / warpSize;
      subExecIdx = blockIdx.y * warpsPerBlock + warpIdx;
    }

    SubExecParam& p = params[subExecIdx];

    // For warp-level dispatch, inactive warps should return early
    if (seType == 1 && p.N == 0) return;

    // Filter by XCC
#if !defined(__NVCC__)
    int32_t xccId;
    GetXccId(xccId);
    if (p.preferredXccId != -1 && xccId != p.preferredXccId) return;
#endif

    // Collect data information
    int32_t const  numSrcs  = p.numSrcs;
    int32_t const  numDsts  = p.numDsts;
    PACKED_FLOAT const* __restrict__ srcFloatPacked[MAX_SRCS];
    PACKED_FLOAT*       __restrict__ dstFloatPacked[MAX_DSTS];
    for (int i = 0; i < numSrcs; i++) srcFloatPacked[i] = (PACKED_FLOAT const*)p.src[i];
    for (int i = 0; i < numDsts; i++) dstFloatPacked[i] = (PACKED_FLOAT*)p.dst[i];

    // Operate on wavefront granularity
    int32_t const nTeams   = p.teamSize;             // Number of threadblocks working together on this subarray
    int32_t const teamIdx  = p.teamIdx;              // Index of this threadblock within the team
    int32_t nWaves, waveIdx;
    if (seType == 0) {
      // Threadblock-level: all wavefronts in block work together
      nWaves  = blockDim.x  / warpSize;              // Number of wavefronts within this threadblock
      waveIdx = threadIdx.x / warpSize;              // Index of this wavefront within the threadblock
    } else {
      // Warp-level: each warp works independently
      nWaves  = 1;
      waveIdx = 0;
    }
    int32_t const tIdx     = threadIdx.x % warpSize; // Thread index within wavefront

    size_t  const numPackedFloat = p.N / (sizeof(PACKED_FLOAT)/sizeof(float));

    int32_t teamStride, waveStride, unrlStride, teamStride2, waveStride2;
    switch (waveOrder) {
    case 0: /* U,W,C */ unrlStride = 1; waveStride = UNROLL; teamStride = UNROLL * nWaves;  teamStride2 = nWaves; waveStride2 = 1     ; break;
    case 1: /* U,C,W */ unrlStride = 1; teamStride = UNROLL; waveStride = UNROLL * nTeams;  teamStride2 = 1;      waveStride2 = nTeams; break;
    case 2: /* W,U,C */ waveStride = 1; unrlStride = nWaves; teamStride = nWaves * UNROLL;  teamStride2 = nWaves; waveStride2 = 1     ; break;
    case 3: /* W,C,U */ waveStride = 1; teamStride = nWaves; unrlStride = nWaves * nTeams;  teamStride2 = nWaves; waveStride2 = 1     ; break;
    case 4: /* C,U,W */ teamStride = 1; unrlStride = nTeams; waveStride = nTeams * UNROLL;  teamStride2 = 1;      waveStride2 = nTeams; break;
    case 5: /* C,W,U */ teamStride = 1; waveStride = nTeams; unrlStride = nTeams * nWaves;  teamStride2 = 1;      waveStride2 = nTeams; break;
    }

    int subIterations = 0;
    while (1) {
      // First loop: Each wavefront in the team works on UNROLL PACKED_FLOAT per thread
      size_t const loop1Stride = nTeams * nWaves * UNROLL * warpSize;
      size_t const loop1Limit  = numPackedFloat / loop1Stride * loop1Stride;
      {
        PACKED_FLOAT val[UNROLL];
        PACKED_FLOAT tmp[UNROLL];
        if (numSrcs == 0) {
          #pragma unroll
          for (int u = 0; u < UNROLL; u++)
            val[u] = MemsetVal<PACKED_FLOAT>();
        }

        for (size_t idx = (teamIdx * teamStride + waveIdx * waveStride) * warpSize + tIdx; idx < loop1Limit; idx += loop1Stride) {
          // Read sources into memory and accumulate in registers
          if (numSrcs) {
            #pragma unroll
            for (int u = 0; u < UNROLL; u++)
              Load<TEMPORAL_MODE>(&srcFloatPacked[0][idx + u * unrlStride * warpSize], val[u]);

            for (int s = 1; s < numSrcs; s++) {
              #pragma unroll
              for (int u = 0; u < UNROLL; u++)
                Load<TEMPORAL_MODE>(&srcFloatPacked[s][idx + u * unrlStride * warpSize], tmp[u]);
              #pragma unroll
              for (int u = 0; u < UNROLL; u++)
                val[u] += tmp[u];
            }
          }

          // Write accumulation to all outputs
          for (int d = 0; d < numDsts; d++) {
            #pragma unroll
            for (int u = 0; u < UNROLL; u++)
              Store<TEMPORAL_MODE>(val[u], &dstFloatPacked[d][idx + u * unrlStride * warpSize]);
          }
        }
      }

      // Second loop: Deal with remaining PACKED_FLOAT
      {
        if (loop1Limit < numPackedFloat) {
          PACKED_FLOAT val, tmp;
          if (numSrcs == 0) val = MemsetVal<PACKED_FLOAT>();

          size_t const loop2Stride = nTeams * nWaves * warpSize;
          for (size_t idx = loop1Limit + (teamIdx * teamStride2 + waveIdx * waveStride2) * warpSize + tIdx;
               idx < numPackedFloat; idx += loop2Stride) {
            if (numSrcs) {
              Load<TEMPORAL_MODE>(&srcFloatPacked[0][idx], val);
              for (int s = 1; s < numSrcs; s++) {
                Load<TEMPORAL_MODE>(&srcFloatPacked[s][idx], tmp);
                val += tmp;
              }
            }
            for (int d = 0; d < numDsts; d++)
              Store<TEMPORAL_MODE>(val, &dstFloatPacked[d][idx]);
          }
        }
      }

      // Third loop; Deal with remaining floats
      {
        if (numPackedFloat * (sizeof(PACKED_FLOAT)/sizeof(float)) < p.N) {
          float val, tmp;
          if (numSrcs == 0) val = MemsetVal<float>();

          size_t const loop3Stride = nTeams * nWaves * warpSize;
          for (size_t idx = numPackedFloat * (sizeof(PACKED_FLOAT)/sizeof(float)) + (teamIdx * teamStride2 + waveIdx * waveStride2) * warpSize + tIdx; idx < p.N; idx += loop3Stride) {
            if (numSrcs) {
              Load<TEMPORAL_MODE>(&p.src[0][idx], val);
              for (int s = 1; s < numSrcs; s++) {
                Load<TEMPORAL_MODE>(&p.src[s][idx], tmp);
                val += tmp;
              }
            }

            for (int d = 0; d < numDsts; d++)
              Store<TEMPORAL_MODE>(val, &p.dst[d][idx]);
          }
        }
      }

      // Wait for all threads to finish this subiteration
      if (seType == 1) {
        // For warp-level, sync within warp only
#if defined(__HIP_PLATFORM_AMD__) && (HIP_VERSION_MAJOR < 7)
        __builtin_amdgcn_wave_barrier();
#else
        __syncwarp();
#endif
      } else {
        // For threadblock-level, sync all threads
        __syncthreads();
      }

      // Allows for numSubiterations == 0 to run infinitely
      if (++subIterations == numSubIterations) break;
    }

    if (shouldRecordTiming) {
      // Previous sync ensures data from all threads in this subexecutor has landed in cache
      // so only one thread needs to do system level threadfence.
      __threadfence_system();
      p.stopCycle  = GetTimestamp();
      p.startCycle = startCycle;
      GetHwId(p.hwId);
      GetXccId(p.xccId);
    }
  }

  // Must match ordering in GfxKernelType
#define GPU_KERNEL_KERNEL_DECL(LAUNCH_BOUND, UNROLL, DWORD, TEMPORAL) \
  {GpuReduceKernel<DWORD, LAUNCH_BOUND, UNROLL, TEMPORAL>,            \
   GpuCopyKernel  <DWORD, LAUNCH_BOUND, UNROLL, TEMPORAL>}

  // Must match mapping in GetGpuKernelTemporalIdx
  constexpr int KERN_TEMPORALS = 4;
#define GPU_KERNEL_TEMPORAL_DECL(LAUNCH_BOUND, UNROLL, DWORD)           \
  {GPU_KERNEL_KERNEL_DECL(LAUNCH_BOUND, UNROLL, DWORD, TEMPORAL_NONE),  \
   GPU_KERNEL_KERNEL_DECL(LAUNCH_BOUND, UNROLL, DWORD, TEMPORAL_LOAD),  \
   GPU_KERNEL_KERNEL_DECL(LAUNCH_BOUND, UNROLL, DWORD, TEMPORAL_STORE), \
   GPU_KERNEL_KERNEL_DECL(LAUNCH_BOUND, UNROLL, DWORD, TEMPORAL_BOTH)}

  int GetGpuKernelTemporalIdx(int temporalMode) {
    if (temporalMode == TEMPORAL_NONE)  return 0;
    if (temporalMode == TEMPORAL_LOAD)  return 1;
    if (temporalMode == TEMPORAL_STORE) return 2;
    if (temporalMode == TEMPORAL_BOTH)  return 3;
    return -1;
  }

  // Must match mapping in GetGpuKernelWordsizeIdx
  constexpr int KERN_WORDSIZES = 3;
#define GPU_KERNEL_DWORD_DECL(LAUNCH_BOUND, UNROLL)        \
  {GPU_KERNEL_TEMPORAL_DECL(LAUNCH_BOUND, UNROLL, float),  \
   GPU_KERNEL_TEMPORAL_DECL(LAUNCH_BOUND, UNROLL, float2), \
   GPU_KERNEL_TEMPORAL_DECL(LAUNCH_BOUND, UNROLL, float4)}

  int GetGpuKernelWordsizeIdx(int wordsize) {
    if (wordsize == 1) return 0;
    if (wordsize == 2) return 1;
    if (wordsize == 4) return 2;
    return -1;
  }

  // Must match mapping in GetGpuKernelUnrollIdx
  constexpr int KERN_UNROLLS = 10;
#define GPU_KERNEL_UNROLL_DECL(LAUNCH_BOUND) \
  {GPU_KERNEL_DWORD_DECL(LAUNCH_BOUND,  1),  \
   GPU_KERNEL_DWORD_DECL(LAUNCH_BOUND,  2),  \
   GPU_KERNEL_DWORD_DECL(LAUNCH_BOUND,  3),  \
   GPU_KERNEL_DWORD_DECL(LAUNCH_BOUND,  4),  \
   GPU_KERNEL_DWORD_DECL(LAUNCH_BOUND,  5),  \
   GPU_KERNEL_DWORD_DECL(LAUNCH_BOUND,  6),  \
   GPU_KERNEL_DWORD_DECL(LAUNCH_BOUND,  7),  \
   GPU_KERNEL_DWORD_DECL(LAUNCH_BOUND,  8),  \
   GPU_KERNEL_DWORD_DECL(LAUNCH_BOUND, 16),  \
   GPU_KERNEL_DWORD_DECL(LAUNCH_BOUND, 32)}

  // Must match the unroll mapping in GPU_KERNEL_UNROLL_DECL
  int GetGpuKernelUnrollIdx(int unroll) {
    if (1 <= unroll && unroll <= 8) return unroll - 1;
    if (unroll == 16)               return 8;
    if (unroll == 32)               return 9;
    return -1;
  }

  // Table of all GPU Reduction kernel functions (templated blocksize / unroll / dword size / temporal / kernel)
  typedef void (*GpuKernelFuncPtr)(SubExecParam*, int, int, int);
  constexpr int KERN_BOUNDS = 4;
#ifndef SINGLE_KERNEL
  GpuKernelFuncPtr GpuKernelsTable[KERN_BOUNDS][KERN_UNROLLS][KERN_WORDSIZES][KERN_TEMPORALS][NUM_GFX_KERNELS] =
  {
    GPU_KERNEL_UNROLL_DECL(256),
    GPU_KERNEL_UNROLL_DECL(512),
    GPU_KERNEL_UNROLL_DECL(768),
    GPU_KERNEL_UNROLL_DECL(1024),
  };
#endif

  #undef GPU_KERNEL_UNROLL_DECL
  #undef GPU_KERNEL_DWORD_DECL
  #undef GPU_KERNEL_TEMPORAL_DECL

  int GetGpuKernelBlocksizeIdx(int blocksize) {
    return (blocksize + 255) / 256 - 1;
  }

  // Execute a single GPU Transfer (when using 1 stream per Transfer)
  static ErrResult ExecuteGpuTransfer(int           const  iteration,
                                      int           const  exeTotalSubExecs,
                                      SubExecParam*        exeSubExecParam,
                                      hipStream_t   const  stream,
                                      hipEvent_t    const  startEvent,
                                      hipEvent_t    const  stopEvent,
                                      int           const  xccDim,
                                      ConfigOptions const& cfg,
                                      int           const  gfxKernelIdx,
                                      bool          const  subExecParamHostAccessible,
                                      TransferResources&   rss)
  {
    // Determine which kernel to launch
    [[maybe_unused]] int const blockSizeIdx = GetGpuKernelBlocksizeIdx(cfg.gfx.blockSize);
    [[maybe_unused]] int const unrollIdx    = GetGpuKernelUnrollIdx(cfg.gfx.unrollFactor);
    [[maybe_unused]] int const wordSizeIdx  = GetGpuKernelWordsizeIdx(cfg.gfx.wordSize);
    [[maybe_unused]] int const temporalIdx  = GetGpuKernelTemporalIdx(cfg.gfx.temporalMode);

#ifdef SINGLE_KERNEL
    auto gpuKernel = GpuReduceKernel<float4, 1024, 1, 0>;
#else
    auto gpuKernel = GpuKernelsTable[blockSizeIdx][unrollIdx][wordSizeIdx][temporalIdx][gfxKernelIdx];
#endif

    // Compute kernel launch parameters
    int  const numSubExecs = cfg.general.useMultiStream ? rss.subExecParamCpu.size() : exeTotalSubExecs;
    int  const gridY = CalculateGridY(cfg.gfx.seType, cfg.gfx.blockSize, numSubExecs);
    dim3 const gridSize(xccDim, gridY, 1);
    dim3 const blockSize(cfg.gfx.blockSize);

    auto cpuStart = std::chrono::high_resolution_clock::now();

    SubExecParam* params = cfg.general.useMultiStream ? rss.subExecParamGpuPtr : exeSubExecParam;

#if defined(__NVCC__)
    if (cfg.general.useHipEvents)
      ERR_CHECK(hipEventRecord(startEvent, stream));
    gpuKernel<<<gridSize, blockSize, 0 , stream>>>(params, cfg.gfx.seType, cfg.gfx.waveOrder, cfg.general.numSubIterations);
    if (cfg.general.useHipEvents)
      ERR_CHECK(hipEventRecord(stopEvent, stream));
#else
    hipExtLaunchKernelGGL(gpuKernel, gridSize, blockSize, 0, stream,
                          cfg.general.useHipEvents ? startEvent : NULL,
                          cfg.general.useHipEvents ? stopEvent  : NULL, 0,
                          params, cfg.gfx.seType, cfg.gfx.waveOrder, cfg.general.numSubIterations);
#endif

    ERR_CHECK(hipStreamSynchronize(stream));

    // Record this timing if this Transfer is being run in multistream mode
    if (cfg.general.useMultiStream) {
      if (iteration >= 0) {
        auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
        double cpuDeltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0 / cfg.general.numSubIterations;

        double deltaMsec = cpuDeltaMsec;
        if (startEvent != NULL) {
          float gpuDeltaMsec;
          ERR_CHECK(hipEventElapsedTime(&gpuDeltaMsec, startEvent, stopEvent));
          deltaMsec = gpuDeltaMsec / cfg.general.numSubIterations;
        }

        rss.totalDurationMsec += deltaMsec;
        if (cfg.general.recordPerIteration) {
          rss.perIterMsec.push_back(deltaMsec);
          std::set<std::pair<int,int>> CUs;
          std::vector<SubExecParam> subExecParamHost;
          SubExecParam const* subExecParam = rss.subExecParamGpuPtr;
          if (!subExecParamHostAccessible) {
            subExecParamHost.resize(numSubExecs);
            ERR_CHECK(hipMemcpy(subExecParamHost.data(), rss.subExecParamGpuPtr,
                                numSubExecs * sizeof(SubExecParam), hipMemcpyDefault));
            subExecParam = subExecParamHost.data();
          }
          for (int i = 0; i < numSubExecs; i++) {
            CUs.insert(std::make_pair(subExecParam[i].xccId,
                                      GetId(subExecParam[i].hwId)));
          }
          rss.perIterCUs.push_back(CUs);
        }
      }
    }
    return ERR_NONE;
  }

  // Execute a single GPU executor
  static ErrResult RunGpuExecutor(int           const  iteration,
                                  ConfigOptions const& cfg,
                                  int           const  exeIndex,
                                  ExeInfo&             exeInfo)
  {
    auto cpuStart = std::chrono::high_resolution_clock::now();
    ERR_CHECK(hipSetDevice(exeIndex));

    int xccDim = exeInfo.useSubIndices ? exeInfo.numSubIndices : 1;

    if (cfg.general.useMultiStream) {
      // Launch one thread per Transfer in separate streams
      vector<std::future<ErrResult>> asyncTransfers;
      for (int i = 0; i < exeInfo.streams.size(); i++) {
        asyncTransfers.emplace_back(std::async(std::launch::async,
                                               ExecuteGpuTransfer,
                                               iteration,
                                               exeInfo.totalSubExecs,
                                               exeInfo.subExecParamGpu,
                                               exeInfo.streams[i],
                                               cfg.general.useHipEvents ? exeInfo.startEvents[i] : NULL,
                                               cfg.general.useHipEvents ? exeInfo.stopEvents[i] : NULL,
                                               xccDim,
                                               std::cref(cfg),
                                               exeInfo.gfxKernelToUse,
                                               exeInfo.subExecParamHostAccessible,
                                               std::ref(exeInfo.resources[i])));
      }
      for (auto& asyncTransfer : asyncTransfers)
        ERR_CHECK(asyncTransfer.get());
    } else {
      // Launch all Transfers in one kernel launch (avoid extra thread creation)
      ExecuteGpuTransfer(iteration, exeInfo.totalSubExecs, exeInfo.subExecParamGpu, exeInfo.streams[0],
                         cfg.general.useHipEvents ? exeInfo.startEvents[0] : NULL,
                         cfg.general.useHipEvents ? exeInfo.stopEvents[0] : NULL,
                         xccDim, cfg, exeInfo.gfxKernelToUse,
                         exeInfo.subExecParamHostAccessible, exeInfo.resources[0]);
    }

    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;

    if (iteration >= 0) {
      // Determine executor timing
      // - Use HIP event timing if enabled and not using multi-stream
      // - Otherwise, Use CPU timing
      if (cfg.general.useHipEvents && !cfg.general.useMultiStream) {
        float gpuDeltaMsec;
        ERR_CHECK(hipEventElapsedTime(&gpuDeltaMsec, exeInfo.startEvents[0], exeInfo.stopEvents[0]));
        gpuDeltaMsec /= cfg.general.numSubIterations;
        exeInfo.totalDurationMsec += gpuDeltaMsec;
      } else {
        double cpuDeltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0
          / cfg.general.numSubIterations;
        exeInfo.totalDurationMsec += cpuDeltaMsec;
      }

      // If Transfers were combined into a single launch, figure out per-Transfer timing
      // Determine timing for each of the individual transfers that were part of this launch
      if (!cfg.general.useMultiStream) {
        std::vector<SubExecParam> subExecParamHost;
        SubExecParam const* subExecParam = exeInfo.subExecParamGpu;
        if (!exeInfo.subExecParamHostAccessible) {
          subExecParamHost.resize(exeInfo.totalSubExecs);
          ERR_CHECK(hipMemcpy(subExecParamHost.data(), exeInfo.subExecParamGpu,
                              exeInfo.totalSubExecs * sizeof(SubExecParam), hipMemcpyDefault));
          subExecParam = subExecParamHost.data();
        }

        for (int i = 0; i < exeInfo.resources.size(); i++) {
          TransferResources& rss = exeInfo.resources[i];
          int64_t minStartCycle = std::numeric_limits<int64_t>::max();
          int64_t maxStopCycle  = std::numeric_limits<int64_t>::min();
          std::set<std::pair<int, int>> CUs;

          for (auto subExecIdx : rss.subExecIdx) {
            if (exeInfo.subExecParamCpu[subExecIdx].N != 0) {
              minStartCycle = std::min(minStartCycle, subExecParam[subExecIdx].startCycle);
              maxStopCycle  = std::max(maxStopCycle,  subExecParam[subExecIdx].stopCycle);
              if (cfg.general.recordPerIteration) {
                CUs.insert(std::make_pair(subExecParam[subExecIdx].xccId,
                                          GetId(subExecParam[subExecIdx].hwId)));
              }
            }
          }

          double deltaMsec = (maxStopCycle - minStartCycle) / (double)(exeInfo.wallClockRate);
          deltaMsec /= cfg.general.numSubIterations;
          rss.totalDurationMsec += deltaMsec;
          if (cfg.general.recordPerIteration) {
            rss.perIterMsec.push_back(deltaMsec);
            rss.perIterCUs.push_back(CUs);
          }
        }
      }
    }
    return ERR_NONE;
  }

// DMA Executor-related functions
//========================================================================================

  // Execute a single DMA Transfer
  static ErrResult ExecuteDmaTransfer(int           const  iteration,
                                      bool          const  useSubIndices,
                                      int           const  exeIndex,
                                      hipStream_t   const  stream,
                                      hipEvent_t    const  startEvent,
                                      hipEvent_t    const  stopEvent,
                                      ConfigOptions const& cfg,
                                      TransferResources&   resources)
  {
    auto cpuStart = std::chrono::high_resolution_clock::now();

    int numDsts = (int)resources.dstMem.size();
    ERR_CHECK(hipSetDevice(exeIndex));
    int subIterations = 0;
    size_t const initOffset = cfg.data.byteOffset / sizeof(float);
    float* const src = resources.srcMem[0] + initOffset;
    if (!useSubIndices && !cfg.dma.useHsaCopy) {
      if (cfg.general.useHipEvents)
        ERR_CHECK(hipEventRecord(startEvent, stream));

      // Force the use of SDMA engine if possible
#if defined(__HIP_PLATFORM_AMD__) && defined(HIP_VERSION_MAJOR) && (HIP_VERSION_MAJOR >= 6)
      hipMemcpyKind memcpyKind = hipMemcpyDeviceToDeviceNoCU;
#endif

      // Use DMA copy engine
      do {
        // Queue for each output location
        for (int dstIdx = 0; dstIdx < numDsts; dstIdx++) {
          float* const dst = resources.dstMem[dstIdx] + initOffset;
#if defined(CUMEM_ENABLED)
          ERR_CHECK(cuMemcpyAsync((CUdeviceptr)dst,
                                  (CUdeviceptr)src,
                                  resources.numBytes, stream));
#else
          ERR_CHECK(hipMemcpyAsync(dst, src, resources.numBytes,
                                   memcpyKind, stream));
#endif
        }
      } while (++subIterations != cfg.general.numSubIterations);

      if (cfg.general.useHipEvents)
        ERR_CHECK(hipEventRecord(stopEvent, stream));
      ERR_CHECK(hipStreamSynchronize(stream));
    } else {
#if defined(__NVCC__)
      return {ERR_FATAL, "HSA copy not supported on NVIDIA hardware"};
#else
      // Use HSA async copy
      do {
        hsa_signal_store_screlease(resources.signal, numDsts);
        for (int dstIdx = 0; dstIdx < numDsts; dstIdx++) {
          float* const dst = resources.dstMem[dstIdx] + initOffset;
          if (!useSubIndices) {
            ERR_CHECK(hsa_amd_memory_async_copy(dst, resources.dstAgent[dstIdx],
                                                src, resources.srcAgent,
                                                resources.numBytes, 0, NULL,
                                                resources.signal));
          } else {
            HSA_CALL(hsa_amd_memory_async_copy_on_engine(dst, resources.dstAgent[dstIdx],
                                                         src, resources.srcAgent,
                                                         resources.numBytes, 0, NULL,
                                                         resources.signal,
                                                         resources.sdmaEngineId, true));
          }
        }
        // Wait for SDMA transfer(s) to complete
        while(hsa_signal_wait_scacquire(resources.signal,
                                        HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                        HSA_WAIT_STATE_ACTIVE) >= 1);
      } while (++subIterations != cfg.general.numSubIterations);
#endif
    }
    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double cpuDeltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0 / cfg.general.numSubIterations;

    if (iteration >= 0) {
      double deltaMsec = cpuDeltaMsec;
      if (!useSubIndices && !cfg.dma.useHsaCopy && cfg.general.useHipEvents) {
        float gpuDeltaMsec;
        ERR_CHECK(hipEventElapsedTime(&gpuDeltaMsec, startEvent, stopEvent));
        deltaMsec = gpuDeltaMsec / cfg.general.numSubIterations;
      }
      resources.totalDurationMsec += deltaMsec;
      if (cfg.general.recordPerIteration)
        resources.perIterMsec.push_back(deltaMsec);
    }
    return ERR_NONE;
  }

  // Execute a single DMA executor
  static ErrResult RunDmaExecutor(int           const  iteration,
                                  ConfigOptions const& cfg,
                                  int           const  exeIndex,
                                  ExeInfo&             exeInfo)
  {
    auto cpuStart = std::chrono::high_resolution_clock::now();
    ERR_CHECK(hipSetDevice(exeIndex));

    vector<std::future<ErrResult>> asyncTransfers;
    for (int i = 0; i < exeInfo.resources.size(); i++) {
      asyncTransfers.emplace_back(std::async(std::launch::async,
                                             ExecuteDmaTransfer,
                                             iteration,
                                             exeInfo.useSubIndices,
                                             exeIndex,
                                             exeInfo.streams[i],
                                             cfg.general.useHipEvents ? exeInfo.startEvents[i] : NULL,
                                             cfg.general.useHipEvents ? exeInfo.stopEvents[i]  : NULL,
                                             std::cref(cfg),
                                             std::ref(exeInfo.resources[i])));
    }

    for (auto& asyncTransfer : asyncTransfers)
      ERR_CHECK(asyncTransfer.get());

    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double deltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0 / cfg.general.numSubIterations;
    if (iteration >= 0)
      exeInfo.totalDurationMsec += deltaMsec;
    return ERR_NONE;
  }

// BMA Executor-related functions
//========================================================================================
#ifdef BMA_EXEC_ENABLED
  // Execute a single BMA Transfer (one hipMemcpyBatchAsync per sub-iteration; each subexecutor is one batch entry)
  static ErrResult ExecuteBatchDmaTransfer(int           const  iteration,
                                           int           const  exeIndex,
                                           hipStream_t   const  stream,
                                           hipEvent_t    const  startEvent,
                                           hipEvent_t    const  stopEvent,
                                           ConfigOptions const& cfg,
                                           TransferResources&   resources)
  {
    auto cpuStart = std::chrono::high_resolution_clock::now();

    ERR_CHECK(hipSetDevice(exeIndex));

    int subIterations = 0;
    if (cfg.general.useHipEvents)
      ERR_CHECK(hipEventRecord(startEvent, stream));

    [[maybe_unused]] size_t failIdx = 0;
    do {
      ERR_CHECK(hipMemcpyBatchAsync(resources.batchDsts.data(),
                                    resources.batchSrcs.data(),
                                    resources.batchBytes.data(),
                                    resources.batchDsts.size(),
                                    nullptr, nullptr, 0,
    // In CUDA 13.0 the failIdx argument was removed from the original CUDA 12.8 API call
#if !defined(__NVCC__) || (defined(CUDA_VERSION) && (CUDA_VERSION < 13000))
                                    &failIdx,
#endif
                                    stream));
    } while (++subIterations != cfg.general.numSubIterations);

    if (cfg.general.useHipEvents)
      ERR_CHECK(hipEventRecord(stopEvent, stream));
    ERR_CHECK(hipStreamSynchronize(stream));

    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double cpuDeltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0 / cfg.general.numSubIterations;

    if (iteration >= 0) {
      double deltaMsec = cpuDeltaMsec;
      if (cfg.general.useHipEvents) {
        float gpuDeltaMsec;
        ERR_CHECK(hipEventElapsedTime(&gpuDeltaMsec, startEvent, stopEvent));
        deltaMsec = gpuDeltaMsec / cfg.general.numSubIterations;
      }
      resources.totalDurationMsec += deltaMsec;
      if (cfg.general.recordPerIteration)
        resources.perIterMsec.push_back(deltaMsec);
    }
    return ERR_NONE;
  }

  static ErrResult RunBmaExecutor(int           const  iteration,
                                  ConfigOptions const& cfg,
                                  int           const  exeIndex,
                                  ExeInfo&             exeInfo)
  {
    auto cpuStart = std::chrono::high_resolution_clock::now();
    ERR_CHECK(hipSetDevice(exeIndex));

    vector<std::future<ErrResult>> asyncTransfers;
    for (int i = 0; i < exeInfo.resources.size(); i++) {
      asyncTransfers.emplace_back(std::async(std::launch::async,
                                             ExecuteBatchDmaTransfer,
                                             iteration,
                                             exeIndex,
                                             exeInfo.streams[i],
                                             cfg.general.useHipEvents ? exeInfo.startEvents[i] : NULL,
                                             cfg.general.useHipEvents ? exeInfo.stopEvents[i]  : NULL,
                                             std::cref(cfg),
                                             std::ref(exeInfo.resources[i])));
    }

    for (auto& asyncTransfer : asyncTransfers)
      ERR_CHECK(asyncTransfer.get());

    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double deltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0 / cfg.general.numSubIterations;
    if (iteration >= 0)
      exeInfo.totalDurationMsec += deltaMsec;
    return ERR_NONE;
  }
#endif // BMA_EXEC_ENABLED

// TDM Executor-related functions
//========================================================================================
#if TDM_SUPPORTED
  // Copy kernel: single SRC -> single DST via tensor-DMA staged through shared memory
  __global__ void  TdmCopyKernel(SubExecParam* params,
                                uint32_t      ldsBytes,
                                int           numSubIterations)
  {
    int64_t startCycle;
    bool const shouldRecordTiming = (threadIdx.x == 0);
    if (shouldRecordTiming) startCycle = GetTimestamp();

    extern __shared__ __align__(128) float shmem[];

    // Each threadblock is a subexecutor (mirrors GpuCopyKernel).
    SubExecParam& p = params[blockIdx.x];
    if (p.N == 0) return;

    float const* __restrict__ src = (float const*)p.src[0];
    float*       __restrict__ dst = (float*)p.dst[0];
    size_t const sizeBytes        = p.N * sizeof(float);

    int subIterations = 0;
    while (1) {
      tdm::tdmCopy(dst, src, sizeBytes, shmem, ldsBytes);
      __syncthreads(); // Wait for all warps to finish
      if (++subIterations == numSubIterations) break;
    }

    if (shouldRecordTiming) {
      __threadfence_system();
      p.stopCycle  = GetTimestamp();
      p.startCycle = startCycle;
      GetHwId(p.hwId);
      GetXccId(p.xccId);
    }
  }

  // Reduce kernel: sum-reduce any number of SRCs into any number of DSTs
  __global__ void  TdmReduceKernel(SubExecParam* params,
                                   uint32_t      ldsBytes,
                                   int           numSubIterations)
  {
    int64_t startCycle;
    bool const shouldRecordTiming = (threadIdx.x == 0);
    if (shouldRecordTiming) startCycle = GetTimestamp();

    extern __shared__ __align__(128) float shmem[];

    // Each threadblock is a subexecutor (mirrors GpuCopyKernel).
    SubExecParam& p = params[blockIdx.x];
    if (p.N == 0) return;

    int32_t const numSrcs = p.numSrcs;
    int32_t const numDsts = p.numDsts;
    size_t const sizeBytes        = p.N * sizeof(float);

    int subIterations = 0;
    while (1) {
      tdm::tdmReduce(p.dst, p.src, numSrcs, numDsts, sizeBytes, shmem, ldsBytes);
      __syncthreads(); // Wait for all warps to finish this subiteration
      if (++subIterations == numSubIterations) break;
    }

    if (shouldRecordTiming) {
      __threadfence_system();
      p.stopCycle  = GetTimestamp();
      p.startCycle = startCycle;
      GetHwId(p.hwId);
      GetXccId(p.xccId);
    }
  }
#else
  // gfx1250 tensor TDM builtins unavailable for this translation: emit empty kernel stubs with
  // the exact launch signatures so the host-side launch path still links. They are never
  // dispatched on non-gfx1250 or nvidia hardware (see TransfersHaveErrors).
  __global__ void TdmCopyKernel(SubExecParam*, uint32_t, int) {}
  __global__ void TdmReduceKernel(SubExecParam*, uint32_t, int) {}
#endif // TDM_SUPPORTED

  // Table of all TDM kernel functions - must match ordering in TdmKernelType
  typedef void (*TdmKernelFuncPtr)(SubExecParam*, uint32_t, int);
  TdmKernelFuncPtr TdmKernelsTable[NUM_TDM_KERNELS] =
  {
    TdmCopyKernel,    // TDM_KERNEL_COPY
    TdmReduceKernel,  // TDM_KERNEL_REDUCE
  };

  static ErrResult ExecuteTdmTransfer(int           const  iteration,
                                      int           const  exeTotalSubExecs,
                                      SubExecParam*        exeSubExecParam,
                                      hipStream_t   const  stream,
                                      hipEvent_t    const  startEvent,
                                      hipEvent_t    const  stopEvent,
                                      ConfigOptions const& cfg,
                                      bool          const  subExecParamHostAccessible,
                                      uint32_t      const  ldsBytes,
                                      int           const  tdmKernelIdx,
                                      TransferResources&   rss)
  {
    // Compute kernel launch parameters
    int  const numSubExecs = cfg.general.useMultiStream ? rss.subExecParamCpu.size() : exeTotalSubExecs;
    dim3 const gridSize(numSubExecs);
    dim3 const blockSize(cfg.tdm.blockSize);
    SubExecParam* params = cfg.general.useMultiStream ? rss.subExecParamGpuPtr : exeSubExecParam;

    // Select which TDM kernel to launch (must match ordering in TdmKernelType)
    auto tdmKernel = TdmKernelsTable[tdmKernelIdx];

    auto cpuStart = std::chrono::high_resolution_clock::now();

#if defined(__NVCC__)
    if (cfg.general.useHipEvents)
      ERR_CHECK(hipEventRecord(startEvent, stream));
    tdmKernel<<<gridSize, blockSize, ldsBytes, stream>>>(params,
                                                         ldsBytes,
                                                         cfg.general.numSubIterations);
    if (cfg.general.useHipEvents)
      ERR_CHECK(hipEventRecord(stopEvent, stream));
#else
    hipExtLaunchKernelGGL(tdmKernel, gridSize, blockSize, (int)ldsBytes, stream,
                          startEvent, stopEvent, 0,
                          params, ldsBytes, cfg.general.numSubIterations);
#endif
    ERR_CHECK(hipStreamSynchronize(stream));

    // Record this timing if this Transfer is being run in multistream mode
    if (cfg.general.useMultiStream) {
      if (iteration >= 0) {
        auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
        double cpuDeltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0 / cfg.general.numSubIterations;

        double deltaMsec = cpuDeltaMsec;
        if (startEvent != NULL) {
          float gpuDeltaMsec;
          ERR_CHECK(hipEventElapsedTime(&gpuDeltaMsec, startEvent, stopEvent));
          deltaMsec = gpuDeltaMsec / cfg.general.numSubIterations;
        }

        rss.totalDurationMsec += deltaMsec;
        if (cfg.general.recordPerIteration) {
          rss.perIterMsec.push_back(deltaMsec);
          std::set<std::pair<int,int>> CUs;
          std::vector<SubExecParam> subExecParamHost;
          SubExecParam const* subExecParam = rss.subExecParamGpuPtr;
          if (!subExecParamHostAccessible) {
            subExecParamHost.resize(numSubExecs);
            ERR_CHECK(hipMemcpy(subExecParamHost.data(), rss.subExecParamGpuPtr,
                                numSubExecs * sizeof(SubExecParam), hipMemcpyDefault));
            subExecParam = subExecParamHost.data();
          }
          for (int i = 0; i < numSubExecs; i++) {
            CUs.insert(std::make_pair(subExecParam[i].xccId,
                                      GetId(subExecParam[i].hwId)));
          }
          rss.perIterCUs.push_back(CUs);
        }
      }
    }
    return ERR_NONE;
  }

  static ErrResult RunTdmExecutor(int           const  iteration,
                                  ConfigOptions const& cfg,
                                  int           const  exeIndex,
                                  ExeInfo&             exeInfo)
  {
    auto cpuStart = std::chrono::high_resolution_clock::now();
    ERR_CHECK(hipSetDevice(exeIndex));

    if (cfg.general.useMultiStream && exeInfo.streams.size() > 1 ) {
      // Launch one thread per Transfer in separate streams
      vector<std::future<ErrResult>> asyncTransfers;
      for (int i = 0; i < exeInfo.streams.size(); i++) {
        asyncTransfers.emplace_back(std::async(std::launch::async,
                                               ExecuteTdmTransfer,
                                               iteration,
                                               exeInfo.totalSubExecs,
                                               exeInfo.subExecParamGpu,
                                               exeInfo.streams[i],
                                               cfg.general.useHipEvents ? exeInfo.startEvents[i] : NULL,
                                               cfg.general.useHipEvents ? exeInfo.stopEvents[i] : NULL,
                                               std::cref(cfg),
                                               exeInfo.subExecParamHostAccessible,
                                               exeInfo.ldsBytesActual,
                                               exeInfo.tdmKernelToUse,
                                               std::ref(exeInfo.resources[i])));
      }
      for (auto& asyncTransfer : asyncTransfers)
        ERR_CHECK(asyncTransfer.get());
    } else {
      // Launch all Transfers in one kernel launch (avoid extra thread creation)
      ExecuteTdmTransfer(iteration, exeInfo.totalSubExecs, exeInfo.subExecParamGpu, exeInfo.streams[0],
                         cfg.general.useHipEvents ? exeInfo.startEvents[0] : NULL,
                         cfg.general.useHipEvents ? exeInfo.stopEvents[0] : NULL,
                         cfg, exeInfo.subExecParamHostAccessible, exeInfo.ldsBytesActual,
                         exeInfo.tdmKernelToUse, exeInfo.resources[0]);
    }

    if (iteration >= 0) {
      // Determine executor timing
      // - Use HIP event timing if enabled and not using multi-stream
      // - Otherwise, Use CPU timing
      if (cfg.general.useHipEvents && !cfg.general.useMultiStream) {
        float gpuDeltaMsec;
        ERR_CHECK(hipEventElapsedTime(&gpuDeltaMsec, exeInfo.startEvents[0], exeInfo.stopEvents[0]));
        gpuDeltaMsec /= cfg.general.numSubIterations;
        exeInfo.totalDurationMsec += gpuDeltaMsec;
      } else {
        auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
        double cpuDeltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0
          / cfg.general.numSubIterations;
        exeInfo.totalDurationMsec += cpuDeltaMsec;
      }

      // If Transfers were combined into a single launch, figure out per-Transfer timing
      // Determine timing for each of the individual transfers that were part of this launch
      if (!cfg.general.useMultiStream) {
        std::vector<SubExecParam> subExecParamHost;
        SubExecParam const* subExecParam = exeInfo.subExecParamGpu;
        if (!exeInfo.subExecParamHostAccessible) {
          subExecParamHost.resize(exeInfo.totalSubExecs);
          ERR_CHECK(hipMemcpy(subExecParamHost.data(), exeInfo.subExecParamGpu,
                              exeInfo.totalSubExecs * sizeof(SubExecParam), hipMemcpyDefault));
          subExecParam = subExecParamHost.data();
        }

        for (int i = 0; i < exeInfo.resources.size(); i++) {
          TransferResources& rss = exeInfo.resources[i];
          int64_t minStartCycle = std::numeric_limits<int64_t>::max();
          int64_t maxStopCycle  = std::numeric_limits<int64_t>::min();
          std::set<std::pair<int, int>> CUs;

          for (auto subExecIdx : rss.subExecIdx) {
            if (exeInfo.subExecParamCpu[subExecIdx].N != 0) {
              minStartCycle = std::min(minStartCycle, subExecParam[subExecIdx].startCycle);
              maxStopCycle  = std::max(maxStopCycle,  subExecParam[subExecIdx].stopCycle);
              if (cfg.general.recordPerIteration) {
                CUs.insert(std::make_pair(subExecParam[subExecIdx].xccId,
                                          GetId(subExecParam[subExecIdx].hwId)));
              }
            }
          }

          double deltaMsec = (maxStopCycle - minStartCycle) / (double)(exeInfo.wallClockRate);
          deltaMsec /= cfg.general.numSubIterations;
          rss.totalDurationMsec += deltaMsec;
          if (cfg.general.recordPerIteration) {
            rss.perIterMsec.push_back(deltaMsec);
            rss.perIterCUs.push_back(CUs);
          }
        }
      }
    }
    return ERR_NONE;
  }

// Executor-related functions
//========================================================================================
  static ErrResult RunExecutor(int           const  iteration,
                               ConfigOptions const& cfg,
                               ExeDevice     const& exeDevice,
                               ExeInfo&             exeInfo)
  {
    switch (exeDevice.exeType) {
    case EXE_CPU:           return RunCpuExecutor(iteration, cfg, exeDevice.exeIndex, exeInfo);
    case EXE_GPU_DMA:       return RunDmaExecutor(iteration, cfg, exeDevice.exeIndex, exeInfo);
    case EXE_GPU_GFX:       return RunGpuExecutor(iteration, cfg, exeDevice.exeIndex, exeInfo);
    case EXE_GPU_TDM:       return RunTdmExecutor(iteration, cfg, exeDevice.exeIndex, exeInfo);
    case EXE_NIC:           return RunNicExecutor(iteration, cfg, exeDevice.exeIndex, exeInfo);
#ifdef BMA_EXEC_ENABLED
    case EXE_GPU_BDMA: return RunBmaExecutor(iteration, cfg, exeDevice.exeIndex, exeInfo);
#endif
    default:            return {ERR_FATAL, "Unsupported executor (%d)", exeDevice.exeType};
    }
  }

#if defined(__NVCC__)
  static bool MnnvlCheck() {
    int flag = 0;
#ifdef POD_COMM_ENABLED
    CUresult err = cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 0);
    if (err || !flag) return false;
    err = cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, 0);
#endif
    if (!flag) return false;
    return true;
  }
#endif

} // End of anonymous namespace
//========================================================================================
/// @endcond

  ErrResult::ErrResult(ErrType err) : errType(err), errMsg("") {};

  ErrResult::ErrResult(hipError_t err)
  {
    if (err == hipSuccess) {
      this->errType = ERR_NONE;
      this->errMsg  = "";
    } else {
      this->errType = ERR_FATAL;
#if defined(__NVCC__)
      this->errMsg  = std::string("CUDA Runtime Error: ") + hipGetErrorString(err);
#else
      this->errMsg  = std::string("HIP Error: ") + hipGetErrorString(err);
#endif
    }
  }

#if !defined(__NVCC__)
  ErrResult::ErrResult(hsa_status_t err)
  {
    if (err == HSA_STATUS_SUCCESS) {
      this->errType = ERR_NONE;
      this->errMsg  = "";
    } else {
      const char *errString = NULL;
      hsa_status_string(err, &errString);
      this->errType = ERR_FATAL;
      this->errMsg  = std::string("HSA Error: ") + errString;
    }
  }
#elif defined(CUMEM_ENABLED)
  ErrResult::ErrResult(CUresult err)
  {
    if (err == CUDA_SUCCESS) {
      this->errType = ERR_NONE;
      this->errMsg  = "";
    } else {
      const char *errString = NULL, *errName = NULL;
      cuGetErrorName(err, &errName);
      cuGetErrorString(err, &errString);
      this->errType = ERR_FATAL;
      this->errMsg  = std::string("CUDA Driver Error: ") + errName
                      + " (" + errString + ")";
    }
  }
#endif

  ErrResult::ErrResult(ErrType errType, const char* format, ...)
  {
    this->errType = errType;
    va_list args, args_temp;
    va_start(args, format);
    va_copy(args_temp, args);

    int len = vsnprintf(nullptr, 0, format, args);
    if (len < 0) {
      va_end(args_temp);
      va_end(args);
    } else {
      this->errMsg.resize(len);
      vsnprintf(this->errMsg.data(), len+1, format, args_temp);
    }
    va_end(args_temp);
    va_end(args);
  }

  bool RunTransfers(ConfigOptions         const& cfg,
                    std::vector<Transfer> const& transfers,
                    TestResults&                 results)
  {
    // Clear all errors;
    auto& errResults = results.errResults;
    errResults.clear();

    // Check for valid configuration and quit if any rank has fatal error
    if (System::Get().Any(ConfigOptionsHaveErrors(cfg, errResults))) {
      System::Get().AllGatherErrors(errResults);
      return false;
    }

    // Check for valid transfers and quit if any rank has fatal error
    if (System::Get().Any(TransfersHaveErrors(cfg, transfers, errResults))) {
      System::Get().AllGatherErrors(errResults);
      return false;
    }

    // Log transfers (if requested)
    System::Get().LogTransfers(transfers);

    // Collect up transfers by executor
    int minNumSrcs = MAX_SRCS + 1;
    int maxNumSrcs = 0;
    size_t maxNumBytes = 0;
    std::map<ExeDevice, ExeInfo> executorMap;
    for (int i = 0; i < transfers.size(); i++) {
      Transfer const& t = transfers[i];
      ExeDevice exeDevice;
      ERR_APPEND(GetActualExecutor(t.exeDevice, exeDevice), errResults);

      TransferResources resource = {};
      resource.transferIdx = i;

      ExeInfo& exeInfo = executorMap[exeDevice];
      exeInfo.totalBytes    += t.numBytes;
      exeInfo.totalSubExecs += t.numSubExecs;
      exeInfo.useSubIndices |= (t.exeSubIndex != -1 || (t.exeDevice.exeType == EXE_GPU_GFX && !cfg.gfx.prefXccTable.empty()));
      exeInfo.resources.push_back(resource);
      minNumSrcs  = std::min(minNumSrcs, (int)t.srcs.size());
      maxNumSrcs  = std::max(maxNumSrcs, (int)t.srcs.size());
      maxNumBytes = std::max(maxNumBytes, t.numBytes);
    }

    // Empty transfer list leaves minNumSrcs at its sentinel (MAX_SRCS + 1);
    // clamp so the dstReference cleanup loop below can't index out of bounds.
    if (transfers.empty()) minNumSrcs = 0;

    // Loop over each executor and prepare
    // - Allocates memory for each Transfer
    // - Set up work for subexecutors
    int const localRank = GetRank();
    vector<ExeDevice> localExecutors;
    vector<TransferResources*> transferResources;
    for (auto& exeInfoPair : executorMap) {
      ExeDevice const& exeDevice = exeInfoPair.first;
      ExeInfo&         exeInfo   = exeInfoPair.second;
      ERR_APPEND(PrepareExecutor(cfg, transfers, exeDevice, exeInfo), errResults);

      for (auto& resource : exeInfo.resources) {
        transferResources.push_back(&resource);
      }
      // Track executors that are on this rank
      if (exeDevice.exeRank == localRank) {
        localExecutors.push_back(exeDevice);
      }

      // Select which GFX kernel to use for this executor
      if (exeDevice.exeType == EXE_GPU_GFX) {
        ERR_APPEND(SelectGfxKernel(cfg, transfers, exeInfo), errResults);
      }

      // Select which TDM kernel to use for this executor
      if (exeDevice.exeType == EXE_GPU_TDM) {
        ERR_APPEND(SelectTdmKernel(cfg, transfers, exeInfo), errResults);

        // For the TDM reduce kernel, warn about any SRC/DST buffer that is not
        // 128B aligned (the tensor data mover reaches peak bandwidth on
        // 128B-aligned addresses; misaligned buffers still work but slower).
        if (exeInfo.tdmKernelToUse == TDM_KERNEL_REDUCE) {
          for (auto const& rss : exeInfo.resources) {
            for (int iSrc = 0; iSrc < (int)rss.srcMem.size(); ++iSrc) {
              if (reinterpret_cast<uintptr_t>(rss.srcMem[iSrc]) & 127u)
                errResults.push_back({ERR_WARN,
                  "Transfer %d: TDM reduce SRC[%d] (%p) is not 128B aligned; performance may be reduced",
                  rss.transferIdx, iSrc, (void*)rss.srcMem[iSrc]});
            }
            for (int iDst = 0; iDst < (int)rss.dstMem.size(); ++iDst) {
              if (reinterpret_cast<uintptr_t>(rss.dstMem[iDst]) & 127u)
                errResults.push_back({ERR_WARN,
                  "Transfer %d: TDM reduce DST[%d] (%p) is not 128B aligned; performance may be reduced",
                  rss.transferIdx, iDst, (void*)rss.dstMem[iDst]});
            }
          }
        }
      }
    }

    // Prepare reference src/dst arrays - only once for largest size.
    // dstReference (expected results) is only needed when validation is enabled.
    bool const validateEnabled = (cfg.data.alwaysValidate >= 0);
    size_t maxN = maxNumBytes / sizeof(float);
    vector<float> outputBuffer(validateEnabled ? maxN : 0);
    vector<vector<float>> dstReference;
    {
      size_t initOffset = cfg.data.byteOffset / sizeof(float);
      vector<vector<float>> srcReference(maxNumSrcs, vector<float>(maxN));

      if (validateEnabled) {
        dstReference.assign(maxNumSrcs + 1, vector<float>(maxN));
        memset(dstReference[0].data(), MEMSET_CHAR, maxNumBytes);
      }
      for (int numSrcs = 0; numSrcs < maxNumSrcs; numSrcs++) {
        PrepareReference(cfg, srcReference[numSrcs], numSrcs);
        if (validateEnabled)
          for (int i = 0; i < maxN; i++)
            dstReference[numSrcs+1][i] = (numSrcs == 0 ? 0 : dstReference[numSrcs][i]) + srcReference[numSrcs][i];
      }
      // Release un-used partial sums
      if (validateEnabled)
        for (int numSrcs = 0; numSrcs < minNumSrcs; numSrcs++)
          dstReference[numSrcs].clear();

      // Initialize all src memory buffers (if on local rank)
      bool const verbose = System::Get().IsVerbose();
      for (auto resource : transferResources) {
        Transfer const& t = transfers[resource->transferIdx];
        for (int srcIdx = 0; srcIdx < resource->srcMem.size(); srcIdx++) {
          if (t.srcs[srcIdx].memRank == localRank) {
            if (IsGpuMemType(t.srcs[srcIdx].memType)) {
              ERR_APPEND(hipSetDevice(t.srcs[srcIdx].memIndex), errResults);
            }
            if (verbose) {
              MemDevice const& md = t.srcs[srcIdx];
              System::Get().Log("[INFO] Data prep memcpy: transfer %d SRC[%d] host->%s%d  %zu bytes  dst=%p src=%p\n",
                                resource->transferIdx, srcIdx,
                                GetMemTypeName(md.memType), md.memIndex,
                                resource->numBytes,
                                resource->srcMem[srcIdx] + initOffset,
                                srcReference[srcIdx].data());
            }
            ERR_APPEND(hipMemcpy(resource->srcMem[srcIdx] + initOffset, srcReference[srcIdx].data(), resource->numBytes,
                                 hipMemcpyDefault), errResults);
            ERR_APPEND(hipDeviceSynchronize(), errResults);
          }
        }
      }
    }

    // Pause before starting when running in iteractive mode
    if (cfg.general.useInteractive) {
      if (localRank == 0) {
        System::Get().Log("Memory prepared:\n");

        for (int i = 0; i < transfers.size(); i++) {
          Transfer const& t     = transfers[i];
          ExeDevice const& exe  = t.exeDevice;

          // Executor info
          std::string exeBdf = IsGpuExeType(exe.exeType) ? GetGpuBdf(exe.exeIndex) : "";
          int exeNuma = IsGpuExeType(exe.exeType)
                        ? System::Get().GetClosestCpuNumaToGpu(exe.exeIndex, exe.exeRank)
                        : IsNicExeType(exe.exeType)
                          ? System::Get().GetClosestCpuNumaToNic(exe.exeIndex, exe.exeRank)
                          : exe.exeIndex;
          System::Get().Log("Transfer %03d: EXE=R%d%c%d NUMA=%d%s%s  %zu bytes\n",
                            i, exe.exeRank, ExeTypeStr[exe.exeType], exe.exeIndex, exeNuma,
                            exeBdf.empty() ? "" : " BDF=", exeBdf.c_str(), t.numBytes);

          for (int iSrc = 0; iSrc < t.srcs.size(); ++iSrc) {
            MemDevice const& md = t.srcs[iSrc];
            std::string bdf = IsGpuMemType(md.memType) ? GetGpuBdf(md.memIndex) : "";
            System::Get().Log("  SRC[%d]: %p  type=%-18s idx=%d NUMA=%s Rank=%d%s%s\n",
                              iSrc, transferResources[i]->srcMem[iSrc],
                              GetMemTypeName(md.memType), md.memIndex,
                              GetMemDeviceNuma(md).c_str(), md.memRank,
                              bdf.empty() ? "" : " BDF=", bdf.c_str());
          }
          for (int iDst = 0; iDst < t.dsts.size(); ++iDst) {
            MemDevice const& md = t.dsts[iDst];
            std::string bdf = IsGpuMemType(md.memType) ? GetGpuBdf(md.memIndex) : "";
            System::Get().Log("  DST[%d]: %p  type=%-18s idx=%d NUMA=%s Rank=%d%s%s\n",
                              iDst, transferResources[i]->dstMem[iDst],
                              GetMemTypeName(md.memType), md.memIndex,
                              GetMemDeviceNuma(md).c_str(), md.memRank,
                              bdf.empty() ? "" : " BDF=", bdf.c_str());
          }
        }
        System::Get().Log("Hit <Enter> to continue: ");
        fflush(stdout);
        if (scanf("%*c") != 0) {
          System::Get().Log("[ERROR] Unexpected input\n");
          exit(1);
        }
        System::Get().Log("\n");
      }
      System::Get().Barrier();
    }

    // Perform iterations
    size_t numTimedIterations = 0;
    double totalCpuTimeSec = 0.0;
    for (int iteration = -cfg.general.numWarmups; ; iteration++) {
      // Stop if number of iterations/seconds has reached limit
      if (cfg.general.numIterations > 0 && iteration >= cfg.general.numIterations) break;

      // NOTE: Time-based limit is based on first rank to avoid any skew issues
      bool shouldStop = (cfg.general.numIterations < 0 && totalCpuTimeSec > -cfg.general.numIterations);
      System::Get().Broadcast(0, sizeof(shouldStop), &shouldStop);
      if (shouldStop) break;

      // Wait for all ranks before starting any timing
      System::Get().Barrier();

      // Start CPU timing for this iteration
      auto cpuStart = std::chrono::high_resolution_clock::now();

      // Execute all Transfers in parallel
      std::vector<std::future<ErrResult>> asyncExecutors;
      for (auto const& exeDevice : localExecutors) {
        asyncExecutors.emplace_back(std::async(std::launch::async, RunExecutor,
                                               iteration,
                                               std::cref(cfg),
                                               std::cref(exeDevice),
                                               std::ref(executorMap[exeDevice])));
      }

      // Wait for all threads to finish
      for (auto& asyncExecutor : asyncExecutors) {
        ERR_APPEND(asyncExecutor.get(), errResults);
      }

      // Wait for all ranks to finish
      System::Get().Barrier();

      // Stop CPU timing for this iteration
      auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
      double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() / cfg.general.numSubIterations;

      if (cfg.data.alwaysValidate > 0) {
        ERR_APPEND(ValidateAllTransfers(cfg, transfers, transferResources, dstReference, outputBuffer),
                   errResults);
      }

      if (iteration >= 0) {
        ++numTimedIterations;
        totalCpuTimeSec += deltaSec;
      }
    }

    // Interactive per-transfer PASS/FAIL display (skipped when validation disabled). This does its
    // own comparison pass to show results; authoritative error reporting still comes from
    // ValidateAllTransfers (mode 0) or the inline per-iteration validation (mode >0).
    if (cfg.general.useInteractive && cfg.data.alwaysValidate >= 0) {
      bool inputOk = true;
      if (localRank == 0) {
        if (cfg.data.alwaysValidate == 0) {
          System::Get().Log("Transfers complete. Hit <Enter> to run validation: ");
          fflush(stdout);
          if (getchar() == EOF) {
            System::Get().Log("[ERROR] Unexpected EOF while waiting for input\n");
            inputOk = false;
          } else {
            System::Get().Log("\nValidation results:\n");
          }
        } else {
          System::Get().Log("Transfers complete.\nValidation results:\n");
        }
      }

      // Coordinate any EOF abort so no rank is left waiting at the Barrier below
      System::Get().Broadcast(0, sizeof(inputOk), &inputOk);
      if (!inputOk)
        ERR_APPEND((ErrResult{ERR_FATAL, "Unexpected EOF while waiting for interactive input"}), errResults);

      if (localRank == 0) {
        size_t initOffset = cfg.data.byteOffset / sizeof(float);
        int numPass = 0, numFail = 0;
        for (auto rss : transferResources) {
          int transferIdx = rss->transferIdx;
          Transfer const& t = transfers[transferIdx];
          size_t N = t.numBytes / sizeof(float);
          float const* expected = dstReference[t.srcs.size()].data();
          bool transferOk = true;
          bool anyLocalDst = false;

          System::Get().Log("  Transfer %03d:", transferIdx);
          for (int dstIdx = 0; dstIdx < (int)rss->dstMem.size(); dstIdx++) {
            if (t.dsts[dstIdx].memRank != GetRank()) {
              System::Get().Log("  DST[%d]=SKIP(remote)", dstIdx);
              continue;
            }
            anyLocalDst = true;
            float* output;
            if (IsCpuMemType(t.dsts[dstIdx].memType) || cfg.data.validateDirect) {
              output = rss->dstMem[dstIdx] + initOffset;
            } else {
              (void)hipSetDevice(t.dsts[dstIdx].memIndex);
              (void)hipMemcpy(outputBuffer.data(), rss->dstMem[dstIdx] + initOffset, t.numBytes, hipMemcpyDefault);
              (void)hipDeviceSynchronize();
              output = outputBuffer.data();
            }

            if (memcmp(output, expected, t.numBytes) == 0) {
              System::Get().Log("  DST[%d]=PASS", dstIdx);
            } else {
              size_t firstErr = 0;
              for (; firstErr < N; firstErr++)
                if (output[firstErr] != expected[firstErr]) break;
              if (firstErr < N)
                System::Get().Log("  DST[%d]=FAIL(first mismatch idx=%zu exp=%.5f got=%.5f)",
                                  dstIdx, firstErr, expected[firstErr], output[firstErr]);
              else
                System::Get().Log("  DST[%d]=FAIL(bitwise mismatch, no float-level diff found)",
                                  dstIdx);
              transferOk = false;
            }
          }
          System::Get().Log("\n");
          if (anyLocalDst) { if (transferOk) ++numPass; else ++numFail; }
        }
        System::Get().Log("  Summary: %d PASS  %d FAIL\n", numPass, numFail);
        fflush(stdout);
      }
      System::Get().Barrier();
    } else if (cfg.general.useInteractive) {
      if (localRank == 0)
        System::Get().Log("Transfers complete. Validation disabled (ALWAYS_VALIDATE < 0)\n");
      System::Get().Barrier();
    }

    // Validate results
    if (cfg.data.alwaysValidate == 0) {
      ERR_APPEND(ValidateAllTransfers(cfg, transfers, transferResources, dstReference, outputBuffer),
                 errResults);
    }

    // Prepare results
    results.exeResults.clear();
    results.tfrResults.clear();
    results.tfrResults.resize(transfers.size());
    results.numTimedIterations = numTimedIterations;
    results.totalBytesTransferred = 0;
    results.avgTotalDurationMsec = (totalCpuTimeSec * 1000.0) / numTimedIterations;
    results.overheadMsec = results.avgTotalDurationMsec;
    for (auto& exeInfoPair : executorMap) {
      ExeDevice const& exeDevice = exeInfoPair.first;
      ExeInfo&         exeInfo   = exeInfoPair.second;

      results.totalBytesTransferred += exeInfo.totalBytes;

      // Copy over executor results
      ExeResult exeResult;
      if (exeDevice.exeRank == localRank) {
        // Local executor collects results
        exeResult.numBytes             = exeInfo.totalBytes;
        exeResult.avgDurationMsec      = exeInfo.totalDurationMsec / numTimedIterations;
        exeResult.avgBandwidthGbPerSec = (exeResult.numBytes / 1.0e6) /  exeResult.avgDurationMsec;
        exeResult.sumBandwidthGbPerSec = 0.0;
        exeResult.transferIdx.clear();

        // Copy over transfer results
        for (auto const& rss : exeInfo.resources) {
          int const transferIdx = rss.transferIdx;
          exeResult.transferIdx.push_back(transferIdx);

          TransferResult& tfrResult      = results.tfrResults[transferIdx];
          tfrResult.exeDevice            = exeDevice;
          tfrResult.exeDstDevice         = {exeDevice.exeType, rss.dstNicIndex};
          tfrResult.numBytes             = rss.numBytes;
          tfrResult.avgDurationMsec      = rss.totalDurationMsec / numTimedIterations;
          tfrResult.avgBandwidthGbPerSec = (rss.numBytes / 1.0e6) / tfrResult.avgDurationMsec;
          if (cfg.general.recordPerIteration) {
            tfrResult.perIterMsec = rss.perIterMsec;
            tfrResult.perIterCUs  = rss.perIterCUs;
          }
          exeResult.sumBandwidthGbPerSec += tfrResult.avgBandwidthGbPerSec;
        }
      }

      // Send executor and transfer result to all ranks
      System::Get().BroadcastExeResult(exeDevice.exeRank, exeResult);
      for (int const transferIdx : exeResult.transferIdx) {
        System::Get().BroadcastTfrResult(exeDevice.exeRank, results.tfrResults[transferIdx]);
      }

      results.exeResults[exeDevice] = exeResult;
      results.overheadMsec = std::min(results.overheadMsec, (results.avgTotalDurationMsec -
                                                             exeResult.avgDurationMsec));
    }
    results.avgTotalBandwidthGbPerSec = (results.totalBytesTransferred / 1.0e6) / results.avgTotalDurationMsec;

    // Teardown executors
    for (auto& exeInfoPair : executorMap) {
      ExeDevice const& exeDevice = exeInfoPair.first;
      ExeInfo&         exeInfo   = exeInfoPair.second;
      ERR_APPEND(TeardownExecutor(cfg, exeDevice, transfers, exeInfo), errResults);
    }

    System::Get().AllGatherErrors(errResults);

    for (auto const& err : errResults) {
      if (err.errType == ERR_FATAL) return false;
    }
    return true;
  }

  std::string GetStrAttribute(StrAttribute attribute)
  {
    switch (attribute) {
    case ATR_SRC_PREP_DESCRIPTION:
      return "Element i = ((i * 517) modulo 383 + 31) * (srcBufferIdx + 1)";
    default:
      return "";
    }
  }

  bool RecursiveWildcardTransferExpansion(WildcardTransfer& wc,
                                          int const& baseRankIndex,
                                          size_t const& numBytes,
                                          int const& numSubExecs,
                                          std::vector<Transfer>& transfers)
  {
    // Basic implementation idea:
    // - This recursive function procedes through each Transfer characteristic that has multiple possible values,
    //   selects one, then proceeds.
    // - At the "end", each characteristic will only have one option, which will then be used to specify the
    //   Transfer to be added to transfers
    bool result = false;

    // Resolve memory wildcards first
    for (int isDst = 0; isDst <= 1; isDst++) {
      for (int iMem = 0; iMem < wc.mem[isDst].size(); iMem++) {

        // Resolve mem rank wildcards first
        if (wc.mem[isDst][iMem].memRanks.size() == 0) {
          // Replace empty rank with baseRankIndex
          wc.mem[isDst][iMem].memRanks = {baseRankIndex};
          RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
          wc.mem[isDst][iMem].memRanks.clear();
          return true;
        } else if (wc.mem[isDst][iMem].memRanks.size() > 1) {
          // Loop over each possible rank and recurse
          std::vector<int> memRanks;
          memRanks.swap(wc.mem[isDst][iMem].memRanks);
          for (auto x : memRanks) {
            wc.mem[isDst][iMem].memRanks = {x};
            result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
          }
          wc.mem[isDst][iMem].memRanks.swap(memRanks);
          return result;
        }
        // At this point, there should be only 1 (valid) rank assigned to this SRC
        if (wc.mem[isDst][iMem].memRanks.size() != 1 || wc.mem[isDst][iMem].memRanks[0] < 0) {
          System::Get().Log("[ERROR] Unexpected number of ranks / invalid number of ranks for %s %d\n", isDst ? "DST" : "SRC", iMem);
          exit(1);
        }

        // Resolve mem index wildcards
        // Mem devices should have at least one index
        if (wc.mem[isDst][iMem].memIndices.size() == 0) {
          System::Get().Log("[ERROR] MemIndex for %s %d cannot be empty\n", isDst ? "DST" : "SRC", iMem);
          exit(1);
        }

        // Loop over user provided list of device indices
        if (wc.mem[isDst][iMem].memIndices.size() > 1) {
          std::vector<int> memIndices;
          memIndices.swap(wc.mem[isDst][iMem].memIndices);
          for (auto x : memIndices) {
            wc.mem[isDst][iMem].memIndices = {x};
            result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
          }
          wc.mem[isDst][iMem].memIndices.swap(memIndices);
          return result;
        } else if (wc.mem[isDst][iMem].memIndices.size() == 1 && wc.mem[isDst][iMem].memIndices[0] == -1) {
          // Wildcard - loop over all possible device indices for this memory type
          int numExecutors = GetNumExecutors(wc.mem[isDst][iMem].memType, wc.mem[isDst][iMem].memRanks[0]);
          for (int x = 0; x < numExecutors; x++) {
            wc.mem[isDst][iMem].memIndices[0] = x;
            result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
          }
          wc.mem[isDst][iMem].memIndices[0] = -1;
          return result;
        }
      }
    }

    // Check for NIC wildcard (device index) first
    if (wc.exe.exeType == EXE_NIC_NEAREST &&
        wc.exe.exeRanks.size() == 0 &&
        wc.exe.exeIndices.size() == 0 &&
        wc.exe.exeSlots.size() == 0 &&
        wc.exe.exeSubIndices.size() == 0 &&
        wc.exe.exeSubSlots.size() == 0) {

      // Find (first) closest NIC to the SRC memory location
      std::vector<int> srcNicIndices;
      if (IsCpuMemType(wc.mem[0][0].memType)) {
        GetClosestNicsToCpu(srcNicIndices, wc.mem[0][0].memIndices[0], wc.mem[0][0].memRanks[0]);
      } else {
        GetClosestNicsToGpu(srcNicIndices, wc.mem[0][0].memIndices[0], wc.mem[0][0].memRanks[0]);
      }
      // Find (first) closest NIC to the DST memory location
      std::vector<int> dstNicIndices;
      if (IsCpuMemType(wc.mem[1][0].memType)) {
        GetClosestNicsToCpu(dstNicIndices, wc.mem[1][0].memIndices[0], wc.mem[1][0].memRanks[0]);
      } else {
        GetClosestNicsToGpu(dstNicIndices, wc.mem[1][0].memIndices[0], wc.mem[1][0].memRanks[0]);
      }

      // If valid, fill in all wildcards
      if (srcNicIndices.size() > 0 && dstNicIndices.size() > 0) {
        wc.exe.exeRanks      = {wc.mem[0][0].memRanks[0]};
        wc.exe.exeIndices    = {srcNicIndices[0]};
        wc.exe.exeSlots      = {0};
        wc.exe.exeSubIndices = {dstNicIndices[0]};
        wc.exe.exeSubSlots   = {0};

        result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);

        wc.exe.exeRanks.clear();
        wc.exe.exeIndices.clear();
        wc.exe.exeSlots.clear();
        wc.exe.exeSubIndices.clear();
        wc.exe.exeSubSlots.clear();
        return result;
      } else {
        return false;
      }
    }

    // Resolve EXE rank
    if (wc.exe.exeRanks.size() == 0)  {
      // No rank provided - Assign the current base rank index
      wc.exe.exeRanks = {baseRankIndex};
      RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
      wc.exe.exeRanks.clear();
      return true;
    } else if (wc.exe.exeRanks.size() > 1) {
      // Loop over user provided ranks
      std::vector<int> exeRanks;
      exeRanks.swap(wc.exe.exeRanks);
      for (auto x : exeRanks) {
        wc.exe.exeRanks = {x};
        result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
      }
      wc.exe.exeRanks.swap(exeRanks);
      return result;
    } else if (wc.exe.exeRanks[0] == -1) {
      System::Get().Log("[ERROR] Exe rank should not be -1\n");
      exit(1);
    }

    // Resolve EXE indices
    if (wc.exe.exeIndices.size() == 0) {
      System::Get().Log("[ERROR] Exe index should never be empty\n");
      exit(1);
    } else if (wc.exe.exeIndices.size() > 1) {
      // Loop over user provided indices
      std::vector<int> exeIndices;
      exeIndices.swap(wc.exe.exeIndices);
      for (auto x : exeIndices) {
        wc.exe.exeIndices = {x};
        result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
      }
      wc.exe.exeIndices.swap(exeIndices);
      return result;
    } else if (wc.exe.exeIndices[0] == -1) {
      // Wildcard - loop over all possible executor indices
      int numExecutors = GetNumExecutors(wc.exe.exeType, wc.exe.exeRanks[0]);
      for (int x = 0; x < numExecutors; x++) {
        wc.exe.exeIndices[0] = x;
        result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
      }
      wc.exe.exeIndices[0] = -1;
      return result;
    }

    // Resolve EXE slots (only apples to EXE_NIC_NEAREST)
    if (wc.exe.exeSlots.size() == 0) {
      // Slot won't be used, so just assign 0
      wc.exe.exeSlots = {0};
      result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
      wc.exe.exeSlots.clear();
      return result;
    } else if (wc.exe.exeSlots.size() > 1) {
      // Loop over user provided slots
      std::vector<int> exeSlots;
      exeSlots.swap(wc.exe.exeSlots);
      for (auto x : exeSlots) {
        wc.exe.exeSlots = {x};
        result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
      }
      wc.exe.exeSlots.swap(exeSlots);
      return result;
    } else if (wc.exe.exeSlots[0] == -1) {
      // Wildcard - Loop over all possible slots, based on SRC memory type
      std::vector<int> srcNicIndices;
      if (IsCpuMemType(wc.mem[0][0].memType)) {
        GetClosestNicsToCpu(srcNicIndices, wc.mem[0][0].memIndices[0], wc.mem[0][0].memRanks[0]);
      } else {
        GetClosestNicsToGpu(srcNicIndices, wc.mem[0][0].memIndices[0], wc.mem[0][0].memRanks[0]);
      }
      for (auto x : srcNicIndices) {
        wc.exe.exeSlots = {x};
        result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
      }
      wc.exe.exeSlots = {-1};
      return result;
    }

    // Resolve EXE subindex
    if (wc.exe.exeSubIndices.size() == 0) {
      if (IsCpuExeType(wc.exe.exeType) || IsGpuExeType(wc.exe.exeType)) {
        wc.exe.exeSubIndices = {-1};
        result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
        wc.exe.exeSubIndices.clear();
        return result;
      } else if (wc.exe.exeType == EXE_NIC) {
        System::Get().Log("[ERROR] NIC executor requires a subindex be specified\n");
        exit(1);
      } else if (wc.exe.exeType == EXE_NIC_NEAREST) {
        // Assign NIC closest to DST mem
        std::vector<int> dstNicIndices;
        if (IsCpuMemType(wc.mem[1][0].memType)) {
          GetClosestNicsToCpu(dstNicIndices, wc.mem[1][0].memIndices[0], wc.mem[1][0].memRanks[0]);
        } else {
          GetClosestNicsToGpu(dstNicIndices, wc.mem[1][0].memIndices[0], wc.mem[1][0].memRanks[0]);
        }
        if (dstNicIndices.size() > 0) {
          wc.exe.exeSubIndices = {dstNicIndices[0]};
          result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
          wc.exe.exeSubIndices.clear();
        }
        return result;
      }
    } else if (wc.exe.exeSubIndices.size() > 1) {
      // Loop over all user provided subindices
      std::vector<int> exeSubIndices;
      exeSubIndices.swap(wc.exe.exeSubIndices);
      for (auto x : exeSubIndices) {
        wc.exe.exeSubIndices = {x};
        result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
      }
      wc.exe.exeSubIndices.swap(exeSubIndices);
      return result;
    } else if (wc.exe.exeSubIndices[0] == -2) {
      switch (wc.exe.exeType) {
      case EXE_CPU: case EXE_GPU_TDM:
        // These Executors do no support subindices
        wc.exe.exeSubIndices[0] = -1;
        result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
        wc.exe.exeSubIndices[0] = -2;
        return result;
      case EXE_GPU_GFX: case EXE_GPU_DMA: case EXE_GPU_BDMA:
      {
        // Iterate over all available subindices
        ExeDevice exeDevice = {wc.exe.exeType, wc.exe.exeIndices[0], wc.exe.exeRanks[0], 0};
        int numSubIndices = GetNumExecutorSubIndices(exeDevice);
        for (int x = 0; x < numSubIndices; x++) {
          wc.exe.exeSubIndices = {x};
          result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
        }
        wc.exe.exeSubIndices = {-1};
        return result;
      }
      case EXE_NIC: case EXE_NIC_NEAREST:
      {
        // Iterates over total number of DST NICs
        int numIndices = 0;
        if (wc.exe.exeType == EXE_NIC) {
          numIndices = GetNumExecutors(EXE_NIC, wc.mem[1][0].memRanks[0]);
        } else {
          numIndices = GetNumExecutors(EXE_GPU_GFX, wc.mem[1][0].memRanks[0]);
        }
        for (int x = 0; x < numIndices; x++) {
          wc.exe.exeSubIndices = {x};
          result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
        }
        wc.exe.exeSubIndices = {-1};
        return result;
      }
      }
      return result;
    }

    // Resolve EXE subslots (only apples to EXE_NIC_NEAREST)
    if (wc.exe.exeSubSlots.size() == 0) {
      // Subslot won't be used, so just assign 0
      wc.exe.exeSubSlots = {0};
      result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
      wc.exe.exeSubSlots.clear();
      return result;
    } else if (wc.exe.exeSubSlots.size() > 1) {
      // Loop over user provided slots
      std::vector<int> exeSubSlots;
      exeSubSlots.swap(wc.exe.exeSubSlots);
      for (auto x : exeSubSlots) {
        wc.exe.exeSubSlots = {x};
        result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
      }
      wc.exe.exeSubSlots.swap(exeSubSlots);
      return result;
    } else if (wc.exe.exeSubSlots[0] == -1) {
      // Wildcard - Loop over all possible slots, based on DST memory type
      std::vector<int> dstNicIndices;
      if (IsCpuMemType(wc.mem[1][0].memType)) {
        GetClosestNicsToCpu(dstNicIndices, wc.mem[1][0].memIndices[0], wc.mem[1][0].memRanks[0]);
      } else {
        GetClosestNicsToGpu(dstNicIndices, wc.mem[1][0].memIndices[0], wc.mem[1][0].memRanks[0]);
      }
      for (auto x : dstNicIndices) {
        wc.exe.exeSubSlots = {x};
        result |= RecursiveWildcardTransferExpansion(wc, baseRankIndex, numBytes, numSubExecs, transfers);
      }
      wc.exe.exeSubSlots = {-1};
      return result;
    }

    // Only reach here when each candidate has been narrowed down to 1 option
    // Create Transfer and add to list
    Transfer t;
    t.numBytes    = numBytes;
    t.numSubExecs = numSubExecs;

    for (int iSrc = 0; iSrc < wc.mem[0].size(); iSrc++)
      t.srcs.push_back({wc.mem[0][iSrc].memType, wc.mem[0][iSrc].memIndices[0], wc.mem[0][iSrc].memRanks[0]});
    for (int iDst = 0; iDst < wc.mem[1].size(); iDst++)
      t.dsts.push_back({wc.mem[1][iDst].memType, wc.mem[1][iDst].memIndices[0], wc.mem[1][iDst].memRanks[0]});
    t.exeDevice.exeType  = wc.exe.exeType;
    t.exeDevice.exeIndex = wc.exe.exeIndices[0];
    t.exeDevice.exeRank  = wc.exe.exeRanks[0];
    t.exeDevice.exeSlot  = wc.exe.exeSlots[0];
    t.exeSubIndex        = wc.exe.exeSubIndices[0];
    t.exeSubSlot         = wc.exe.exeSubSlots[0];

    transfers.push_back(t);

    return false;
  }

  ErrResult ParseTransfers(std::string            line,
                           std::vector<Transfer>& transfers)
  {
    // Replace any round brackets or '->' with spaces,
    for (int i = 1; line[i]; i++)
      if (line[i] == '(' || line[i] == ')' || line[i] == '-'  || line[i] == ':' || line[i] == '>' ) line[i] = ' ';

    transfers.clear();

    // Read in number of transfers descriptions
    // NOTE: Transfers descriptions with wildcards get expanded to multiple transfers
    int numTransfers = 0;
    std::istringstream iss(line);
    iss >> numTransfers;
    if (iss.fail()) return ERR_NONE;

    // If numTransfers < 0, read 5-tuple (srcMem, exeMem, dstMem, #CUs, #Bytes)
    // otherwise read triples (srcMem, exeMem, dstMem)
    bool const advancedMode = (numTransfers < 0);
    numTransfers = abs(numTransfers);

    int numSubExecs;
    std::string srcStr, exeStr, dstStr, numBytesToken;

    if (!advancedMode) {
      iss >> numSubExecs;
      if (numSubExecs < 0 || iss.fail()) {
        return {ERR_FATAL,
                "Parsing error: Number of blocks to use (%d) must be non-negative", numSubExecs};
      }
    }

    for (int i = 0; i < numTransfers; i++) {
      size_t numBytes;
      if (!advancedMode) {
        iss >> srcStr >> exeStr >> dstStr;
        if (iss.fail()) {
          return {ERR_FATAL,
            "Parsing error: Unable to read valid Transfer %d (SRC EXE DST) triplet", i+1};
        }
        numBytes = 0;
      } else {
        iss >> srcStr >> exeStr >> dstStr >> numSubExecs >> numBytesToken;
        if (iss.fail()) {
          return {ERR_FATAL,
            "Parsing error: Unable to read valid Transfer %d (SRC EXE DST $CU #Bytes) tuple", i+1};
        }
        if (sscanf(numBytesToken.c_str(), "%lu", &numBytes) != 1) {
          return {ERR_FATAL,
            "Parsing error: Unable to read valid Transfer %d (SRC EXE DST #CU #Bytes) tuple", i+1};
        }

        char units = numBytesToken.back();
        switch (toupper(units)) {
        case 'G': numBytes *= 1024;
        case 'M': numBytes *= 1024;
        case 'K': numBytes *= 1024;
        }
      }

      WildcardTransfer wct;
      ERR_CHECK(ParseMemType(srcStr, wct.mem[0]));
      ERR_CHECK(ParseMemType(dstStr, wct.mem[1]));
      ERR_CHECK(ParseExeType(exeStr, wct.exe));

      // Perform wildcard expansion
      int numRanks = GetNumRanks();
      for (int localRankIndex = 0; localRankIndex < numRanks; localRankIndex++) {
        bool localRankModified = RecursiveWildcardTransferExpansion(wct, localRankIndex, numBytes, numSubExecs, transfers);
        if (!localRankModified) break;
      }
    }

    return ERR_NONE;
  }

  // System related
  //========================================================================================
  System::System() :
    rank(0), numRanks(1), commMode(COMM_NONE)
  {
    // Collect env vars
    // TB_VERBOSE       = enables extra logging
    // TB_SINGLE_LOG    = Only rank 0 will produce output (useful if spawning multi-node socket)
    // TB_DUMP_CFG_FILE = Config file to dump executed Transfers
    // TB_PAUSE         = Insert a pause for debug attachment

    verbose = getenv("TB_VERBOSE") ? atoi(getenv("TB_VERBOSE")) : 0;
    bool singleLog = getenv("TB_SINGLE_LOG") ? atoi(getenv("TB_SINGLE_LOG")) : 0;

    char* dumpCfgFilename = getenv("TB_DUMP_CFG_FILE");
    if (dumpCfgFilename) {
      dumpCfgFile = fopen(dumpCfgFilename, "w");
    }

    if (getenv("TB_PAUSE")) {
      Log("Pausing for debug attachment (e.g. sudo gdb -p %d)\n", getpid());
      volatile bool pause = true;
      while (pause);
    }

#ifdef AMD_SMI_ENABLED
    if (verbose) {
      Log("[INFO] Initializing AMD System Management Interface Library (AMDSMI)\n");
    }
    amdsmi_init(AMDSMI_INIT_AMD_APUS);
#elif defined (NVML_ENABLED)
    if (verbose) {
      Log("[INFO] Initializing NVIDIA Management Library (NVML)\n");
    }
    nvmlInit_v2();
#endif

    // Priority 1: Socket communicator
    SetupSocketCommunicator();

    // Priority 2: MPI communicator
    if (commMode == COMM_NONE) {
      SetupMpiCommunicator();
    }

    // Establish which ranks will output when logging
    if (rank > 0 && (commMode == COMM_MPI || singleLog))
      rankDoesOutput = false;

    if (verbose && commMode == COMM_NONE) {
      Log("[INFO] Running in single node mode\n");
    }

    // Collect topology and distribute across all ranks
    CollectTopology();
    if (verbose) {
      Log("[INFO] Finished topology exchange\n");
    }
  }

  System::~System()
  {
#ifdef MPI_COMM_ENABLED
    if (commMode == COMM_MPI) {
      if (mpiInit == true)  {
        MPI_Finalize();
      }
    }
#endif
    if (commMode == COMM_SOCKET) {
      // Close all sockets
      for (auto& sock : sockets) {
        if (sock != -1) {
          close(sock);
          sock = -1;
        }
      }

      if (listenSocket != -1) {
        close(listenSocket);
        listenSocket = -1;
      }
    }

    if (dumpCfgFile) {
      fclose(dumpCfgFile);
    }

#ifdef AMD_SMI_ENABLED
    amdsmi_shut_down();
#elif defined(NVML_ENABLED)
    nvmlShutdown();
#endif
  }

  namespace detail {

  inline std::string FormatIpv4(struct in_addr const& addr)
  {
    char buf[INET_ADDRSTRLEN];
    if (inet_ntop(AF_INET, &addr, buf, sizeof(buf)))
      return std::string(buf);
    return std::string();
  }

  inline bool IsUsableIpv4(sockaddr_in const* sin)
  {
    if (!sin || sin->sin_family != AF_INET)
      return false;
    uint32_t a = ntohl(sin->sin_addr.s_addr);
    if (a == INADDR_ANY || a == INADDR_NONE)
      return false;
    if ((a >> 24) == 127)
      return false;
    return true;
  }

  // IPv4 to advertise when TB_MASTER_ADDR is unset on rank 0 (after listen).
  inline std::string DetectPrimaryIpv4(char const* preferredIface)
  {
    ifaddrs* ifap = nullptr;
    if (getifaddrs(&ifap) != 0)
      return std::string();

    auto tryPick = [&](bool allowLinkLocal) -> std::string {
      for (ifaddrs* ifa = ifap; ifa; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET)
          continue;
        if (ifa->ifa_flags & IFF_LOOPBACK)
          continue;
        if (!(ifa->ifa_flags & IFF_UP))
          continue;
        auto* sin = reinterpret_cast<sockaddr_in*>(ifa->ifa_addr);
        if (!IsUsableIpv4(sin))
          continue;
        if (preferredIface && preferredIface[0]) {
          if (!ifa->ifa_name || strcmp(ifa->ifa_name, preferredIface) != 0)
            continue;
        } else {
          uint32_t a = ntohl(sin->sin_addr.s_addr);
          if (!allowLinkLocal && (a & 0xffff0000) == 0xa9fe0000)
            continue;
        }
        return FormatIpv4(sin->sin_addr);
      }
      return std::string();
    };

    std::string chosen;
    if (preferredIface && preferredIface[0]) {
      chosen = tryPick(true);
      freeifaddrs(ifap);
      return chosen;
    }

    chosen = tryPick(false);
    if (chosen.empty())
      chosen = tryPick(true);
    freeifaddrs(ifap);
    return chosen;
  }

  inline bool ResolveMasterAddrV4(char const* host, int port, sockaddr_in* out, char const** gaiErr)
  {
    *gaiErr = nullptr;
    if (!host || !host[0] || !out)
      return false;
    char portBuf[16];
    snprintf(portBuf, sizeof(portBuf), "%d", port);
    addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    int gai = getaddrinfo(host, portBuf, &hints, &res);
    if (gai != 0) {
      *gaiErr = gai_strerror(gai);
      return false;
    }
    for (addrinfo* p = res; p; p = p->ai_next) {
      if (p->ai_family == AF_INET && p->ai_addrlen >= sizeof(sockaddr_in)) {
        memcpy(out, p->ai_addr, sizeof(sockaddr_in));
        freeaddrinfo(res);
        return true;
      }
    }
    freeaddrinfo(res);
    return false;
  }

  } // namespace detail

  void System::SetupSocketCommunicator()
  {
    char* rankStr       = getenv("TB_RANK");
    char* numRanksStr   = getenv("TB_NUM_RANKS");
    char* masterAddrStr = getenv("TB_MASTER_ADDR");
    char* masterPortStr = getenv("TB_MASTER_PORT");

    if (!numRanksStr) {
      if (verbose) {
        Log("[INFO] SocketCommunicator skipped (TB_NUM_RANKS not set)\n");
      }
      return;
    }

    numRanks = atoi(numRanksStr);
    if (numRanks < 2) {
      if (verbose) {
        Log("[INFO] SocketCommunicator skipped (TB_NUM_RANKS=%d requires at least 2 for socket mode)\n", numRanks);
      }
      return;
    }

    rank = (rankStr && rankStr[0]) ? atoi(rankStr) : 0;
    masterAddr = masterAddrStr ? std::string(masterAddrStr) : std::string();
    masterPort = masterPortStr ? atoi(masterPortStr) : 29500;

    if (rank != 0 && masterAddr.empty()) {
      Log("[ERROR] TB_MASTER_ADDR is required when TB_RANK is greater than 0 (socket communicator)\n");
      exit(1);
    }

    if (rank < 0 || rank >= numRanks) {
      Log("[ERROR] Invalid rank index.  Must be between 0 and %d (not %d)\n", numRanks - 1, rank);
      exit(1);
    }

    sockets.resize(numRanks, -1);

    // Rank 0 acts as server for others to connect to
    int opt = 1;
    if (rank == 0) {
      // Create listening socket
      listenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
      if (listenSocket == -1) {
        Log("[ERROR] Unable to create listener socket\n");
        exit(1);
      }

      // Allow address reuse
      setsockopt(listenSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

      // Bind to port
      sockaddr_in serverAddr;
      memset(&serverAddr, 0, sizeof(serverAddr));
      serverAddr.sin_family      = AF_INET;
      serverAddr.sin_addr.s_addr = INADDR_ANY;
      serverAddr.sin_port        = htons(masterPort);

      if (bind(listenSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        Log("[ERROR] Failed to bind listen socket\n");
        exit(1);
      }

      if (listen(listenSocket, numRanks) == -1) {
        Log("[ERROR] Failed to listen on socket\n");
        exit(1);
      }

      if (masterAddr.empty()) {
        char const* ifaceEnv = getenv("TB_MASTER_IFACE");
        masterAddr = detail::DetectPrimaryIpv4(ifaceEnv);
        if (masterAddr.empty()) {
          Log("[ERROR] TB_MASTER_ADDR not set and could not detect a primary IPv4 for workers");
          if (ifaceEnv && ifaceEnv[0])
            Log(" (check TB_MASTER_IFACE=%s)\n", ifaceEnv);
          else
            Log(" (set TB_MASTER_ADDR or TB_MASTER_IFACE)\n");
          exit(1);
        }
        Log("[INFO] TB_MASTER_ADDR not set; using detected IPv4 %s\n", masterAddr.c_str());
      }

      Log("[INFO] Socket rank 0: on each other host set TB_RANK to a unique value in 1..%d, then for example:\n",
          numRanks - 1);
      Log("       TB_NUM_RANKS=%d TB_MASTER_ADDR=%s TB_MASTER_PORT=%d TB_RANK=1\n",
          numRanks, masterAddr.c_str(), masterPort);

      Log("[INFO] Waiting for connections from %d other rank(s) [TB_MASTER_ADDR=%s TB_MASTER_PORT=%d]\n",
          numRanks - 1, masterAddr.c_str(), masterPort);

      for (int i = 1; i < numRanks; i++) {
        sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);

        auto clientSocket = accept(listenSocket, (sockaddr*)&clientAddr, &clientAddrLen);
        if (clientSocket == -1) {
          Log("[ERROR] Failed to accept connection from rank %d\n", i);
          exit(1);
        }

        // Receive rank ID from client
        int clientRank;
        recv(clientSocket, (char*)&clientRank, sizeof(clientRank), 0);

        if (clientRank < 0 || clientRank >= numRanks) {
          close(clientSocket);
          Log("[ERROR] Invalid rank received: %d\n", clientRank);
          exit(1);
        }
        if (verbose) {
          Log("[INFO] Rank 0 accepted connection from rank %d\n", clientRank);
        }
        sockets[clientRank] = clientSocket;
      }
      Log("[INFO] %d other rank(s) have connected\n", numRanks);
    } else {
      // All other ranks connect to rank 0
      int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
      if (sock == -1) {
        Log("[ERROR] Failed to create socket\n");
        exit(1);
      }

      sockaddr_in serverAddr;
      memset(&serverAddr, 0, sizeof(serverAddr));
      serverAddr.sin_family = AF_INET;
      char const* gaiErr = nullptr;
      if (!detail::ResolveMasterAddrV4(masterAddr.c_str(), masterPort, &serverAddr, &gaiErr)) {
        if (gaiErr)
          Log("[ERROR] Invalid master address '%s': %s\n", masterAddr.c_str(), gaiErr);
        else
          Log("[ERROR] Invalid master address: %s\n", masterAddr.c_str());
        exit(1);
      }

      // Retry connection with backoff
      if (verbose)
        Log("[INFO] Rank %d attempting to connect to %s:%d\n", rank, masterAddr.c_str(), masterPort);
      int maxRetries = 50;
      bool connected = false;
      for (int retry = 0; retry < maxRetries; retry++) {
        if (connect(sock, (sockaddr*)&serverAddr, sizeof(serverAddr)) == 0) {
          connected = true;
          break;
        }
        if (retry == maxRetries - 1) {
          Log("[ERROR] Failed to connect to master after %d retries\n", maxRetries);
        }
        sleep(1);
      }
      if (!connected) {
        close(sock);
        exit(1);
      }

      // Send local rank to the server
      send(sock, (char*)&rank, sizeof(rank), 0);
      sockets[0] = sock;
    }

    commMode = COMM_SOCKET;
  };

  void System::SetupMpiCommunicator()
  {
#ifdef MPI_COMM_ENABLED
    int flag;
    MPI_Initialized(&flag);
    if (!flag) {
      MPI_Init(NULL, NULL);
      mpiInit = true;
    }

    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &numRanks);
    if (numRanks > 1) {
      if (verbose) {
        Log("[INFO] Enabling MPI communicator (%d ranks found)\n", numRanks);
      }
      commMode = COMM_MPI;
    } else if (mpiInit) {
      // Drop out of MPI use for single node
      MPI_Finalize();
    }
#endif
  }

  void System::Log(const char* format, ...) const
  {
    if (rankDoesOutput) {
      va_list args;
      va_start(args, format);
      vprintf(format, args);
      va_end(args);
    }
  }

  void System::LogTransfers(std::vector<Transfer> const& transfers)
  {
    if (!dumpCfgFile || !rankDoesOutput) return;

    fprintf(dumpCfgFile, "-%lu ", transfers.size());
    for (auto const& t : transfers) {
      fprintf(dumpCfgFile, "(");

      // Print SRCs
      for (auto const& src : t.srcs) {
        fprintf(dumpCfgFile, "R%d%c%d", src.memRank, MemTypeStr[src.memType], src.memIndex);
      }
      if (t.srcs.empty())
        fprintf(dumpCfgFile, "N");

      fprintf(dumpCfgFile, "->");

      // Print Executor
      fprintf(dumpCfgFile, "R%d%c%d", t.exeDevice.exeRank, ExeTypeStr[t.exeDevice.exeType], t.exeDevice.exeIndex);
      if (t.exeDevice.exeSlot != 0)
        fprintf(dumpCfgFile, "%c", 'A' + t.exeDevice.exeSlot);
      if (t.exeSubIndex != -1) {
        fprintf(dumpCfgFile, ".%d", t.exeSubIndex);
      }
      if (t.exeSubSlot != 0) {
        fprintf(dumpCfgFile, "%c", 'A' + t.exeSubSlot);
      }

      fprintf(dumpCfgFile, "->");

      // Print DSTs
      for (auto const& dst : t.dsts) {
        fprintf(dumpCfgFile, "R%d%c%d", dst.memRank, MemTypeStr[dst.memType], dst.memIndex);
      }
      if (t.dsts.empty())
        fprintf(dumpCfgFile, "N");

      fprintf(dumpCfgFile, " %d %lu)", t.numSubExecs, t.numBytes);
      fflush(dumpCfgFile);
    }
    fprintf(dumpCfgFile, "\n");
  }

  void System::Barrier() const
  {
#ifdef MPI_COMM_ENABLED
    if (commMode == COMM_MPI) {
      MPI_Barrier(comm);
      return;
    }
#endif
    if (commMode == COMM_SOCKET) {
      char dummy = 0;

      // Simple barrier using rank 0 to coordinate
      if (rank == 0) {
        // Wait for notification from all ranks
        for (int peerRank = 1; peerRank < numRanks; peerRank++)
          RecvData(peerRank, 1, &dummy);

        // Release all ranks
        for (int peerRank = 1; peerRank < numRanks; peerRank++)
          SendData(peerRank, 1, &dummy);
      } else {
        // Send notification to root
        SendData(0, 1, &dummy);

        // Wait for release from root
        RecvData(0, 1, &dummy);
      }
    }
  }

  void System::SendData(int dstRank, size_t const numBytes, const void* sendData) const
  {
#ifdef MPI_COMM_ENABLED
    if (commMode == COMM_MPI) {
      MPI_Send(sendData, numBytes, MPI_BYTE, dstRank, 1234, comm);
      return;
    }
#endif
    if (commMode == COMM_SOCKET) {
      if (rank != 0 && dstRank != 0) {
        Log("[ERROR] Socket communicator is limited to sending from/to rank 0\n");
        exit(1);
      }
      auto sock = sockets[dstRank];

      // Send data
      size_t totalSent = 0;
      while (totalSent < numBytes) {
        auto sent = send(sock, (char*)sendData + totalSent, numBytes - totalSent, 0);
        if (sent == -1) {
          Log("[ERROR] Send failed (rank %d to rank %d)\n", rank, dstRank);
          exit(1);
        }
        totalSent += sent;
      }
    }
  }

  void System::RecvData(int srcRank, size_t const numBytes, void* recvData) const
  {
#ifdef MPI_COMM_ENABLED
    if (commMode == COMM_MPI) {
      MPI_Status status;
      MPI_Recv(recvData, numBytes, MPI_BYTE, srcRank, 1234, comm, &status);
      return;
    }
#endif
    if (commMode == COMM_SOCKET) {
      if (rank != 0 && srcRank != 0) {
        Log("[ERROR] Socket communicator is limited to receiving from/at rank 0\n");
        exit(1);
      }

      auto sock = sockets[srcRank];
      size_t totalRecv = 0;
      while (totalRecv < numBytes) {
        auto recvd = recv(sock, (char*)recvData + totalRecv, numBytes - totalRecv, 0);
        if (recvd == -1 || recvd == 0) {
          Log("[ERROR] Recv failed (rank %d from rank %d)\n", rank, srcRank);
          perror("recv");
          exit(1);
        }
        totalRecv += recvd;
      }
    }
  }

  void System::Broadcast(int root, size_t const numBytes, void* data) const
  {
    if (numBytes == 0) return;

#ifdef MPI_COMM_ENABLED
    if (commMode == COMM_MPI) {
      int err = MPI_Bcast(data, numBytes, MPI_CHAR, root, comm);
      if (err != MPI_SUCCESS) {
        Log("[ERROR] MPI_Bcast failed with error code %d\n", err);
      }
      return;
    }
#endif
    if (commMode == COMM_SOCKET) {
      // Relay through rank 0 first
      if (root != 0) {
        if (rank == root) {
          SendData(0, numBytes, data);
        } else if (rank == 0) {
          RecvData(root, numBytes, data);
        }
      }
      if (rank == 0) {
        for (int peer = 1; peer < numRanks; peer++) {
          SendData(peer, numBytes, data);
        }
      } else {
        RecvData(0, numBytes, data);
      }
    }

    Barrier(); // temporary workaround
  }

  bool System::Any(bool const flag) const
  {
    bool result = false;
    for (int i = 0; i < numRanks; i++) {
      bool flagToSend = flag;
      Broadcast(i, sizeof(flagToSend), &flagToSend);
      result |= flagToSend;
      if (result) break;
    }
    return result;
  }

  std::string System::GetCpuName() const
  {
    std::ifstream cpuInfo("/proc/cpuinfo");
    std::string line;

    if (cpuInfo.is_open()) {
      while (std::getline(cpuInfo, line)) {
        if (line.find("model name") != std::string::npos) {
          size_t colonIdx = line.find(":");
          if (colonIdx != std::string::npos) {
            return line.substr(colonIdx + 2);
          }
        }
      }
    }
    return "Unknown CPU";
  }

  void System::CollectPodMembership(char* ppodId, int64_t& vpodId)
  {
    memset(ppodId, 0, 16);
    vpodId = -1;

    // TB_FORCE_SINGLE_POD skips any required queries to AMDSMI
    char* forceSinglePod = getenv("TB_FORCE_SINGLE_POD");
    if (forceSinglePod) {
      vpodId = 0;
      return;
    }

    // Check fabric support
#if defined(__NVCC__)
#ifdef NVML_ENABLED
    if (!MnnvlCheck()) return;
    char busId[] = "00000000:00:00.0";
    if (cudaDeviceGetPCIBusId(busId, sizeof(busId), 0)) return;

    nvmlGpuFabricInfoV_t fabricInfo;
    fabricInfo.state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;
    nvmlDevice_t nvmlDev;
    nvmlReturn_t err = nvmlDeviceGetHandleByPciBusId_v2(busId, &nvmlDev);
    if (err != NVML_SUCCESS) {
      if (verbose) {
        System::Get().Log("[WARN] Unable to get processor handle for GPU 0 at %s [%s]\n",
                           busId, nvmlErrorString(err));
      }
      return;
    }
    fabricInfo.version = nvmlGpuFabricInfo_v2;

    err = nvmlDeviceGetGpuFabricInfoV(nvmlDev, &fabricInfo);
    if (err != NVML_SUCCESS || fabricInfo.state == NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {
      System::Get().Log("[WARN] MNNVL not supported\n");
    } else {
      vpodId = fabricInfo.cliqueId;
      memcpy(ppodId, fabricInfo.clusterUuid, 16);
    }
#endif
#else
#ifdef AMD_SMI_ENABLED
    int numGpus = 0;
    if (hipGetDeviceCount(&numGpus) == hipSuccess && numGpus > 0) {
      // Query GPU 0 as the representative for pod membership. All GPUs on a node are
      // expected to share the same pod (ppod_id/vpod_id), so querying any one is sufficient.
      char pciBusId[256] = "";
      hipError_t hipErr = hipDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), 0);
      if (hipErr != hipSuccess) {
        if (verbose) {
          Log("[WARN] Unable to get PCI bus ID for GPU 0; skipping AMD-SMI pod membership query\n");
        }
        return;
      }

      amdsmi_bdf_t bdf = {};
      unsigned domain, bus, device, func;
      if (sscanf(pciBusId, "%x:%x:%x.%x", &domain, &bus, &device, &func) != 4) {
        if (verbose) {
          Log("[WARN] Unable to parse PCI bus ID '%s'; skipping AMD-SMI pod membership query\n", pciBusId);
        }
        return;
      }
      bdf.domain_number   = domain;
      bdf.bus_number      = bus;
      bdf.device_number   = device;
      bdf.function_number = func;

      amdsmi_processor_handle gpuHandle;
      amdsmi_status_t err = amdsmi_get_processor_handle_from_bdf(bdf, &gpuHandle);
      if (err != AMDSMI_STATUS_SUCCESS) {
        if (verbose) {
          const char *errString = NULL;
          amdsmi_status_code_to_string(err, &errString);
          Log("[WARN] Unable to get processor handle for GPU 0 at %s [%s]\n",
                            pciBusId, errString);
        }
      } else {
        amdsmi_fabric_info_t fabricInfo;
        err = amdsmi_get_gpu_fabric_info(gpuHandle, &fabricInfo);
        if (err == AMDSMI_STATUS_SUCCESS) {
          // NOTE: vpod_id is a uint32_t but System holds it as an int64_t to allow for
          //       vpodId == -1 to represent no pod present
          memcpy(ppodId, &fabricInfo.fabric_info.fabric_version.v1.ppod_id,
                 sizeof(fabricInfo.fabric_info.fabric_version.v1.ppod_id));
          vpodId = fabricInfo.fabric_info.fabric_version.v1.vpod_id;
        } else if (verbose) {
          const char *errString = NULL;
          amdsmi_status_code_to_string(err, &errString);
          Log("[WARN] Unable to get fabric info from AMD SMI [%s]\n", errString);
        }
      }
    }
#endif
#endif
  }

  void System::GetRankTopology(RankTopology& topo)
  {
    // Clear topology structure first
    topo.numExecutors.clear();
    topo.numExecutorSubIndices.clear();
    topo.numSubExecutors.clear();
    topo.closestCpuNumaToGpu.clear();
    topo.closestCpuNumaToNic.clear();
    topo.closestNicsToGpu.clear();
    topo.closestGpusToNic.clear();

    memset(topo.hostname, 0, sizeof(topo.hostname));
    gethostname(topo.hostname, 32);
    char* firstDotPtr = std::strchr(topo.hostname, '.');
    if (firstDotPtr) *firstDotPtr = 0;

    // Collect Pod membership
    CollectPodMembership(topo.ppodId, topo.vpodId);

    // CPU Executor
    int numCpus = numa_num_configured_nodes();
    topo.numExecutors[EXE_CPU] = numCpus;

    std::string cpuName = GetCpuName();

    for (int exeIndex = 0; exeIndex < numCpus; exeIndex++) {
      topo.numExecutorSubIndices[{EXE_CPU, exeIndex}] = 0;
      topo.executorName[{EXE_CPU, exeIndex}] = cpuName;
    }

    for (int cpuCore = 0; cpuCore < numa_num_configured_cpus(); cpuCore++) {
      topo.numSubExecutors[{EXE_CPU, numa_node_of_cpu(cpuCore)}]++;
    }

    if (verbose) {
      for (int exeIndex = 0; exeIndex < numCpus; exeIndex++) {
        Log("[INFO] Rank %03d: CPU [%02d/%02d] %03d cores (%s)\n", rank, exeIndex, numCpus,
            topo.numSubExecutors[{EXE_CPU, exeIndex}],
            topo.executorName[{EXE_CPU, exeIndex}].c_str());
      }
    }

    // GPU Executor
    int numGpus = 0;
    hipError_t status = hipGetDeviceCount(&numGpus);
    if (status != hipSuccess) numGpus = 0;
    topo.numExecutors[EXE_GPU_GFX]  = numGpus;
    topo.numExecutors[EXE_GPU_DMA]  = numGpus;
    topo.numExecutors[EXE_GPU_BDMA] = numGpus;
    topo.numExecutors[EXE_GPU_TDM]  = numGpus;

    std::vector<std::string> gpuArchNames(numGpus);

    for (int exeIndex = 0; exeIndex < numGpus; exeIndex++) {
      int numDeviceCUs  = 0;
      int numXccs       = 0;
      int numDmaEngines = 0;
      int closestNuma   = -1;

      if (hipDeviceGetAttribute(&numDeviceCUs, hipDeviceAttributeMultiprocessorCount, exeIndex) != hipSuccess) {
        numDeviceCUs = 0;
      }

      std::string gpuName = "Unknown GPU";
      hipDeviceProp_t props;
      if (hipGetDeviceProperties(&props, exeIndex) == hipSuccess) {
        gpuName = props.name;
        std::string fullName = props.gcnArchName;
        gpuArchNames[exeIndex] = fullName.substr(0, fullName.find(':'));
      }
      topo.executorName[{EXE_GPU_GFX,  exeIndex}] = gpuName;
      topo.executorName[{EXE_GPU_DMA,  exeIndex}] = gpuName;
      topo.executorName[{EXE_GPU_BDMA, exeIndex}] = gpuName;
      topo.executorName[{EXE_GPU_TDM,  exeIndex}] = gpuName;

#if !defined(__NVCC__)
      hsa_agent_t gpuAgent = gpuAgents[exeIndex];
      if (hsa_agent_get_info(gpuAgent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_XCC, &numXccs) != HSA_STATUS_SUCCESS)
        numXccs = 1;

      int numEnginesA, numEnginesB;
      if (hsa_agent_get_info(gpuAgent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SDMA_ENG, &numEnginesA)
          == HSA_STATUS_SUCCESS)
        numDmaEngines += numEnginesA;
      if (hsa_agent_get_info(gpuAgent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG, &numEnginesB)
          == HSA_STATUS_SUCCESS)
        numDmaEngines += numEnginesB;

      hsa_agent_t closestCpuAgent;
      if (hsa_agent_get_info(gpuAgent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NEAREST_CPU, &closestCpuAgent)
          == HSA_STATUS_SUCCESS) {
        for (int cpuIndex = 0; cpuIndex < numCpus; cpuIndex++) {
          hsa_agent_t cpuAgent = cpuAgents[cpuIndex];
          if (cpuAgent.handle == closestCpuAgent.handle) {
            closestNuma = cpuIndex;
            break;
          }
        }
      }
#endif
      topo.numExecutorSubIndices[{EXE_GPU_GFX,  exeIndex}] = numXccs;
      topo.numExecutorSubIndices[{EXE_GPU_DMA,  exeIndex}] = numDmaEngines;
      topo.numExecutorSubIndices[{EXE_GPU_BDMA, exeIndex}] = 0;
      topo.numExecutorSubIndices[{EXE_GPU_TDM,  exeIndex}] = 0;

      topo.numSubExecutors[{EXE_GPU_GFX, exeIndex}]  = numDeviceCUs;
      topo.numSubExecutors[{EXE_GPU_DMA, exeIndex}]  = 1;
      topo.numSubExecutors[{EXE_GPU_BDMA, exeIndex}] = numDmaEngines;
      topo.numSubExecutors[{EXE_GPU_TDM, exeIndex}]  = numDeviceCUs;

      topo.closestCpuNumaToGpu[exeIndex] = closestNuma;
      topo.closestNicsToGpu[exeIndex] = {};
    }

    // NIC Executor
    int numNics = 0;
    if (IsIbvSymbolsReady())
    {
      numNics = GetIbvDeviceList().size();
      for (int exeIndex = 0; exeIndex < numNics; exeIndex++) {
        topo.closestCpuNumaToNic[exeIndex] = GetIbvDeviceList()[exeIndex].numaNode;
        topo.executorName[{EXE_NIC, exeIndex}] = GetIbvDeviceList()[exeIndex].name;
        topo.nicIsActive[exeIndex] = GetIbvDeviceList()[exeIndex].hasActivePort;
        if (verbose) {
          auto const& nic = GetIbvDeviceList()[exeIndex];
          Log("[INFO] Rank %03d: NIC [%02d/%02d] %s BDF %s NUMA %d active=%s\n",
              rank, exeIndex, numNics, nic.name.c_str(),
              nic.busId.empty() ? "?" : nic.busId.c_str(),
              topo.closestCpuNumaToNic[exeIndex],
              nic.hasActivePort ? "yes" : "no");
        }
      }
    }
    topo.numExecutors[EXE_NIC] = topo.numExecutors[EXE_NIC_NEAREST] = numNics;

    for (int nicIndex = 0; nicIndex < numNics; nicIndex++) {
      topo.numSubExecutors[{EXE_NIC, nicIndex}] = 0;
      topo.numExecutorSubIndices[{EXE_NIC, nicIndex}] = 0;
      std::string gpuName = "Unknown GPU";

    }
    for (int gpuIndex = 0; gpuIndex < numGpus; gpuIndex++) {
      topo.numSubExecutors[{EXE_NIC_NEAREST, gpuIndex}] = 0;
      topo.numExecutorSubIndices[{EXE_NIC_NEAREST, gpuIndex}] = 0;
    }

    // Build GPU bus address list once — reused by NIC proximity loops and verbose output
    std::vector<std::string> gpuAddressList;
    for (int gpuIdx = 0; gpuIdx < numGpus; gpuIdx++) {
      char hipPciBusId[64];
      hipError_t err = hipDeviceGetPCIBusId(hipPciBusId, sizeof(hipPciBusId), gpuIdx);
      gpuAddressList.push_back(err == hipSuccess ? std::string(hipPciBusId) : "");
      if (err != hipSuccess)
        Log("Failed to get PCI Bus ID for HIP device %d: %s\n", gpuIdx, hipGetErrorString(err));
    }

    // Figure out closest NICs to GPUs
    // Build up list of NIC bus addresses
    std::vector<std::string> ibvAddressList;
    auto const& ibvDeviceList = GetIbvDeviceList();
    if (IsIbvSymbolsReady()) {
      for (auto const& ibvDevice : ibvDeviceList)
        ibvAddressList.push_back(ibvDevice.hasActivePort ? ibvDevice.busId : "");

      // Track how many times a device has been assigned as "closest"
      // This allows distributed work across devices using multiple ports (sharing the same busID)
      // NOTE: This isn't necessarily optimal, but likely to work in most cases involving multi-port
      // Counter example:
      //
      //  G0 prefers (N0,N1), picks N0
      //  G1 prefers (N1,N2), picks N1
      //  G2 prefers N0,      picks N0
      //
      //  instead of G0->N1, G1->N2, G2->N0

      std::vector<int> assignedCount(ibvDeviceList.size(), 0);

      // Loop over each GPU to find the closest NIC(s) based on PCIe address
      for (int gpuIndex = 0; gpuIndex < numGpus; gpuIndex++) {
        if (gpuAddressList[gpuIndex].empty()) continue;
        const char* hipPciBusId = gpuAddressList[gpuIndex].c_str();

        // Find closest NICs
        std::set<int> closestNicIdxs = GetNearestDevicesInTree(hipPciBusId, ibvAddressList);

        // Pick the least-used NIC to assign as closest
        int closestIdx = -1;
        for (auto idx : closestNicIdxs) {
          if (closestIdx == -1 || assignedCount[idx] < assignedCount[closestIdx])
            closestIdx = idx;
        }

        // The following will only use distance between bus IDs
        // to determine the closest NIC to GPU if the PCIe tree approach fails
        if (closestIdx < 0) {
  #ifdef VERBS_DEBUG
          Log("[WARN] Falling back to PCIe bus ID distance to determine proximity\n");
  #endif
          int minDistance = std::numeric_limits<int>::max();
          for (int nicIndex = 0; nicIndex < numNics; nicIndex++) {
            if (ibvDeviceList[nicIndex].busId != "") {
              int distance = GetBusIdDistance(hipPciBusId, ibvDeviceList[nicIndex].busId);
              if (distance < minDistance && distance >= 0) {
                minDistance = distance;
                closestIdx = nicIndex;
              }
            }
          }
        }
        if (closestIdx != -1) {
          topo.closestNicsToGpu[gpuIndex].push_back(closestIdx);
          assignedCount[closestIdx]++;
        }
      }

      // Compute the reverse mapping: closest GPU(s) for each NIC
      // Loop over each NIC to find the closest GPU(s) based on PCIe address
      for (int nicIndex = 0; nicIndex < numNics; nicIndex++) {
        if (!ibvDeviceList[nicIndex].hasActivePort || ibvDeviceList[nicIndex].busId.empty()) {
          continue;
        }

        // Find closest GPUs using LCA algorithm
        std::set<int> closestGpuIdxs = GetNearestDevicesInTree(ibvDeviceList[nicIndex].busId, gpuAddressList);

        if (closestGpuIdxs.empty()) {
          // Fallback: use bus ID distance
          int minDistance = std::numeric_limits<int>::max();
          int closestIdx = -1;

          for (int gpuIdx = 0; gpuIdx < numGpus; gpuIdx++) {
            if (gpuAddressList[gpuIdx].empty()) continue;

            int distance = GetBusIdDistance(ibvDeviceList[nicIndex].busId, gpuAddressList[gpuIdx]);
            if (distance >= 0 && distance < minDistance) {
              minDistance = distance;
              closestIdx = gpuIdx;
            }
          }

          if (closestIdx != -1) {
            topo.closestGpusToNic[nicIndex].push_back(closestIdx);
          }
        } else {
          // Store all GPUs that are equally close
          for (int idx : closestGpuIdxs) {
            topo.closestGpusToNic[nicIndex].push_back(idx);
          }
        }
      }
    }

    if (verbose) {
      for (int exeIndex = 0; exeIndex < numGpus; exeIndex++) {
        const char* gpuBdf = gpuAddressList[exeIndex].c_str();
        int numXccs    = topo.numExecutorSubIndices[{EXE_GPU_GFX, exeIndex}];
        int numCus     = topo.numSubExecutors[{EXE_GPU_GFX, exeIndex}];
        int numDmaEngs = topo.numExecutorSubIndices[{EXE_GPU_DMA, exeIndex}];
        Log("[INFO] Rank %03d: GPU [%02d/%02d] %03d CUs %d XCCs %d DMA-eng NUMA %d BDF %s (%s) Closest NICs:",
            rank, exeIndex, numGpus, numCus, numXccs, numDmaEngs,
            topo.closestCpuNumaToGpu[exeIndex], gpuBdf, gpuArchNames[exeIndex].c_str());
        if (topo.closestNicsToGpu[exeIndex].size() == 0) {
          Log(" none\n");
        } else {
          for (auto nicIndex : topo.closestNicsToGpu[exeIndex]) {
            Log(" %d", nicIndex);
          }
          Log("\n");
        }
      }
      if (IsIbvSymbolsReady()) {
        for (int nicIndex = 0; nicIndex < numNics; nicIndex++) {
          Log("[INFO] Rank %03d: NIC [%02d/%02d] %s Closest GPUs:", rank, nicIndex, numNics,
                            ibvDeviceList[nicIndex].name.c_str());
          if (topo.closestGpusToNic[nicIndex].size() == 0) {
            Log(" none");
          } else {
            for (auto gpuIndex : topo.closestGpusToNic[nicIndex]) {
              Log(" %d", gpuIndex);
            }
          }
          Log("\n");
        }
      }
    }
  }

  template <typename KeyType, typename ValType>
  void System::SendMap(int peerRank, std::map<KeyType, std::vector<ValType>> const& mapToSend) const
  {
    size_t mapSize = mapToSend.size();
    SendData(peerRank, sizeof(mapSize), &mapSize);
    for (auto const& p : mapToSend) {
      SendData(peerRank, sizeof(p.first), &p.first);
      size_t vectorSize = p.second.size();
      SendData(peerRank, sizeof(vectorSize), &vectorSize);
      for (auto const& v : p.second) {
        SendData(peerRank, sizeof(v), &v);
      }
    }
    fflush(stdout);
  }

  template <typename KeyType, typename ValType>
  void System::SendMap(int peerRank, std::map<KeyType, ValType> const& mapToSend) const
  {
    size_t mapSize = mapToSend.size();
    SendData(peerRank, sizeof(mapSize), &mapSize);
    for (auto const p : mapToSend) {
      SendData(peerRank, sizeof(p), &p);
    }
  }

  template <typename KeyType>
  void System::SendMap(int peerRank, std::map<KeyType, std::string> const& mapToSend) const
  {
    size_t mapSize = mapToSend.size();
    SendData(peerRank, sizeof(mapSize), &mapSize);
    for (auto const p : mapToSend) {
      size_t strlen = p.second.size();
      SendData(peerRank, sizeof(p.first), &p.first);
      SendData(peerRank, sizeof(strlen), &strlen);
      if (strlen) SendData(peerRank, strlen, p.second.data());
    }
  }

  template <typename KeyType, typename ValType>
  void System::RecvMap(int peerRank, std::map<KeyType, std::vector<ValType>>& mapToRecv) const
  {
    mapToRecv.clear();
    size_t mapSize;
    RecvData(peerRank, sizeof(mapSize), &mapSize);
    for (size_t i = 0; i < mapSize; i++) {
      KeyType key;
      size_t vectorSize;
      std::vector<ValType> values;
      RecvData(peerRank, sizeof(key), &key);
      RecvData(peerRank, sizeof(vectorSize), &vectorSize);
      if (vectorSize) {
        values.resize(vectorSize);
        for (size_t j = 0; j < vectorSize; j++) {
          RecvData(peerRank, sizeof(ValType), &values[j]);
        }
      }
      mapToRecv[key] = values;
    }
  }

  template <typename KeyType>
  void System::RecvMap(int peerRank, std::map<KeyType, std::string>& mapToRecv) const
  {
    mapToRecv.clear();
    size_t mapSize;
    RecvData(peerRank, sizeof(mapSize), &mapSize);
    for (size_t i = 0; i < mapSize; i++) {
      KeyType key;
      size_t strlen;
      std::string value;
      RecvData(peerRank, sizeof(key), &key);
      RecvData(peerRank, sizeof(size_t), &strlen);
      if (strlen) {
        value.resize(strlen);
        RecvData(peerRank, strlen, value.data());
      }
      mapToRecv[key] = value;
    }
  }

  template <typename KeyType, typename ValType>
  void System::RecvMap(int peerRank, std::map<KeyType, ValType>& mapToRecv) const
  {
    mapToRecv.clear();
    size_t mapSize;
    RecvData(peerRank, sizeof(mapSize), &mapSize);
    for (size_t i = 0; i < mapSize; i++) {
      std::pair<KeyType, ValType> p;
      RecvData(peerRank, sizeof(p), &p);
      mapToRecv[p.first] = p.second;
    }
  }

  void System::SendRankTopo(int peerRank, RankTopology const& topo) const
  {
    SendData(peerRank, sizeof(topo.hostname), topo.hostname);
    SendData(peerRank, sizeof(topo.ppodId), topo.ppodId);
    SendData(peerRank, sizeof(topo.vpodId), &topo.vpodId);
    SendMap(peerRank, topo.numExecutors);
    SendMap(peerRank, topo.numExecutorSubIndices);
    SendMap(peerRank, topo.numSubExecutors);
    SendMap(peerRank, topo.closestCpuNumaToGpu);
    SendMap(peerRank, topo.closestCpuNumaToNic);
    SendMap(peerRank, topo.nicIsActive);
    SendMap(peerRank, topo.closestNicsToGpu);
    SendMap(peerRank, topo.closestGpusToNic);
    SendMap(peerRank, topo.executorName);
  };

  void System::RecvRankTopo(int peerRank, RankTopology& topo) const
  {
    RecvData(peerRank, sizeof(topo.hostname), topo.hostname);
    RecvData(peerRank, sizeof(topo.ppodId), topo.ppodId);
    RecvData(peerRank, sizeof(topo.vpodId), &topo.vpodId);
    RecvMap(peerRank, topo.numExecutors);
    RecvMap(peerRank, topo.numExecutorSubIndices);
    RecvMap(peerRank, topo.numSubExecutors);
    RecvMap(peerRank, topo.closestCpuNumaToGpu);
    RecvMap(peerRank, topo.closestCpuNumaToNic);
    RecvMap(peerRank, topo.nicIsActive);
    RecvMap(peerRank, topo.closestNicsToGpu);
    RecvMap(peerRank, topo.closestGpusToNic);
    RecvMap(peerRank, topo.executorName);
  }

  template <typename T>
  void System::BroadcastVector(int root, vector<T>& data) const
  {
    // This assumes T is trivially copyable
    static_assert(std::is_trivially_copyable<T>::value);

    size_t len = data.size();
    Broadcast(root, sizeof(len), &len);
    data.resize(len);
    if (len) {
      Broadcast(root, sizeof(T) * len, data.data());
    }
  }

  void System::BroadcastString(int root, std::string& string) const
  {
    size_t len = string.size();
    Broadcast(root, sizeof(len), &len);
    string.resize(len);
    if (len) {
      Broadcast(root, len, string.data());
    }
  }

  void System::BroadcastExeResult(int root, ExeResult& exeResult) const
  {
    #define BROADCAST(X)  Broadcast(root, sizeof(X), &X)
    BROADCAST(exeResult.numBytes);
    BROADCAST(exeResult.avgDurationMsec);
    BROADCAST(exeResult.avgBandwidthGbPerSec);
    BROADCAST(exeResult.sumBandwidthGbPerSec);
    BroadcastVector(root, exeResult.transferIdx);
    #undef BROADCAST
  }

  void System::BroadcastTfrResult(int root, TransferResult& tfrResult) const
  {
    #define BROADCAST(X)  Broadcast(root, sizeof(X), &X)
    BROADCAST(tfrResult.numBytes);
    BROADCAST(tfrResult.avgDurationMsec);
    BROADCAST(tfrResult.avgBandwidthGbPerSec);
    BroadcastVector(root, tfrResult.perIterMsec);
    BROADCAST(tfrResult.exeDevice);
    BROADCAST(tfrResult.exeDstDevice);

    // Per-Iteration CU results need to be handled in a custom manner
    size_t perIterCuSize = tfrResult.perIterCUs.size();
    BROADCAST(perIterCuSize);

    if (perIterCuSize > 0) {
      tfrResult.perIterCUs.resize(perIterCuSize);
      for (size_t i = 0; i < perIterCuSize; i++) {
        size_t setSize;

        //vector<set<pair<int,int>>> perIterCUs;      ///< GFX-Executor only. XCC:CU used per iteration

        if (GetRank() == root) {
          setSize = tfrResult.perIterCUs[i].size();
          BROADCAST(setSize);
          if (setSize > 0) {
            for (pair<int,int> const& x : tfrResult.perIterCUs[i]) {
              pair<int, int> p = x;
              BROADCAST(p);
            }
          }
        } else {
          BROADCAST(setSize);
          tfrResult.perIterCUs[i].clear();
          for (size_t j = 0; j < setSize; j++) {
            pair<int, int> p;
            BROADCAST(p);
            tfrResult.perIterCUs[i].insert(p);
          }
        }
      }
    } else {
      tfrResult.perIterCUs.clear();
    }
    #undef BROADCAST
  };

  void System::AllGatherErrors(vector<ErrResult>& errResults) const
  {
    if (commMode == COMM_NONE) return;

    vector<ErrResult> tempResults = std::move(errResults);

    for (int i = 0; i < numRanks; i++) {
      size_t errListSize = tempResults.size();
      Broadcast(i, sizeof(errListSize), &errListSize);
      for (size_t j = 0; j < errListSize; j++) {
        ErrResult errResult;
        if (rank == i) errResult = tempResults[j];
        Broadcast(i, sizeof(errResult.errType), &errResult.errType);
        BroadcastString(i, errResult.errMsg);
        errResult.errMsg += " (Rank " + std::to_string(i) + ")";
        errResults.push_back(errResult);
      }
    }
  }

#if !defined(__NVCC__)
  // Get the hsa_agent_t associated with a ExeDevice
  ErrResult System::GetHsaAgent(ExeDevice const& exeDevice, hsa_agent_t& agent) const
  {
    int numCpus = static_cast<int>(cpuAgents.size());
    int numGpus = static_cast<int>(gpuAgents.size());
    int exeIndex = exeDevice.exeIndex;

    switch (exeDevice.exeType) {
    case EXE_CPU:
      if (exeIndex < 0 || exeIndex >= numCpus)
        return {ERR_FATAL, "CPU index must be between 0 and %d inclusively", numCpus - 1};
      agent = cpuAgents[exeDevice.exeIndex];
      break;
    case EXE_GPU_GFX: case EXE_GPU_DMA: case EXE_GPU_BDMA: case EXE_GPU_TDM:
      if (exeIndex < 0 || exeIndex >= numGpus)
        return {ERR_FATAL, "GPU index must be between 0 and %d inclusively", numGpus - 1};
      agent = gpuAgents[exeIndex];
      break;
    default:
      return {ERR_FATAL,
              "Attempting to get HSA agent of unknown or unsupported executor type (%d)",
              exeDevice.exeType};
    }
    return ERR_NONE;
  }

  // Get the hsa_agent_t associated with a MemDevice
  ErrResult System::GetHsaAgent(MemDevice const& memDevice, hsa_agent_t& agent) const
  {
    if (memDevice.memType == MEM_CPU_CLOSEST)
      return GetHsaAgent({EXE_CPU, GetClosestCpuNumaToGpu(memDevice.memIndex)}, agent);
    if (IsCpuMemType(memDevice.memType)) return GetHsaAgent({EXE_CPU, memDevice.memIndex}, agent);
    if (IsGpuMemType(memDevice.memType)) return GetHsaAgent({EXE_GPU_GFX, memDevice.memIndex}, agent);
    return {ERR_FATAL,
            "Unable to get HSA agent for memDevice (%d,%d)",
            memDevice.memType, memDevice.memIndex};
  }
#endif

  void System::CollectTopology()
  {
    // Cache the HSA agents for each device
#if !defined(__NVCC__)
    {
      hsa_amd_pointer_info_t info;
      info.size = sizeof(info);

      // Callback to process each agent
      auto cpuAgentCallback = [](hsa_agent_t agent, void* data) -> hsa_status_t {
        std::map<int, hsa_agent_t>* agents = static_cast<std::map<int, hsa_agent_t>*>(data);

        hsa_device_type_t deviceType;
        hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &deviceType);
        if (status != HSA_STATUS_SUCCESS) return status;

        if (deviceType == HSA_DEVICE_TYPE_CPU) {
          uint32_t nodeId;
          status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &nodeId);
          if (status == HSA_STATUS_SUCCESS) {
            (*agents)[nodeId] = agent;
          } else {
            return status;
          }
        }
        return HSA_STATUS_SUCCESS;
      };

      // Index GPU agents
      int numGpus = 0;
      hipError_t status = hipGetDeviceCount(&numGpus);
      if (status != hipSuccess) numGpus = 0;
      gpuAgents.clear();
      char *tempBuffer;
      for (int i = 0; i < numGpus; i++) {
        AllocateMemory({MEM_GPU, i}, 1024, (void**)&tempBuffer);
        hsa_amd_pointer_info(tempBuffer, &info, NULL, NULL, NULL);
        gpuAgents.push_back(info.agentOwner);
        DeallocateMemory(MEM_GPU, tempBuffer, 1024);
      }

      // Index CPU agents (done after HIP initialization)
      std::map<int, hsa_agent_t> cpuAgentMap;
      hsa_iterate_agents(cpuAgentCallback, &cpuAgentMap);

      cpuAgents.clear();
      int numCpus = numa_num_configured_nodes();
      cpuAgents.resize(numCpus);
      for (int i = 0; i < numCpus; i++) {
        if (cpuAgentMap.count(i)) {
          cpuAgents[i] = cpuAgentMap[i];
        }
      }
    }
#endif

    // Collect the topology of the local node
    RankTopology localTopo;
    GetRankTopology(localTopo);

    // Distribute amongst all ranks
    rankInfo.resize(numRanks);

    if (rank == 0) {
      // Receive topology info from each rank
      rankInfo[0] = localTopo;
      for (int peerRank = 1; peerRank < numRanks; peerRank++) {
        if (verbose) {
          Log("[INFO] Rank 0 receives topology from Rank %d\n", peerRank);
        }
        RecvRankTopo(peerRank, rankInfo[peerRank]);
      }

      // Send out full set of info to each rank
      for (int peerRank = 1; peerRank < numRanks; peerRank++) {
        for (int i = 0; i < numRanks; i++) {
          if (verbose) {
            Log("[INFO] Rank 0 sends topology %d to Rank %d\n", i, peerRank);
          }
          SendRankTopo(peerRank, rankInfo[i]);
        }
      }
    } else {
      // Send local topology info back to root
      if (verbose) {
        Log("[INF0] Rank %d sends topology from Rank 0\n", rank);
      }
      SendRankTopo(0, localTopo);

      for (int i = 0; i < numRanks; i++) {
        RecvRankTopo(0, rankInfo[i]);
        if (verbose) {
          Log("[INF0] Rank %d receives topology %d from Rank 0\n", rank, i);
        }
      }
    }
  }

  int System::GetNumExecutors(ExeType exeType, int targetRank) const
  {
    if (targetRank < 0 || targetRank >= numRanks) targetRank = rank;
    if (rankInfo[targetRank].numExecutors.count(exeType) == 0) return 0;
    return rankInfo[targetRank].numExecutors.at(exeType);
  }

  int System::GetNumExecutorSubIndices(ExeDevice exeDevice) const
  {
    int targetRank = exeDevice.exeRank;
    if (targetRank < 0 || targetRank >= numRanks) targetRank = rank;
    if (rankInfo[targetRank].numExecutorSubIndices.count({exeDevice.exeType, exeDevice.exeIndex}) == 0)
      return 0;
    return rankInfo[targetRank].numExecutorSubIndices.at({exeDevice.exeType, exeDevice.exeIndex});
  }

  int System::GetNumSubExecutors(ExeDevice exeDevice) const
  {
    int targetRank = exeDevice.exeRank;
    if (targetRank < 0 || targetRank >= numRanks) targetRank = rank;
    if (rankInfo[targetRank].numSubExecutors.count({exeDevice.exeType, exeDevice.exeIndex}) == 0)
      return 0;
    return rankInfo[targetRank].numSubExecutors.at({exeDevice.exeType, exeDevice.exeIndex});
  }

  int System::GetClosestCpuNumaToGpu(int gpuIndex, int targetRank) const
  {
    if (targetRank < 0 || targetRank >= numRanks) targetRank = rank;
    if (gpuIndex < 0 || gpuIndex >= GetNumExecutors(EXE_GPU_GFX, targetRank)) return 0;
    return rankInfo[targetRank].closestCpuNumaToGpu.at(gpuIndex);
  }

  int System::GetClosestCpuNumaToNic(int nicIndex, int targetRank) const
  {
    if (targetRank < 0 || targetRank >= numRanks) targetRank = rank;
    if (nicIndex < 0 || nicIndex >= GetNumExecutors(EXE_NIC, targetRank)) return 0;
    return rankInfo[targetRank].closestCpuNumaToNic.at(nicIndex);
  }

  void System::GetClosestNicsToGpu(std::vector<int>& nicIndices, int gpuIndex, int targetRank) const
  {
    nicIndices.clear();
    if (targetRank < 0 || targetRank >= numRanks) targetRank = rank;
    if (gpuIndex < 0 || gpuIndex >= GetNumExecutors(EXE_GPU_GFX, targetRank)) return;
    nicIndices = rankInfo[targetRank].closestNicsToGpu.at(gpuIndex);
  }

  void System::GetClosestGpusToNic(std::vector<int>& gpuIndices, int nicIndex, int targetRank) const
  {
    gpuIndices.clear();
    if (targetRank < 0 || targetRank >= numRanks) targetRank = rank;
    if (nicIndex < 0 || nicIndex >= GetNumExecutors(EXE_NIC, targetRank)) return;
    if (rankInfo[targetRank].closestGpusToNic.count(nicIndex) > 0) {
      gpuIndices = rankInfo[targetRank].closestGpusToNic.at(nicIndex);
    }
  }

  std::string System::GetHostname(int targetRank) const
  {
    if (targetRank < 0 || targetRank >= numRanks) targetRank = rank;
    return rankInfo[targetRank].hostname;
  }

  int64_t System::GetPodIdx(int targetRank) const
  {
    using PodKey = std::pair<std::array<char, 16>, int64_t>;

    static std::map<PodKey, int64_t> podIdxMap;
    static bool initialized = false;

    if (!initialized) {
      int64_t nextIdx = 0;
      for (int r = 0; r < numRanks; r++) {
        PodKey key;
        memcpy(key.first.data(), rankInfo[r].ppodId, 16);
        key.second = rankInfo[r].vpodId;

        // vpodIdx == -1 means not part of any pod; assign -1 directly
        if (key.second == -1) continue;

        if (podIdxMap.find(key) == podIdxMap.end()) {
          podIdxMap[key] = nextIdx++;
        }
      }
      initialized = true;
    }

    if (targetRank < 0 || targetRank >= numRanks) targetRank = rank;

    PodKey key;
    memcpy(key.first.data(), rankInfo[targetRank].ppodId, 16);
    key.second = rankInfo[targetRank].vpodId;

    if (key.second == -1) return -1;

    return podIdxMap[key];
  }

  bool System::IsSamePod(int targetRank, int sourceRank) const
  {
    if (sourceRank < 0 || sourceRank >= numRanks) sourceRank = rank;
    if (GetPodIdx(sourceRank) == -1 || GetPodIdx(targetRank) == -1)
      return false;
    return GetPodIdx(sourceRank) == GetPodIdx(targetRank);
  }

  std::string System::GetExecutorName(ExeDevice exeDevice) const
  {
    int targetRank = exeDevice.exeRank;
    if (targetRank < 0 || targetRank >= numRanks) targetRank = rank;

    if (rankInfo[targetRank].executorName.count({exeDevice.exeType, exeDevice.exeIndex}) == 0)
      return "Unknown device";
    return rankInfo[targetRank].executorName.at({exeDevice.exeType, exeDevice.exeIndex});
  }

  int System::NicIsActive(int nicIndex, int targetRank) const
  {
    if (targetRank < 0 || targetRank >= numRanks) targetRank = rank;
    if (rankInfo[targetRank].nicIsActive.count(nicIndex) == 0) return 0;
    return rankInfo[targetRank].nicIsActive.at(nicIndex);
  }

  int GetNumExecutors(ExeType exeType, int targetRank)
  {
    return System::Get().GetNumExecutors(exeType, targetRank);
  }

  int GetNumExecutors(MemType memType, int targetRank)
  {
    if (IsCpuMemType(memType)) return GetNumExecutors(EXE_CPU,     targetRank);
    if (IsGpuMemType(memType)) return GetNumExecutors(EXE_GPU_GFX, targetRank);
    return 0;
  }

  int GetNumSubExecutors(ExeDevice exeDevice)
  {
    return System::Get().GetNumSubExecutors(exeDevice);
  }

  int GetNumExecutorSubIndices(ExeDevice exeDevice)
  {
    return System::Get().GetNumExecutorSubIndices(exeDevice);
  }

  int GetClosestCpuNumaToGpu(int gpuIndex, int targetRank)
  {
    return System::Get().GetClosestCpuNumaToGpu(gpuIndex, targetRank);
  }

  int GetClosestCpuNumaToNic(int nicIndex, int targetRank)
  {
    return System::Get().GetClosestCpuNumaToNic(nicIndex, targetRank);
  }

  int GetClosestNicToGpu(int gpuIndex, int targetRank)
  {
    std::vector<int> nicIndices;
    System::Get().GetClosestNicsToGpu(nicIndices, gpuIndex, targetRank);
    if (nicIndices.size() == 0) return -1;
    return nicIndices[0];
  }

  void GetClosestNicsToGpu(std::vector<int>& nicIndices, int gpuIndex, int targetRank)
  {
    System::Get().GetClosestNicsToGpu(nicIndices, gpuIndex, targetRank);
  }

  int GetClosestGpuToNic(int nicIndex, int targetRank)
  {
    std::vector<int> gpuIndices;
    System::Get().GetClosestGpusToNic(gpuIndices, nicIndex, targetRank);
    if (gpuIndices.size() == 0) return -1;
    return gpuIndices[0];
  }

  void GetClosestGpusToNic(std::vector<int>& gpuIndices, int nicIndex, int targetRank)
  {
    System::Get().GetClosestGpusToNic(gpuIndices, nicIndex, targetRank);
  }

  void GetClosestNicsToCpu(std::vector<int>& nicIndices, int cpuIndex, int targetRank)
  {
    int numNics = GetNumExecutors(EXE_NIC, targetRank);
    nicIndices.clear();
    for (int nicIndex = 0; nicIndex < numNics; nicIndex++) {
      if (GetClosestCpuNumaToNic(nicIndex, targetRank) == cpuIndex) {
        nicIndices.push_back(nicIndex);
      }
    }
  }

  int GetRank()
  {
    return System::Get().GetRank();
  }

  int GetNumRanks()
  {
    return System::Get().GetNumRanks();
  }

  int GetCommMode()
  {
    return System::Get().GetCommMode();
  }

  std::string GetHostname(int targetRank)
  {
    return System::Get().GetHostname(targetRank);
  }

  int64_t GetPodIdx(int targetRank)
  {
    return System::Get().GetPodIdx(targetRank);
  }

  bool IsSamePod(int targetRank, int sourceRank)
  {
    return System::Get().IsSamePod(targetRank, sourceRank);
  }

  std::string GetExecutorName(ExeDevice exeDevice)
  {
    return System::Get().GetExecutorName(exeDevice);
  }

  int NicIsActive(int nicIndex, int targetRank)
  {
    return System::Get().NicIsActive(nicIndex, targetRank);
  }

// Undefine CUDA compatibility macros
#if defined(__NVCC__)

// ROCm specific
#undef wall_clock64
#undef gcnArchName

// Datatypes
#undef hipDeviceProp_t
#undef hipError_t
#undef hipEvent_t
#undef hipStream_t
#undef hipMemAllocationProp
#undef hipMemGenericAllocationHandle_t
#undef hipMemAccessDesc
#undef hipMemFabricHandle_t

// Enumerations
#undef hipDeviceAttributeClockRate
#undef hipDeviceAttributeMultiprocessorCount
#undef hipDeviceAttributeMaxSharedMemoryPerBlock
#undef hipDeviceAttributeWarpSize
#undef hipErrorPeerAccessAlreadyEnabled
#undef hipFuncCachePreferShared
#undef hipMemcpyDefault
#undef hipMemcpyKind
#undef hipMemcpyDeviceToHost
#undef hipMemcpyHostToDevice
#undef hipSuccess
#undef hipMemLocationTypeDevice
#undef hipMemAllocationTypePinned
//#undef hipMemAllocationTypeUncached
#undef hipMemHandleTypeFabric
#undef hipMemAllocationGranularityRecommended
#undef hipMemAccessFlagsProtReadWrite

// Functions
#undef hipDeviceCanAccessPeer
#undef hipDeviceEnablePeerAccess
#undef hipDeviceGetAttribute
#undef hipDeviceGetPCIBusId
#undef hipDeviceSetCacheConfig
#undef hipDeviceSynchronize
#undef hipEventCreate
#undef hipEventDestroy
#undef hipEventElapsedTime
#undef hipEventRecord
#undef hipFree
#undef hipGetDeviceCount
#undef hipGetDeviceProperties
#undef hipGetErrorString
#undef hipGetLastError
#undef hipHostFree
#undef hipHostMalloc
#undef hipMalloc
#undef hipMallocManaged
#undef hipMemcpy
#undef hipMemcpyAsync
#undef hipMemset
#undef hipMemsetAsync
#undef hipSetDevice
#undef hipStreamCreate
#undef hipStreamDestroy
#undef hipStreamSynchronize
#undef hipMemGetAllocationGranularity
#undef hipMemCreate
#undef hipMemAddressReserve
#undef hipMemMap
#undef hipMemSetAccess
#undef hipMemUnmap
#undef hipMemRelease
#undef hipMemAddressFree
#undef hipMemExportToShareableHandle
#undef hipMemImportFromShareableHandle
#endif

// Kernel macros
#undef GetHwId
//#undef GetXccId

// Undefine helper macros
#undef ERR_CHECK
#undef ERR_APPEND
}

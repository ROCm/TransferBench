/*
Copyright (c) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once
#include <cstring>
#include <future>
#include <map>
#include <numa.h> // If not found, try installing libnuma-dev (e.g apt-get install libnuma-dev)
#include <numaif.h>
#include <set>
#include <sstream>
#include <stdarg.h>
#include <thread>
#include <vector>

#if defined(__NVCC__)
#include <cuda_runtime.h>
#else
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif

namespace TransferBench
{
  using std::map;
  using std::pair;
  using std::set;
  using std::vector;

  constexpr char VERSION[] = "1.54";

  /**
   * Enumeration of supported Executor types
   *
   * @note The Executor is the device used to perform a Transfer
   * @note IBVerbs executor is currently not implemented yet
   */
  enum ExeType
  {
    EXE_CPU          = 0,                       ///<  CPU executor              (subExecutor = CPU thread)
    EXE_GPU_GFX      = 1,                       ///<  GPU kernel-based executor (subExecutor = threadblock/CU)
    EXE_GPU_DMA      = 2,                       ///<  GPU SDMA executor         (subExecutor = not supported)
    EXE_IBV          = 3,                       ///<  IBVerbs executor          (subExecutor = queue pair)
  };
  char const ExeTypeStr[5] = "CGDI";
  inline bool IsCpuExeType(ExeType e){ return e == EXE_CPU; }
  inline bool IsGpuExeType(ExeType e){ return e == EXE_GPU_GFX || e == EXE_GPU_DMA; }

  /**
   * A ExeDevice defines a specific Executor
   */
  struct ExeDevice
  {
    ExeType exeType;                            ///< Executor type
    int32_t exeIndex;                           ///< Executor index

    // Default comparison operator
    auto operator<=>(const ExeDevice&) const = default;
  };

  /**
   * Enumeration of supported memory types
   *
   * @note These are possible types of memory to be used as sources/destinations
   */
  enum MemType
  {
    MEM_CPU          = 0,                       ///< Coarse-grained pinned CPU memory
    MEM_GPU          = 1,                       ///< Coarse-grained global GPU memory
    MEM_CPU_FINE     = 2,                       ///< Fine-grained pinned CPU memory
    MEM_GPU_FINE     = 3,                       ///< Fine-grained global GPU memory
    MEM_CPU_UNPINNED = 4,                       ///< Unpinned CPU memory
    MEM_NULL         = 5,                       ///< NULL memory - used for empty
    MEM_MANAGED      = 6                        ///< Managed memory
  };
  char const MemTypeStr[8] = "CGBFUNM";
  inline bool IsCpuMemType(MemType m) { return (m == MEM_CPU || m == MEM_CPU_FINE || m == MEM_CPU_UNPINNED); }
  inline bool IsGpuMemType(MemType m) { return (m == MEM_GPU || m == MEM_GPU_FINE || m == MEM_MANAGED); }

  /**
   * A MemDevice indicates a memory type on a specific device
   */
  struct MemDevice
  {
    MemType memType;                            ///< Memory type
    int32_t memIndex;                           ///< Device index
    auto operator<=>(const MemDevice&) const = default;
  };

  /**
   * A Transfer adds together data from zero or more sources then writes the sum to zero or more desintations
   */
  struct Transfer
  {
    size_t            numBytes    = (1<<26);    ///< # of bytes to Transfer
    vector<MemDevice> srcs        = {};         ///< List of source memory devices
    vector<MemDevice> dsts        = {};         ///< List of destination memory devices
    ExeDevice         exeDevice   = {};         ///< Executor to use
    int32_t           exeDstIndex = -1;         ///< Destination executor index (for RDMA executor only)
    int32_t           exeSubIndex = -1;         ///< Executor subindex
    int               numSubExecs = 0;          ///< Number of subExecutors to use for this Transfer
  };

  /**
   * General options
   */
  struct GeneralOptions
  {
    int numIterations      = 10;                ///< # of timed iterations to perform. If negative, run for -numIterations seconds instead
    int numSubIterations   = 1;                 ///< # of sub-iterations per iteration
    int numWarmups         = 3;                 ///< Number of un-timed warmup iterations to perform
    int recordPerIteration = 0;                 ///< Record per-iteration timing information
    int useInteractive     = 0;                 ///< Pause for user-input before starting transfer loop
  };

  /**
   * Data options
   */
  struct DataOptions
  {
    int           alwaysValidate   = 0;         ///< Validate after each iteration instead of once at end
    int           blockBytes       = 256;       ///< Each subexecutor works on a multiple of this many bytes
    int           byteOffset       = 0;         ///< Byte-offset for memory allocations
    vector<float> fillPattern      = {};        ///< Pattern of floats used to fill source data
    int           validateDirect   = 0;         ///< Validate GPU results directly instead of copying to host
    int           validateSource   = 0;         ///< Validate src GPU memory immediately after preparation
  };

  /**
   * DMA Executor options
   */
  struct DmaOptions
  {
    int useHipEvents = 1;                       ///< Use HIP events for timing DMA Executor
    int useHsaCopy   = 0;                       ///< Use HSA copy instead of HIP copy to perform DMA
  };

  /**
   * GFX Executor options
   */
  struct GfxOptions
  {
    int                 blockSize      = 256;   ///< Size of each threadblock (must be multiple of 64)
    vector<uint32_t>    cuMask         = {};    ///< Bit-vector representing the CU mask
    vector<vector<int>> prefXccTable   = {};    ///< 2D table with preferred XCD to use for a specific [src][dst] GPU device
    int                 unrollFactor   = 4;     ///< GFX-kernel unroll factor
    int                 useHipEvents   = 1;     ///< Use HIP events for timing GFX Executor
    int                 useMultiStream = 0;     ///< Use multiple streams for GFX
    int                 useSingleTeam  = 1;     ///< Team all subExecutors across the data array
    int                 waveOrder      = 0;     ///< GFX-kernel wavefront ordering
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
   * ErrResult consists of error type and error message
   */
  struct ErrResult
  {
    ErrType     errType;                        ///< Error type
    std::string errMsg;                         ///< Error details

    ErrResult() = default;
#if defined(__NVCC__)
    ErrResult(cudaError_t  err);
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

  /**
   * Enumeration of implementation attributes
   */
  enum IntAttribute
  {
    ATR_GFX_MAX_BLOCKSIZE,                      ///< Maximum blocksize for GFX executor
    ATR_GFX_MAX_UNROLL,                         ///< Maximum unroll factor for GFX executor
  };

  enum StrAttribute
  {
    ATR_SRC_PREP_DESCRIPTION                    ///< Description of how source memory is prepared
  };

  /**
   * Query attributes (integer)
   *
   * @note This allows querying of implementation information such as limits
   *
   * @param[in] attribute   Attribute to query
   * @returns Value of the attribute
   */
  int GetIntAttribute(IntAttribute attribute);

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
   * Returns information about number of available available Executors
   *
   * @param[in] exeType    Executor type to query
   * @returns Number of detected Executors of exeType
   */
  int GetNumExecutors(ExeType exeType);

  /**
   * Returns the number of possible Executor subindices
   *
   * @note For CPU, this is 0
   * @note For GFX, this refers to the number of XCDs
   * @note For DMA, this refers to the number of DMA engines
   *
   * @param[in] exeDevice The specific Executor to query
   * @returns Number of detected executor subindices
   */
  int GetNumExecutorSubIndices(ExeDevice exeDevice);

  /**
   * Returns number of subExecutors for a given ExeDevice
   *
   * @param[in] exeDevice   The specific Executor to query
   * @returns Number of detected subExecutors for the given ExePair
   */
  int GetNumSubExecutors(ExeDevice exeDevice);

  /**
   * Returns the index of the NUMA node closest to the given GPU
   *
   * @param[in] gpuIndex Index of the GPU to query
   * @returns NUMA node index closest to GPU gpuIndex, or -1 if unable to detect
   */
  int GetClosestCpuNumaToGpu(int gpuIndex);

  /**
   * Helper function to parse a line containing Transfers into a vector of Transfers
   *
   * @param[in]  str       String containing description of Transfers
   * @param[out] transfers List of Transfers described by 'str'
   * @returns Information about any error that may have occured
   */
  ErrResult ParseTransfers(std::string str,
                           std::vector<Transfer>& transfers);

};
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

  // Enumerations
  #define hipDeviceAttributeClockRate                        cudaDevAttrClockRate
  #define hipDeviceAttributeMultiprocessorCount              cudaDevAttrMultiProcessorCount
  #define hipErrorPeerAccessAlreadyEnabled                   cudaErrorPeerAccessAlreadyEnabled
  #define hipFuncCachePreferShared                           cudaFuncCachePreferShared
  #define hipMemcpyDefault                                   cudaMemcpyDefault
  #define hipMemcpyDeviceToHost                              cudaMemcpyDeviceToHost
  #define hipMemcpyHostToDevice                              cudaMemcpyHostToDevice
  #define hipSuccess                                         cudaSuccess

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
  #define hipHostFree                                        cudaFreeHost
  #define hipHostMalloc                                      cudaMallocHost
  #define hipMalloc                                          cudaMalloc
  #define hipMallocManaged                                   cudaMallocManaged
  #define hipMemcpy                                          cudaMemcpy
  #define hipMemcpyAsync                                     cudaMemcpyAsync
  #define hipMemset                                          cudaMemset
  #define hipMemsetAsync                                     cudaMemsetAsync
  #define hipSetDevice                                       cudaSetDevice
  #define hipStreamCreate                                    cudaStreamCreate
  #define hipStreamDestroy                                   cudaStreamDestroy
  #define hipStreamSynchronize                               cudaStreamSynchronize

  // Define float4 addition operator for NVIDIA platform
  __device__ inline float4& operator +=(float4& a, const float4& b)
  {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
  }
#endif

// Helper macro functions
//==========================================================================================

// Macro for collecting CU/SM GFX kernel is running on
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1200__) || defined(__gfx1201__)
#define GetHwId(hwId) hwId = 0
#elif defined(__NVCC__)
#define GetHwId(hwId) asm("mov.u32 %0, %smid;" : "=r"(hwId))
#else
#define GetHwId(hwId) asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s" (hwId));
#endif

// Macro for collecting XCC GFX kernel is running on
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define GetXccId(val) asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s" (val));
#else
#define GetXccId(val) val = 0
#endif

// Error check macro (NOTE: This will return even for ERR_WARN)
#define ERR_CHECK(cmd)            \
  do {                            \
    ErrResult err = (cmd);        \
    if (err.errType != ERR_NONE)  \
      return err;                 \
  } while (0)

// Appends warn/fatal errors to a list, return false if fatal
#define ERR_APPEND(cmd, list)     \
  do {                            \
    ErrResult err = (cmd);        \
    if (err.errType != ERR_NONE)  \
      list.push_back(err);        \
    if (err.errType == ERR_FATAL) \
      return false;               \
  } while (0)

namespace TransferBench
{
// Helper functions ('hidden' in anonymous namespace)
//========================================================================================
namespace {

// Constants
//========================================================================================
  int   constexpr MAX_BLOCKSIZE  = 512;                       // Max threadblock size
  int   constexpr MAX_WAVEGROUPS = MAX_BLOCKSIZE / 64;        // Max wavegroups/warps
  int   constexpr MAX_UNROLL     = 8;                         // Max unroll factor
  int   constexpr MAX_SRCS       = 8;                         // Max # srcs per Transfer
  int   constexpr MAX_DSTS       = 8;                         // Max # dsts per Transfer
  int   constexpr MEMSET_CHAR    = 75;                        // Value to memset (char)
  float constexpr MEMSET_VAL     = 13323083.0f;               // Value to memset (double)

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

  static ErrResult ParseMemType(std::string const& token,
                                std::vector<MemDevice>& memDevices)
  {
    char memTypeChar;
    int offset = 0, memIndex, inc;
    MemType memType;
    bool found = false;

    memDevices.clear();
    while (sscanf(token.c_str() + offset, " %c %d%n", &memTypeChar, &memIndex, &inc) == 2) {
      offset += inc;

      ErrResult err = CharToMemType(memTypeChar, memType);
      if (err.errType != ERR_NONE) return err;

      if (memType != MEM_NULL)
        memDevices.push_back({memType, memIndex});
      found = true;
    }
    if (found) return ERR_NONE;
    return {ERR_FATAL,
            "Unable to parse memory type token %s.  Expected one of %s followed by an index",
            token.c_str(), MemTypeStr};
  }

  static ErrResult ParseExeType(std::string const& token,
                                ExeDevice& exeDevice,
                                int& exeSubIndex)
  {
    char exeTypeChar;
    exeSubIndex = -1;

    int numTokensParsed = sscanf(token.c_str(),
                                 " %c%d.%d", &exeTypeChar, &exeDevice.exeIndex, &exeSubIndex);
    if (numTokensParsed < 2) {
      return {ERR_FATAL,
              "Unable to parse valid executor token (%s)."
              "Expected one of %s followed by an index",
              token.c_str(), ExeTypeStr};
    }
    return CharToExeType(exeTypeChar, exeDevice.exeType);
  }

// Memory-related functions
//========================================================================================
  // Enable peer access between two GPUs
  static ErrResult EnablePeerAccess(int const deviceId, int const peerDeviceId)
  {
    int canAccess;
    ERR_CHECK(hipDeviceCanAccessPeer(&canAccess, deviceId, peerDeviceId));
    if (!canAccess)
      return {ERR_FATAL,
              "Unable to enable peer access from GPU devices %d to %d", peerDeviceId, deviceId};

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
              " This could be due to hardware memory issues",
              mistakeCount, numPages, targetId};
    }
    return ERR_NONE;
  }

  // Allocate memory
  static ErrResult AllocateMemory(MemDevice memDevice, size_t numBytes, void** memPtr)
  {
    if (numBytes == 0) {
      return {ERR_FATAL, "Unable to allocate 0 bytes"};
    }
    *memPtr = nullptr;

    MemType const& memType = memDevice.memType;

    if (IsCpuMemType(memType)) {
      // Set numa policy prior to call to hipHostMalloc
      numa_set_preferred(memDevice.memIndex);

      // Allocate host-pinned memory (should respect NUMA mem policy)
      if (memType == MEM_CPU_FINE) {
#if defined (__NVCC__)
        return {ERR_FATAL, "Fine-grained CPU memory not supported on NVIDIA platform"};
#else
        ERR_CHECK(hipHostMalloc((void **)memPtr, numBytes, hipHostMallocNumaUser));
#endif
      } else if (memType == MEM_CPU) {
#if defined (__NVCC__)
        ERR_CHECK(hipHostMalloc((void **)memPtr, numBytes, 0));
#else
        ERR_CHECK(hipHostMalloc((void **)memPtr, numBytes, hipHostMallocNumaUser | hipHostMallocNonCoherent));
#endif
      } else if (memType == MEM_CPU_UNPINNED) {
        *memPtr = numa_alloc_onnode(numBytes, memDevice.memIndex);
      }

      // Check that the allocated pages are actually on the correct NUMA node
      memset(*memPtr, 0, numBytes);
      ERR_CHECK(CheckPages((char*)*memPtr, numBytes, memDevice.memIndex));

      // Reset to default numa mem policy
      numa_set_preferred(-1);
    } else if (IsGpuMemType(memType)) {
      // Switch to the appropriate GPU
      ERR_CHECK(hipSetDevice(memDevice.memIndex));

      if (memType == MEM_GPU) {
        // Allocate GPU memory on appropriate device
        ERR_CHECK(hipMalloc((void**)memPtr, numBytes));
      } else if (memType == MEM_GPU_FINE) {
#if defined (__NVCC__)
        return {ERR_FATAL, "Fine-grained GPU memory not supported on NVIDIA platform"};
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
  static ErrResult DeallocateMemory(MemType memType, void *memPtr, size_t const bytes)
  {
    // Avoid deallocating nullptr
    if (memPtr == nullptr)
      return {ERR_FATAL, "Attempted to free null pointer for %lu bytes", bytes};

    switch (memType) {
    case MEM_CPU: case MEM_CPU_FINE:
    {
      ERR_CHECK(hipHostFree(memPtr));
      break;
    }
    case MEM_CPU_UNPINNED:
    {
      numa_free(memPtr, bytes);
      break;
    }
    case MEM_GPU : case MEM_GPU_FINE: case MEM_MANAGED:
    {
      ERR_CHECK(hipFree(memPtr));
      break;
    }
    default:
      return {ERR_FATAL, "Attempting to deallocate unrecognized memory type (%d)", memType};
    }
    return ERR_NONE;
  }

// HSA-related functions
//========================================================================================

#if !defined(__NVCC__)
  // Get the hsa_agent_t associated with a ExeDevice
  static ErrResult GetHsaAgent(ExeDevice const& exeDevice, hsa_agent_t& agent)
  {
    static bool isInitialized = false;
    static std::vector<hsa_agent_t> cpuAgents;
    static std::vector<hsa_agent_t> gpuAgents;

    int const& exeIndex = exeDevice.exeIndex;
    int const numCpus   = GetNumExecutors(EXE_CPU);
    int const numGpus   = GetNumExecutors(EXE_GPU_GFX);

    // Initialize results on first use
    if (!isInitialized) {
      hsa_amd_pointer_info_t info;
      info.size = sizeof(info);

      ErrResult err;
      int32_t* tempBuffer;

      // Index CPU agents
      cpuAgents.clear();
      for (int i = 0; i < numCpus; i++) {
        ERR_CHECK(AllocateMemory({MEM_CPU, i}, 1024, (void**)&tempBuffer));
        ERR_CHECK(hsa_amd_pointer_info(tempBuffer, &info, NULL, NULL, NULL));
        cpuAgents.push_back(info.agentOwner);
        ERR_CHECK(DeallocateMemory(MEM_CPU, tempBuffer, 1024));
      }

      // Index GPU agents
      gpuAgents.clear();
      for (int i = 0; i < numGpus; i++) {
        ERR_CHECK(AllocateMemory({MEM_GPU, i}, 1024, (void**)&tempBuffer));
        ERR_CHECK(hsa_amd_pointer_info(tempBuffer, &info, NULL, NULL, NULL));
        gpuAgents.push_back(info.agentOwner);
        ERR_CHECK(DeallocateMemory(MEM_GPU, tempBuffer, 1024));
      }
      isInitialized = true;
    }

    switch (exeDevice.exeType) {
    case EXE_CPU:
      if (exeIndex < 0 || exeIndex >= numCpus)
        return {ERR_FATAL, "CPU index must be between 0 and %d inclusively", numCpus - 1};
      agent = cpuAgents[exeDevice.exeIndex];
      break;
    case EXE_GPU_GFX: case EXE_GPU_DMA:
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
  static ErrResult GetHsaAgent(MemDevice const& memDevice, hsa_agent_t& agent)
  {
    if (IsCpuMemType(memDevice.memType)) return GetHsaAgent({EXE_CPU, memDevice.memIndex}, agent);
    if (IsGpuMemType(memDevice.memType)) return GetHsaAgent({EXE_GPU_GFX, memDevice.memIndex}, agent);
    return {ERR_FATAL,
            "Unable to get HSA agent for memDevice (%d,%d)",
            memDevice.memType, memDevice.memIndex};
  }
#endif

// Setup validation-related functions
//========================================================================================

  // Validate that MemDevice exists
  static ErrResult CheckMemDevice(MemDevice const& memDevice)
  {
    if (memDevice.memType == MEM_NULL)
      return ERR_NONE;

    if (IsCpuMemType(memDevice.memType)) {
      int numCpus = GetNumExecutors(EXE_CPU);
      if (memDevice.memIndex < 0 || memDevice.memIndex >= numCpus)
        return {ERR_FATAL,
                "CPU index must be between 0 and %d (instead of %d)", numCpus - 1, memDevice.memIndex};
      return ERR_NONE;
    }

    if (IsGpuMemType(memDevice.memType)) {
    int numGpus = GetNumExecutors(EXE_GPU_GFX);
      if (memDevice.memIndex < 0 || memDevice.memIndex >= numGpus)
        return {ERR_FATAL,
                "GPU index must be between 0 and %d (instead of %d)", numGpus - 1, memDevice.memIndex};
      return ERR_NONE;
    }
    return {ERR_FATAL, "Unsupported memory type (%d)", memDevice.memType};
  }

  // Validate configuration options - return trues if and only if an fatal error is detected
  static bool ConfigOptionsHaveErrors(ConfigOptions const&    cfg,
                                      std::vector<ErrResult>& errors)
  {
    // Check general options
    if (cfg.general.numWarmups < 0)
      errors.push_back({ERR_FATAL, "[general.numWarmups] must be a non-negative number"});

    // Check data options
    if (cfg.data.blockBytes == 0 || cfg.data.blockBytes % 4)
      errors.push_back({ERR_FATAL, "[data.blockBytes] must be positive multiple of %lu", sizeof(float)});
    if (cfg.data.byteOffset < 0 || cfg.data.byteOffset % sizeof(float))
      errors.push_back({ERR_FATAL, "[data.byteOffset] must be positive multiple of %lu", sizeof(float)});

    // Check GFX options
    int gfxMaxBlockSize = GetIntAttribute(ATR_GFX_MAX_BLOCKSIZE);
    if (cfg.gfx.blockSize < 0 || cfg.gfx.blockSize % 64 || cfg.gfx.blockSize > gfxMaxBlockSize)
      errors.push_back({ERR_FATAL,
                        "[gfx.blockSize] must be positive multiple of 64 less than or equal to %d",
                        gfxMaxBlockSize});

    int gfxMaxUnroll = GetIntAttribute(ATR_GFX_MAX_UNROLL);
    if (cfg.gfx.unrollFactor < 0 || cfg.gfx.unrollFactor > gfxMaxUnroll)
      errors.push_back({ERR_FATAL,
                        "[gfx.unrollFactor] must be non-negative and less than or equal to %d",
                        gfxMaxUnroll});
    if (cfg.gfx.waveOrder < 0 || cfg.gfx.waveOrder >= 6)
      errors.push_back({ERR_FATAL,
                        "[gfx.waveOrder] must be non-negative and less than 6"});

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

  // Validate Transfers to execute - returns true if and only if fatal error detected
  static bool TransfersHaveErrors(ConfigOptions         const& cfg,
                                  std::vector<Transfer> const& transfers,
                                  std::vector<ErrResult>&      errors)
  {
    int numCpus = GetNumExecutors(EXE_CPU);
    int numGpus = GetNumExecutors(EXE_GPU_GFX);

    std::set<ExeDevice>      executors;
    std::map<ExeDevice, int> transferCount;
    std::map<ExeDevice, int> useSubIndexCount;
    std::map<ExeDevice, int> totalSubExecs;

    // Per-Transfer checks
    for (size_t i = 0; i < transfers.size(); i++) {
      Transfer const& t = transfers[i];

      if (t.numBytes == 0)
        errors.push_back({ERR_FATAL, "Transfer %d: Cannot perform 0-byte transfers", i});

      if (t.exeDevice.exeType == EXE_GPU_GFX || t.exeDevice.exeType == EXE_CPU) {
        size_t const N               = t.numBytes / sizeof(float);
        int    const targetMultiple  = cfg.data.blockBytes / sizeof(float);
        int    const maxSubExecToUse = std::min((size_t)(N + targetMultiple - 1) / targetMultiple,
                                                (size_t)t.numSubExecs);

        if (maxSubExecToUse < t.numSubExecs)
          errors.push_back({ERR_WARN,
                            "Transfer %d data size is too small - will only use %d of %d subexecutors",
                            i, maxSubExecToUse, t.numSubExecs});
      }

      // Check sources and destinations
      if (t.srcs.empty() && t.dsts.empty())
        errors.push_back({ERR_FATAL, "Transfer %d: Must have at least one source or destination", i});

      for (int j = 0; j < t.srcs.size(); j++) {
        ErrResult err = CheckMemDevice(t.srcs[j]);
        if (err.errType != ERR_NONE)
          errors.push_back({ERR_FATAL, "Transfer %d: SRC %d: %s", i, j, err.errMsg.c_str()});
      }
      for (int j = 0; j < t.dsts.size(); j++) {
        ErrResult err = CheckMemDevice(t.dsts[j]);
        if (err.errType != ERR_NONE)
          errors.push_back({ERR_FATAL, "Transfer %d: DST %d: %s", i, j, err.errMsg.c_str()});
      }

      // Check executor
      executors.insert(t.exeDevice);
      transferCount[t.exeDevice]++;
      switch (t.exeDevice.exeType) {
      case EXE_CPU:
        if (t.exeDevice.exeIndex < 0 || t.exeDevice.exeIndex >= numCpus)
          errors.push_back({ERR_FATAL,
                            "Transfer %d: CPU index must be between 0 and %d (instead of %d)",
                            i, numCpus - 1, t.exeDevice.exeIndex});
        break;
      case EXE_GPU_GFX:
        if (t.exeDevice.exeIndex < 0 || t.exeDevice.exeIndex >= numGpus) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: GFX index must be between 0 and %d (instead of %d)",
                            i, numGpus - 1, t.exeDevice.exeIndex});
        } else {
          if (t.exeSubIndex != -1) {
#if defined(__NVCC__)
            errors.push_back({ERR_FATAL,
                              "Transfer %d: GFX executor subindex not supported on NVIDIA hardware", i});
#else
            useSubIndexCount[t.exeDevice]++;
            int numSubIndices = GetNumExecutorSubIndices(t.exeDevice);
            if (t.exeSubIndex >= numSubIndices)
              errors.push_back({ERR_FATAL,
                                "Transfer %d: GFX subIndex (XCC) must be between 0 and %d", i, numSubIndices - 1});
#endif
          }
        }
        break;
      case EXE_GPU_DMA:
        if (t.srcs.size() != 1 || t.dsts.size() != 1) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: DMA executor must have exactly 1 source and 1 destination", i});
        }

        if (t.exeDevice.exeIndex < 0 || t.exeDevice.exeIndex >= numGpus) {
          errors.push_back({ERR_FATAL,
                            "Transfer %d: DMA index must be between 0 and %d (instead of %d)",
                            i, numGpus - 1, t.exeDevice.exeIndex});
          // Cannot proceed with any further checks
          continue;
        }

        if (t.exeSubIndex != -1) {
#if defined(__NVCC__)
          errors.push_back({ERR_FATAL,
                            "Transfer %d: DMA executor subindex not supported on NVIDIA hardware", i});
#else
          useSubIndexCount[t.exeDevice]++;
          int numSubIndices = GetNumExecutorSubIndices(t.exeDevice);
          if (t.exeSubIndex >= numSubIndices)
            errors.push_back({ERR_FATAL,
                              "Transfer %d: DMA subIndex (engine) must be between 0 and %d",
                              i, numSubIndices - 1});

          // Check that engine Id exists between agents
          hsa_agent_t srcAgent, dstAgent;
          ErrResult err;
          err = GetHsaAgent(t.srcs[0], srcAgent);
          if (err.errType != ERR_NONE) {
            errors.push_back(err);
            if (err.errType == ERR_FATAL) break;
          }
          err = GetHsaAgent(t.dsts[0], dstAgent);
          if (err.errType != ERR_NONE) {
            errors.push_back(err);
            if (err.errType == ERR_FATAL) break;
          }

          uint32_t engineIdMask = 0;
          err = hsa_amd_memory_copy_engine_status(dstAgent, srcAgent, &engineIdMask);
          if (err.errType != ERR_NONE) {
            errors.push_back(err);
            if (err.errType == ERR_FATAL) break;
          }
          hsa_amd_sdma_engine_id_t sdmaEngineId = (hsa_amd_sdma_engine_id_t)(1U << t.exeSubIndex);
          if (!(sdmaEngineId & engineIdMask)) {
            errors.push_back({ERR_FATAL,
                "Transfer %d: DMA %d.%d does not exist or cannot copy between src/dst",
                i, t.exeDevice.exeIndex, t.exeSubIndex});
          }
#endif
        }

        if (!IsGpuMemType(t.srcs[0].memType) || !IsGpuMemType(t.dsts[0].memType)) {
          errors.push_back({ERR_WARN,
              "Transfer %d: No GPU memory for source or destination.  Copy might not execute on DMA %d",
              i, t.exeDevice.exeIndex});
        } else {
          // Currently HIP will use src agent if source memory is GPU, otherwise dst agent
          if (IsGpuMemType(t.srcs[0].memType)) {
            if (t.srcs[0].memIndex != t.exeDevice.exeIndex) {
              errors.push_back({ERR_WARN,
                  "Transfer %d: DMA executor will automatically switch to using the source memory device (%d) not (%d)",
                  i, t.srcs[0].memIndex, t.exeDevice.exeIndex});
            }
          } else if (t.dsts[0].memIndex != t.exeDevice.exeIndex) {
            errors.push_back({ERR_WARN,
                "Transfer %d: DMA executor will automatically switch to using the destination memory device (%d) not (%d)",
                i, t.dsts[0].memIndex, t.exeDevice.exeIndex});
          }
        }
        break;
      case EXE_IBV:
        errors.push_back({ERR_FATAL, "Transfer %d: IBV executor currently not supported", i});
        break;
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
        if (totalSubExecs[exeDevice] > numGpuSubExec)
          errors.push_back({ERR_WARN,
                            "GPU %d requests %d total CUs however only %d available. "
                            "Serialization will occur",
                            exeDevice.exeIndex, totalSubExecs[exeDevice], numGpuSubExec});
        // Check that if executor subindices are used, all Transfers specify executor subindices
        if (useSubIndexCount[exeDevice] > 0 && useSubIndexCount[exeDevice] != transferCount[exeDevice]) {
          errors.push_back({ERR_FATAL,
                            "GPU %d specifies XCC on only %d of %d Transfers. "
                            "Must either specific none or all",
                            exeDevice.exeIndex, useSubIndexCount[exeDevice], transferCount[exeDevice]});
        }

        if (cfg.gfx.useMultiStream && transferCount[exeDevice] > gpuMaxHwQueues) {
          errors.push_back({ERR_WARN,
                            "GPU %d attempting %d parallel transfers, however GPU_MAX_HW_QUEUES only set to %d",
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
    long long                  startCycle;        ///< Start timestamp for in-kernel timing (GPU-GFX executor)
    long long                  stopCycle;         ///< Stop  timestamp for in-kernel timing (GPU-GFX executor)
    uint32_t                   hwId;              ///< Hardware ID
    uint32_t                   xccId;             ///< XCC ID
  };

  // Internal resources allocated per Transfer
  struct TransferResources
  {
    int                        transferIdx;       ///< The associated Transfer
    size_t                     numBytes;          ///< Number of bytes to Transfer
    vector<float*>             srcMem;            ///< Source memory
    vector<float*>             dstMem;            ///< Destination memory
    vector<SubExecParam>       subExecParamCpu;   ///< Defines subarrays for each subexecutor
    vector<int>                subExecIdx;        ///< Indices into subExecParamGpu

    // For GFX executor
    SubExecParam*              subExecParamGpuPtr;

    // For targeted-SDMA
#if !defined(__NVCC__)
    hsa_agent_t                dstAgent;          ///< DMA destination memory agent
    hsa_agent_t                srcAgent;          ///< DMA source memory agent
    hsa_signal_t               signal;            ///< HSA signal for completion
    hsa_amd_sdma_engine_id_t   sdmaEngineId;      ///< DMA engine ID
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
    int                        wallClockRate;     ///< (GFX-only) Device wall clock rate
    vector<SubExecParam>       subExecParamCpu;   ///< Subexecutor parameters for this executor
    vector<TransferResources>  resources;         ///< Per-Transfer resources

    // For GPU-Executors
    SubExecParam*              subExecParamGpu;   ///< GPU copy of subExecutor parameters
    vector<hipStream_t>        streams;           ///< HIP streams to launch on
    vector<hipEvent_t>         startEvents;       ///< HIP start timing event
    vector<hipEvent_t>         stopEvents;        ///< HIP stop timing event
  };

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

    // Source buffer
    if (bufferIdx >= 0) {
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
        for (size_t i = 0; i < N; ++i)
          cpuBuffer[i] = PrepSrcValue(bufferIdx, i);
      }
    } else { // Destination buffer
      int numSrcs = -bufferIdx - 1;

      if (numSrcs == 0) {
        // Note: 0x75757575 = 13323083.0
        memset(cpuBuffer.data(), MEMSET_CHAR, N * sizeof(float));
      } else {
        PrepareReference(cfg, cpuBuffer, 0);
        if (numSrcs > 1) {
          std::vector<float> temp(N);
          for (int i = 1; i < numSrcs; i++) {
            PrepareReference(cfg, temp, i);
            for (int j = 0; j < N; j++) {
              cpuBuffer[i] += temp[i];
            }
          }
        }
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
    size_t initOffset = cfg.data.byteOffset / sizeof(float);

    for (auto resource : transferResources) {
      int transferIdx = resource->transferIdx;
      Transfer const& t = transfers[transferIdx];
      size_t N = t.numBytes / sizeof(float);

      float const* expected = dstReference[t.srcs.size()].data();
      for (int dstIdx = 0; dstIdx < resource->dstMem.size(); dstIdx++) {
        if (IsCpuMemType(t.dsts[dstIdx].memType) || cfg.data.validateDirect) {
          output = (resource->dstMem[dstIdx]) + initOffset;
        } else {
          ERR_CHECK(hipMemcpy(outputBuffer.data(), (resource->dstMem[dstIdx]) + initOffset, t.numBytes, hipMemcpyDefault));
          ERR_CHECK(hipDeviceSynchronize());
          output = outputBuffer.data();
        }

        if (memcmp(output, expected, t.numBytes)) {
          // Difference found - find first error
          for (size_t i = 0; i < N; i++) {
            if (output[i] != expected[i]) {
              return {ERR_FATAL, "Transfer %d: Unexpected mismatch at index %lu of destination %d: Expected %10.5f Actual: %10.5f",
                transferIdx, i, dstIdx, expected[i], output[i]};
            }
          }
          return {ERR_FATAL, "Transfer %d: Unexpected output mismatch for destination %d", transferIdx, dstIdx};
        }
      }
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
                                        TransferResources&   resources)
  {
    // Each subExecutor needs to know src/dst pointers and how many elements to transfer
    // Figure out the sub-array each subExecutor works on for this Transfer
    // - Partition N as evenly as possible, but try to keep subarray sizes as multiples of data.blockBytes
    //   except the very last one, for alignment reasons
    size_t const N              = transfer.numBytes / sizeof(float);
    int    const initOffset     = cfg.data.byteOffset / sizeof(float);
    int    const targetMultiple = cfg.data.blockBytes / sizeof(float);

    // In some cases, there may not be enough data for all subExectors
    int const maxSubExecToUse = std::min((size_t)(N + targetMultiple - 1) / targetMultiple,
                                         (size_t)transfer.numSubExecs);

    vector<SubExecParam>& subExecParam = resources.subExecParamCpu;
    subExecParam.clear();
    subExecParam.resize(transfer.numSubExecs);

    size_t assigned = 0;
    for (int i = 0; i < transfer.numSubExecs; ++i) {
      SubExecParam& p  = subExecParam[i];
      p.numSrcs        = resources.srcMem.size();
      p.numDsts        = resources.dstMem.size();
      p.startCycle     = 0;
      p.stopCycle      = 0;
      p.hwId           = 0;
      p.xccId          = 0;

      // In single team mode, subexecutors stripe across the entire array
      if (cfg.gfx.useSingleTeam && transfer.exeDevice.exeType == EXE_GPU_GFX) {
        p.N        = N;
        p.teamSize = transfer.numSubExecs;
        p.teamIdx  = i;
        for (int iSrc = 0; iSrc < p.numSrcs; ++iSrc) p.src[iSrc] = resources.srcMem[iSrc] + initOffset;
        for (int iDst = 0; iDst < p.numDsts; ++iDst) p.dst[iDst] = resources.dstMem[iDst] + initOffset;
      } else {
        // Otherwise, each subexecutor works on separate subarrays
        int    const subExecLeft = std::max(0, maxSubExecToUse - i);
        size_t const leftover    = N - assigned;
        size_t const roundedN    = (leftover + targetMultiple - 1) / targetMultiple;

        p.N        = subExecLeft ? std::min(leftover, ((roundedN / subExecLeft) * targetMultiple)) : 0;
        p.teamSize = 1;
        p.teamIdx  = 0;
        for (int iSrc = 0; iSrc < p.numSrcs; ++iSrc) p.src[iSrc] = resources.srcMem[iSrc] + initOffset + assigned;
        for (int iDst = 0; iDst < p.numDsts; ++iDst) p.dst[iDst] = resources.dstMem[iDst] + initOffset + assigned;
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

    // Clear counters
    resources.totalDurationMsec = 0.0;

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

    // Loop over each transfer this executor is involved in
    for (auto& resources : exeInfo.resources) {
      Transfer const& t = transfers[resources.transferIdx];
      resources.numBytes = t.numBytes;

      // Allocate source memory
      resources.srcMem.resize(t.srcs.size());
      for (int iSrc = 0; iSrc < t.srcs.size(); ++iSrc) {
        MemDevice const& srcMemDevice = t.srcs[iSrc];

        // Ensure executing GPU can access source memory
        if (exeDevice.exeType == EXE_GPU_GFX && IsGpuMemType(srcMemDevice.memType) &&
            srcMemDevice.memIndex != exeDevice.exeIndex) {
          ERR_CHECK(EnablePeerAccess(exeDevice.exeIndex, srcMemDevice.memIndex));
        }
        ERR_CHECK(AllocateMemory(srcMemDevice, t.numBytes + cfg.data.byteOffset, (void**)&resources.srcMem[iSrc]));
      }

      // Allocate destination memory
      resources.dstMem.resize(t.dsts.size());
      for (int iDst = 0; iDst < t.dsts.size(); ++iDst) {
        MemDevice const& dstMemDevice = t.dsts[iDst];

        // Ensure executing GPU can access destination memory
        if (exeDevice.exeType == EXE_GPU_GFX && IsGpuMemType(dstMemDevice.memType) &&
            dstMemDevice.memIndex != exeDevice.exeIndex) {
          ERR_CHECK(EnablePeerAccess(exeDevice.exeIndex, dstMemDevice.memIndex));
        }
        ERR_CHECK(AllocateMemory(dstMemDevice, t.numBytes + cfg.data.byteOffset, (void**)&resources.dstMem[iDst]));
      }

      if (exeDevice.exeType == EXE_GPU_DMA && (t.exeSubIndex != -1 || cfg.dma.useHsaCopy)) {
#if !defined(__NVCC__)
        // Collect HSA agent information
        hsa_amd_pointer_info_t info;
        info.size = sizeof(info);
        ERR_CHECK(hsa_amd_pointer_info(resources.dstMem[0], &info, NULL, NULL, NULL));
        resources.dstAgent = info.agentOwner;

        ERR_CHECK(hsa_amd_pointer_info(resources.srcMem[0], &info, NULL, NULL, NULL));
        resources.srcAgent = info.agentOwner;

        // Create HSA completion signal
        ERR_CHECK(hsa_signal_create(1, 0, NULL, &resources.signal));
#endif
      }

      // Prepare subexecutor parameters
      ERR_CHECK(PrepareSubExecParams(cfg, t, resources));
    }

    // Prepare additional requirements for GPU-based executors
    if (exeDevice.exeType == EXE_GPU_GFX || exeDevice.exeType == EXE_GPU_DMA) {
      ERR_CHECK(hipSetDevice(exeDevice.exeIndex));

      // Determine how many streams to use
      int const numStreamsToUse = (exeDevice.exeType == EXE_GPU_DMA ||
                                   (exeDevice.exeType == EXE_GPU_GFX && cfg.gfx.useMultiStream))
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

      if (cfg.gfx.useHipEvents || cfg.dma.useHipEvents) {
        exeInfo.startEvents.resize(numStreamsToUse);
        exeInfo.stopEvents.resize(numStreamsToUse);
        for (int i = 0; i < numStreamsToUse; ++i) {
          ERR_CHECK(hipEventCreate(&exeInfo.startEvents[i]));
          ERR_CHECK(hipEventCreate(&exeInfo.stopEvents[i]));
        }
      }
    }

    // Prepare for GPU GFX executor
    if (exeDevice.exeType == EXE_GPU_GFX) {
      // Allocate one contiguous chunk of GPU memory for threadblock parameters
      // This allows support for executing one transfer per stream, or all transfers in a single stream
#if !defined(__NVCC__)
      MemType memType = MEM_GPU;      // AMD hardware can directly access GPU memory from host
#else
      MemType memType = MEM_MANAGED;  // NVIDIA hardware requires managed memory to access from host
#endif
      ERR_CHECK(AllocateMemory({memType, exeDevice.exeIndex}, exeInfo.totalSubExecs * sizeof(SubExecParam),
                               (void**)&exeInfo.subExecParamGpu));

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
      for (auto& resources : exeInfo.resources) {
        Transfer const& t = transfers[resources.transferIdx];
        resources.subExecParamGpuPtr = exeInfo.subExecParamGpu + transferOffset;
        for (auto p : resources.subExecParamCpu) {
          resources.subExecIdx.push_back(exeInfo.subExecParamCpu.size());
          exeInfo.subExecParamCpu.push_back(p);
          transferOffset++;
        }
      }

      // Copy sub executor parameters to GPU
      ERR_CHECK(hipSetDevice(exeDevice.exeIndex));
      ERR_CHECK(hipMemcpy(exeInfo.subExecParamGpu,
                          exeInfo.subExecParamCpu.data(),
                          exeInfo.totalSubExecs * sizeof(SubExecParam),
                          hipMemcpyHostToDevice));
      ERR_CHECK(hipDeviceSynchronize());
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
    // Loop over each transfer this executor is involved in
    for (auto& resources : exeInfo.resources) {
      Transfer const& t = transfers[resources.transferIdx];

      // Deallocate source memory
      for (int iSrc = 0; iSrc < t.srcs.size(); ++iSrc) {
        ERR_CHECK(DeallocateMemory(t.srcs[iSrc].memType, resources.srcMem[iSrc], t.numBytes + cfg.data.byteOffset));
      }

      // Deallocate destination memory
      for (int iDst = 0; iDst < t.dsts.size(); ++iDst) {
        ERR_CHECK(DeallocateMemory(t.dsts[iDst].memType, resources.dstMem[iDst], t.numBytes + cfg.data.byteOffset));
      }

      // Destroy HSA signal for DMA executor
#if !defined(__NVCC__)
      if (exeDevice.exeType == EXE_GPU_DMA && (t.exeSubIndex != -1 || cfg.dma.useHsaCopy)) {
        ERR_CHECK(hsa_signal_destroy(resources.signal));
      }
#endif
    }

    // Teardown additional requirements for GPU-based executors
    if (exeDevice.exeType == EXE_GPU_GFX || exeDevice.exeType == EXE_GPU_DMA) {
      for (auto stream : exeInfo.streams)
        ERR_CHECK(hipStreamDestroy(stream));
      if (cfg.gfx.useHipEvents || cfg.dma.useHipEvents) {
        for (auto event : exeInfo.startEvents)
          ERR_CHECK(hipEventDestroy(event));
        for (auto event : exeInfo.stopEvents)
          ERR_CHECK(hipEventDestroy(event));
      }
    }

    if (exeDevice.exeType == EXE_GPU_GFX) {
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
  static void CpuReduceKernel(SubExecParam const& p)
  {
    if (p.N == 0) return;

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
          printf("[ERROR] Nan detected\n");
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
  }

  // Execution of a single CPU Transfers
  static void ExecuteCpuTransfer(int           const  iteration,
                                 ConfigOptions const& cfg,
                                 int           const  exeIndex,
                                 TransferResources&   resources)
  {
    auto cpuStart = std::chrono::high_resolution_clock::now();
    vector<std::thread> childThreads;
    int subIteration = 0;
    do {
      for (auto const& subExecParam : resources.subExecParamCpu)
        childThreads.emplace_back(std::thread(CpuReduceKernel, std::cref(subExecParam)));

      for (auto& subExecThread : childThreads)
        subExecThread.join();
      childThreads.clear();
    } while (++subIteration != cfg.general.numSubIterations);

    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double deltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0;

    if (iteration >= 0) {
      resources.totalDurationMsec += deltaMsec;
      if (cfg.general.recordPerIteration)
        resources.perIterMsec.push_back(deltaMsec);
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
    for (auto& resource : exeInfo.resources) {
      asyncTransfers.emplace_back(std::thread(ExecuteCpuTransfer,
                                              iteration,
                                              std::cref(cfg),
                                              exeIndex,
                                              std::ref(resource)));
    }
    for (auto& asyncTransfer : asyncTransfers)
      asyncTransfer.join();

    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double deltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0;
    if (iteration >= 0)
      exeInfo.totalDurationMsec += deltaMsec;
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
  template <>           __device__ __forceinline__ float4 MemsetVal(){ return make_float4(MEMSET_VAL,
                                                                                          MEMSET_VAL,
                                                                                          MEMSET_VAL,
                                                                                          MEMSET_VAL); }

  // Kernel for GFX execution
  template <int BLOCKSIZE, int UNROLL>
  __global__ void __launch_bounds__(BLOCKSIZE)
    GpuReduceKernel(SubExecParam* params, int waveOrder, int numSubIterations)
  {
    int64_t startCycle;
    if (threadIdx.x == 0) startCycle = GetTimestamp();

    SubExecParam& p = params[blockIdx.y];

    // Filter by XCC
#if !defined(__NVCC__)
    int32_t xccId;
    GetXccId(xccId);
    if (p.preferredXccId != -1 && xccId != p.preferredXccId) return;
#endif

    // Collect data information
    int32_t const  numSrcs  = p.numSrcs;
    int32_t const  numDsts  = p.numDsts;
    float4  const* __restrict__ srcFloat4[MAX_SRCS];
    float4*        __restrict__ dstFloat4[MAX_DSTS];
    for (int i = 0; i < numSrcs; i++) srcFloat4[i] = (float4*)p.src[i];
    for (int i = 0; i < numDsts; i++) dstFloat4[i] = (float4*)p.dst[i];

    // Operate on wavefront granularity
    int32_t const nTeams   = p.teamSize;             // Number of threadblocks working together on this subarray
    int32_t const teamIdx  = p.teamIdx;              // Index of this threadblock within the team
    int32_t const nWaves   = BLOCKSIZE   / warpSize; // Number of wavefronts within this threadblock
    int32_t const waveIdx  = threadIdx.x / warpSize; // Index of this wavefront within the threadblock
    int32_t const tIdx     = threadIdx.x % warpSize; // Thread index within wavefront

    size_t  const numFloat4 = p.N / 4;

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
      // First loop: Each wavefront in the team works on UNROLL float4s per thread
      size_t const loop1Stride = nTeams * nWaves * UNROLL * warpSize;
      size_t const loop1Limit  = numFloat4 / loop1Stride * loop1Stride;
      {
        float4 val[UNROLL];
        if (numSrcs == 0) {
          #pragma unroll
          for (int u = 0; u < UNROLL; u++)
            val[u] = MemsetVal<float4>();
        }

        for (size_t idx = (teamIdx * teamStride + waveIdx * waveStride) * warpSize + tIdx; idx < loop1Limit; idx += loop1Stride) {
          // Read sources into memory and accumulate in registers
          if (numSrcs) {
            for (int u = 0; u < UNROLL; u++)
              val[u] = srcFloat4[0][idx + u * unrlStride * warpSize];
            for (int s = 1; s < numSrcs; s++)
              for (int u = 0; u < UNROLL; u++)
                val[u] += srcFloat4[s][idx + u * unrlStride * warpSize];
          }

          // Write accumulation to all outputs
          for (int d = 0; d < numDsts; d++) {
            #pragma unroll
            for (int u = 0; u < UNROLL; u++)
              dstFloat4[d][idx + u * unrlStride * warpSize] = val[u];
          }
        }
      }

      // Second loop: Deal with remaining float4s
      {
        if (loop1Limit < numFloat4) {
          float4 val;
          if (numSrcs == 0) val = MemsetVal<float4>();

          size_t const loop2Stride = nTeams * nWaves * warpSize;
          for (size_t idx = loop1Limit + (teamIdx * teamStride2 + waveIdx * waveStride2) * warpSize + tIdx;
               idx < numFloat4; idx += loop2Stride) {
            if (numSrcs) {
              val = srcFloat4[0][idx];
              for (int s = 1; s < numSrcs; s++)
                val += srcFloat4[s][idx];
            }
            for (int d = 0; d < numDsts; d++)
              dstFloat4[d][idx] = val;
          }
        }
      }

      // Third loop; Deal with remaining floats
      {
        if (numFloat4 * 4 < p.N) {
          float val;
          if (numSrcs == 0) val = MemsetVal<float>();

          size_t const loop3Stride = nTeams * nWaves * warpSize;
          for( size_t idx = numFloat4 * 4 + (teamIdx * teamStride2 + waveIdx * waveStride2) * warpSize + tIdx; idx < p.N; idx += loop3Stride) {
            if (numSrcs) {
              val = p.src[0][idx];
              for (int s = 1; s < numSrcs; s++)
                val += p.src[s][idx];
            }

            for (int d = 0; d < numDsts; d++)
              p.dst[d][idx] = val;
          }
        }
      }

      if (++subIterations == numSubIterations) break;
    }

    // Wait for all threads to finish
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence_system();
      p.stopCycle  = GetTimestamp();
      p.startCycle = startCycle;
      GetHwId(p.hwId);
      GetXccId(p.xccId);
    }
  }

#define GPU_KERNEL_UNROLL_DECL(BLOCKSIZE)   \
    {GpuReduceKernel<BLOCKSIZE, 1>,         \
     GpuReduceKernel<BLOCKSIZE, 2>,         \
     GpuReduceKernel<BLOCKSIZE, 3>,         \
     GpuReduceKernel<BLOCKSIZE, 4>,         \
     GpuReduceKernel<BLOCKSIZE, 5>,         \
     GpuReduceKernel<BLOCKSIZE, 6>,         \
     GpuReduceKernel<BLOCKSIZE, 7>,         \
     GpuReduceKernel<BLOCKSIZE, 8>}

  // Table of all GPU Reduction kernel functions (templated blocksize / unroll)
  typedef void (*GpuKernelFuncPtr)(SubExecParam*, int, int);
  GpuKernelFuncPtr GpuKernelTable[MAX_WAVEGROUPS][MAX_UNROLL] =
  {
    GPU_KERNEL_UNROLL_DECL(64),
    GPU_KERNEL_UNROLL_DECL(128),
    GPU_KERNEL_UNROLL_DECL(192),
    GPU_KERNEL_UNROLL_DECL(256),
    GPU_KERNEL_UNROLL_DECL(320),
    GPU_KERNEL_UNROLL_DECL(384),
    GPU_KERNEL_UNROLL_DECL(448),
    GPU_KERNEL_UNROLL_DECL(512)
  };
  #undef GPU_KERNEL_UNROLL_DECL

  // Execute a single GPU Transfer (when using 1 stream per Transfer)
  static ErrResult ExecuteGpuTransfer(int           const  iteration,
                                      hipStream_t   const  stream,
                                      int           const  xccDim,
                                      ConfigOptions const& cfg,
                                      TransferResources&   resources)
  {
    auto cpuStart = std::chrono::high_resolution_clock::now();

    int numSubExecs = resources.subExecParamCpu.size();
    dim3 const gridSize(xccDim, numSubExecs, 1);
    dim3 const blockSize(cfg.gfx.blockSize, 1);

#if defined(__NVCC__)
    GpuKernelTable[cfg.gfx.blockSize/64 - 1][cfg.gfx.unrollFactor - 1]
      <<<gridSize, blockSize, 0, stream>>>
      (resources.subExecParamGpuPtr, cfg.gfx.waveOrder, cfg.general.numSubIterations);
#else
    hipExtLaunchKernelGGL(GpuKernelTable[cfg.gfx.blockSize/64 - 1][cfg.gfx.unrollFactor - 1],
                          gridSize, blockSize, 0, stream,
                          NULL, NULL,
                          0, resources.subExecParamGpuPtr, cfg.gfx.waveOrder, cfg.general.numSubIterations);
#endif

    ERR_CHECK(hipStreamSynchronize(stream));

    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double deltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0;

    if (iteration >= 0) {
      resources.totalDurationMsec += deltaMsec;
      if (cfg.general.recordPerIteration) {
        resources.perIterMsec.push_back(deltaMsec);
#if !defined(__NVCC__)
        std::set<std::pair<int,int>> CUs;
        for (int i = 0; i < numSubExecs; i++) {
          CUs.insert(std::make_pair(resources.subExecParamGpuPtr[i].xccId,
                                    GetId(resources.subExecParamGpuPtr[i].hwId)));
        }
        resources.perIterCUs.push_back(CUs);
#endif
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

    if (cfg.gfx.useMultiStream) {
      // Launch each Transfer separately in its own stream
      vector<std::future<ErrResult>> asyncTransfers;
      for (int i = 0; i < exeInfo.streams.size(); i++) {
        asyncTransfers.emplace_back(std::async(std::launch::async,
                                               ExecuteGpuTransfer,
                                               iteration,
                                               exeInfo.streams[i],
                                               xccDim,
                                               std::cref(cfg),
                                               std::ref(exeInfo.resources[i])));
      }
      for (auto& asyncTransfer : asyncTransfers)
        ERR_CHECK(asyncTransfer.get());
    } else {
      // Combine all the Transfers into a single kernel launch
      int numSubExecs = exeInfo.totalSubExecs;
      dim3 const gridSize(xccDim, numSubExecs, 1);
      dim3 const blockSize(cfg.gfx.blockSize, 1);
      hipStream_t stream = exeInfo.streams[0];

#if defined(__NVCC__)
      if (cfg.gfx.useHipEvents)
        ERR_CHECK(hipEventRecord(exeInfo.startEvents[0], stream));

      GpuKernelTable[cfg.gfx.blockSize/64 - 1][cfg.gfx.unrollFactor - 1]
        <<<gridSize, blockSize, 0 , stream>>>
        (exeInfo.subExecParamGpu, cfg.gfx.waveOrder, cfg.general.numSubIterations);

      if (cfg.gfx.useHipEvents)
        ERR_CHECK(hipEventRecord(exeInfo.stopEvents[0], stream));
#else
      hipExtLaunchKernelGGL(GpuKernelTable[cfg.gfx.blockSize/64 - 1][cfg.gfx.unrollFactor - 1],
                            gridSize, blockSize, 0, stream,
                            cfg.gfx.useHipEvents ? exeInfo.startEvents[0] : NULL,
                            cfg.gfx.useHipEvents ? exeInfo.stopEvents[0] : NULL, 0,
                            exeInfo.subExecParamGpu, cfg.gfx.waveOrder, cfg.general.numSubIterations);
#endif
      ERR_CHECK(hipStreamSynchronize(stream));
    }
    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double cpuDeltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0;

    if (iteration >= 0) {
      if (cfg.gfx.useHipEvents) {
        float gpuDeltaMsec;
        ERR_CHECK(hipEventElapsedTime(&gpuDeltaMsec, exeInfo.startEvents[0], exeInfo.stopEvents[0]));
        exeInfo.totalDurationMsec += gpuDeltaMsec;
      } else {
        exeInfo.totalDurationMsec += cpuDeltaMsec;
      }

      // Determine timing for each of the individual transfers that were part of this launch
      for (int i = 0; i < exeInfo.resources.size(); i++) {
        TransferResources& resources = exeInfo.resources[i];
        long long minStartCycle = std::numeric_limits<long long>::max();
        long long maxStopCycle  = std::numeric_limits<long long>::min();
        std::set<std::pair<int, int>> CUs;

        for (auto subExecIdx : resources.subExecIdx) {
          minStartCycle = std::min(minStartCycle, exeInfo.subExecParamGpu[subExecIdx].startCycle);
          maxStopCycle  = std::max(maxStopCycle,  exeInfo.subExecParamGpu[subExecIdx].stopCycle);
          if (cfg.general.recordPerIteration) {
#if !defined(__NVCC__)
            CUs.insert(std::make_pair(exeInfo.subExecParamGpu[subExecIdx].xccId,
                                      GetId(exeInfo.subExecParamGpu[subExecIdx].hwId)));
#endif
          }
        }
        double deltaMsec = (maxStopCycle - minStartCycle) / (double)(exeInfo.wallClockRate);

        resources.totalDurationMsec += deltaMsec;
        if (cfg.general.recordPerIteration) {
          resources.perIterMsec.push_back(deltaMsec);
#if !defined(__NVCC__)
          resources.perIterCUs.push_back(CUs);
#endif
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
                                      hipStream_t   const  stream,
                                      hipEvent_t    const  startEvent,
                                      hipEvent_t    const  stopEvent,
                                      ConfigOptions const& cfg,
                                      TransferResources&   resources)
  {
    auto cpuStart = std::chrono::high_resolution_clock::now();

    int subIterations = 0;
    if (!useSubIndices && !cfg.dma.useHsaCopy) {
      if (cfg.dma.useHipEvents)
        ERR_CHECK(hipEventRecord(startEvent, stream));

      // Use hipMemcpy
      do {
        ERR_CHECK(hipMemcpyAsync(resources.dstMem[0], resources.srcMem[0], resources.numBytes,
                                 hipMemcpyDefault, stream));
      } while (++subIterations != cfg.general.numSubIterations);

      if (cfg.dma.useHipEvents)
        ERR_CHECK(hipEventRecord(stopEvent, stream));
      ERR_CHECK(hipStreamSynchronize(stream));
    } else {
#if defined(__NVCC__)
      return {ERR_FATAL, "HSA copy not supported on NVIDIA hardware"};
#else
      // Use HSA async copy
      do {
        hsa_signal_store_screlease(resources.signal, 1);
        if (cfg.dma.useHsaCopy) {
          ERR_CHECK(hsa_amd_memory_async_copy(resources.dstMem[0], resources.dstAgent,
                                              resources.srcMem[0], resources.srcAgent,
                                              resources.numBytes, 0, NULL,
                                              resources.signal));
        } else {
          HSA_CALL(hsa_amd_memory_async_copy_on_engine(resources.dstMem[0], resources.dstAgent,
                                                       resources.srcMem[0], resources.srcAgent,
                                                       resources.numBytes, 0, NULL,
                                                       resources.signal,
                                                       resources.sdmaEngineId, true));
        }
        // Wait for SDMA transfer to complete
        while(hsa_signal_wait_scacquire(resources.signal,
                                        HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                        HSA_WAIT_STATE_ACTIVE) >= 1);
      } while (++subIterations != cfg.general.numSubIterations);
#endif
    }
    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double cpuDeltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0;

    if (iteration >= 0) {
      double deltaMsec = cpuDeltaMsec;
      if (!useSubIndices && !cfg.dma.useHsaCopy && cfg.dma.useHipEvents) {
        float gpuDeltaMsec;
        ERR_CHECK(hipEventElapsedTime(&gpuDeltaMsec, startEvent, stopEvent));
        deltaMsec = gpuDeltaMsec;
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
                                             exeInfo.streams[i],
                                             cfg.dma.useHipEvents ? exeInfo.startEvents[i] : NULL,
                                             cfg.dma.useHipEvents ? exeInfo.stopEvents[i]  : NULL,
                                             std::cref(cfg),
                                             std::ref(exeInfo.resources[i])));
    }

    for (auto& asyncTransfer : asyncTransfers)
      ERR_CHECK(asyncTransfer.get());

    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double deltaMsec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0;
    if (iteration >= 0)
      exeInfo.totalDurationMsec += deltaMsec;
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
    case EXE_CPU:     return RunCpuExecutor(iteration, cfg, exeDevice.exeIndex, exeInfo);
    case EXE_GPU_GFX: return RunGpuExecutor(iteration, cfg, exeDevice.exeIndex, exeInfo);
    case EXE_GPU_DMA: return RunDmaExecutor(iteration, cfg, exeDevice.exeIndex, exeInfo);
    default:          return {ERR_FATAL, "Unsupported executor (%d)", exeDevice.exeType};
    }
  }

} // End of anonymous namespace
//========================================================================================

  ErrResult::ErrResult(ErrType err) : errType(err), errMsg("") {};

  ErrResult::ErrResult(hipError_t err)
  {
    if (err == hipSuccess) {
      this->errType = ERR_NONE;
      this->errMsg  = "";
    } else {
      this->errType = ERR_FATAL;
      this->errMsg  = std::string("HIP Error: ") + hipGetErrorString(err);
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

    // Check for valid configuration
    if (ConfigOptionsHaveErrors(cfg, errResults)) return false;

    // Check for valid transfers
    if (TransfersHaveErrors(cfg, transfers, errResults)) return false;

    // Collect up transfers by executor
    int minNumSrcs = MAX_SRCS + 1;
    int maxNumSrcs = 0;
    size_t maxNumBytes = 0;
    std::map<ExeDevice, ExeInfo> executorMap;
    for (int i = 0; i < transfers.size(); i++) {
      Transfer const& t = transfers[i];

      ExeInfo& exeInfo = executorMap[t.exeDevice];
      exeInfo.totalBytes    += t.numBytes;
      exeInfo.totalSubExecs += t.numSubExecs;
      exeInfo.useSubIndices |= (t.exeSubIndex != -1);

      TransferResources resource = {};
      resource.transferIdx = i;
      exeInfo.resources.push_back(resource);

      minNumSrcs  = std::min(minNumSrcs, (int)t.srcs.size());
      maxNumSrcs  = std::max(maxNumSrcs, (int)t.srcs.size());
      maxNumBytes = std::max(maxNumBytes, t.numBytes);
    }

    // Loop over each executor and prepare
    // - Allocates memory for each Transfer
    // - Set up work for subexecutors
    vector<TransferResources*> transferResources;
    for (auto& exeInfoPair : executorMap) {
      ExeDevice const& exeDevice = exeInfoPair.first;
      ExeInfo&         exeInfo   = exeInfoPair.second;
      ERR_APPEND(PrepareExecutor(cfg, transfers, exeDevice, exeInfo), errResults);

      for (auto& resource : exeInfo.resources) {
        transferResources.push_back(&resource);
      }
    }

    // Prepare reference src/dst arrays - only once for largest size
    size_t maxN = maxNumBytes / sizeof(float);
    vector<float> outputBuffer(maxN);
    vector<vector<float>> dstReference(maxNumSrcs + 1, vector<float>(maxN));
    {
      vector<vector<float>> srcReference(maxNumSrcs, vector<float>(maxN));
      memset(dstReference[0].data(), MEMSET_CHAR, maxNumBytes);

      for (int numSrcs = 0; numSrcs < maxNumSrcs; numSrcs++) {
        PrepareReference(cfg, srcReference[numSrcs], numSrcs);
        for (int i = 0; i < maxN; i++) {
          dstReference[numSrcs+1][i] = (numSrcs == 0 ? 0 : dstReference[numSrcs][i]) + srcReference[numSrcs][i];
        }
      }
      // Release un-used partial sums
      for (int numSrcs = 0; numSrcs < minNumSrcs; numSrcs++)
        dstReference[numSrcs].clear();

      // Initialize all src memory buffers
      for (auto resource : transferResources) {
        for (int srcIdx = 0; srcIdx < resource->srcMem.size(); srcIdx++) {
          ERR_APPEND(hipMemcpy(resource->srcMem[srcIdx], srcReference[srcIdx].data(), resource->numBytes,
                               hipMemcpyDefault), errResults);
        }
      }
    }

    // Perform iterations
    size_t numTimedIterations = 0;
    double totalCpuTimeSec = 0.0;
    for (int iteration = -cfg.general.numWarmups; ; iteration++) {
      // Stop if number of iterations/seconds has reached limit
      if (cfg.general.numIterations > 0 && iteration >= cfg.general.numIterations) break;
      if (cfg.general.numIterations < 0 && totalCpuTimeSec > -cfg.general.numIterations) break;

      // Pause before starting first timed iteration in iteractive mode
      if (cfg.general.useInteractive && iteration == 0) {
        printf("Memory prepared:\n");

        for (int i = 0; i < transfers.size(); i++) {
          ExeInfo const& exeInfo = executorMap[transfers[i].exeDevice];
          printf("Transfer %03d:\n", i);
          for (int iSrc = 0; iSrc < transfers[i].srcs.size(); ++iSrc)
            printf("  SRC %0d: %p\n", iSrc, exeInfo.resources[i].srcMem[iSrc]);
          for (int iDst = 0; iDst < transfers[i].dsts.size(); ++iDst)
            printf("  DST %0d: %p\n", iDst, exeInfo.resources[i].dstMem[iDst]);
        }
        printf("Hit <Enter> to continue: ");
        if (scanf("%*c") != 0) {
          printf("[ERROR] Unexpected input\n");
          exit(1);
        }
        printf("\n");
      }

      // Start CPU timing for this iteration
      auto cpuStart = std::chrono::high_resolution_clock::now();

      // Execute all Transfers in parallel
      std::vector<std::future<ErrResult>> asyncExecutors;
      for (auto& exeInfoPair : executorMap) {
        asyncExecutors.emplace_back(std::async(std::launch::async, RunExecutor,
                                               iteration,
                                               std::cref(cfg),
                                               std::cref(exeInfoPair.first),
                                               std::ref(exeInfoPair.second)));
      }

      // Wait for all threads to finish
      for (auto& asyncExecutor : asyncExecutors) {
        ERR_APPEND(asyncExecutor.get(), errResults);
      }

       // Stop CPU timing for this iteration
      auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
      double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count();

      if (cfg.data.alwaysValidate) {
        ERR_APPEND(ValidateAllTransfers(cfg, transfers, transferResources, dstReference, outputBuffer),
                   errResults);
      }

      if (iteration >= 0) {
        ++numTimedIterations;
        totalCpuTimeSec += deltaSec;
      }
    }

    // Pause for interactive mode
    if (cfg.general.useInteractive) {
      printf("Transfers complete. Hit <Enter> to continue: ");
      if (scanf("%*c") != 0)  {
        printf("[ERROR] Unexpected input\n");
        exit(1);
      }
      printf("\n");
    }

    // Validate results
    if (!cfg.data.alwaysValidate) {
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
    results.overheadMsec = 0.0;
    for (auto& exeInfoPair : executorMap) {
      ExeDevice const& exeDevice = exeInfoPair.first;
      ExeInfo&         exeInfo   = exeInfoPair.second;

      // Copy over executor results
      ExeResult& exeResult = results.exeResults[exeDevice];
      exeResult.numBytes = exeInfo.totalBytes;
      exeResult.avgDurationMsec = exeInfo.totalDurationMsec / numTimedIterations;
      exeResult.avgBandwidthGbPerSec = (exeResult.numBytes / 1.0e6) /  exeResult.avgDurationMsec;
      exeResult.sumBandwidthGbPerSec = 0.0;
      exeResult.transferIdx.clear();
      results.totalBytesTransferred += exeInfo.totalBytes;
      results.overheadMsec = std::max(results.overheadMsec, (results.avgTotalDurationMsec -
                                                             exeResult.avgDurationMsec));

      // Copy over transfer results
      for (auto const& resources : exeInfo.resources) {
        int const transferIdx = resources.transferIdx;
        TransferResult& tfrResult = results.tfrResults[transferIdx];
        exeResult.transferIdx.push_back(transferIdx);
        tfrResult.numBytes = resources.numBytes;
        tfrResult.avgDurationMsec = resources.totalDurationMsec / numTimedIterations;
        tfrResult.avgBandwidthGbPerSec = (resources.numBytes / 1.0e6) / tfrResult.avgDurationMsec;
        if (cfg.general.recordPerIteration) {
          tfrResult.perIterMsec = resources.perIterMsec;
          tfrResult.perIterCUs  = resources.perIterCUs;
        }
        exeResult.sumBandwidthGbPerSec += tfrResult.avgBandwidthGbPerSec;
      }
    }
    results.avgTotalBandwidthGbPerSec = (results.totalBytesTransferred / 1.0e6) / results.avgTotalDurationMsec;

    // Teardown executors
    for (auto& exeInfoPair : executorMap) {
      ExeDevice const& exeDevice = exeInfoPair.first;
      ExeInfo&         exeInfo   = exeInfoPair.second;
      ERR_APPEND(TeardownExecutor(cfg, exeDevice, transfers, exeInfo), errResults);
    }

    return true;
  }

  int GetIntAttribute(IntAttribute attribute)
  {
    switch (attribute) {
    case ATR_GFX_MAX_BLOCKSIZE: return MAX_BLOCKSIZE;
    case ATR_GFX_MAX_UNROLL:    return MAX_UNROLL;
    default:                    return -1;
    }
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

  ErrResult ParseTransfers(std::string            line,
                           std::vector<Transfer>& transfers)
  {
    // Replace any round brackets or '->' with spaces,
    for (int i = 1; line[i]; i++)
      if (line[i] == '(' || line[i] == ')' || line[i] == '-' || line[i] == '>' ) line[i] = ' ';

    transfers.clear();

    // Read in number of transfers
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
      Transfer transfer;

      if (!advancedMode) {
        iss >> srcStr >> exeStr >> dstStr;
        transfer.numSubExecs = numSubExecs;
        if (iss.fail()) {
          return {ERR_FATAL,
                  "Parsing error: Unable to read valid Transfer %d (SRC EXE DST) triplet", i+1};
        }
      } else {
        iss >> srcStr >> exeStr >> dstStr >> transfer.numSubExecs >> numBytesToken;
        if (iss.fail()) {
          return {ERR_FATAL,
                  "Parsing error: Unable to read valid Transfer %d (SRC EXE DST $CU #Bytes) tuple", i+1};
        }
        if (sscanf(numBytesToken.c_str(), "%lu", &transfer.numBytes) != 1) {
          return {ERR_FATAL,
                  "Parsing error: Unable to read valid Transfer %d (SRC EXE DST #CU #Bytes) tuple", i+1};
        }

        char units = numBytesToken.back();
        switch (toupper(units)) {
        case 'G': transfer.numBytes *= 1024;
        case 'M': transfer.numBytes *= 1024;
        case 'K': transfer.numBytes *= 1024;
        }
      }

      ERR_CHECK(ParseMemType(srcStr, transfer.srcs));
      ERR_CHECK(ParseMemType(dstStr, transfer.dsts));
      ERR_CHECK(ParseExeType(exeStr, transfer.exeDevice, transfer.exeSubIndex));

      transfers.push_back(transfer);
    }
    return ERR_NONE;
  }

  int GetNumExecutors(ExeType exeType)
  {
    switch (exeType) {
    case EXE_CPU:
      return numa_num_configured_nodes();
    case EXE_GPU_GFX: case EXE_GPU_DMA:
    {
      int numDetectedGpus = 0;
      hipError_t status = hipGetDeviceCount(&numDetectedGpus);
      if (status != hipSuccess) numDetectedGpus = 0;
      return numDetectedGpus;
    }
    default:
      return 0;
    }
  }

  int GetNumSubExecutors(ExeDevice exeDevice)
  {
    int const& exeIndex = exeDevice.exeIndex;

    switch(exeDevice.exeType) {
    case EXE_CPU:
    {
      int numCores = 0;
      for (int i = 0; i < numa_num_configured_cpus(); i++)
        if (numa_node_of_cpu(i) == exeIndex) numCores++;
      return numCores;
    }
    case EXE_GPU_GFX:
    {
      int numGpus = GetNumExecutors(EXE_GPU_GFX);
      if (exeIndex < 0 || numGpus <= exeIndex) return 0;
      int numDeviceCUs = 0;
      hipError_t status = hipDeviceGetAttribute(&numDeviceCUs, hipDeviceAttributeMultiprocessorCount, exeIndex);
      if (status != hipSuccess) numDeviceCUs = 0;
      return numDeviceCUs;
    }
    case EXE_GPU_DMA:
    {
      return 1;
    }
    default:
      return 0;
    }
  }

  int GetNumExecutorSubIndices(ExeDevice exeDevice)
  {
    // Executor subindices are not supported on NVIDIA hardware
#if defined(__NVCC__)
    return 0;
#else
    int const& exeIndex = exeDevice.exeIndex;

    switch(exeDevice.exeType) {
    case EXE_CPU: return 0;
    case EXE_GPU_GFX:
    {
      hsa_agent_t agent;
      ErrResult err = GetHsaAgent(exeDevice, agent);
      if (err.errType != ERR_NONE) return 0;
      int numXccs = 1;
      if (hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_XCC, &numXccs) != HSA_STATUS_SUCCESS)
        return 1;
      return numXccs;
    }
    case EXE_GPU_DMA:
    {
      std::set<int> engineIds;
      ErrResult err;

      // Get HSA agent for this GPU
      hsa_agent_t agent;
      err = GetHsaAgent(exeDevice, agent);
      if (err.errType != ERR_NONE) return 0;

      int numTotalEngines = 0, numEnginesA = 0, numEnginesB = 0;
      if (hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SDMA_ENG, &numEnginesA)
          == HSA_STATUS_SUCCESS)
        numTotalEngines += numEnginesA;
      if (hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG, &numEnginesB)
          == HSA_STATUS_SUCCESS)
        numTotalEngines += numEnginesB;

      return numTotalEngines;
    }
    default:
      return 0;
    }
#endif
  }

  int GetClosestCpuNumaToGpu(int gpuIndex)
  {
    // Closest NUMA is not supported on NVIDIA hardware at this time
#if defined(__NVCC__)
    return -1;
#else
    hsa_agent_t gpuAgent;
    ErrResult err = GetHsaAgent({EXE_GPU_GFX, gpuIndex}, gpuAgent);
    if (err.errType != ERR_NONE) return -1;

    hsa_agent_t closestCpuAgent;
    if (hsa_agent_get_info(gpuAgent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NEAREST_CPU, &closestCpuAgent)
        == HSA_STATUS_SUCCESS) {
      int numCpus = GetNumExecutors(EXE_CPU);
      for (int i = 0; i < numCpus; i++) {
        hsa_agent_t cpuAgent;
        err = GetHsaAgent({EXE_CPU, i}, cpuAgent);
        if (err.errType != ERR_NONE) return -1;
        if (cpuAgent.handle == closestCpuAgent.handle) return i;
      }
    }
    return -1;
#endif
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

// Enumerations
#undef hipDeviceAttributeClockRate
#undef hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
#undef hipDeviceAttributeMultiprocessorCount
#undef hipErrorPeerAccessAlreadyEnabled
#undef hipFuncCachePreferShared
#undef hipMemcpyDefault
#undef hipMemcpyDeviceToHost
#undef hipMemcpyHostToDevice
#undef hipSuccess

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
#endif

// Kernel macros
#undef GetHwId
#undef GetXccId

// Undefine helper macros
#undef ERR_CHECK
#undef ERR_APPEND
}

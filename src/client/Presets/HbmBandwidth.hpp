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

#pragma once

#include "EnvVars.hpp"
#include "Utilities.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace TransferBench;

// CUDA translation
#if defined(__NVCC__)
#define hipEvent_t           cudaEvent_t
#define hipEventCreate       cudaEventCreate
#define hipEventDestroy      cudaEventDestroy
#define hipEventElapsedTime  cudaEventElapsedTime
#define hipEventRecord       cudaEventRecord
#define hipSetDevice         cudaSetDevice
#define hipStream_t          cudaStream_t
#define hipStreamCreate      cudaStreamCreate
#define hipStreamDestroy     cudaStreamDestroy
#define hipStreamSynchronize cudaStreamSynchronize
#endif

// Load a value
template<bool USE_NT, typename T>
__device__ __forceinline__ T Load(const T& ref)
{
#if !defined(__NVCC__)
  if (USE_NT) return __builtin_nontemporal_load(&ref);
#endif
  return ref;
}

// Main kernel for HBM bandwidth testing
template<int LAUNCH_BOUND, int UNROLL, typename T, bool USE_NT>
__global__ __launch_bounds__(LAUNCH_BOUND)
void HbmReadBwKernel(const void* __restrict pSrcBuffer,
                     void*       __restrict dummy,
                     const size_t           numSteps,
                     long long*  __restrict minStartCycle,
                     long long*  __restrict maxStopCycle)
{
  int64_t startTime;
  if (threadIdx.x == 0) {
    startTime = GetTimestamp();
  }

  // Cast src/dst buffers to the correct type
  T const* __restrict srcBuffer = reinterpret_cast<T const*>(pSrcBuffer);
  T*       __restrict dstBuffer = reinterpret_cast<T*      >(dummy);
  T v{};

  // Determine the total number of elements this threadblock handles
  size_t elemPerThreadblock = numSteps * blockDim.x * UNROLL;

  // Determine the initial offset for this threadblock
  size_t srcOffset = blockIdx.x * elemPerThreadblock + threadIdx.x;

  #pragma unroll 1
  for (size_t step = 0; step < numSteps; step++) {
    #pragma unroll
    for (uint32_t i = 0; i < UNROLL; i++) {
      v |= Load<USE_NT>(srcBuffer[srcOffset]);
      srcOffset += blockDim.x;
    }
  }

  // This statement is never true, but is required to make sure compiler
  // doesn't optimize away the reads
  if (elemPerThreadblock == 0)
    *dstBuffer = v;

  // Update min/max start times
  __syncthreads();
  if (threadIdx.x == 0 && minStartCycle != NULL) {
    int64_t stopTime = GetTimestamp();
    atomicMin(minStartCycle, startTime);
    atomicMax(maxStopCycle, stopTime);
  }
}

// Build up function pointer table
typedef void (*HbmReadBwKernelFuncPtr)(const void*, void *, size_t, long long*, long long*);

#define HBM_KERNEL_TEMPORAL_DECL(LAUNCH_BOUND, UNROLL, DTYPE) \
  {HbmReadBwKernel<LAUNCH_BOUND, UNROLL, DTYPE, false>,       \
   HbmReadBwKernel<LAUNCH_BOUND, UNROLL, DTYPE, true>}

#define HBM_KERNEL_DTYPE_DECL(LAUNCH_BOUND, UNROLL)             \
  {HBM_KERNEL_TEMPORAL_DECL(LAUNCH_BOUND, UNROLL, uint32_t),    \
   HBM_KERNEL_TEMPORAL_DECL(LAUNCH_BOUND, UNROLL, uint64_t),    \
   HBM_KERNEL_TEMPORAL_DECL(LAUNCH_BOUND, UNROLL, __uint128_t)}

#define HBM_KERNEL_UNROLL_DECL(LAUNCH_BOUND) \
  {HBM_KERNEL_DTYPE_DECL(LAUNCH_BOUND, 1),   \
   HBM_KERNEL_DTYPE_DECL(LAUNCH_BOUND, 2),   \
   HBM_KERNEL_DTYPE_DECL(LAUNCH_BOUND, 4),   \
   HBM_KERNEL_DTYPE_DECL(LAUNCH_BOUND, 8),   \
   HBM_KERNEL_DTYPE_DECL(LAUNCH_BOUND, 16)}

  HbmReadBwKernelFuncPtr HbmReadKernelTable[4][5][3][2] =
  {
    HBM_KERNEL_UNROLL_DECL(256),
    HBM_KERNEL_UNROLL_DECL(512),
    HBM_KERNEL_UNROLL_DECL(768),
    HBM_KERNEL_UNROLL_DECL(1024)
  };

// Kernel to fill buffer with random data
__global__ void FillPsuedoRandomData(size_t N, uint32_t* p, uint32_t shift)
{
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
    uint32_t d = static_cast<uint32_t>(idx + shift);
    uint32_t val = 2166136261u;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      val ^= d & 0xff;
      val *= 16777619u;
      d >>= 8;
    }
    p[idx] = val;
  }
}

struct HbmBwResult
{
  int rank;
  int gpuIdx;
  int numSubExec;
  int blockSize;
  int unroll;
  int elemByte;

  double bw[3];  // MAX | AVG | MIN
};

int HbmBandwidthPreset(EnvVars&          ev,
                       size_t      const numBytesPerTransfer,
                       std::string const presetName,
                       bool        const bytesSpecified)
{
  // If bytes aren't specified, default to 1GB
  size_t numBytesAtLeast = (bytesSpecified ? numBytesPerTransfer : 1024 * 1024 * 1024);

  // Determine rank information
  int numRanks = TransferBench::GetNumRanks();
  int myRank   = TransferBench::GetRank();

  // Make sure each rank has at least one GPU
  for (int rank = 0; rank < numRanks; rank++) {
    if (TransferBench::GetNumExecutors(EXE_GPU_GFX, rank) == 0) {
      Utils::Print("[ERROR] Each rank must have at least GPU.  Rank %d has no GPUs\n", rank);
      return 1;
    }
  }
  int defSubExec = TransferBench::GetNumSubExecutors({EXE_GPU_GFX, 0});

  // Collect environment variables
  std::vector<int> blockSizes    = EnvVars::GetEnvVarArray("BLOCKSIZES"    ,   {256, 512});
  int              criteria      = EnvVars::GetEnvVar     ("CRITERIA"      ,            0);
  std::vector<int> elemBytes     = EnvVars::GetEnvVarArray("ELEM_BYTES"    ,       {16,8});
  std::vector<int> gpuIndices    = EnvVars::GetEnvVarArray("GPU_INDICES"   ,           {});
  int              memTypeIdx    = EnvVars::GetEnvVar     ("MEM_TYPE"      ,            0);
  int              numBuffers    = EnvVars::GetEnvVar     ("NUM_BUFFERS"   ,            2);
  int              numIterations = EnvVars::GetEnvVar     ("NUM_ITERATIONS",          100);
  std::vector<int> numSesList    = EnvVars::GetEnvVarArray("NUM_SUB_EXECS" , {defSubExec});
  int              outputToCsv   = EnvVars::GetEnvVar     ("OUTPUT_TO_CSV" ,            0);
  int              prewarmMsec   = EnvVars::GetEnvVar     ("PREWARM_MSEC"  ,           50);
  int              showBorders   = EnvVars::GetEnvVar     ("SHOW_BORDERS"  ,            1);
  int              showDetails   = EnvVars::GetEnvVar     ("SHOW_DETAILS"  ,            0);
  int              showExtra     = EnvVars::GetEnvVar     ("SHOW_EXTRA"    ,            0);
  int              temporalMask  = EnvVars::GetEnvVar     ("TEMPORAL_MASK" ,            3);
  std::vector<int> unrolls       = EnvVars::GetEnvVarArray("UNROLLS"       ,     {16,8,4});
  int              useWallClock  = EnvVars::GetEnvVar     ("USE_WALLCLOCK" ,            1);

  // SHOW_DETAILS is not supported in multi-rank runs
  if (numRanks > 1) showDetails = 0;

  // Non-temporal reads are not supported for CUDA
#if defined(__NVCC__)
  temporalMask = 1;
#endif

  // Check for consistency across ranks
  IS_UNIFORM(blockSizes,    "BLOCKSIZES");
  IS_UNIFORM(criteria,      "CRITERIA");
  IS_UNIFORM(elemBytes,     "ELEM_BYTES");
  // GPU_INDICES may be different per rank
  IS_UNIFORM(memTypeIdx,    "MEM_TYPE");
  IS_UNIFORM(numBuffers,    "NUM_BUFFERS");
  IS_UNIFORM(numIterations, "NUM_ITERATIONS");
  IS_UNIFORM(numSesList,    "NUM_SUB_EXECS");
  IS_UNIFORM(prewarmMsec,   "PREWARM_MSEC");
  IS_UNIFORM(showDetails,   "SHOW_DETAILS");
  IS_UNIFORM(showExtra,     "SHOW_EXTRA");
  IS_UNIFORM(temporalMask,  "TEMPORAL_MASK");
  IS_UNIFORM(unrolls,       "UNROLLS");
  IS_UNIFORM(useWallClock,  "USE_WALLCLOCK");

  // Validate environment variables and set defaults
  if (blockSizes.empty()) {
    Utils::Print("[ERROR] BLOCKSIZES may not be empty\n");
    return 1;
  }
  for (auto blockSize : blockSizes) {
    if (blockSize <= 0 || blockSize % 128 != 0 || blockSize > 1024) {
      Utils::Print("[ERROR] BLOCKSIZES must only contain positive multiples of 128 up to 1024 (not %d)\n", blockSize);
      return 1;
    }
  }

  if (criteria < 0 || criteria > 2) {
    Utils::Print("[ERROR] CRITERIA must be either 0 (for MAX), 1 (for AVG), or 2 (for MIN) (not %d)\n", criteria);
    return 1;
  }

  if (elemBytes.empty()) {
    Utils::Print("[ERROR] ELEM_BYTES may not be empty\n");
    return 1;
  }
  for (auto elemByte : elemBytes) {
    if (elemByte != 4 && elemByte != 8 && elemByte != 16) {
      Utils::Print("[ERROR] ELEM_BYTES may only contain {4,8 or 16}\n");
      return 1;
    }
  }

  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);
  if (!gpuIndices.empty()) {
    for (auto gpuIdx : gpuIndices) {
      if (gpuIdx < 0 || gpuIdx >= numDetectedGpus) {
        Utils::Print("[ERROR] GPU_INDICES index out of range (%d) (rank %d)\n", gpuIdx, myRank);
        return 1;
      }
    }
  }

  if (numBuffers < 1) {
    Utils::Print("[ERROR] NUM_BUFFERS must be a positive number (not %d)\n", numBuffers);
    return 1;
  }
  if (numBuffers > numIterations) {
    Utils::Print("[WARN] NUM_BUFFERS (%d) exceeds NUM_ITERATIONS (%d), so some buffers will not be used\n",
                 numBuffers, numIterations);
    numBuffers = numIterations;
  }

  if (numIterations <= 0) {
    Utils::Print("[ERROR] NUM_ITERATIONS must be positive (not %d)\n", numIterations);
    return 1;
  }


  if (numSesList.empty()) {
    // By default, use all available sub executors
    numSesList.push_back(defSubExec);
  } else {
    for (auto x : numSesList) {
      if (x <= 0 || x > defSubExec) {
        Utils::Print("[ERROR] Number of subexecutors must be positive and less than %d\n", defSubExec);
        return 1;
      }
    }
  }

  if (prewarmMsec < 0) {
    Utils::Print("[ERROR] PREWARM_MSEC must be non-negative (not %d)\n", prewarmMsec);
    return 1;
  }

  if (temporalMask < 1 || temporalMask > 3) {
    Utils::Print("[ERROR] TEMPORAL_MASK must be between 1 to 3 (not %d)\n", temporalMask);
    return 1;
  }

  if (unrolls.empty()) {
    Utils::Print("[ERROR] UNROLLS may not be empty");
    return 1;
  }
  for (auto unroll : unrolls) {
    if (unroll != 1 && unroll != 2 && unroll != 4 && unroll != 8 && unroll != 16) {
      Utils::Print("[ERROR] UNROLLS must only contain {1,2,4,8 or 16} (not %d)\n", unroll);
      return 1;
    }
  }

  MemType memType = Utils::GetGpuMemType(memTypeIdx);
  std::string devMemTypeStr = Utils::GetGpuMemTypeStr(memTypeIdx);

  if (!ev.hideEnv)
  {
    if (!ev.outputToCsv) Utils::Print("[HBM Bandwidth Related]\n");
    if (Utils::RankDoesOutput()) {
      ev.Print("BLOCKSIZES"    , EnvVars::ToStr(blockSizes).c_str(), "Threadblock sizes to sweep over (multiple of 128 up to 1024)");
      ev.Print("CRITERIA"      , criteria                          , "Reporting highest %s bandwidth (0=MAX,1=AVG,2=MIN)", criteria == 0 ? "MAX" : criteria == 1 ? "AVG" : "MIN");
      ev.Print("ELEM_BYTES"    , EnvVars::ToStr(elemBytes).c_str() , "Element sizes in bytes to sweep over (must contain only 4,8 or 16)");
      ev.Print("GPU_INDICES"   , EnvVars::ToStr(gpuIndices).c_str(), "GPU indices to test.  Leave empty for all");
      ev.Print("MEM_TYPE"      , memTypeIdx                        , "Using %s memory (%s)", devMemTypeStr.c_str(), Utils::GetAllGpuMemTypeStr().c_str());
      ev.Print("NUM_BUFFERS"   , numBuffers                        , "Number of buffers to rotate through (1 per iteration)");
      ev.Print("NUM_ITERATIONS", numIterations                     , "Number of iterations to time");
      ev.Print("NUM_SUB_EXECS" , EnvVars::ToStr(numSesList).c_str(), "Number of subexecutors to sweep over (default to all available)");
      ev.Print("PREWARM_MSEC"  , prewarmMsec                       , "Prewarm duration in msec");
      ev.Print("SHOW_DETAILS"  , showDetails                       , "Show sweep details (ignored for multi-rank).  Setting to 2 shows per iteration output");
      ev.Print("SHOW_EXTRA"    , showExtra                         , "Show best sweep config details");
      ev.Print("TEMPORAL_MASK" , temporalMask                      , "Temporal mask (1 = temporal, 2 = non-temporal, 3 = both)");
      ev.Print("UNROLLS"       , EnvVars::ToStr(unrolls).c_str()   , "Unroll factors to sweep over (must contain only 1,2,4,8 or 16)");
      ev.Print("USE_WALLCLOCK" , useWallClock                      , useWallClock ? "Using GPU wall-clock for timing" : "Using events for timing");
      Utils::Print("\n");
    }
  }

  if (gpuIndices.empty()) {
    // If empty, use all available GPUs on local rank
    for (int gpuIdx = 0; gpuIdx < numDetectedGpus; gpuIdx++)
      gpuIndices.push_back(gpuIdx);
  }

  // Determine how how much memory to allocate based on sweep setting
  // During each Step each threadblock works on BLOCKSIZE * UNROLL * ELEM_BYTES bytes
  // Each buffer will be allocated as the smallest multiple of this, larger than numBytesAtLeast,
  // NOTE: It's not safe to just base this on maximums values in each sweep parameter,
  //       (e.g if maximum size divides numBytesAtLeast perfectly) so looping over entire space is safer
  size_t largestTotalBytesPerBuffer = 0;
  for (int numSubExec : numSesList) {
    for (int blockSize : blockSizes) {
      for (int unroll : unrolls) {
        for (int elemByte : elemBytes) {
          size_t totalBytesPerStep = numSubExec * blockSize * unroll * elemByte;
          size_t numSteps = std::max((size_t)1, (numBytesAtLeast + totalBytesPerStep - 1) / totalBytesPerStep);
          size_t totalBytesPerBuffer = numSteps * totalBytesPerStep;
          if (totalBytesPerBuffer > largestTotalBytesPerBuffer) largestTotalBytesPerBuffer = totalBytesPerBuffer;
        }
      }
    }
  }

  if (showDetails) {
    Utils::Print("GPU ## | #SE | BKSZ | UR | EB | TOTALBYTES | #STEP | MAX GB/s | AVG GB/s | MIN GB/s\n");
  }

  // Test all local GPUs
  std::vector<HbmBwResult> localResults;

  if (!showDetails) {
    // Calculate total number of tests that will be executed per GPU
    size_t numTests = numSesList.size() * blockSizes.size() * unrolls.size() * elemBytes.size() * (temporalMask == 3 ? 2 : 1);

    Utils::Print("Testing on at least %lu bytes (%lu configs per GPU): ", numBytesAtLeast, numTests);
    fflush(stdout);
  }

  for (int gpuIdx : gpuIndices) {
    HIP_CALL(hipSetDevice(gpuIdx));

    // Create streams/events for this GPU
    hipStream_t stream;
    hipEvent_t startEvent, stopEvent;
    HIP_CALL(hipStreamCreate(&stream));
    HIP_CALL(hipEventCreate(&startEvent));
    HIP_CALL(hipEventCreate(&stopEvent));

    // Allocate pinned host memory closest to this GPU to capture timestamps (if enabled)
    int wallClockRate;
    long long* minStartCycle = nullptr;
    long long* maxStopCycle = nullptr;

    if (useWallClock) {
    #if defined(__NVCC__)
      wallClockRate = 1000000;
#else
      HIP_CALL(hipDeviceGetAttribute(&wallClockRate, hipDeviceAttributeWallClockRate, gpuIdx));
#endif
      if (Utils::AllocateMemory({MEM_CPU_CLOSEST, gpuIdx, myRank}, sizeof(int64_t), (void**)&minStartCycle) ||
          Utils::AllocateMemory({MEM_CPU_CLOSEST, gpuIdx, myRank}, sizeof(int64_t), (void**)&maxStopCycle)) {
        Utils::Print("[ERROR] Unable to allocate pinned host memory on rank %d closest to GPU %d\n", myRank, gpuIdx);
        return 1;
      }
    }

    // Allocate and initialize each GPU buffer
    MemDevice memDevice = {memType, gpuIdx, myRank};
    std::vector<void*> inputBuffers(numBuffers);
    for (int bufferIdx = 0; bufferIdx < numBuffers; bufferIdx++) {
      ErrResult err = AllocateMemory(memDevice, largestTotalBytesPerBuffer, &inputBuffers[bufferIdx]);
      if (err.errType != ERR_NONE) {
        Utils::Print("[ERROR] Error when allocating memory (%s)\n", err.errMsg.c_str());
        return 1;
      }
      FillPsuedoRandomData<<<32, 256, 0, stream>>>(largestTotalBytesPerBuffer / sizeof(uint32_t),
                                                   (uint32_t*)inputBuffers[bufferIdx], bufferIdx);
    }
    HIP_CALL(hipStreamSynchronize(stream));

    HbmBwResult bestResult = {};

    // Run sweep to find fastest result
    for (int numSubExec : numSesList) {
      dim3 gridDim(numSubExec, 1, 1);
      for (int blockSize : blockSizes) {
        if (!showDetails) {
          Utils::Print(".");
          fflush(stdout);
        }
        dim3 blockDim(blockSize, 1, 1);
        int launchBoundIdx = (blockSize + 255) / 256 - 1;
        for (int unroll : unrolls) {
          int unrollIdx = (int)log2(unroll);
          for (int elemByte : elemBytes) {
            int elemByteIdx = (int)log2(elemByte) - 2;
            size_t totalBytesPerStep = numSubExec * blockSize * unroll * elemByte;
            size_t numSteps = std::max((size_t)1, (numBytesAtLeast + totalBytesPerStep - 1) / totalBytesPerStep);
            size_t totalBytes = numSteps * totalBytesPerStep;

            for (int useNt = 0; useNt <= 1; useNt++) {
              if (!(temporalMask & (1<<useNt))) continue;
              auto kernel = HbmReadKernelTable[launchBoundIdx][unrollIdx][elemByteIdx][useNt];

              double minBw = std::numeric_limits<double>::max();
              double maxBw = std::numeric_limits<double>::lowest();
              double sumBw = 0.0;

              /* Run warmups for user-specified time */
              int currBufferIdx = 0;
              auto prewarmEnd = std::chrono::steady_clock::now() + std::chrono::milliseconds(prewarmMsec);
              do {
                kernel<<<gridDim, blockDim, 0, stream>>>(inputBuffers[currBufferIdx++], nullptr, numSteps, minStartCycle, maxStopCycle);
                HIP_CALL(hipStreamSynchronize(stream));
                if (currBufferIdx == numBuffers) currBufferIdx = 0;
              } while (std::chrono::steady_clock::now() < prewarmEnd);

              /* Run timed iterations */
              currBufferIdx = 0;
              for (int iteration = 0; iteration < numIterations; iteration++) {
                *minStartCycle = std::numeric_limits<long long int>::max();
                *maxStopCycle = 0;

#if defined(__NVCC__)
                if (!useWallClock) {
                  HIP_CALL(hipEventRecord(startEvent, stream));
                }
                kernel<<<gridDim, blockDim, 0, stream>>>(inputBuffers[currBufferIdx++], nullptr, numSteps, minStartCycle, maxStopCycle);
                if (!useWallClock) {
                  HIP_CALL(hipEventRecord(stopEvent, stream));
                }
#else
                hipExtLaunchKernelGGL(kernel, gridDim, blockDim, 0, stream, useWallClock ? nullptr : startEvent, useWallClock ? nullptr : stopEvent, 0,
                                      inputBuffers[currBufferIdx++], nullptr, numSteps, minStartCycle, maxStopCycle);
#endif
                HIP_CALL(hipStreamSynchronize(stream));
                if (currBufferIdx == numBuffers) currBufferIdx = 0;

                float elapsedMsec;
                if (useWallClock) {
                  elapsedMsec = (*maxStopCycle - *minStartCycle) / (double)wallClockRate;
                } else {
                  HIP_CALL(hipEventElapsedTime(&elapsedMsec, startEvent, stopEvent));
                }

                double bw = totalBytes / (elapsedMsec / 1000.0) / 1e9;

                if (showDetails > 1) {
                  Utils::Print("GPU %02d | %3d | %4d | %2d | %2d | %10lu | %5d | %8.3f\n",
                               gpuIdx, numSubExec, blockSize, unroll, elemByte, totalBytes, numSteps, bw);
                  fflush(stdout);
                }

                minBw = std::min(minBw, bw);
                maxBw = std::max(maxBw, bw);
                sumBw += bw;
              }

              double avgBw = sumBw / numIterations;

              if (showDetails) {
                Utils::Print("GPU %02d | %3d | %4d | %2d | %2d | %10lu | %5d | %8.3f | %8.3f | %8.3f\n",
                             gpuIdx, numSubExec, blockSize, unroll, elemByte, totalBytes, numSteps, maxBw, avgBw, minBw);
                fflush(stdout);
              }

              double bw[3] = {maxBw, avgBw, minBw};
              if (bw[criteria] > bestResult.bw[criteria]) {
                bestResult.rank       = myRank;
                bestResult.gpuIdx     = gpuIdx;
                bestResult.numSubExec = numSubExec;
                bestResult.blockSize  = blockSize;
                bestResult.unroll     = unroll;
                bestResult.elemByte   = elemByte;
                bestResult.bw[0]      = bw[0];
                bestResult.bw[1]      = bw[1];
                bestResult.bw[2]      = bw[2];
              }
            }
          }
        }
      }
    }

    localResults.push_back(bestResult);

    // Deallocate memory buffers
    for (int bufferIdx = 0; bufferIdx < numBuffers; bufferIdx++) {
      ErrResult err = DeallocateMemory(memType, inputBuffers[bufferIdx], largestTotalBytesPerBuffer);
      if (err.errType != ERR_NONE) {
        Utils::Print("[ERROR] Error when deallocating memory (%s)\n", err.errMsg.c_str());
        return 1;
      }
    }

    if (useWallClock) {
      if (Utils::DeallocateMemory(MEM_CPU_CLOSEST, minStartCycle, sizeof(int64_t)) ||
          Utils::DeallocateMemory(MEM_CPU_CLOSEST, maxStopCycle,  sizeof(int64_t))) {
        Utils::Print("[ERROR] Unable to deallocate pinned host memory on rank %d closest to GPU %d\n", myRank, gpuIdx);
        return 1;
      }
    }

    // Cleanup streams and events
    HIP_CALL(hipStreamDestroy(stream));
    HIP_CALL(hipEventDestroy(startEvent));
    HIP_CALL(hipEventDestroy(stopEvent));
  }
  if (!showDetails) {
    Utils::Print("\n"); fflush(stdout);
  }

  // Determine the total number of results
  std::vector<int> numGpusOnRank(numRanks);
  int totalGpus = 0;
  for (int rank = 0; rank < numRanks; rank++) {
    numGpusOnRank[rank] = (int)gpuIndices.size();
    TransferBench::System::Get().Broadcast(rank, sizeof(int), &numGpusOnRank[rank]);
    totalGpus += numGpusOnRank[rank];
  }

  int numRows = 1 + totalGpus;
  int numCols = 5 + (showExtra ? 4 : 0);
  int precision = 2;
  Utils::TableHelper table(numRows, numCols, precision);

  table.DrawRowBorder(0);
  table.DrawRowBorder(1);
  table.DrawColBorder(0);
  table.DrawColBorder(2);
  table.DrawColBorder(5);
  table.DrawColBorder(numCols);

  // Header row
  table.Set(0, 0, " Rank ");
  table.Set(0, 1, " GPU ");
  table.Set(0, 2, " MaxBw (GB/s) ");
  table.Set(0, 3, " AvgBw (GB/s) ");
  table.Set(0, 4, " MinBw (GB/s) ");
  if (showExtra) {
    table.Set(0, 5, " #SE ");
    table.Set(0, 6, " Blocksize ");
    table.Set(0, 7, " Unroll ");
    table.Set(0, 8, " EBytes ");
  }

  // Data rows
  int rowIdx = 1;
  for (int rank = 0; rank < numRanks; rank++) {
    for (int gpu = 0; gpu < numGpusOnRank[rank]; gpu++) {
      HbmBwResult result;
      if (rank == myRank) result = localResults[gpu];
      TransferBench::System::Get().Broadcast(rank, sizeof(result), &result);

      table.Set(rowIdx, 0, " %d "   , result.rank);
      table.Set(rowIdx, 1, " %d "   , result.gpuIdx);
      table.Set(rowIdx, 2, " %8.2f ", result.bw[0]);
      table.Set(rowIdx, 3, " %8.2f ", result.bw[1]);
      table.Set(rowIdx, 4, " %8.2f ", result.bw[2]);
      if (showExtra) {
        table.Set(rowIdx, 5, " %d ", result.numSubExec);
        table.Set(rowIdx, 6, " %d ", result.blockSize);
        table.Set(rowIdx, 7, " %d ", result.unroll);
        table.Set(rowIdx, 8, " %d ", result.elemByte);
      }
      rowIdx++;
    }
    table.DrawRowBorder(rowIdx);
  }
  table.PrintTable(outputToCsv, showBorders);

  return 0;
}

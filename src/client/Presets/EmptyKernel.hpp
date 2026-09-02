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
#include <chrono>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using namespace TransferBench;

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

__global__ void EmptyKernelDeviceKernel()
{
}

static void inline EmptyKernelRunBatch(int         const batchSize,
                                       int         const gridX,
                                       int         const blockX,
                                       hipStream_t const stream)
{
  for (int i = 0; i < batchSize; i++) {
    EmptyKernelDeviceKernel<<<gridX, blockX, 0, stream>>>();
  }
}

// Event timing for one batch: CUDA uses explicit record/stop around <<<>>>; HIP uses
// hipExtLaunchKernelGGL with start on the first launch and stop on the last
static void inline EmptyKernelRunBatchEvtOnly(int         const batchSize,
                                              int         const useExtLaunch,
                                              int         const gridX,
                                              int         const blockX,
                                              hipStream_t const stream,
                                              hipEvent_t  const startEvent,
                                              hipEvent_t  const stopEvent)
{
#if defined(__NVCC__)
  HIP_CALL(hipEventRecord(startEvent, stream));
  EmptyKernelRunBatch(batchSize, gridX, blockX, stream);
  HIP_CALL(hipEventRecord(stopEvent, stream));
#else
  if (batchSize == 0)
  {
     HIP_CALL(hipEventRecord(startEvent, stream));
     HIP_CALL(hipEventRecord(stopEvent, stream));
  } else {
    if (useExtLaunch) {
      for (int i = 0; i < batchSize; i++) {
        hipExtLaunchKernelGGL(EmptyKernelDeviceKernel, gridX, blockX, 0, stream,
                              (i ==           0) ? startEvent : NULL,
                              (i == batchSize-1) ?  stopEvent : NULL, 0);
      }
    } else {
      HIP_CALL(hipEventRecord(startEvent, stream));
      EmptyKernelRunBatch(batchSize, gridX, blockX, stream);
      HIP_CALL(hipEventRecord(stopEvent, stream));
    }
  }
#endif
}

int EmptyKernelPreset(EnvVars&          ev,
                      size_t      const /*numBytesPerTransfer*/,
                      std::string const presetName,
                      [[maybe_unused]] bool const bytesSpecified)
{
  if (!Utils::AllRanksHaveSameGpuCount()) {
    Utils::Print("[ERROR] %s preset requires all ranks to have the same number of GPUs\n", presetName.c_str());
    Utils::Print("[ERROR] Run ./TransferBench without any args to display topology information\n");
    return ERR_FATAL;
  }

  int const numRanks = GetNumRanks();
  int const myRank   = GetRank();

  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);
  int numSubExec      = TransferBench::GetNumSubExecutors({EXE_GPU_GFX, 0});

#if defined(__NVCC__)
  int defaultExtLaunch = 0;
#else
  int defaultExtLaunch = 1;
#endif

  // Collect env vars
  int numGpuDevices   = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int numIterations   = EnvVars::GetEnvVar("NUM_ITERATIONS",                5);
  int useExtLaunch    = EnvVars::GetEnvVar("USE_EXT_LAUNCH", defaultExtLaunch);
  std::vector<int> batchSizes = EnvVars::GetEnvVarArray("BATCHSIZES", {0, 1, 16, 256});
  std::vector<int> gridSizes  = EnvVars::GetEnvVarArray("GRIDSIZES",  {numSubExec});
  std::vector<int> blockSizes = EnvVars::GetEnvVarArray("BLOCKSIZES", {256});

  if (Utils::RankDoesOutput() && !ev.hideEnv) {
    if (!ev.outputToCsv) {
      Utils::Print("[EmptyKernel Related]\n");
    }
    std::string const batchStr = EnvVars::ToStr(batchSizes);
    std::string const gridStr  = EnvVars::ToStr(gridSizes);
    std::string const blockStr = EnvVars::ToStr(blockSizes);
    ev.Print("BATCHSIZES",       batchStr,          "Kernels per batch before hipStreamSynchronize");
    ev.Print("GRIDSIZES",        gridStr,           "Grid X dimension (# threadblocks per kernel launch, set to ',' to sweep all)");
    ev.Print("BLOCKSIZES",       blockStr,          "Thread-block width (blockDim.x)");
    ev.Print("NUM_GPU_DEVICES",  numGpuDevices,     "GPUs per rank to benchmark");
    ev.Print("NUM_ITERATIONS",   numIterations,     "Timed passes per cell (HIP and CPU measured separately each pass)");
    ev.Print("NUM_WARMUPS",      ev.numWarmups,     "Untimed warmup iterations");
    ev.Print("OUTPUT_TO_CSV",    ev.outputToCsv,    "CSV formatting for result table");
    ev.Print("SHOW_ITERATIONS",  ev.showIterations, "Show per-iteration EVT/CPU columns before MIN/AVG/MAX");
    ev.Print("SHOW_PERCENTILES", ev.showPercentiles.empty() ? 0 : 1, "%s",
             ev.showPercentiles.empty() ? "Disabled" : ev.GetStr(ev.showPercentiles).c_str());
    ev.Print("USE_EXT_LAUNCH",   useExtLaunch,      "Use hipExtLaunchKernelGGL to combine hipEvents with kernel launch");
    Utils::Print("\n");
  }

  // Check that input parameters are uniform across all ranks
  IS_UNIFORM(batchSizes,         "BATCHSIZES");
  IS_UNIFORM(gridSizes,          "GRIDSIZES");
  IS_UNIFORM(blockSizes,         "BLOCKSIZES");
  IS_UNIFORM(numGpuDevices,      "NUM_GPU_DEVICES");
  IS_UNIFORM(numIterations,      "NUM_ITERATIONS");
  IS_UNIFORM(ev.numWarmups,      "NUM_WARMUPS");
  IS_UNIFORM(ev.showIterations,  "SHOW_ITERATIONS");
  IS_UNIFORM(ev.showPercentiles, "SHOW_PERCENTILES");
  IS_UNIFORM(useExtLaunch,       "USE_EXT_LAUNCH");

  if (batchSizes.empty()) {
    Utils::Print("[ERROR] BATCHSIZES may not be empty\n");
    return ERR_FATAL;
  }
  for (int b : batchSizes) {
    if (b < 0) {
      Utils::Print("[ERROR] BATCHSIZES entries must be >= 0 (got %d)\n", b);
      return ERR_FATAL;
    }
  }

  if (gridSizes.empty()) {
    for (int i = 1; i <= 65535; i++)
      gridSizes.push_back(i);
  }

  for (int g : gridSizes) {
    if (g < 1 || g > 65535) {
      Utils::Print("[ERROR] GRIDSIZES entries must be in [1, 65535] (got %d)\n", g);
      return ERR_FATAL;
    }
  }

  if (blockSizes.empty()) {
    Utils::Print("[ERROR] BLOCKSIZES may not be empty\n");
    return ERR_FATAL;
  }
  for (int blockSize : blockSizes) {
    if (blockSize <= 0 || blockSize > 1024) {
      Utils::Print("[ERROR] BLOCKSIZES must be positive number up to 1024 (not %d)\n", blockSize);
      return ERR_FATAL;
    }
  }

  if (numGpuDevices <= 0 || numGpuDevices > numDetectedGpus) {
    Utils::Print("[ERROR] empty preset requires 1 <= NUM_GPU_DEVICES <= %d (got %d)\n", numDetectedGpus, numGpuDevices);
    return ERR_FATAL;
  }
  if (numIterations <= 0) {
    Utils::Print("[ERROR] empty preset requires NUM_ITERATIONS > 0 (got %d)\n", numIterations);
    return ERR_FATAL;
  }
  if (ev.numWarmups < 0) {
    Utils::Print("[ERROR] NUM_WARMUPS must be non-negative (got %d)\n", ev.numWarmups);
    return ERR_FATAL;
  }

#if defined(__NVCC__)
  if (useExtLaunch) {
    Utils::Print("[ERROR] USE_EXT_LAUNCH is not supported on NVIDIA hardware\n");
    return ERR_FATAL;
  }
#endif

  char const sep = ev.outputToCsv ? ',' : ' ';

  // Printer header row
  if (Utils::RankDoesOutput()) {
    if (!ev.outputToCsv) {
      Utils::Print("EmptyKernel preset: times in microseconds per kernel launch (averaged across batch size).\n\n");
      Utils::Print("Evt = hipEvent measured time\n");
      Utils::Print("Cpu = CPU wallclock measured time\n\n");
    }
    Utils::Print("%4s%c%8s%c%4s%c%4s%c%3s", "BatS", sep, "GrdS", sep, "BlkS", sep, "Rank", sep, "GPU");
    for (int mode = 0; mode < 2; mode++) {
      std::string modeStr = (mode == 0 ? "Evt" : "Cpu");
      if (ev.showIterations) {
        for (int it = 0; it < numIterations; it++) {
          char col[32];
          snprintf(col, sizeof(col), "%s%d", modeStr.c_str(), it);
          Utils::Print("%c%6s", sep, col);
        }
      }
      if (!ev.showPercentiles.empty()) {
        for (int pct : ev.showPercentiles) {
          char col[32];
          snprintf(col, sizeof(col), "%sp%d", modeStr.c_str(), pct);
          Utils::Print("%c%6s", sep, col);
        }
      }
      Utils::Print("%c%sMin%c%sAvg%c%sMax", sep, modeStr.c_str(), sep, modeStr.c_str(), sep, modeStr.c_str());
    }
    Utils::Print("\n");
    fflush(stdout);
  }

  // Create local stream/events per GPU
  std::vector<hipStream_t> stream(numGpuDevices);
  std::vector<hipEvent_t>  startEvent(numGpuDevices);
  std::vector<hipEvent_t>  stopEvent(numGpuDevices);
  for (int gpu = 0; gpu < numGpuDevices; gpu++) {
    HIP_CALL(hipSetDevice(gpu));
    HIP_CALL(hipStreamCreate(&stream[gpu]));
    HIP_CALL(hipEventCreate(&startEvent[gpu]));
    HIP_CALL(hipEventCreate(&stopEvent[gpu]));
  }

  // Build table and collect data on the fly
  for (auto const batchSize : batchSizes) {
    for (auto const gridSize : gridSizes) {
      for (auto const blockSize : blockSizes) {

        // All ranks collect for their GPUs in parallel
        std::vector<std::vector<double>> results(numGpuDevices, std::vector<double>(2*numIterations));
        for (int gpu = 0; gpu < numGpuDevices; gpu++) {
          HIP_CALL(hipSetDevice(gpu));
          for (int iteration = -ev.numWarmups; iteration < numIterations; iteration++) {

            EmptyKernelRunBatchEvtOnly(batchSize, useExtLaunch, gridSize, blockSize, stream[gpu], startEvent[gpu], stopEvent[gpu]);
            HIP_CALL(hipStreamSynchronize(stream[gpu]));
            float elapsedMsec = 0.0f;
            HIP_CALL(hipEventElapsedTime(&elapsedMsec, startEvent[gpu], stopEvent[gpu]));
            double const evtUsec = static_cast<double>(elapsedMsec) * 1000.0 / (batchSize == 0 ? 1 : batchSize);

            auto const t0 = std::chrono::steady_clock::now();
            EmptyKernelRunBatch(batchSize, gridSize, blockSize, stream[gpu]);
            HIP_CALL(hipStreamSynchronize(stream[gpu]));
            auto const t1 = std::chrono::steady_clock::now();
            double const cpuUsec = std::chrono::duration<double, std::micro>(t1 - t0).count() / (batchSize == 0 ? 1 : batchSize);

            if (iteration >= 0) {
              results[gpu][iteration]               = evtUsec;
              results[gpu][iteration+numIterations] = cpuUsec;
            }
          }
        }

        // Broadcast results to ranks that output
        std::vector<double> data(2 * numIterations);
        for (int rank = 0; rank < numRanks; rank++) {
          for (int gpu = 0; gpu < numGpuDevices; gpu++) {
            if (rank == myRank) {
              std::copy(results[gpu].begin(), results[gpu].end(), data.begin());
            }
            TransferBench::System::Get().Broadcast(rank, data.size()*sizeof(double), data.data());

            if (Utils::RankDoesOutput()) {
              Utils::Print("%4d%c%8d%c%4d%c%4d%c%3d", batchSize, sep, gridSize, sep, blockSize, sep, rank, sep, gpu);

              for (int mode = 0; mode < 2; mode++) {
                size_t baseOffset = mode * numIterations;
                double minTime = std::numeric_limits<double>::max(), sumTime = 0, maxTime = 0;
                for (int it = 0; it < numIterations; it++) {
                  double val = data[baseOffset + it];
                  minTime  = std::min(minTime, val);
                  sumTime += val;
                  maxTime  = std::max(maxTime, val);
                  if (ev.showIterations) {
                    Utils::Print("%c%6.2f", sep, val);
                  }
                }
                if (!ev.showPercentiles.empty()) {
                  std::vector<double> sorted(data.begin() + baseOffset,
                                             data.begin() + baseOffset + numIterations);
                  std::sort(sorted.begin(), sorted.end());
                  for (int pct : ev.showPercentiles) {
                    Utils::Print("%c%6.2f", sep, TransferBench::Utils::PercentileDurationMsecFromSorted(sorted, pct));
                  }
                }
                Utils::Print("%c%6.2f%c%6.2f%c%6.2f", sep, minTime, sep, sumTime / numIterations, sep, maxTime);
              }
              Utils::Print("\n");
              fflush(stdout);
            }
          }
        }
      }
    }
  }

  for (int gpu = 0; gpu < numGpuDevices; gpu++) {
    HIP_CALL(hipStreamDestroy(stream[gpu]));
    HIP_CALL(hipEventDestroy(startEvent[gpu]));
    HIP_CALL(hipEventDestroy(stopEvent[gpu]));
  }

  return ERR_NONE;
}

#if defined(__NVCC__)
#undef hipEvent_t
#undef hipEventCreate
#undef hipEventDestroy
#undef hipEventElapsedTime
#undef hipEventRecord
#undef hipSetDevice
#undef hipStream_t
#undef hipStreamCreate
#undef hipStreamDestroy
#undef hipStreamSynchronize
#endif

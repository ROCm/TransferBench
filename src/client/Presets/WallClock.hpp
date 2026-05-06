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

#include <limits>

__global__ void GetXccTimestamps(uint64_t* timestamps, volatile int* readyFlag)
{
  // Only first thread does any work
  if (threadIdx.x != 0) return;

  // Threadblocks in first "row" handle timestamps
  if (blockIdx.y == 0) {

    // Collect XCD for this
    int xccId;
    GetXccId(xccId);

    // All threadblocks wait for ready signal (no timeout — assumes signaling block is live)
    while (*readyFlag == 0);

    // Collect timestamp and save to memory; guard against unexpected XCC IDs
    auto w = GetTimestamp();
    if (xccId >= 0 && xccId < (int)gridDim.x) timestamps[xccId] = w;
  } else if (blockIdx.x == 0) {

    // Sleep for some number of cycles to ensure that other threadblocks are active
    auto w = GetTimestamp();
    while (GetTimestamp() - w < 10000);

    // Signal start to the other threadblocks
    *readyFlag = 1;
  }
}

#if defined(__NVCC__)
#define hipDeviceSynchronize                               cudaDeviceSynchronize
#define hipMemset                                          cudaMemset
#define hipSetDevice                                       cudaSetDevice
#endif

int WallClockPreset(EnvVars&          ev,
                    size_t      const numBytesPerTransfer,
                    std::string const presetName,
                    bool        const bytesSpecified)
{
  // Gather results and print
  int numRanks = GetNumRanks();
  int myRank   = GetRank();

  // Check for single homogenous group
  if (Utils::GetNumRankGroups() > 1) {
    Utils::Print("[ERROR] wallclock preset can only be run across ranks that are homogenous\n");
    Utils::Print("[ERROR] Run ./TransferBench without any args to display topology information\n");
    Utils::Print("[ERROR] TB_NIC_FILTER may also be used to limit NIC visibility\n");
    return ERR_FATAL;
  }

  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);
  int numGpuDevices   = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);

  // Print off env vars
  if (Utils::RankDoesOutput()) {
    if (!ev.hideEnv) {
      if (!ev.outputToCsv) printf("[WallClock Related]\n");
      ev.Print("NUM_GPU_DEVICES", numGpuDevices,     "Limit to using %d GPUs (per rank)", numGpuDevices);
      ev.Print("NUM_ITERATIONS" , ev.numIterations,  "Number of iterations");
      ev.Print("NUM_WARMUPS"    , ev.numWarmups,     "Number of warmup iterations");
      ev.Print("SHOW_ITERATIONS", ev.showIterations, "Showing per iteration details. Set to 2 to see raw wallclock values");
    }
  }

  // Check for env var consistency across ranks
  IS_UNIFORM(numGpuDevices,     "NUM_GPU_DEVICES");
  IS_UNIFORM(ev.numIterations,  "NUM_ITERATIONS");
  IS_UNIFORM(ev.numWarmups,     "NUM_WARMUPS");
  IS_UNIFORM(ev.showIterations, "SHOW_ITERATIONS");

  if (numGpuDevices <= 0 || numGpuDevices > numDetectedGpus) {
    Utils::Print("[ERROR] wallclock preset requires 1 <= NUM_GPU_DEVICES <= %d (got %d)\n", numDetectedGpus, numGpuDevices);
    return ERR_FATAL;
  }
  // Seconds-based (numIterations < 0) and infinite (numIterations == 0) modes are not supported
  if (ev.numIterations <= 0) {
    Utils::Print("[ERROR] wallclock preset requires NUM_ITERATIONS > 0 (seconds-based and infinite modes are not supported)\n");
    return ERR_FATAL;
  }

  // Collect local results
  // Query XCC count per device; all must match since results are sized from GPU 0
  int numXccs = GetNumExecutorSubIndices({EXE_GPU_GFX, 0});
  if (numXccs <= 0) {
    Utils::Print("[ERROR] wallclock preset requires at least one XCC (GPU 0 reports 0); topology data may be missing\n");
    return ERR_FATAL;
  }
  for (int deviceId = 1; deviceId < numGpuDevices; deviceId++) {
    int devXccs = GetNumExecutorSubIndices({EXE_GPU_GFX, deviceId});
    if (devXccs != numXccs) {
      Utils::Print("[ERROR] GPU device %d has %d XCCs but GPU 0 has %d; heterogeneous XCC counts are not supported\n",
                   deviceId, devXccs, numXccs);
      return ERR_FATAL;
    }
  }

  // Compute wall clock rate (based on GPU 0)
  int wallClockKhz;
#if defined(__NVCC__)
  wallClockKhz = 1000000;
#else
  HIP_CALL(hipDeviceGetAttribute(&wallClockKhz, hipDeviceAttributeWallClockRate, 0));
#endif
  if (wallClockKhz == 0) wallClockKhz = 100000;
  double uSecPerCycle = 1000.0 / wallClockKhz;

  Utils::Print("\nRunning %d iterations.  Detected wall clock rate of %dKhz = %.2f usec per cycle\n\n",
               ev.numIterations, wallClockKhz, uSecPerCycle);

  std::vector<std::vector<std::vector<uint64_t>>> results(numGpuDevices,
                                                          std::vector<std::vector<uint64_t>>(ev.numIterations,
                                                                                             std::vector<uint64_t>(numXccs, 0)));
  for (int deviceId = 0; deviceId < numGpuDevices; deviceId++) {
    HIP_CALL(hipSetDevice(deviceId));

    uint64_t* timestamps;
    int32_t* readyFlag;

    if (Utils::AllocateMemory({MEM_CPU_CLOSEST, deviceId}, numXccs * sizeof(uint64_t), (void**)&timestamps)) {
      Utils::Print("[ERROR] Unable to allocate pinned host memory for storing timestamps for GPU device %d on rank %d\n",
                   deviceId, myRank);
      return ERR_FATAL;
    }
    if (Utils::AllocateMemory({MEM_GPU, deviceId}, sizeof(int32_t), (void**)&readyFlag)) {
      Utils::Print("[ERROR] Unable to allocate readyFlag on GPU device %d on rank %d\n", deviceId, myRank);
      return ERR_FATAL;
    }

    for (int i = -ev.numWarmups; i < ev.numIterations; i++)
    {
      memset(timestamps, 0, numXccs * sizeof(uint64_t));
      HIP_CALL(hipMemset(readyFlag, 0, sizeof(*readyFlag)));
      HIP_CALL(hipDeviceSynchronize());
      GetXccTimestamps<<<dim3(numXccs,2,1), 1>>>(timestamps, readyFlag);
      HIP_CALL(hipDeviceSynchronize());
      if (i >= 0) {
        memcpy(results[deviceId][i].data(), timestamps, numXccs * sizeof(uint64_t));
      }
    }

    Utils::DeallocateMemory(MEM_CPU_CLOSEST, timestamps, numXccs * sizeof(uint64_t));
    Utils::DeallocateMemory(MEM_GPU, readyFlag, sizeof(int32_t));
  }

  // Prepare table of results
  int numRows   = 1 + numRanks * numGpuDevices * (ev.showIterations ? (ev.numIterations+1) : 1);
  int numCols   = 5 + (ev.showIterations ? numXccs : 0);
  Utils::TableHelper table(numRows, numCols);

  for (int i = 0; i < numCols; i++) {
    table.SetColAlignment(i, Utils::TableHelper::ALIGN_CENTER);
  }

  // Prepare header row
  int currRow = 0;
  int currCol = 0;
  table.Set(currRow, currCol++, "Rank");
  table.Set(currRow, currCol++, "GPU");
  table.Set(currRow, currCol++, "Iter");
  table.Set(currRow, currCol++, "Delta(cycles)");
  table.Set(currRow, currCol++, "Delta(usec)");
  if (ev.showIterations) {
    for (int i = 0; i < numXccs; i++) {
      table.Set(currRow, currCol++, " XCC %d ", i);
    }
  }
  currRow++;

  double minDelta = std::numeric_limits<double>::max();
  double maxDelta = std::numeric_limits<double>::lowest();

  for (int rank = 0; rank < numRanks; rank++) {
    table.DrawRowBorder(currRow);
    for (int deviceId = 0; deviceId < numGpuDevices; deviceId++) {
      size_t totalCycles = 0;
      std::vector<uint64_t> timestamps(numXccs, 0);

      for (int iteration = 0; iteration < ev.numIterations; iteration++) {
        if (rank == myRank) timestamps = results[deviceId][iteration];
        TransferBench::System::Get().Broadcast(rank, numXccs * sizeof(uint64_t), timestamps.data());

        const auto [min,max] = std::minmax_element(timestamps.begin(), timestamps.end());

        uint64_t cycles = (*max - *min);
        totalCycles += cycles;

        if (ev.showIterations) {
          currCol = 0;
          table.Set(currRow, currCol++, "%d", rank);
          table.Set(currRow, currCol++, "%d", deviceId);
          table.Set(currRow, currCol++, "%d", iteration);
          table.Set(currRow, currCol++, "%lu", cycles);
          table.Set(currRow, currCol++, "%.2f", cycles * uSecPerCycle);
          for (int i = 0; i < numXccs; i++) {
            table.Set(currRow, currCol++, "%lu", timestamps[i] - (ev.showIterations > 1 ? 0 : *min));
          }
          currRow++;
        }
      }

      double avgCycles = totalCycles * 1.0 / ev.numIterations;
      minDelta = std::min(minDelta, avgCycles);
      maxDelta = std::max(maxDelta, avgCycles);
      currCol = 0;
      table.Set(currRow, currCol++, "%d", rank);
      table.Set(currRow, currCol++, "%d", deviceId);
      table.Set(currRow, currCol++, "AVG");
      table.Set(currRow, currCol++, "%.2f", avgCycles);
      table.Set(currRow, currCol++, "%.2f", avgCycles * uSecPerCycle);
      currRow++;
    }
  }

  table.PrintTable(ev.outputToCsv, ev.showBorders);

  Utils::Print("\n");
  Utils::Print("Minimum Delta detected: %.2f cycles (%.2f usec)\n", minDelta, minDelta * uSecPerCycle);
  Utils::Print("Maximum Delta detected: %.2f cycles (%.2f usec)\n", maxDelta, maxDelta * uSecPerCycle);

  if (Utils::HasDuplicateHostname()) {
    Utils::Print("[WARN] It is recommended to run TransferBench with one rank per host to avoid potential aliasing of executors\n");
  }
  return ERR_NONE;
}

#if defined(__NVCC__)
#undef hipDeviceSynchronize
#undef hipMemset
#endif

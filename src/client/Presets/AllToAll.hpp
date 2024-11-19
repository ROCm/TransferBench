/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "EnvVars.hpp"

void AllToAllPreset(EnvVars&           ev,
                    size_t      const  numBytesPerTransfer,
                    std::string const  presetName)
{
  enum
  {
    A2A_COPY       = 0,
    A2A_READ_ONLY  = 1,
    A2A_WRITE_ONLY = 2
  };
  char a2aModeStr[3][20] = {"Copy", "Read-Only", "Write-Only"};

  // Force single-stream mode for all-to-all benchmark
  ev.useSingleStream = 1;

  // Force to gfx unroll 2 unless explicitly set
  ev.gfxUnroll      = EnvVars::GetEnvVar("GFX_UNROLL", 2);

  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  // Collect env vars for this preset
  int a2aDirect     = EnvVars::GetEnvVar("A2A_DIRECT"     , 1);
  int a2aLocal      = EnvVars::GetEnvVar("A2A_LOCAL"      , 0);
  int a2aMode       = EnvVars::GetEnvVar("A2A_MODE"       , 0);
  int numGpus       = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int numSubExecs   = EnvVars::GetEnvVar("NUM_SUB_EXEC"   , 8);
  int useDmaExec    = EnvVars::GetEnvVar("USE_DMA_EXEC"   , 0);
  int useFineGrain  = EnvVars::GetEnvVar("USE_FINE_GRAIN" , 1);
  int useRemoteRead = EnvVars::GetEnvVar("USE_REMOTE_READ", 0);

  // Print off environment variables
  ev.DisplayEnvVars();
  if (!ev.hideEnv) {
    if (!ev.outputToCsv) printf("[AllToAll Related]\n");
    ev.Print("A2A_DIRECT"     , a2aDirect    , a2aDirect ? "Only using direct links" : "Full all-to-all");
    ev.Print("A2A_LOCAL"      , a2aLocal     , "%s local transfers", a2aLocal ? "Include" : "Exclude");
    ev.Print("A2A_MODE"       , a2aMode      , a2aModeStr[a2aMode]);
    ev.Print("NUM_GPU_DEVICES", numGpus      , "Using %d GPUs", numGpus);
    ev.Print("NUM_SUB_EXEC"   , numSubExecs  , "Using %d subexecutors/CUs per Transfer", numSubExecs);
    ev.Print("USE_DMA_EXEC"   , useDmaExec   , "Using %s executor", useDmaExec ? "DMA" : "GFX");
    ev.Print("USE_FINE_GRAIN" , useFineGrain , "Using %s-grained memory", useFineGrain ? "fine" : "coarse");
    ev.Print("USE_REMOTE_READ", useRemoteRead, "Using %s as executor", useRemoteRead ? "DST" : "SRC");
    printf("\n");
  }

  // Validate env vars
  if (a2aMode < 0 || a2aMode > 2) {
    printf("[ERROR] a2aMode must be between 0 and 2\n");
    exit(1);
  }
  if (numGpus < 0 || numGpus > numDetectedGpus) {
    printf("[ERROR] Cannot use %d GPUs.  Detected %d GPUs\n", numGpus, numDetectedGpus);
    exit(1);
  }

  // Collect the number of GPU devices to use
  int const numSrcs = (a2aMode == A2A_WRITE_ONLY ? 0 : 1);
  int const numDsts = (a2aMode == A2A_READ_ONLY  ? 0 : 1);

  MemType memType = useFineGrain ? MEM_GPU_FINE : MEM_GPU;
  ExeType exeType = useDmaExec ? EXE_GPU_DMA : EXE_GPU_GFX;

  std::map<std::pair<int, int>, int> reIndex;
  std::vector<Transfer> transfers;
  for (int i = 0; i < numGpus; i++) {
    for (int j = 0; j < numGpus; j++) {

      // Check whether or not to execute this pair
      if (i == j) {
        if (!a2aLocal) continue;
      } else if (a2aDirect) {
#if !defined(__NVCC__)
        uint32_t linkType, hopCount;
        HIP_CALL(hipExtGetLinkTypeAndHopCount(i, j, &linkType, &hopCount));
        if (hopCount != 1) continue;
#endif
      }

      // Build Transfer and add it to list
      TransferBench::Transfer transfer;
      transfer.numBytes = numBytesPerTransfer;
      if (numSrcs) transfer.srcs.push_back({memType, i});
      if (numDsts) transfer.dsts.push_back({memType, j});
      transfer.exeDevice = {exeType, (useRemoteRead ? j : i)};
      transfer.exeSubIndex = -1;
      transfer.numSubExecs = numSubExecs;

      reIndex[std::make_pair(i,j)] = transfers.size();
      transfers.push_back(transfer);
    }
  }

  printf("GPU-GFX All-To-All benchmark:\n");
  printf("==========================\n");
  printf("- Copying %lu bytes between %s pairs of GPUs using %d CUs (%lu Transfers)\n",
         numBytesPerTransfer, a2aDirect ? "directly connected" : "all", numSubExecs, transfers.size());
  if (transfers.size() == 0) return;

  // Execute Transfers
  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  TransferBench::TestResults results;
  if (!TransferBench::RunTransfers(cfg, transfers, results)) {
    for (auto const& err : results.errResults)
      printf("%s\n", err.errMsg.c_str());
    exit(0);
  } else {
    PrintResults(ev, 1, transfers, results);
  }

  // Print results
  char separator = (ev.outputToCsv ? ',' : ' ');
  printf("\nSummary: [%lu bytes per Transfer]\n", numBytesPerTransfer);
  printf("==========================================================\n");
  printf("SRC\\DST ");
  for (int dst = 0; dst < numGpus; dst++)
    printf("%cGPU %02d    ", separator, dst);
  printf("   %cSTotal     %cActual\n", separator, separator);

  double totalBandwidthGpu = 0.0;
  double minExecutorBandwidth = std::numeric_limits<double>::max();
  double maxExecutorBandwidth = 0.0;
  std::vector<double> colTotalBandwidth(numGpus+1, 0.0);
  for (int src = 0; src < numGpus; src++) {
    double rowTotalBandwidth = 0;
    double executorBandwidth = 0;
    printf("GPU %02d", src);
    for (int dst = 0; dst < numGpus; dst++) {
      if (reIndex.count(std::make_pair(src, dst))) {
        int const transferIdx = reIndex[std::make_pair(src,dst)];
        TransferBench::TransferResult const& r = results.tfrResults[transferIdx];
        colTotalBandwidth[dst]  += r.avgBandwidthGbPerSec;
        rowTotalBandwidth       += r.avgBandwidthGbPerSec;
        totalBandwidthGpu       += r.avgBandwidthGbPerSec;
        executorBandwidth        = std::max(executorBandwidth,
                                            results.exeResults[transfers[transferIdx].exeDevice].avgBandwidthGbPerSec);
        printf("%c%8.3f  ", separator, r.avgBandwidthGbPerSec);
      } else {
        printf("%c%8s  ", separator, "N/A");
      }
    }
    printf("   %c%8.3f   %c%8.3f\n", separator, rowTotalBandwidth, separator, executorBandwidth);
    minExecutorBandwidth = std::min(minExecutorBandwidth, executorBandwidth);
    maxExecutorBandwidth = std::max(maxExecutorBandwidth, executorBandwidth);
    colTotalBandwidth[numGpus] += rowTotalBandwidth;
  }
  printf("\nRTotal");
  for (int dst = 0; dst < numGpus; dst++) {
    printf("%c%8.3f  ", separator, colTotalBandwidth[dst]);
  }
  printf("   %c%8.3f   %c%8.3f   %c%8.3f\n", separator, colTotalBandwidth[numGpus],
         separator, minExecutorBandwidth, separator, maxExecutorBandwidth);
  printf("\n");

  printf("Average   bandwidth (GPU Timed): %8.3f GB/s\n", totalBandwidthGpu / transfers.size());
  printf("Aggregate bandwidth (GPU Timed): %8.3f GB/s\n", totalBandwidthGpu);
  printf("Aggregate bandwidth (CPU Timed): %8.3f GB/s\n", results.avgTotalBandwidthGbPerSec);

  PrintErrors(results.errResults);
}

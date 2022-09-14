/*
Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.

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

// This program measures simultaneous copy performance across multiple GPUs
// on the same node
#include <numa.h>
#include <numaif.h>
#include <random>
#include <stack>
#include <thread>

#include "TransferBench.hpp"
#include "GetClosestNumaNode.hpp"
#include "Kernels.hpp"

int main(int argc, char **argv)
{
  // Check for NUMA library support
  if (numa_available() == -1)
  {
    printf("[ERROR] NUMA library not supported. Check to see if libnuma has been installed on this system\n");
    exit(1);
  }

  // Display usage instructions and detected topology
  if (argc <= 1)
  {
    int const outputToCsv = EnvVars::GetEnvVar("OUTPUT_TO_CSV", 0);
    if (!outputToCsv) DisplayUsage(argv[0]);
    DisplayTopology(outputToCsv);
    exit(0);
  }

  // Collect environment variables / display current run configuration
  EnvVars ev;

  // Determine number of bytes to run per Transfer
  size_t numBytesPerTransfer = argc > 2 ? atoll(argv[2]) : DEFAULT_BYTES_PER_TRANSFER;
  if (argc > 2)
  {
    // Adjust bytes if unit specified
    char units = argv[2][strlen(argv[2])-1];
    switch (units)
    {
    case 'K': case 'k': numBytesPerTransfer *= 1024; break;
    case 'M': case 'm': numBytesPerTransfer *= 1024*1024; break;
    case 'G': case 'g': numBytesPerTransfer *= 1024*1024*1024; break;
    }
  }
  if (numBytesPerTransfer % 4)
  {
    printf("[ERROR] numBytesPerTransfer (%lu) must be a multiple of 4\n", numBytesPerTransfer);
    exit(1);
  }

  // Check for preset tests
  // - Tests that sweep across possible sets of Transfers
  if (!strcmp(argv[1], "sweep") || !strcmp(argv[1], "rsweep"))
  {
    ev.configMode = CFG_SWEEP;
    RunSweepPreset(ev, numBytesPerTransfer, !strcmp(argv[1], "rsweep"));
    exit(0);
  }
  // - Tests that benchmark peer-to-peer performance
  else if (!strcmp(argv[1], "p2p") || !strcmp(argv[1], "p2p_rr") ||
           !strcmp(argv[1], "g2g") || !strcmp(argv[1], "g2g_rr"))
  {
    int numBlocksToUse = 0;
    if (argc > 3)
      numBlocksToUse = atoi(argv[3]);
    else
      HIP_CALL(hipDeviceGetAttribute(&numBlocksToUse, hipDeviceAttributeMultiprocessorCount, 0));

    // Perform either local read (+remote write) [EXE = SRC] or
    // remote read (+local write)                [EXE = DST]
    int readMode = (!strcmp(argv[1], "p2p_rr") || !strcmp(argv[1], "g2g_rr") ? 1 : 0);
    int skipCpu  = (!strcmp(argv[1], "g2g"   ) || !strcmp(argv[1], "g2g_rr") ? 1 : 0);

    // Execute peer to peer benchmark mode
    ev.configMode = CFG_P2P;
    RunPeerToPeerBenchmarks(ev, numBytesPerTransfer / sizeof(float), numBlocksToUse, readMode, skipCpu);
    exit(0);
  }

  // Check that Transfer configuration file can be opened
  ev.configMode = CFG_FILE;
  FILE* fp = fopen(argv[1], "r");
  if (!fp)
  {
    printf("[ERROR] Unable to open transfer configuration file: [%s]\n", argv[1]);
    exit(1);
  }

  // Print environment variables and CSV header
  ev.DisplayEnvVars();
  if (ev.outputToCsv)
  {
    printf("Test#,Transfer#,NumBytes,Src,Exe,Dst,CUs,BW(GB/s),Time(ms),"
           "ExeToSrcLinkType,ExeToDstLinkType,SrcAddr,DstAddr\n");
  }

  int testNum = 0;
  char line[2048];
  while(fgets(line, 2048, fp))
  {
    // Check if line is a comment to be echoed to output (starts with ##)
    if (!ev.outputToCsv && line[0] == '#' && line[1] == '#') printf("%s", line);

    // Parse set of parallel Transfers to execute
    std::vector<Transfer> transfers;
    ParseTransfers(line, ev.numCpuDevices, ev.numGpuDevices, transfers);
    if (transfers.empty()) continue;

    // If the number of bytes is specified, use it
    if (numBytesPerTransfer != 0)
    {
      size_t N = numBytesPerTransfer / sizeof(float);
      ExecuteTransfers(ev, ++testNum, N, transfers);
    }
    else
    {
      // Otherwise generate a range of values
      for (int N = 256; N <= (1<<27); N *= 2)
      {
        int delta = std::max(32, N / ev.samplingFactor);
        int curr = N;
        while (curr < N * 2)
        {
          ExecuteTransfers(ev, ++testNum, N, transfers);
          curr += delta;
        }
      }
    }
  }
  fclose(fp);

  return 0;
}

void ExecuteTransfers(EnvVars const& ev,
                      int const testNum,
                      size_t const N,
                      std::vector<Transfer>& transfers,
                      bool verbose)
{
  int const initOffset = ev.byteOffset / sizeof(float);

  // Map transfers by executor
  TransferMap transferMap;
  for (Transfer& transfer : transfers)
  {
    Executor executor(transfer.exeMemType, transfer.exeIndex);
    ExecutorInfo& executorInfo = transferMap[executor];
    executorInfo.transfers.push_back(&transfer);
  }

  // Loop over each executor and prepare GPU resources
  std::vector<Transfer*> transferList;
  for (auto& exeInfoPair : transferMap)
  {
    Executor const& executor = exeInfoPair.first;
    ExecutorInfo& exeInfo = exeInfoPair.second;
    exeInfo.totalTime = 0.0;
    exeInfo.totalBlocks = 0;

    // Loop over each transfer this executor is involved in
    for (Transfer* transfer : exeInfo.transfers)
    {
      // Get some aliases to transfer variables
      MemType const& exeMemType  = transfer->exeMemType;
      MemType const& srcMemType  = transfer->srcMemType;
      MemType const& dstMemType  = transfer->dstMemType;
      int     const& blocksToUse = transfer->numBlocksToUse;

      // Get potentially remapped device indices
      int const srcIndex = RemappedIndex(transfer->srcIndex, srcMemType);
      int const exeIndex = RemappedIndex(transfer->exeIndex, exeMemType);
      int const dstIndex = RemappedIndex(transfer->dstIndex, dstMemType);

      // Enable peer-to-peer access if necessary (can only be called once per unique pair)
      if (exeMemType == MEM_GPU)
      {
        // Ensure executing GPU can access source memory
        if ((srcMemType == MEM_GPU || srcMemType == MEM_GPU_FINE) && srcIndex != exeIndex)
          EnablePeerAccess(exeIndex, srcIndex);

        // Ensure executing GPU can access destination memory
        if ((dstMemType == MEM_GPU || dstMemType == MEM_GPU_FINE) && dstIndex != exeIndex)
          EnablePeerAccess(exeIndex, dstIndex);
      }

      // Allocate (maximum) source / destination memory based on type / device index
      transfer->numBytesToCopy = (transfer->numBytes ? transfer->numBytes : N * sizeof(float));
      AllocateMemory(srcMemType, srcIndex, transfer->numBytesToCopy + ev.byteOffset, (void**)&transfer->srcMem);
      AllocateMemory(dstMemType, dstIndex, transfer->numBytesToCopy + ev.byteOffset, (void**)&transfer->dstMem);

      transfer->blockParam.resize(exeMemType == MEM_CPU ? ev.numCpuPerTransfer : blocksToUse);
      exeInfo.totalBlocks += transfer->blockParam.size();
      transferList.push_back(transfer);
    }

    // Prepare per-threadblock parameters for GPU executors
    MemType const exeMemType = executor.first;
    int     const exeIndex   = RemappedIndex(executor.second, exeMemType);
    if (exeMemType == MEM_GPU)
    {
      // Allocate one contiguous chunk of GPU memory for threadblock parameters
      // This allows support for executing one transfer per stream, or all transfers in a single stream
      AllocateMemory(exeMemType, exeIndex, exeInfo.totalBlocks * sizeof(BlockParam),
                     (void**)&exeInfo.blockParamGpu);

      int const numTransfersToRun = ev.useSingleStream ? 1 : exeInfo.transfers.size();
      exeInfo.streams.resize(numTransfersToRun);
      exeInfo.startEvents.resize(numTransfersToRun);
      exeInfo.stopEvents.resize(numTransfersToRun);
      for (int i = 0; i < numTransfersToRun; ++i)
      {
        HIP_CALL(hipSetDevice(exeIndex));
        HIP_CALL(hipStreamCreate(&exeInfo.streams[i]));
        HIP_CALL(hipEventCreate(&exeInfo.startEvents[i]));
        HIP_CALL(hipEventCreate(&exeInfo.stopEvents[i]));
      }

      // Assign each transfer its portion of threadblock parameters
      int transferOffset = 0;
      for (int i = 0; i < exeInfo.transfers.size(); i++)
      {
        exeInfo.transfers[i]->blockParamGpuPtr = exeInfo.blockParamGpu + transferOffset;
        transferOffset += exeInfo.transfers[i]->blockParam.size();
      }
    }
  }

  if (verbose && !ev.outputToCsv) printf("Test %d:\n", testNum);

  // Prepare input memory and block parameters for current N
  for (auto& exeInfoPair : transferMap)
  {
    ExecutorInfo& exeInfo = exeInfoPair.second;
    exeInfo.totalBytes = 0;

    int transferOffset = 0;
    for (int i = 0; i < exeInfo.transfers.size(); ++i)
    {
      // Prepare subarrays each threadblock works on and fill src memory with patterned data
      Transfer* transfer = exeInfo.transfers[i];
      transfer->PrepareBlockParams(ev, transfer->numBytesToCopy / sizeof(float));
      exeInfo.totalBytes += transfer->numBytesToCopy;

      // Copy block parameters to GPU for GPU executors
      if (transfer->exeMemType == MEM_GPU)
      {
        HIP_CALL(hipMemcpy(&exeInfo.blockParamGpu[transferOffset],
                           transfer->blockParam.data(),
                           transfer->blockParam.size() * sizeof(BlockParam),
                           hipMemcpyHostToDevice));
        transferOffset += transfer->blockParam.size();
      }
    }
  }

  // Launch kernels (warmup iterations are not counted)
  double totalCpuTime = 0;
  size_t numTimedIterations = 0;
  std::stack<std::thread> threads;
  for (int iteration = -ev.numWarmups; ; iteration++)
  {
    if (ev.numIterations > 0 && iteration >= ev.numIterations) break;
    if (ev.numIterations < 0 && totalCpuTime > -ev.numIterations) break;

    // Pause before starting first timed iteration in interactive mode
    if (verbose && ev.useInteractive && iteration == 0)
    {
      printf("Hit <Enter> to continue: ");
      scanf("%*c");
      printf("\n");
    }

    // Start CPU timing for this iteration
    auto cpuStart = std::chrono::high_resolution_clock::now();

    // Execute all Transfers in parallel
    for (auto& exeInfoPair : transferMap)
    {
      ExecutorInfo& exeInfo = exeInfoPair.second;
      int const numTransfersToRun = (IsGpuType(exeInfoPair.first.first) && ev.useSingleStream) ?
        1 : exeInfo.transfers.size();
      for (int i = 0; i < numTransfersToRun; ++i)
        threads.push(std::thread(RunTransfer, std::ref(ev), iteration, std::ref(exeInfo), i));
    }

    // Wait for all threads to finish
    int const numTransfers = threads.size();
    for (int i = 0; i < numTransfers; i++)
    {
      threads.top().join();
      threads.pop();
    }

    // Stop CPU timing for this iteration
    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count();

    if (iteration >= 0)
    {
      ++numTimedIterations;
      totalCpuTime += deltaSec;
    }
  }

  // Pause for interactive mode
  if (verbose && ev.useInteractive)
  {
    printf("Transfers complete. Hit <Enter> to continue: ");
    scanf("%*c");
    printf("\n");
  }

  // Validate that each transfer has transferred correctly
  size_t totalBytesTransferred = 0;
  int const numTransfers = transferList.size();
  for (auto transfer : transferList)
  {
    CheckOrFill(MODE_CHECK, transfer->numBytesToCopy / sizeof(float), ev.useMemset, ev.useHipCall, ev.fillPattern, transfer->dstMem + initOffset);
    totalBytesTransferred += transfer->numBytesToCopy;
  }

  // Report timings
  totalCpuTime = totalCpuTime / (1.0 * numTimedIterations) * 1000;
  double totalBandwidthGbs = (totalBytesTransferred / 1.0E6) / totalCpuTime;
  double maxGpuTime = 0;

  if (ev.useSingleStream)
  {
    for (auto& exeInfoPair : transferMap)
    {
      ExecutorInfo  exeInfo    = exeInfoPair.second;
      MemType const exeMemType = exeInfoPair.first.first;
      int     const exeIndex   = exeInfoPair.first.second;

      // Compute total time for CPU executors
      if (!IsGpuType(exeMemType))
      {
        exeInfo.totalTime = 0;
        for (auto const& transfer : exeInfo.transfers)
          exeInfo.totalTime = std::max(exeInfo.totalTime, transfer->transferTime);
      }

      double exeDurationMsec = exeInfo.totalTime / (1.0 * numTimedIterations);
      double exeBandwidthGbs = (exeInfo.totalBytes / 1.0E9) / exeDurationMsec * 1000.0f;
      maxGpuTime = std::max(maxGpuTime, exeDurationMsec);

      if (verbose && !ev.outputToCsv)
      {
        printf(" Executor: %cPU %02d        (# Transfers %02lu)| %9.3f GB/s | %8.3f ms | %12lu bytes\n",
               MemTypeStr[exeMemType], exeIndex, exeInfo.transfers.size(), exeBandwidthGbs, exeDurationMsec, exeInfo.totalBytes);
      }

      int totalCUs = 0;
      for (auto const& transfer : exeInfo.transfers)
      {
        double transferDurationMsec = transfer->transferTime / (1.0 * numTimedIterations);
        double transferBandwidthGbs = (N * sizeof(float) / 1.0E9) / transferDurationMsec * 1000.0f;
        totalCUs += transfer->exeMemType == MEM_CPU ? ev.numCpuPerTransfer : transfer->numBlocksToUse;

        if (!verbose) continue;
        if (!ev.outputToCsv)
        {
          printf("                            Transfer  %02d | %9.3f GB/s | %8.3f ms | %12lu bytes | %c%02d -> %c%02d:(%03d) -> %c%02d\n",
                 transfer->transferIndex,
                 transferBandwidthGbs,
                 transferDurationMsec,
                 transfer->numBytesToCopy,
                 MemTypeStr[transfer->srcMemType], transfer->srcIndex,
                 MemTypeStr[transfer->exeMemType], transfer->exeIndex,
                 transfer->exeMemType == MEM_CPU ? ev.numCpuPerTransfer : transfer->numBlocksToUse,
                 MemTypeStr[transfer->dstMemType], transfer->dstIndex);

        }
        else
        {
          printf("%d,%d,%lu,%c%02d,%c%02d,%c%02d,%d,%.3f,%.3f,%s,%s,%p,%p\n",
                 testNum, transfer->transferIndex, transfer->numBytesToCopy,
                 MemTypeStr[transfer->srcMemType], transfer->srcIndex,
                 MemTypeStr[transfer->exeMemType], transfer->exeIndex,
                 MemTypeStr[transfer->dstMemType], transfer->dstIndex,
                 transfer->exeMemType == MEM_CPU ? ev.numCpuPerTransfer : transfer->numBlocksToUse,
                 transferBandwidthGbs, transferDurationMsec,
                 GetDesc(transfer->exeMemType, transfer->exeIndex, transfer->srcMemType, transfer->srcIndex).c_str(),
                 GetDesc(transfer->exeMemType, transfer->exeIndex, transfer->dstMemType, transfer->dstIndex).c_str(),
                 transfer->srcMem + initOffset, transfer->dstMem + initOffset);
        }
      }

      if (verbose && ev.outputToCsv)
      {
        printf("%d,ALL,%lu,ALL,%c%02d,ALL,%d,%.3f,%.3f,ALL,ALL,ALL,ALL\n",
               testNum, totalBytesTransferred,
               MemTypeStr[exeMemType], exeIndex, totalCUs,
               exeBandwidthGbs, exeDurationMsec);
      }
    }
  }
  else
  {
    for (auto const& transfer : transferList)
    {
      double transferDurationMsec = transfer->transferTime / (1.0 * numTimedIterations);
      double transferBandwidthGbs = (transfer->numBytesToCopy / 1.0E9) / transferDurationMsec * 1000.0f;
      maxGpuTime = std::max(maxGpuTime, transferDurationMsec);
      if (!verbose) continue;
      if (!ev.outputToCsv)
      {
        printf(" Transfer %02d: %c%02d -> [%cPU %02d:%03d] -> %c%02d | %9.3f GB/s | %8.3f ms | %12lu bytes | %-16s\n",
               transfer->transferIndex,
               MemTypeStr[transfer->srcMemType], transfer->srcIndex,
               MemTypeStr[transfer->exeMemType], transfer->exeIndex,
               transfer->exeMemType == MEM_CPU ? ev.numCpuPerTransfer : transfer->numBlocksToUse,
               MemTypeStr[transfer->dstMemType], transfer->dstIndex,
               transferBandwidthGbs, transferDurationMsec,
               transfer->numBytesToCopy,
               GetTransferDesc(*transfer).c_str());
      }
      else
      {
        printf("%d,%d,%lu,%c%02d,%c%02d,%c%02d,%d,%.3f,%.3f,%s,%s,%p,%p\n",
               testNum, transfer->transferIndex, transfer->numBytesToCopy,
               MemTypeStr[transfer->srcMemType], transfer->srcIndex,
               MemTypeStr[transfer->exeMemType], transfer->exeIndex,
               MemTypeStr[transfer->dstMemType], transfer->dstIndex,
               transfer->exeMemType == MEM_CPU ? ev.numCpuPerTransfer : transfer->numBlocksToUse,
               transferBandwidthGbs, transferDurationMsec,
               GetDesc(transfer->exeMemType, transfer->exeIndex, transfer->srcMemType, transfer->srcIndex).c_str(),
               GetDesc(transfer->exeMemType, transfer->exeIndex, transfer->dstMemType, transfer->dstIndex).c_str(),
               transfer->srcMem + initOffset, transfer->dstMem + initOffset);
      }
    }
  }

  // Display aggregate statistics
  if (verbose)
  {
    if (!ev.outputToCsv)
    {
      printf(" Aggregate Bandwidth (CPU timed)         | %9.3f GB/s | %8.3f ms | %12lu bytes | Overhead: %.3f ms\n",
             totalBandwidthGbs, totalCpuTime, totalBytesTransferred, totalCpuTime - maxGpuTime);
    }
    else
    {
      printf("%d,ALL,%lu,ALL,ALL,ALL,ALL,%.3f,%.3f,ALL,ALL,ALL,ALL\n",
             testNum, totalBytesTransferred, totalBandwidthGbs, totalCpuTime);
    }
  }

  // Release GPU memory
  for (auto exeInfoPair : transferMap)
  {
    ExecutorInfo& exeInfo = exeInfoPair.second;
    for (auto& transfer : exeInfo.transfers)
    {
      // Get some aliases to Transfer variables
      MemType const& exeMemType = transfer->exeMemType;
      MemType const& srcMemType = transfer->srcMemType;
      MemType const& dstMemType = transfer->dstMemType;

      // Allocate (maximum) source / destination memory based on type / device index
      DeallocateMemory(srcMemType, transfer->srcMem,  N * sizeof(float) + ev.byteOffset);
      DeallocateMemory(dstMemType, transfer->dstMem,  N * sizeof(float) + ev.byteOffset);
      transfer->blockParam.clear();
    }

    MemType const exeMemType = exeInfoPair.first.first;
    int     const exeIndex   = RemappedIndex(exeInfoPair.first.second, exeMemType);
    if (exeMemType == MEM_GPU)
    {
      DeallocateMemory(exeMemType, exeInfo.blockParamGpu);
      int const numTransfersToRun = ev.useSingleStream ? 1 : exeInfo.transfers.size();
      for (int i = 0; i < numTransfersToRun; ++i)
      {
        HIP_CALL(hipEventDestroy(exeInfo.startEvents[i]));
        HIP_CALL(hipEventDestroy(exeInfo.stopEvents[i]));
        HIP_CALL(hipStreamDestroy(exeInfo.streams[i]));
      }
    }
  }
}

void DisplayUsage(char const* cmdName)
{
  printf("TransferBench v%s\n", TB_VERSION);
  printf("========================================\n");

  if (numa_available() == -1)
  {
    printf("[ERROR] NUMA library not supported. Check to see if libnuma has been installed on this system\n");
    exit(1);
  }
  int numGpuDevices;
  HIP_CALL(hipGetDeviceCount(&numGpuDevices));
  int const numCpuDevices = numa_num_configured_nodes();

  printf("Usage: %s config <N>\n", cmdName);
  printf("  config: Either:\n");
  printf("          - Filename of configFile containing Transfers to execute (see example.cfg for format)\n");
  printf("          - Name of preset benchmark:\n");
  printf("              p2p{_rr} - All CPU/GPU pairs benchmark {with remote reads}\n");
  printf("              g2g{_rr} - All GPU/GPU pairs benchmark {with remote reads}\n");
  printf("              sweep    - Sweep across possible sets of Transfers\n");
  printf("              rsweep   - Randomly sweep across possible sets of Transfers\n");
  printf("            - 3rd optional argument will be used as # of CUs to use (uses all by default)\n");
  printf("  N     : (Optional) Number of bytes to copy per Transfer.\n");
  printf("          If not specified, defaults to %lu bytes. Must be a multiple of 4 bytes\n",
         DEFAULT_BYTES_PER_TRANSFER);
  printf("          If 0 is specified, a range of Ns will be benchmarked\n");
  printf("          May append a suffix ('K', 'M', 'G') for kilobytes / megabytes / gigabytes\n");
  printf("\n");

  EnvVars::DisplayUsage();
}

int RemappedIndex(int const origIdx, MemType const memType)
{
  static std::vector<int> remapping;

  // No need to re-map CPU devices
  if (memType == MEM_CPU) return origIdx;

  // Build remapping on first use
  if (remapping.empty())
  {
    int numGpuDevices;
    HIP_CALL(hipGetDeviceCount(&numGpuDevices));
    remapping.resize(numGpuDevices);

    int const usePcieIndexing = getenv("USE_PCIE_INDEX") ? atoi(getenv("USE_PCIE_INDEX")) : 0;
    if (!usePcieIndexing)
    {
      // For HIP-based indexing no remapping is necessary
      for (int i = 0; i < numGpuDevices; ++i)
        remapping[i] = i;
    }
    else
    {
      // Collect PCIe address for each GPU
      std::vector<std::pair<std::string, int>> mapping;
      char pciBusId[20];
      for (int i = 0; i < numGpuDevices; ++i)
      {
        HIP_CALL(hipDeviceGetPCIBusId(pciBusId, 20, i));
        mapping.push_back(std::make_pair(pciBusId, i));
      }
      // Sort GPUs by PCIe address then use that as mapping
      std::sort(mapping.begin(), mapping.end());
      for (int i = 0; i < numGpuDevices; ++i)
        remapping[i] = mapping[i].second;
    }
  }
  return remapping[origIdx];
}

void DisplayTopology(bool const outputToCsv)
{
  int numCpuDevices = numa_num_configured_nodes();
  int numGpuDevices;
  HIP_CALL(hipGetDeviceCount(&numGpuDevices));

  if (outputToCsv)
  {
    printf("NumCpus,%d\n", numCpuDevices);
    printf("NumGpus,%d\n", numGpuDevices);
  }
  else
  {
    printf("\nDetected topology: %d CPU NUMA node(s)   %d GPU device(s)\n", numa_num_configured_nodes(), numGpuDevices);
  }

  // Print out detected CPU topology
  if (outputToCsv)
  {
    printf("NUMA");
    for (int j = 0; j < numCpuDevices; j++)
      printf(",NUMA%02d", j);
    printf(",# CPUs,ClosestGPUs\n");
  }
  else
  {
    printf("        |");
    for (int j = 0; j < numCpuDevices; j++)
      printf("NUMA %02d |", j);
    printf(" # Cpus | Closest GPU(s)\n");
    for (int j = 0; j <= numCpuDevices; j++)
      printf("--------+");
    printf("--------+-------------\n");
  }

  for (int i = 0; i < numCpuDevices; i++)
  {
    printf("NUMA %02d%s", i, outputToCsv ? "," : " |");
    for (int j = 0; j < numCpuDevices; j++)
    {
      int numaDist = numa_distance(i,j);
      if (outputToCsv)
        printf("%d,", numaDist);
      else
        printf(" %6d |", numaDist);
    }

    int numCpus = 0;
    for (int j = 0; j < numa_num_configured_cpus(); j++)
      if (numa_node_of_cpu(j) == i) numCpus++;
    if (outputToCsv)
      printf("%d,", numCpus);
    else
      printf(" %6d | ", numCpus);

    bool isFirst = true;
    for (int j = 0; j < numGpuDevices; j++)
    {
      if (GetClosestNumaNode(RemappedIndex(j, MEM_GPU)) == i)
      {
        if (isFirst) isFirst = false;
        else printf(",");
        printf("%d", j);
      }
    }
    printf("\n");
  }
  printf("\n");

  // Print out detected GPU topology
  if (outputToCsv)
  {
    printf("GPU");
    for (int j = 0; j < numGpuDevices; j++)
      printf(",GPU %02d", j);
    printf(",PCIe Bus ID,ClosestNUMA\n");
  }
  else
  {
    printf("        |");
    for (int j = 0; j < numGpuDevices; j++)
      printf(" GPU %02d |", j);
    printf(" PCIe Bus ID  | Closest NUMA\n");
    for (int j = 0; j <= numGpuDevices; j++)
      printf("--------+");
    printf("--------------+-------------\n");
  }

  char pciBusId[20];

  for (int i = 0; i < numGpuDevices; i++)
  {
    printf("%sGPU %02d%s", outputToCsv ? "" : " ", i, outputToCsv ? "," : " |");
    for (int j = 0; j < numGpuDevices; j++)
    {
      if (i == j)
      {
        if (outputToCsv)
          printf("-,");
        else
          printf("    -   |");
      }
      else
      {
        uint32_t linkType, hopCount;
        HIP_CALL(hipExtGetLinkTypeAndHopCount(RemappedIndex(i, MEM_GPU),
                                              RemappedIndex(j, MEM_GPU),
                                              &linkType, &hopCount));
        printf("%s%s-%d%s",
               outputToCsv ? "" : " ",
               linkType == HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT ? "  HT" :
               linkType == HSA_AMD_LINK_INFO_TYPE_QPI            ? " QPI" :
               linkType == HSA_AMD_LINK_INFO_TYPE_PCIE           ? "PCIE" :
               linkType == HSA_AMD_LINK_INFO_TYPE_INFINBAND      ? "INFB" :
               linkType == HSA_AMD_LINK_INFO_TYPE_XGMI           ? "XGMI" : "????",
               hopCount, outputToCsv ? "," : " |");
      }
    }
    HIP_CALL(hipDeviceGetPCIBusId(pciBusId, 20, RemappedIndex(i, MEM_GPU)));
    if (outputToCsv)
      printf("%s,%d\n", pciBusId, GetClosestNumaNode(RemappedIndex(i, MEM_GPU)));
    else
      printf(" %11s |  %d  \n", pciBusId, GetClosestNumaNode(RemappedIndex(i, MEM_GPU)));
  }
}

void ParseMemType(std::string const& token, int const numCpus, int const numGpus, MemType* memType, int* memIndex)
{
  char typeChar;
  if (sscanf(token.c_str(), " %c %d", &typeChar, memIndex) != 2)
  {
    printf("[ERROR] Unable to parse memory type token %s - expecting either 'B,C,G or F' followed by an index\n",
           token.c_str());
    exit(1);
  }

  switch (typeChar)
  {
  case 'C': case 'c': case 'B': case 'b': case 'U': case 'u':
    *memType = (typeChar == 'C' || typeChar == 'c') ? MEM_CPU : ((typeChar == 'B' || typeChar == 'b') ? MEM_CPU_FINE : MEM_CPU_UNPINNED);
    if (*memIndex < 0 || *memIndex >= numCpus)
    {
      printf("[ERROR] CPU index must be between 0 and %d (instead of %d)\n", numCpus-1, *memIndex);
      exit(1);
    }
    break;
  case 'G': case 'g': case 'F': case 'f':
    *memType = (typeChar == 'G' || typeChar == 'g') ? MEM_GPU : MEM_GPU_FINE;
    if (*memIndex < 0 || *memIndex >= numGpus)
    {
      printf("[ERROR] GPU index must be between 0 and %d (instead of %d)\n", numGpus-1, *memIndex);
      exit(1);
    }
    break;
  default:
    printf("[ERROR] Unrecognized memory type %s.  Expecting either 'B','C','U','G' or 'F'\n", token.c_str());
    exit(1);
  }
}

// Helper function to parse a list of Transfer definitions
void ParseTransfers(char* line, int numCpus, int numGpus, std::vector<Transfer>& transfers)
{
  // Replace any round brackets or '->' with spaces,
  for (int i = 1; line[i]; i++)
    if (line[i] == '(' || line[i] == ')' || line[i] == '-' || line[i] == '>' ) line[i] = ' ';

  transfers.clear();

  int numTransfers = 0;
  std::istringstream iss(line);
  iss >> numTransfers;
  if (iss.fail()) return;

  std::string exeMem;
  std::string srcMem;
  std::string dstMem;

  // If numTransfers < 0, read quads (srcMem, exeMem, dstMem, #CUs)
  // otherwise read triples (srcMem, exeMem, dstMem)
  bool const perTransferCUs = (numTransfers < 0);
  numTransfers = abs(numTransfers);

  int numBlocksToUse;
  if (!perTransferCUs)
  {
    iss >> numBlocksToUse;
    if (numBlocksToUse <= 0 || iss.fail())
    {
      printf("Parsing error: Number of blocks to use (%d) must be greater than 0\n", numBlocksToUse);
      exit(1);
    }
  }

  for (int i = 0; i < numTransfers; i++)
  {
    Transfer transfer;
    transfer.transferIndex = i;
    transfer.numBytes = 0;
    transfer.numBytesToCopy = 0;
    iss >> srcMem >> exeMem >> dstMem;
    if (perTransferCUs) iss >> numBlocksToUse;
    if (iss.fail())
    {
      if (perTransferCUs)
        printf("Parsing error: Unable to read valid Transfer quadruple (possibly missing a SRC or EXE or DST or #CU)\n");
      else
        printf("Parsing error: Unable to read valid Transfer triplet (possibly missing a SRC or EXE or DST)\n");
      exit(1);
    }

    ParseMemType(srcMem, numCpus, numGpus, &transfer.srcMemType, &transfer.srcIndex);
    ParseMemType(exeMem, numCpus, numGpus, &transfer.exeMemType, &transfer.exeIndex);
    ParseMemType(dstMem, numCpus, numGpus, &transfer.dstMemType, &transfer.dstIndex);
    transfer.numBlocksToUse = numBlocksToUse;
    transfers.push_back(transfer);
  }
}

void EnablePeerAccess(int const deviceId, int const peerDeviceId)
{
  int canAccess;
  HIP_CALL(hipDeviceCanAccessPeer(&canAccess, deviceId, peerDeviceId));
  if (!canAccess)
  {
    printf("[ERROR] Unable to enable peer access from GPU devices %d to %d\n", peerDeviceId, deviceId);
    exit(1);
  }
  HIP_CALL(hipSetDevice(deviceId));
  hipError_t error = hipDeviceEnablePeerAccess(peerDeviceId, 0);
  if (error != hipSuccess && error != hipErrorPeerAccessAlreadyEnabled)
  {
    printf("[ERROR] Unable to enable peer to peer access from %d to %d (%s)\n",
           deviceId, peerDeviceId, hipGetErrorString(error));
    exit(1);
  }
}

void AllocateMemory(MemType memType, int devIndex, size_t numBytes, void** memPtr)
{
  if (numBytes == 0)
  {
    printf("[ERROR] Unable to allocate 0 bytes\n");
    exit(1);
  }

  if (IsCpuType(memType))
  {
    // Set numa policy prior to call to hipHostMalloc
    // NOTE: It may be possible that the actual configured numa nodes do not start at 0
    //       so remapping may be necessary
    // Find the 'deviceId'-th available NUMA node
    int numaIdx = 0;
    for (int i = 0; i <= devIndex; i++)
      while (!numa_bitmask_isbitset(numa_get_mems_allowed(), numaIdx))
        ++numaIdx;

    unsigned long nodemask = (1ULL << numaIdx);
    long retCode = set_mempolicy(MPOL_BIND, &nodemask, sizeof(nodemask)*8);
    if (retCode)
    {
      printf("[ERROR] Unable to set NUMA memory policy to bind to NUMA node %d\n", numaIdx);
      exit(1);
    }

    // Allocate host-pinned memory (should respect NUMA mem policy)

    if (memType == MEM_CPU_FINE)
    {
      HIP_CALL(hipHostMalloc((void **)memPtr, numBytes, hipHostMallocNumaUser));
    }
    else if (memType == MEM_CPU)
    {
      HIP_CALL(hipHostMalloc((void **)memPtr, numBytes, hipHostMallocNumaUser | hipHostMallocNonCoherent));
    }
    else if (memType == MEM_CPU_UNPINNED)
    {
      *memPtr = numa_alloc_onnode(numBytes, numaIdx);
    }

    // Check that the allocated pages are actually on the correct NUMA node
    memset(*memPtr, 0, numBytes);
    CheckPages((char*)*memPtr, numBytes, numaIdx);

    // Reset to default numa mem policy
    retCode = set_mempolicy(MPOL_DEFAULT, NULL, 8);
    if (retCode)
    {
      printf("[ERROR] Unable reset to default NUMA memory policy\n");
      exit(1);
    }
  }
  else if (memType == MEM_GPU)
  {
    // Allocate GPU memory on appropriate device
    HIP_CALL(hipSetDevice(devIndex));
    HIP_CALL(hipMalloc((void**)memPtr, numBytes));
  }
  else if (memType == MEM_GPU_FINE)
  {
    HIP_CALL(hipSetDevice(devIndex));
    HIP_CALL(hipExtMallocWithFlags((void**)memPtr, numBytes, hipDeviceMallocFinegrained));
  }
  else
  {
    printf("[ERROR] Unsupported memory type %d\n", memType);
    exit(1);
  }
}

void DeallocateMemory(MemType memType, void* memPtr, size_t const bytes)
{
  if (memType == MEM_CPU || memType == MEM_CPU_FINE)
  {
    HIP_CALL(hipHostFree(memPtr));
  }
  else if (memType == MEM_CPU_UNPINNED)
  {
    numa_free(memPtr, bytes);
  }
  else if (memType == MEM_GPU || memType == MEM_GPU_FINE)
  {
    HIP_CALL(hipFree(memPtr));
  }
}

void CheckPages(char* array, size_t numBytes, int targetId)
{
  unsigned long const pageSize = getpagesize();
  unsigned long const numPages = (numBytes + pageSize - 1) / pageSize;

  std::vector<void *> pages(numPages);
  std::vector<int> status(numPages);

  pages[0] = array;
  for (int i = 1; i < numPages; i++)
  {
    pages[i] = (char*)pages[i-1] + pageSize;
  }

  long const retCode = move_pages(0, numPages, pages.data(), NULL, status.data(), 0);
  if (retCode)
  {
    printf("[ERROR] Unable to collect page info\n");
    exit(1);
  }

  size_t mistakeCount = 0;
  for (int i = 0; i < numPages; i++)
  {
    if (status[i] < 0)
    {
      printf("[ERROR] Unexpected page status %d for page %d\n", status[i], i);
      exit(1);
    }
    if (status[i] != targetId) mistakeCount++;
  }
  if (mistakeCount > 0)
  {
    printf("[ERROR] %lu out of %lu pages for memory allocation were not on NUMA node %d\n", mistakeCount, numPages, targetId);
    printf("[ERROR] Ensure up-to-date ROCm is installed\n");
    exit(1);
  }
}

// Helper function to either fill a device pointer with pseudo-random data, or to check to see if it matches
void CheckOrFill(ModeType mode, int N, bool isMemset, bool isHipCall, std::vector<float>const& fillPattern, float* ptr)
{
  // Prepare reference resultx
  float* refBuffer = (float*)malloc(N * sizeof(float));
  if (isMemset)
  {
    if (isHipCall)
    {
      memset(refBuffer, 42, N * sizeof(float));
    }
    else
    {
      for (int i = 0; i < N; i++)
        refBuffer[i] = 1234.0f;
    }
  }
  else
  {
    // Fill with repeated pattern if specified
    size_t patternLen = fillPattern.size();
    if (patternLen > 0)
    {
      for (int i = 0; i < N; i++)
        refBuffer[i] = fillPattern[i % patternLen];
    }
    else // Otherwise fill with pseudo-random values
    {
      for (int i = 0; i < N; i++)
        refBuffer[i] = (i % 383 + 31);
    }
  }

  // Either fill the memory with the reference buffer, or compare against it
  if (mode == MODE_FILL)
  {
    HIP_CALL(hipMemcpy(ptr, refBuffer, N * sizeof(float), hipMemcpyDefault));
  }
  else if (mode == MODE_CHECK)
  {
    float* hostBuffer = (float*) malloc(N * sizeof(float));
    HIP_CALL(hipMemcpy(hostBuffer, ptr, N * sizeof(float), hipMemcpyDefault));
    for (int i = 0; i < N; i++)
    {
      if (refBuffer[i] != hostBuffer[i])
      {
        printf("[ERROR] Mismatch at element %d Ref: %f Actual: %f\n", i, refBuffer[i], hostBuffer[i]);
        exit(1);
      }
    }
    free(hostBuffer);
  }

  free(refBuffer);
}

std::string GetLinkTypeDesc(uint32_t linkType, uint32_t hopCount)
{
  char result[10];

  switch (linkType)
  {
  case HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT: sprintf(result, "  HT-%d", hopCount); break;
  case HSA_AMD_LINK_INFO_TYPE_QPI           : sprintf(result, " QPI-%d", hopCount); break;
  case HSA_AMD_LINK_INFO_TYPE_PCIE          : sprintf(result, "PCIE-%d", hopCount); break;
  case HSA_AMD_LINK_INFO_TYPE_INFINBAND     : sprintf(result, "INFB-%d", hopCount); break;
  case HSA_AMD_LINK_INFO_TYPE_XGMI          : sprintf(result, "XGMI-%d", hopCount); break;
  default: sprintf(result, "??????");
  }
  return result;
}

std::string GetDesc(MemType srcMemType, int srcIndex,
                    MemType dstMemType, int dstIndex)
{
  if (IsCpuType(srcMemType))
  {
    if (IsCpuType(dstMemType)) return (srcIndex == dstIndex) ? "LOCAL" : "NUMA";
    if (IsGpuType(dstMemType)) return "PCIE";
    goto error;
  }
  if (IsGpuType(srcMemType))
  {
    if (IsCpuType(dstMemType)) return "PCIE";
    if (IsGpuType(dstMemType))
    {
      if (srcIndex == dstIndex) return "LOCAL";
      else
      {
        uint32_t linkType, hopCount;
        HIP_CALL(hipExtGetLinkTypeAndHopCount(RemappedIndex(srcIndex, MEM_GPU),
                                              RemappedIndex(dstIndex, MEM_GPU),
                                              &linkType, &hopCount));
        return GetLinkTypeDesc(linkType, hopCount);
      }
    }
  }
error:
  printf("[ERROR] Unrecognized memory type\n");
  exit(1);
}

std::string GetTransferDesc(Transfer const& transfer)
{
  return GetDesc(transfer.srcMemType, transfer.srcIndex, transfer.exeMemType, transfer.exeIndex) + "-"
    + GetDesc(transfer.exeMemType, transfer.exeIndex, transfer.dstMemType, transfer.dstIndex);
}

void RunTransfer(EnvVars const& ev, int const iteration,
                 ExecutorInfo& exeInfo, int const transferIdx)
{
  Transfer* transfer = exeInfo.transfers[transferIdx];

  // GPU execution agent
  if (transfer->exeMemType == MEM_GPU)
  {
    // Switch to executing GPU
    int const exeIndex = RemappedIndex(transfer->exeIndex, MEM_GPU);
    HIP_CALL(hipSetDevice(exeIndex));

    hipStream_t& stream     = exeInfo.streams[transferIdx];
    hipEvent_t&  startEvent = exeInfo.startEvents[transferIdx];
    hipEvent_t&  stopEvent  = exeInfo.stopEvents[transferIdx];

    int const initOffset = ev.byteOffset / sizeof(float);

    if (ev.useHipCall)
    {
      // Record start event
      HIP_CALL(hipEventRecord(startEvent, stream));

      // Execute hipMemset / hipMemcpy
      if (ev.useMemset)
        HIP_CALL(hipMemsetAsync(transfer->dstMem + initOffset, 42, transfer->numBytesToCopy, stream));
      else
        HIP_CALL(hipMemcpyAsync(transfer->dstMem + initOffset,
                                transfer->srcMem + initOffset,
                                transfer->numBytesToCopy, hipMemcpyDefault,
                                stream));
      // Record stop event
      HIP_CALL(hipEventRecord(stopEvent, stream));
    }
    else
    {
      int const numBlocksToRun = ev.useSingleStream ? exeInfo.totalBlocks : transfer->numBlocksToUse;
      hipExtLaunchKernelGGL(ev.useMemset ? GpuMemsetKernel : GpuCopyKernel,
                            dim3(numBlocksToRun, 1, 1),
                            dim3(BLOCKSIZE, 1, 1),
                            ev.sharedMemBytes, stream,
                            startEvent, stopEvent,
                            0, transfer->blockParamGpuPtr);
    }

    // Synchronize per iteration, unless in single sync mode, in which case
    // synchronize during last warmup / last actual iteration
    HIP_CALL(hipStreamSynchronize(stream));

    if (iteration >= 0)
    {
      // Record GPU timing
      float gpuDeltaMsec;
      HIP_CALL(hipEventElapsedTime(&gpuDeltaMsec, startEvent, stopEvent));

      if (ev.useSingleStream)
      {
        for (Transfer* currTransfer : exeInfo.transfers)
        {
          long long minStartCycle = currTransfer->blockParamGpuPtr[0].startCycle;
          long long maxStopCycle  = currTransfer->blockParamGpuPtr[0].stopCycle;
          for (int i = 1; i < currTransfer->numBlocksToUse; i++)
          {
            minStartCycle = std::min(minStartCycle, currTransfer->blockParamGpuPtr[i].startCycle);
            maxStopCycle  = std::max(maxStopCycle,  currTransfer->blockParamGpuPtr[i].stopCycle);
          }
          int const wallClockRate = GetWallClockRate(exeIndex);
          double iterationTimeMs = (maxStopCycle - minStartCycle) / (double)(wallClockRate);
          currTransfer->transferTime += iterationTimeMs;
        }
        exeInfo.totalTime += gpuDeltaMsec;
      }
      else
      {
        transfer->transferTime += gpuDeltaMsec;
      }
    }
  }
  else if (transfer->exeMemType == MEM_CPU) // CPU execution agent
  {
    // Force this thread and all child threads onto correct NUMA node
    if (numa_run_on_node(transfer->exeIndex))
    {
      printf("[ERROR] Unable to set CPU to NUMA node %d\n", transfer->exeIndex);
      exit(1);
    }

    std::vector<std::thread> childThreads;

    auto cpuStart = std::chrono::high_resolution_clock::now();

    // Launch child-threads to perform memcopies
    for (int i = 0; i < ev.numCpuPerTransfer; i++)
      childThreads.push_back(std::thread(ev.useMemset ? CpuMemsetKernel : CpuCopyKernel, std::ref(transfer->blockParam[i])));

    // Wait for child-threads to finish
    for (int i = 0; i < ev.numCpuPerTransfer; i++)
      childThreads[i].join();

    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;

    // Record time if not a warmup iteration
    if (iteration >= 0)
      transfer->transferTime += (std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count() * 1000.0);
  }
}

void RunPeerToPeerBenchmarks(EnvVars const& ev, size_t N, int numBlocksToUse, int readMode, int skipCpu)
{
  // Collect the number of available CPUs/GPUs on this machine
  int numGpus;
  HIP_CALL(hipGetDeviceCount(&numGpus));
  int const numCpus = numa_num_configured_nodes();
  int const numDevices = numCpus + numGpus;

  // Enable peer to peer for each GPU
  for (int i = 0; i < numGpus; i++)
    for (int j = 0; j < numGpus; j++)
      if (i != j) EnablePeerAccess(i, j);

  if (!ev.outputToCsv)
  {
    printf("Performing copies in each direction of %lu bytes\n", N * sizeof(float));
    printf("Using %d threads per NUMA node for CPU copies\n", ev.numCpuPerTransfer);
    printf("Using %d CUs per transfer\n", numBlocksToUse);
  }
  else
  {
    printf("SRC,DST,Direction,ReadMode,BW(GB/s),Bytes\n");
  }

  // Perform unidirectional / bidirectional
  for (int isBidirectional = 0; isBidirectional <= 1; isBidirectional++)
  {
    // Print header
    if (!ev.outputToCsv)
    {
      printf("%sdirectional copy peak bandwidth GB/s [%s read / %s write]\n", isBidirectional ? "Bi" : "Uni",
             readMode == 0 ? "Local" : "Remote",
             readMode == 0 ? "Remote" : "Local");
      printf("%10s", "D/D");
      if (!skipCpu)
      {
        for (int i = 0; i < numCpus; i++)
          printf("%7s %02d", "CPU", i);
      }
      for (int i = 0; i < numGpus; i++)
        printf("%7s %02d", "GPU", i);
      printf("\n");
    }

    // Loop over all possible src/dst pairs
    for (int src = 0; src < numDevices; src++)
    {
      MemType const& srcMemType = (src < numCpus ? MEM_CPU : MEM_GPU);
      if (skipCpu && srcMemType == MEM_CPU) continue;
      int srcIndex = (srcMemType == MEM_CPU ? src : src - numCpus);
      if (!ev.outputToCsv)
        printf("%7s %02d", (srcMemType == MEM_CPU) ? "CPU" : "GPU", srcIndex);
      for (int dst = 0; dst < numDevices; dst++)
      {
        MemType const& dstMemType = (dst < numCpus ? MEM_CPU : MEM_GPU);
        if (skipCpu && dstMemType == MEM_CPU) continue;
        int dstIndex = (dstMemType == MEM_CPU ? dst : dst - numCpus);
        double bandwidth = GetPeakBandwidth(ev, N, isBidirectional, readMode, numBlocksToUse,
                                            srcMemType, srcIndex, dstMemType, dstIndex);
        if (!ev.outputToCsv)
        {
          if (bandwidth == 0)
            printf("%10s", "N/A");
          else
            printf("%10.2f", bandwidth);
        }
        else
        {
          printf("%s %02d,%s %02d,%s,%s,%.2f,%lu\n",
                 srcMemType == MEM_CPU ? "CPU" : "GPU",
                 srcIndex,
                 dstMemType == MEM_CPU ? "CPU" : "GPU",
                 dstIndex,
                 isBidirectional ? "bidirectional" : "unidirectional",
                 readMode == 0 ? "Local" : "Remote",
                 bandwidth,
                 N * sizeof(float));
        }
        fflush(stdout);
      }
      if (!ev.outputToCsv) printf("\n");
    }
    if (!ev.outputToCsv) printf("\n");
  }
}

double GetPeakBandwidth(EnvVars const& ev,
                        size_t  const  N,
                        int     const  isBidirectional,
                        int     const  readMode,
                        int     const  numBlocksToUse,
                        MemType const  srcMemType,
                        int     const  srcIndex,
                        MemType const  dstMemType,
                        int     const  dstIndex)
{
  // Skip bidirectional on same device
  if (isBidirectional && srcMemType == dstMemType && srcIndex == dstIndex) return 0.0f;

  int const initOffset = ev.byteOffset / sizeof(float);

  // Prepare Transfers
  std::vector<Transfer> transfers(2);
  transfers[0].srcMemType     = transfers[1].dstMemType     = srcMemType;
  transfers[0].dstMemType     = transfers[1].srcMemType     = dstMemType;
  transfers[0].srcIndex       = transfers[1].dstIndex       = RemappedIndex(srcIndex, srcMemType);
  transfers[0].dstIndex       = transfers[1].srcIndex       = RemappedIndex(dstIndex, dstMemType);
  transfers[0].numBytes       = transfers[1].numBytes       = N * sizeof(float);
  transfers[0].numBlocksToUse = transfers[1].numBlocksToUse = numBlocksToUse;

  // Either perform (local read + remote write), or (remote read + local write)
  transfers[0].exeMemType = (readMode == 0 ? srcMemType : dstMemType);
  transfers[1].exeMemType = (readMode == 0 ? dstMemType : srcMemType);
  transfers[0].exeIndex   = RemappedIndex((readMode == 0 ? srcIndex : dstIndex), transfers[0].exeMemType);
  transfers[1].exeIndex   = RemappedIndex((readMode == 0 ? dstIndex : srcIndex), transfers[1].exeMemType);

  transfers.resize(isBidirectional + 1);

  // Abort if executing on NUMA node with no CPUs
  for (int i = 0; i <= isBidirectional; i++)
  {
    if (transfers[i].exeMemType == MEM_CPU && ev.numCpusPerNuma[transfers[i].exeIndex] == 0)
      return 0;
  }

  ExecuteTransfers(ev, 0, N, transfers, false);

  // Collect aggregate bandwidth
  double totalBandwidth = 0;
  for (int i = 0; i <= isBidirectional; i++)
  {
    double transferDurationMsec = transfers[i].transferTime / (1.0 * ev.numIterations);
    double transferBandwidthGbs = (transfers[i].numBytesToCopy / 1.0E9) / transferDurationMsec * 1000.0f;
    totalBandwidth += transferBandwidthGbs;
  }

  return totalBandwidth;
}

void Transfer::PrepareBlockParams(EnvVars const& ev, size_t const N)
{
  int const initOffset = ev.byteOffset / sizeof(float);

  // Initialize source memory with patterned data
  CheckOrFill(MODE_FILL, N, ev.useMemset, ev.useHipCall, ev.fillPattern, this->srcMem + initOffset);

  // Each block needs to know src/dst pointers and how many elements to transfer
  // Figure out the sub-array each block does for this Transfer
  // - Partition N as evenly as possible, but try to keep blocks as multiples of BLOCK_BYTES bytes,
  //   except the very last one, for alignment reasons
  int const targetMultiple = ev.blockBytes / sizeof(float);
  int const maxNumBlocksToUse = std::min((N + targetMultiple - 1) / targetMultiple, this->blockParam.size());
  size_t assigned = 0;
  for (int j = 0; j < this->blockParam.size(); j++)
  {
    int    const blocksLeft = std::max(0, maxNumBlocksToUse - j);
    size_t const leftover   = N - assigned;
    size_t const roundedN   = (leftover + targetMultiple - 1) / targetMultiple;

    BlockParam& param = this->blockParam[j];
    param.N          = blocksLeft ? std::min(leftover, ((roundedN / blocksLeft) * targetMultiple)) : 0;
    param.src        = this->srcMem + assigned + initOffset;
    param.dst        = this->dstMem + assigned + initOffset;
    param.startCycle = 0;
    param.stopCycle  = 0;
    assigned += param.N;
  }

  this->transferTime = 0.0;
}

// NOTE: This is a stop-gap solution until HIP provides wallclock values
int GetWallClockRate(int deviceId)
{
  static std::vector<int> wallClockPerDeviceMhz;

  if (wallClockPerDeviceMhz.size() == 0)
  {
    int numGpuDevices;
    HIP_CALL(hipGetDeviceCount(&numGpuDevices));
    wallClockPerDeviceMhz.resize(numGpuDevices);

    hipDeviceProp_t prop;
    for (int i = 0; i < numGpuDevices; i++)
    {
      HIP_CALL(hipGetDeviceProperties(&prop, i));
      int value = 25000;
      switch (prop.gcnArch)
      {
      case 906: case 910: value = 25000; break;
      default:
        printf("Unrecognized GCN arch %d\n", prop.gcnArch);
      }
      wallClockPerDeviceMhz[i] = value;
    }
  }
  return wallClockPerDeviceMhz[deviceId];
}

void RunSweepPreset(EnvVars const& ev, size_t const numBytesPerTransfer, bool const isRandom)
{
  ev.DisplaySweepEnvVars();

  // Compute how many possible Transfers are permitted (unique SRC/EXE/DST triplets)
  std::vector<std::pair<MemType, int>> exeList;
  for (auto exe : ev.sweepExe)
  {
    MemType const exeMemType = CharToMemType(exe);
    if (IsGpuType(exeMemType))
    {
      for (int exeIndex = 0; exeIndex < ev.numGpuDevices; ++exeIndex)
        exeList.push_back(std::make_pair(exeMemType, exeIndex));
    }
    else
    {
      for (int exeIndex = 0; exeIndex < ev.numCpuDevices; ++exeIndex)
      {
        // Skip NUMA nodes that have no CPUs (e.g. CXL)
        if (ev.numCpusPerNuma[exeIndex] == 0) continue;
        exeList.push_back(std::make_pair(exeMemType, exeIndex));
      }
    }
  }
  int numExes = exeList.size();

  std::vector<std::pair<MemType, int>> srcList;
  for (auto src : ev.sweepSrc)
  {
    MemType const srcMemType = CharToMemType(src);
    int const numDevices = IsGpuType(srcMemType) ? ev.numGpuDevices : ev.numCpuDevices;

    for (int srcIndex = 0; srcIndex < numDevices; ++srcIndex)
      srcList.push_back(std::make_pair(srcMemType, srcIndex));
  }
  int numSrcs = srcList.size();


  std::vector<std::pair<MemType, int>> dstList;
  for (auto dst : ev.sweepDst)
  {
    MemType const dstMemType = CharToMemType(dst);
    int const numDevices = IsGpuType(dstMemType) ? ev.numGpuDevices : ev.numCpuDevices;

    for (int dstIndex = 0; dstIndex < numDevices; ++dstIndex)
      dstList.push_back(std::make_pair(dstMemType, dstIndex));
  }
  int numDsts = dstList.size();

  // Build array of possibilities, respecting any additional restrictions (e.g. XGMI hop count)
  struct TransferInfo
  {
    MemType srcMemType; int srcIndex;
    MemType exeMemType; int exeIndex;
    MemType dstMemType; int dstIndex;
  };

  // If either XGMI minimum is non-zero, or XGMI maximum is specified and non-zero then both links must be XGMI
  bool const useXgmiOnly = (ev.sweepXgmiMin > 0 || ev.sweepXgmiMax > 0);

  std::vector<TransferInfo> possibleTransfers;
  TransferInfo tinfo;
  for (int i = 0; i < numExes; ++i)
  {
    // Skip CPU executors if XGMI link must be used
    if (useXgmiOnly && !IsGpuType(exeList[i].first)) continue;
    tinfo.exeMemType = exeList[i].first;
    tinfo.exeIndex   = exeList[i].second;

    bool isXgmiSrc = false;
    int  numHopsSrc = 0;
    for (int j = 0; j < numSrcs; ++j)
    {
      if (IsGpuType(exeList[i].first) && IsGpuType(srcList[j].first))
      {
        if (exeList[i].second != srcList[j].second)
        {
          uint32_t exeToSrcLinkType, exeToSrcHopCount;
          HIP_CALL(hipExtGetLinkTypeAndHopCount(RemappedIndex(exeList[i].second, MEM_GPU),
                                                RemappedIndex(srcList[j].second, MEM_GPU),
                                                &exeToSrcLinkType,
                                                &exeToSrcHopCount));
          isXgmiSrc = (exeToSrcLinkType == HSA_AMD_LINK_INFO_TYPE_XGMI);
          if (isXgmiSrc) numHopsSrc = exeToSrcHopCount;
        }
        else
        {
          isXgmiSrc = true;
          numHopsSrc = 0;
        }

        // Skip this SRC if it is not XGMI but only XGMI links may be used
        if (useXgmiOnly && !isXgmiSrc) continue;

        // Skip this SRC if XGMI distance is already past limit
        if (ev.sweepXgmiMax >= 0 && isXgmiSrc && numHopsSrc > ev.sweepXgmiMax) continue;
      }
      else if (useXgmiOnly) continue;

      tinfo.srcMemType = srcList[j].first;
      tinfo.srcIndex   = srcList[j].second;

      bool isXgmiDst = false;
      int  numHopsDst = 0;
      for (int k = 0; k < numDsts; ++k)
      {
        if (IsGpuType(exeList[i].first) && IsGpuType(dstList[k].first))
        {
          if (exeList[i].second != dstList[k].second)
          {
            uint32_t exeToDstLinkType, exeToDstHopCount;
            HIP_CALL(hipExtGetLinkTypeAndHopCount(RemappedIndex(exeList[i].second, MEM_GPU),
                                                  RemappedIndex(dstList[k].second, MEM_GPU),
                                                  &exeToDstLinkType,
                                                  &exeToDstHopCount));
            isXgmiDst = (exeToDstLinkType == HSA_AMD_LINK_INFO_TYPE_XGMI);
            if (isXgmiDst) numHopsDst = exeToDstHopCount;
          }
          else
          {
            isXgmiDst = true;
            numHopsDst = 0;
          }
        }

        // Skip this DST if it is not XGMI but only XGMI links may be used
        if (useXgmiOnly && !isXgmiDst) continue;

        // Skip this DST if total XGMI distance (SRC + DST) is less than min limit
        if (ev.sweepXgmiMin > 0 && (numHopsSrc + numHopsDst < ev.sweepXgmiMin)) continue;

        // Skip this DST if total XGMI distance (SRC + DST) is greater than max limit
        if (ev.sweepXgmiMax >= 0 && (numHopsSrc + numHopsDst) > ev.sweepXgmiMax) continue;

        tinfo.dstMemType = dstList[k].first;
        tinfo.dstIndex   = dstList[k].second;

        possibleTransfers.push_back(tinfo);
      }
    }
  }

  int const numPossible = (int)possibleTransfers.size();
  int maxParallelTransfers = (ev.sweepMax == 0 ? numPossible : ev.sweepMax);

  if (ev.sweepMin > numPossible)
  {
    printf("No valid test configurations exist\n");
    return;
  }

  if (ev.outputToCsv)
  {
    printf("\nTest#,Transfer#,NumBytes,Src,Exe,Dst,CUs,BW(GB/s),Time(ms),"
           "ExeToSrcLinkType,ExeToDstLinkType,SrcAddr,DstAddr\n");
  }

  int numTestsRun = 0;
  int M = ev.sweepMin;
  std::uniform_int_distribution<int> randSize(1, numBytesPerTransfer / sizeof(float));
  std::uniform_int_distribution<int> distribution(ev.sweepMin, maxParallelTransfers);

  // Create bitmask of numPossible triplets, of which M will be chosen
  std::string bitmask(M, 1);  bitmask.resize(numPossible, 0);
  auto cpuStart = std::chrono::high_resolution_clock::now();
  while (1)
  {
    if (isRandom)
    {
      // Pick random number of simultaneous transfers to execute
      // NOTE: This currently skews distribution due to some #s having more possibilities than others
      M = distribution(*ev.generator);

      // Generate a random bitmask
      for (int i = 0; i < numPossible; i++)
        bitmask[i] = (i < M) ? 1 : 0;
      std::shuffle(bitmask.begin(), bitmask.end(), *ev.generator);
    }

    // Convert bitmask to list of Transfers
    std::vector<Transfer> transfers;
    for (int value = 0; value < numPossible; ++value)
    {
      if (bitmask[value])
      {
        // Convert integer value to (SRC->EXE->DST) triplet
        Transfer transfer;
        transfer.srcMemType     = possibleTransfers[value].srcMemType;
        transfer.srcIndex       = possibleTransfers[value].srcIndex;
        transfer.exeMemType     = possibleTransfers[value].exeMemType;
        transfer.exeIndex       = possibleTransfers[value].exeIndex;
        transfer.dstMemType     = possibleTransfers[value].dstMemType;
        transfer.dstIndex       = possibleTransfers[value].dstIndex;
        transfer.numBlocksToUse = IsGpuType(transfer.exeMemType) ? 4 : ev.numCpuPerTransfer;
        transfer.transferIndex  = transfers.size();
        transfer.numBytes       = ev.sweepRandBytes ? randSize(*ev.generator) * sizeof(float) : 0;
        transfers.push_back(transfer);
      }
    }

    ExecuteTransfers(ev, ++numTestsRun, numBytesPerTransfer / sizeof(float), transfers);

    // Check for test limit
    if (numTestsRun == ev.sweepTestLimit)
    {
      printf("Test limit reached\n");
      break;
    }

    // Check for time limit
    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double totalCpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count();
    if (ev.sweepTimeLimit && totalCpuTime > ev.sweepTimeLimit)
    {
      printf("Time limit exceeded\n");
      break;
    }

    // Increment bitmask if not random sweep
    if (!isRandom && !std::prev_permutation(bitmask.begin(), bitmask.end()))
    {
      M++;
      // Check for completion
      if (M > maxParallelTransfers)
      {
        printf("Sweep complete\n");
        break;
      }
      for (int i = 0; i < numPossible; i++)
        bitmask[i] = (i < M) ? 1 : 0;
    }
  }
}

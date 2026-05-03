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

#include <numeric>

int RingsPreset(EnvVars&          ev,
                size_t      const numBytesPerTransfer,
                std::string const presetName,
                bool        const bytesSpecified)
{
  // Check for homogeneous ranks
  if (Utils::GetNumRankGroups() > 1) {
    Utils::Print("[ERROR] rings preset can only be run across ranks that are homogeneous\n");
    Utils::Print("[ERROR] Run ./TransferBench without any args to display topology information\n");
    Utils::Print("[ERROR] TB_NIC_FILTER may also be used to limit NIC visibility\n");
    return ERR_FATAL;
  }

  // Check for pod support (if multi-node)
  int numRanks = TransferBench::GetNumRanks();
  if (numRanks > 1 && Utils::GetRankPerPodMap().size() != 1) {
    Utils::Print("[ERROR] Multi-rank runs must be within a single pod.  Set TB_FORCE_SINGLE_POD=1 to treat all ranks as a single pod.\n");
    return ERR_FATAL;
  }

  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  int memTypeIdx    = EnvVars::GetEnvVar("MEM_TYPE"       , 0);
  int numGpus       = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int numQueuePairs = EnvVars::GetEnvVar("NUM_QUEUE_PAIRS", 0);
  int numSubExecs   = EnvVars::GetEnvVar("NUM_SUB_EXEC"   , 8);
  int showDetails   = EnvVars::GetEnvVar("SHOW_DETAILS"   , 0);
  int useDmaExec    = EnvVars::GetEnvVar("USE_DMA_EXEC"   , 0);
  int useRemoteRead = EnvVars::GetEnvVar("USE_REMOTE_READ", 0);
  int stride        = EnvVars::GetEnvVar("STRIDE"         , 1);
  int ringSize      = EnvVars::GetEnvVar("RING_SIZE"      , numRanks * numGpus);


  if (numGpus <= 0 || numGpus > numDetectedGpus) {
    Utils::Print("[ERROR] Cannot use %d GPUs.  Detected %d GPUs\n", numGpus, numDetectedGpus);
    return ERR_FATAL;
  }
  if (ringSize <= 0) {
    Utils::Print("[ERROR] Ring size must be greater than 0\n");
    return ERR_FATAL;
  }
  if (numQueuePairs < 0) {
    Utils::Print("[ERROR] Num queue pairs must be non-negative\n");
    return ERR_FATAL;
  }

  int totalGpus = numRanks * numGpus;
  if (totalGpus % ringSize) {
    Utils::Print("[ERROR] Ring size %d must evenly divide the total number of GPUs %d\n", ringSize, totalGpus);
    return ERR_FATAL;
  }

  MemType memType           = Utils::GetGpuMemType(memTypeIdx);
  std::string devMemTypeStr = Utils::GetGpuMemTypeStr(memTypeIdx);

  if (Utils::RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
      if (!ev.outputToCsv) printf("[Rings Related]\n");
      ev.Print("MEM_TYPE"       , memTypeIdx   , "Using %s GPU memory (%s)", devMemTypeStr.c_str(), Utils::GetAllGpuMemTypeStr().c_str());
      ev.Print("NUM_GPU_DEVICES", numGpus      , "Using %d GPUs", numGpus);
      ev.Print("NUM_QUEUE_PAIRS", numQueuePairs, "Using %d queue pairs for NIC transfers", numQueuePairs);
      ev.Print("NUM_SUB_EXEC"   , numSubExecs  , "Using %d subexecutors/CUs per Transfer", numSubExecs);
      ev.Print("USE_DMA_EXEC"   , useDmaExec   , "Using %s executor", useDmaExec ? "DMA" : "GFX");
      ev.Print("USE_REMOTE_READ", useRemoteRead, "Using %s as executor", useRemoteRead ? "DST" : "SRC");
      ev.Print("STRIDE"         , stride       , "Reordering devices by taking %d steps", stride);
      ev.Print("RING_SIZE"      , ringSize     , "Building rings of size %d", ringSize);
      printf("\n");
    }
  }

  Utils::Print("GPU-%s Rings benchmark:\n", useDmaExec ? "DMA" : "GFX");
  Utils::Print("==============================\n");
  Utils::Print("[%lu bytes per Transfer] [%s:%d] [MemType:%s] [NIC QueuePairs:%d] [#Ranks:%d]\n",
               numBytesPerTransfer, useDmaExec ? "DMA" : "GFX", numSubExecs,
               devMemTypeStr.c_str(), numQueuePairs, numRanks);

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  ExeType exeType = useDmaExec ? EXE_GPU_DMA : EXE_GPU_GFX;

  int numRings = totalGpus / ringSize;
  Utils::Print("Running %d parallel ring(s) each of %d devices.  All numbers in GB/s:\n", numRings, ringSize);

  // Determine ordering of GPUs for the rings based on stride
  std::vector<int> indices(totalGpus);
  std::iota(indices.begin(), indices.end(), 0);
  Utils::StrideGenerate(indices, stride);

  // Establish memory devices for all GPUs
  // Assumes ranks are numbered 0..numRanks-1 and each has exactly numGpus devices
  std::vector<MemDevice> memDevices(totalGpus);
  for (int i = 0; i < totalGpus; i++) {
    memDevices[i] = {memType, indices[i] % numGpus, indices[i] / numGpus};
  }

  // Build list of Transfers
  std::vector<Transfer> transfers;
  for (int ringIdx = 0; ringIdx < numRings; ringIdx++) {
    int const ringBase = ringIdx * ringSize;

    // Build GFX or DMA transfers for this ring
    for (int i = 0; i < ringSize; i++) {
      Transfer t;
      int srcIdx    = ringBase + i;
      int dstIdx    = ringBase + (i + 1) % ringSize;
      int exeIdx    = useRemoteRead ? dstIdx : srcIdx;
      t.numBytes    = numBytesPerTransfer;
      t.srcs        = {memDevices[srcIdx]};
      t.dsts        = {memDevices[dstIdx]};
      t.exeDevice   = {exeType, memDevices[exeIdx].memIndex, memDevices[exeIdx].memRank};
      t.numSubExecs = numSubExecs;
      transfers.push_back(t);

      // Build NIC transfers between these GPUs as well if requested
      if (numQueuePairs > 0) {
        Transfer nicTransfer = t;
        nicTransfer.exeDevice   = {EXE_NIC_NEAREST, memDevices[exeIdx].memIndex, memDevices[exeIdx].memRank};
        nicTransfer.exeSubIndex = memDevices[useRemoteRead ? srcIdx : dstIdx].memIndex;
        nicTransfer.numSubExecs = numQueuePairs;
        transfers.push_back(nicTransfer);
      }
    }
  }

  TransferBench::TestResults results;
  if (!TransferBench::RunTransfers(cfg, transfers, results)) {
    for (auto const& err : results.errResults)
      Utils::Print("%s\n", err.errMsg.c_str());
    return ERR_FATAL;
  }
  if (showDetails) {
    Utils::PrintResults(ev, 1, transfers, results);
    Utils::Print("\n");
  }

  if (Utils::RankDoesOutput()) {

    // Limit the number of columns of output
    int maxColumns   = 24;
    int colsPerRing  = (numQueuePairs ? 3 : 2);
    int ringsPerPage = maxColumns / colsPerRing;
    int numPages     = (numRings + ringsPerPage - 1) / ringsPerPage;


    // Compute table size
    int numRows = numPages * (2 + ringSize + 4);
    int numCols = std::min(numRings, ringsPerPage) * colsPerRing;
    Utils::TableHelper table(numRows, numCols);

    std::vector<std::vector<double>> ringMin(numQueuePairs ? 2 : 1, std::vector<double>(numRings, std::numeric_limits<double>::max()));
    std::vector<std::vector<double>> ringSum(numQueuePairs ? 2 : 1, std::vector<double>(numRings, 0.0));
    std::vector<std::vector<double>> ringMax(numQueuePairs ? 2 : 1, std::vector<double>(numRings, 0.0));

    for (int pageIdx = 0; pageIdx < numPages; pageIdx++) {
      int headerRow = pageIdx * (2 + ringSize + 4);

      table.DrawRowBorder(headerRow);
      table.DrawRowBorder(headerRow+2);
      for (int r = 0; r < ringsPerPage; r++) {
        int ringIdx = pageIdx * ringsPerPage + r;
        if (ringIdx >= numRings) break;
        int currCol = colsPerRing * r;

        // Set header for ring
        table.DrawColBorder(currCol);
        table.DrawColBorder(currCol + colsPerRing);
        for (int i = 0; i < colsPerRing; i++)
          table.Set(headerRow, currCol+i, "Ring%02d", ringIdx);
        table.Set(headerRow+1, currCol, "Device");
        table.Set(headerRow+1, currCol+1, "%s BW", useDmaExec ? "DMA" : "GFX");
        if (numQueuePairs) {
          table.Set(headerRow+1, currCol+2, "NIC BW");
        }

        // Fill results for ring
        int baseRow = headerRow + 2;
        table.DrawRowBorder(baseRow);
        for (int i = 0; i < ringSize; i++) {
          int tfrIdx = (ringIdx * ringSize + i) * (colsPerRing - 1);
          Transfer const& t = transfers[tfrIdx];
          if (numRanks > 1) {
            table.Set(baseRow + i, currCol, "R%02d:%d", t.srcs[0].memRank, t.srcs[0].memIndex);
          } else {
            table.Set(baseRow + i, currCol, "%d", t.srcs[0].memIndex);
          }

          for (int j = 0; j < colsPerRing - 1; j++) {
            double bw = results.tfrResults[tfrIdx + j].avgBandwidthGbPerSec;
            table.Set(baseRow + i, currCol+1+j, "%7.2f", bw);
            ringMin[j][ringIdx] = std::min(ringMin[j][ringIdx], bw);
            ringSum[j][ringIdx] += bw;
            ringMax[j][ringIdx] = std::max(ringMax[j][ringIdx], bw);
          }
        }
        int statRow = baseRow + ringSize;
        table.DrawRowBorder(statRow);
        table.Set(statRow  , currCol, "MIN");
        table.Set(statRow+1, currCol, "AVG");
        table.Set(statRow+2, currCol, "MAX");
        table.Set(statRow+3, currCol, "SUM");

        for (int j = 0; j < colsPerRing - 1; j++) {
          table.Set(statRow  , currCol+1+j, "%7.2f", ringMin[j][ringIdx]);
          table.Set(statRow+1, currCol+1+j, "%7.2f", ringSum[j][ringIdx] / ringSize);
          table.Set(statRow+2, currCol+1+j, "%7.2f", ringMax[j][ringIdx]);
          table.Set(statRow+3, currCol+1+j, "%7.2f", ringSum[j][ringIdx]);
        }
        table.DrawRowBorder(statRow+3);
        table.DrawRowBorder(statRow+4);
      }
    }
    table.PrintTable(ev.outputToCsv, ev.showBorders);
    Utils::Print("Aggregate bandwidth (CPU Timed): %8.3f GB/s\n", results.avgTotalBandwidthGbPerSec);

    if (Utils::HasDuplicateHostname())
      Utils::Print("[WARN] It is recommended to run TransferBench with one rank per host to avoid potential aliasing of executors\n");
  }

  return ERR_NONE;
}

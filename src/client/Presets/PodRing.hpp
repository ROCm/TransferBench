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

int PodRingPreset(EnvVars&          ev,
                  size_t      const numBytesPerTransfer,
                  std::string const presetName,
                  bool        const bytesSpecified)
{
  // Check for homogeneous ranks
  if (Utils::GetNumRankGroups() > 1) {
    Utils::Print("[ERROR] PodRing preset can only be run across ranks that are homogeneous\n");
    Utils::Print("[ERROR] Run ./TransferBench without any args to display topology information\n");
    Utils::Print("[ERROR] TB_NIC_FILTER may also be used to limit NIC visibility\n");
    return 1;
  }

  // Check for pod support (if multi-node)
  int numRanks = TransferBench::GetNumRanks();
  if (numRanks > 1 && Utils::GetRankPerPodMap().empty()) {
    Utils::Print("[ERROR] No pods detected. Set TB_FORCE_SINGLE_POD=1 to treat all ranks as a single pod.\n");
    return 1;
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
  int groupSize     = EnvVars::GetEnvVar("GROUP_SIZE"     , numRanks * numGpus);

  if (numGpus <= 0 || numGpus > numDetectedGpus) {
    Utils::Print("[ERROR] Cannot use %d GPUs.  Detected %d GPUs\n", numGpus, numDetectedGpus);
    return 1;
  }
  if (groupSize <= 0) {
    Utils::Print("[ERROR] Group size must be greater than 0\n");
    return 1;
  }
  if (numRanks * numGpus % groupSize) {
    Utils::Print("[ERROR] Group size %d cannot evenly divide %d total devices from %d ranks.\n",
                 groupSize, numRanks * numGpus, numRanks);
    return 1;
  }

  int numNics = TransferBench::GetNumExecutors(EXE_NIC, 0);
  bool nicDifference = false;
  for (int rank = 0; rank < numRanks; rank++) {
    if (numGpus > TransferBench::GetNumExecutors(EXE_GPU_GFX, rank)) {
      Utils::Print("[ERROR] PodRing preset requires each rank to have the same number of GPUs\n");
      return 1;
    }
    if (numQueuePairs > 0 && numNics != TransferBench::GetNumExecutors(EXE_NIC, rank))
      nicDifference = true;
  }
  if (nicDifference)
    Utils::Print("[WARN] Not all ranks have the same number of NICs\n");

  MemType memType = Utils::GetGpuMemType(memTypeIdx);
  std::string devMemTypeStr = Utils::GetGpuMemTypeStr(memTypeIdx);

  if (Utils::RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
      if (!ev.outputToCsv) printf("[PodRing Related]\n");
      ev.Print("MEM_TYPE"       , memTypeIdx   , "Using %s GPU memory (%s)", devMemTypeStr.c_str(), Utils::GetAllGpuMemTypeStr().c_str());
      ev.Print("NUM_GPU_DEVICES", numGpus      , "Using %d GPUs", numGpus);
      ev.Print("NUM_QUEUE_PAIRS", numQueuePairs, "Using %d queue pairs for NIC transfers", numQueuePairs);
      ev.Print("NUM_SUB_EXEC"   , numSubExecs  , "Using %d subexecutors/CUs per Transfer", numSubExecs);
      ev.Print("USE_DMA_EXEC"   , useDmaExec   , "Using %s executor", useDmaExec ? "DMA" : "GFX");
      ev.Print("USE_REMOTE_READ", useRemoteRead, "Using %s as executor", useRemoteRead ? "DST" : "SRC");
      ev.Print("STRIDE"         , stride       , "Reordering devices by taking %d steps", stride);
      ev.Print("GROUP_SIZE"     , groupSize    , "Dividing all devices into ring groups of %d", groupSize);
      printf("\n");
    }
  }

  Utils::Print("GPU-%s IntraPod Ring benchmark:\n", useDmaExec ? "DMA" : "GFX");
  Utils::Print("==============================\n");
  Utils::Print("[%lu bytes per Transfer] [%s:%d] [MemType:%s] [NIC QueuePairs:%d] [#Ranks:%d]\n",
               numBytesPerTransfer, useDmaExec ? "DMA" : "GFX", numSubExecs,
               devMemTypeStr.c_str(), numQueuePairs, numRanks);

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  ExeType exeType = useDmaExec ? EXE_GPU_DMA : EXE_GPU_GFX;

  int n = numRanks * numGpus;
  int numGroups = n / groupSize;

  std::vector<int> indices(n);
  for (int k = 0; k < n; k++) indices[k] = k;
  Utils::StrideGenerate(indices, stride);

  std::vector<MemDevice> devices(n);
  for (int i = 0; i < n; i++) {
    int const globalIdx = indices[i];
    int const rank      = globalIdx / numGpus;
    int const devIdx    = globalIdx % numGpus;
    devices[i] = {memType, devIdx, rank};
  }

  Utils::Print("%d ring(s) of %d devices:\n", numGroups, groupSize);
  for (int group = 0; group < numGroups; group++) {
    int const groupBase = group * groupSize;
    Utils::Print("  Ring %d: ", group);
    for (int i = 0; i < groupSize; i++) {
      Utils::Print("R%d:G%d -> ", devices[groupBase + i].memRank, devices[groupBase + i].memIndex);
    }
    Utils::Print("R%d:G%d\n", devices[groupBase].memRank, devices[groupBase].memIndex);
  }
  Utils::Print("\n");

  for (int group = 0; group < numGroups; group++) {
    int const groupBase = group * groupSize;
    std::vector<Transfer> transfers;

    for (int i = 0; i < groupSize; i++) {
      int srcIdx = groupBase + i;
      int dstIdx = groupBase + (i + 1) % groupSize;

      TransferBench::Transfer transfer;
      transfer.numBytes = numBytesPerTransfer;
      transfer.srcs.push_back(devices[srcIdx]);
      transfer.dsts.push_back(devices[dstIdx]);
      transfer.exeDevice = {exeType,
                           (int32_t)(useRemoteRead ? devices[dstIdx].memIndex : devices[srcIdx].memIndex),
                           (int32_t)(useRemoteRead ? devices[dstIdx].memRank  : devices[srcIdx].memRank)};
      transfer.exeSubIndex = -1;
      transfer.numSubExecs = numSubExecs;
      transfers.push_back(transfer);

      if (numQueuePairs > 0) {
        TransferBench::Transfer nicTransfer;
        nicTransfer.numBytes = numBytesPerTransfer;
        nicTransfer.srcs.push_back(devices[srcIdx]);
        nicTransfer.dsts.push_back(devices[dstIdx]);
        nicTransfer.exeDevice = {TransferBench::EXE_NIC_NEAREST,
                                (int32_t)devices[srcIdx].memIndex, (int32_t)devices[srcIdx].memRank};
        nicTransfer.exeSubIndex = devices[dstIdx].memIndex;
        nicTransfer.numSubExecs = numQueuePairs;
        transfers.push_back(nicTransfer);
      }
    }

    TransferBench::TestResults results;
    if (!TransferBench::RunTransfers(cfg, transfers, results)) {
      for (auto const& err : results.errResults)
        Utils::Print("%s\n", err.errMsg.c_str());
      return 1;
    }
    if (showDetails) {
      Utils::PrintResults(ev, 1, transfers, results);
      Utils::Print("\n");
    }

    if (Utils::RankDoesOutput()) {
      Utils::Print("\n--- Pod Ring Group %d ---\n", group);

      int const numHops   = groupSize;
      int const numRows   = 2 + numHops + 3;
      int const numCols   = 6;
      int const precision = 2;
      Utils::TableHelper table(numRows, numCols, precision);

      table.DrawRowBorder(0);
      table.DrawColBorder(0);
      table.DrawColBorder(numCols);
      table.DrawRowBorder(numRows);

      table.Set(0, 0, " Src ");
      table.Set(0, 1, " Src ");
      table.Set(0, 2, " Dst ");
      table.Set(0, 3, " Dst ");
      table.Set(0, 4, " GFX BW ");
      table.Set(1, 0, " Rank ");
      table.Set(1, 1, " GPU ");
      table.Set(1, 2, " Rank ");
      table.Set(1, 3, " GPU ");
      table.Set(1, 4, " (GB/s) ");
      table.DrawColBorder(2);
      table.DrawColBorder(4);

      if (numQueuePairs > 0) {
        table.Set(0, 5, " NIC BW ");
        table.Set(1, 5, " (GB/s) ");
      } else {
        table.Set(0, 5, " ");
        table.Set(1, 5, " ");
      }

      table.DrawRowBorder(2);

      double gfxMin = std::numeric_limits<double>::max();
      double gfxAvg = 0.0;
      double gfxMax = std::numeric_limits<double>::lowest();
      double nicMin = std::numeric_limits<double>::max();
      double nicAvg = 0.0;
      double nicMax = std::numeric_limits<double>::lowest();

      int tfrIdx = 0;
      for (int i = 0; i < numHops; i++) {
        int srcIdx = groupBase + i;
        int dstIdx = groupBase + (i + 1) % groupSize;
        int row    = 2 + i;

        double gfxBw = results.tfrResults[tfrIdx].avgBandwidthGbPerSec;
        tfrIdx++;

        table.Set(row, 0, " %d ", devices[srcIdx].memRank);
        table.Set(row, 1, " %d ", devices[srcIdx].memIndex);
        table.Set(row, 2, " %d ", devices[dstIdx].memRank);
        table.Set(row, 3, " %d ", devices[dstIdx].memIndex);
        table.Set(row, 4, " %.2f ", gfxBw);

        gfxMin = std::min(gfxMin, gfxBw);
        gfxAvg += gfxBw;
        gfxMax = std::max(gfxMax, gfxBw);

        if (numQueuePairs > 0) {
          double nicBw = results.tfrResults[tfrIdx].avgBandwidthGbPerSec;
          tfrIdx++;
          table.Set(row, 5, " %.2f ", nicBw);
          nicMin = std::min(nicMin, nicBw);
          nicAvg += nicBw;
          nicMax = std::max(nicMax, nicBw);
        }
      }

      int summaryBase = 2 + numHops;
      table.DrawRowBorder(summaryBase);
      table.Set(summaryBase    , 1, " MAX ");
      table.Set(summaryBase + 1, 1, " AVG ");
      table.Set(summaryBase + 2, 1, " MIN ");
      table.Set(summaryBase    , 4, " %.2f ", gfxMax);
      table.Set(summaryBase + 1, 4, " %.2f ", gfxAvg / numHops);
      table.Set(summaryBase + 2, 4, " %.2f ", gfxMin);

      if (numQueuePairs > 0) {
        table.Set(summaryBase    , 5, " %.2f ", nicMax);
        table.Set(summaryBase + 1, 5, " %.2f ", nicAvg / numHops);
        table.Set(summaryBase + 2, 5, " %.2f ", nicMin);
      }

      table.PrintTable(ev.outputToCsv, ev.showBorders);

      Utils::Print("Aggregate bandwidth (CPU Timed): %8.3f GB/s\n", results.avgTotalBandwidthGbPerSec);
    }
  }

  if (!Utils::RankDoesOutput()) return 0;

  if (Utils::HasDuplicateHostname()) {
    printf("[WARN] It is recommended to run TransferBench with one rank per host to avoid potential aliasing of executors\n");
  }

  return 0;
}

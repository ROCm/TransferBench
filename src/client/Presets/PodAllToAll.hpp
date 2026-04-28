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

int PodAllToAllPreset(EnvVars&          ev,
                      size_t      const numBytesPerTransfer,
                      std::string const presetName,
                      bool        const bytesSpecified)
{
  enum
  {
    A2A_COPY       = 0,
    A2A_READ_ONLY  = 1,
    A2A_WRITE_ONLY = 2,
    A2A_CUSTOM     = 3,
  };
  char a2aModeStr[4][20] = {"Copy", "Read-Only", "Write-Only", "Custom"};

  // Force single-stream mode for all-to-all benchmark
  ev.useSingleStream = 1;

  // Force to gfx unroll 2 unless explicitly set
  ev.gfxUnroll      = EnvVars::GetEnvVar("GFX_UNROLL", 2);

  int numRanks = TransferBench::GetNumRanks();
  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  // Collect env vars for this preset
  int a2aLocal      = EnvVars::GetEnvVar("A2A_LOCAL"      , 0);
  int memTypeIdx    = EnvVars::GetEnvVar("MEM_TYPE"       , 0);
  int numGpus       = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int numQueuePairs = EnvVars::GetEnvVar("NUM_QUEUE_PAIRS", 0);
  int numSubExecs   = EnvVars::GetEnvVar("NUM_SUB_EXEC"   , 8);
  int showDetails   = EnvVars::GetEnvVar("SHOW_DETAILS"   , 0);
  int useDmaExec    = EnvVars::GetEnvVar("USE_DMA_EXEC"   , 0);
  int useRemoteRead = EnvVars::GetEnvVar("USE_REMOTE_READ", 0);
  int stride        = EnvVars::GetEnvVar("STRIDE"         , 1);
  int groupSize     = EnvVars::GetEnvVar("GROUP_SIZE"     , numRanks * numDetectedGpus);

  // Check that all ranks have at least the number of GPUs requested
  // Warn if NIC configuration is slightly different from one another
  int numNics  = TransferBench::GetNumExecutors(EXE_NIC, 0);
  bool nicDifference = false;
  for (int rank = 0; rank < numRanks; rank++) {
    if (numGpus > TransferBench::GetNumExecutors(EXE_GPU_GFX, rank)) {
      Utils::Print("[ERROR] All-to-All preset requires each rank to have the same number of GPUs\n");
      return 1;
    }
    if (numQueuePairs > 0 && numNics != TransferBench::GetNumExecutors(EXE_NIC, rank))
      nicDifference = true;
  }
  if (nicDifference)
    Utils::Print("[WARN] Not all ranks have the same number of NICs\n");

  // A2A_MODE may be 0,1,2 or else custom numSrcs:numDsts
  int numSrcs, numDsts;
  int a2aMode = 0;
  if (getenv("A2A_MODE") && sscanf(getenv("A2A_MODE"), "%d:%d", &numSrcs, &numDsts) == 2) {
    a2aMode = A2A_CUSTOM;
  } else {
    a2aMode = EnvVars::GetEnvVar("A2A_MODE", 0);
    if (a2aMode < 0 || a2aMode > 2) {
      Utils::Print("[ERROR] a2aMode must be between 0 and 2, or else numSrcs:numDsts\n");
      return 1;
    }
    numSrcs = (a2aMode == A2A_WRITE_ONLY ? 0 : 1);
    numDsts = (a2aMode == A2A_READ_ONLY  ? 0 : 1);
  }

  MemType memType = Utils::GetGpuMemType(memTypeIdx);
  std::string devMemTypeStr = Utils::GetGpuMemTypeStr(memTypeIdx);

  // Print off environment variables
  if (Utils::RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
      if (!ev.outputToCsv) printf("[AllToAll Related]\n");
      ev.Print("A2A_LOCAL"      , a2aLocal     , "%s local transfers", a2aLocal ? "Include" : "Exclude");
      ev.Print("A2A_MODE"       , (a2aMode == A2A_CUSTOM) ?  std::to_string(numSrcs) + ":" + std::to_string(numDsts) : std::to_string(a2aMode),
                                  (a2aMode == A2A_CUSTOM) ? (std::to_string(numSrcs) + " read(s) " +
                                                             std::to_string(numDsts) + " write(s)").c_str(): a2aModeStr[a2aMode]);
      ev.Print("MEM_TYPE"       , memTypeIdx   , "Using %s GPU memory (%s)", devMemTypeStr.c_str(), Utils::GetAllGpuMemTypeStr().c_str());
      ev.Print("NUM_GPU_DEVICES", numGpus      , "Using %d GPUs", numGpus);
      ev.Print("NUM_QUEUE_PAIRS", numQueuePairs, "Using %d queue pairs for NIC transfers", numQueuePairs);
      ev.Print("NUM_SUB_EXEC"   , numSubExecs  , "Using %d subexecutors/CUs per Transfer", numSubExecs);
      ev.Print("USE_DMA_EXEC"   , useDmaExec   , "Using %s executor", useDmaExec ? "DMA" : "GFX");
      ev.Print("USE_REMOTE_READ", useRemoteRead, "Using %s as executor", useRemoteRead ? "DST" : "SRC");
      ev.Print("STRIDE"         , stride       , "Reordering devices by taking %d steps", stride);
      ev.Print("GROUP_SIZE"     , groupSize    , "Dividing all devices into groups of %d for a2a", groupSize);
      printf("\n");
    }
  }
  // Validate env vars
  if (numGpus < 0 || numGpus > numDetectedGpus) {
    Utils::Print("[ERROR] Cannot use %d GPUs.  Detected %d GPUs\n", numGpus, numDetectedGpus);
    return 1;
  }
  if (useDmaExec && (numSrcs != 1 || numDsts != 1)) {
    Utils::Print("[ERROR] DMA execution can only be used for copies (A2A_MODE=0)\n");
    return 1;
  }

  if (numRanks * numDetectedGpus % groupSize) {
    Utils::Print("[ERROR] Group size %d cannot evenly divide %d total devices from %d ranks.\n", groupSize, numRanks * numDetectedGpus, numRanks);
    return 1;
  }

  Utils::Print("GPU-%s IntraPod All-To-All benchmark:\n", useDmaExec ? "DMA" : "GFX");
  Utils::Print("==============================\n");
  Utils::Print("[%lu bytes per Transfer] [%s:%d] [%d Read(s) %d Write(s)] [MemType:%s] [NIC QueuePairs:%d] [#Ranks:%d]\n",
               numBytesPerTransfer, useDmaExec ? "DMA" : "GFX", numSubExecs, numSrcs, numDsts,
               devMemTypeStr.c_str(), numQueuePairs, numRanks);

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  ExeType exeType = useDmaExec ? EXE_GPU_DMA : EXE_GPU_GFX;

  Utils::RankPerPodMap& rankToPod = Utils::GetRankPerPodMap();
  if (rankToPod.empty()) {
    Utils::Print("[ERROR] No pods detected. Set TB_FORCE_SINGLE_POD=1 to treat all ranks as a single pod.\n");
    return 1;
  }
  for (auto const& [pod, ranks] : rankToPod) {
    int n = ranks.size() * numGpus;
    int numGroups = n / groupSize;
    std::vector<MemDevice> devices(n);
    std::vector<int> indices(n);
    for (int k = 0; k < n; k++) indices[k] = k;
    Utils::StrideGenerate(indices, stride);
    int idx = 0;
    for (int rank : ranks) {
      for (int devIdx = 0; devIdx < numGpus; devIdx++) {
        devices[indices[idx++]] = {memType, devIdx, rank};
      }
    }

    for (int group = 0; group < numGroups; group++) {
      std::vector<std::vector<int>> groupReIndex(groupSize, std::vector<int>(groupSize, -1));
      std::vector<Transfer> transfers;
      for (int i = group * groupSize; i < (group + 1) * groupSize; i++) {
        for (int j = group * groupSize; j < (group + 1) * groupSize; j++) {
          if (i == j) {
            if (!a2aLocal) continue;
          }
          TransferBench::Transfer transfer;
          transfer.numBytes = numBytesPerTransfer;
          for (int x = 0; x < numSrcs; x++) transfer.srcs.push_back(devices[i]);
          if (numDsts) transfer.dsts.push_back(devices[j]);
          for (int x = 1; x < numDsts; x++) transfer.dsts.push_back(devices[i]);
          transfer.exeDevice = {exeType,
                               (int32_t)(useRemoteRead ? devices[j].memIndex : devices[i].memIndex),
                               (int32_t)(useRemoteRead ? devices[j].memRank : devices[i].memRank)};
          transfer.exeSubIndex = -1;
          transfer.numSubExecs = numSubExecs;
          int const localI = i - group * groupSize;
          int const localJ = j - group * groupSize;
          groupReIndex[localI][localJ] = (int)transfers.size();
          transfers.push_back(transfer);
        }

        if (numQueuePairs > 0) {
          TransferBench::Transfer transfer;
          transfer.numBytes = numBytesPerTransfer;
          transfer.srcs.push_back(devices[i]);
          int next = group * groupSize + (i - group * groupSize + 1) % groupSize;
          transfer.dsts.push_back(devices[next]);
          transfer.exeDevice = {TransferBench::EXE_NIC_NEAREST,
                               (int32_t)devices[i].memIndex, (int32_t)devices[i].memRank};
          transfer.exeSubIndex = devices[next].memIndex;
          transfer.numSubExecs = numQueuePairs;
          transfers.push_back(transfer);
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

      // Per-group bandwidth table
      std::vector<std::vector<double>> groupBw(groupSize, std::vector<double>(groupSize, -1.0));
      for (int localI = 0; localI < groupSize; localI++) {
        for (int localJ = 0; localJ < groupSize; localJ++) {
          int const k = groupReIndex[localI][localJ];
          if (k >= 0)
            groupBw[localI][localJ] = results.tfrResults[k].avgBandwidthGbPerSec;
        }
      }
      if (Utils::RankDoesOutput()) {
        Utils::Print("\n--- Pod AllToAll Group %d ---\n", group);
        int const groupBase = group * groupSize;
        int const numRows = 2 + groupSize;
        int const numCols = 2 + groupSize;
        int const precision = 2;
        Utils::TableHelper table(numRows, numCols, precision);
        table.DrawRowBorder(0);
        table.DrawColBorder(0);
        table.DrawColBorder(numCols);
        table.DrawRowBorder(numRows);
        table.Set(0, 0, useRemoteRead ? " SRC\\DST+EXE " : " SRC+EXE\\DST ");
        table.DrawRowBorder(1);
        table.DrawColBorder(1);
        table.Set(1, 1, " Mem Device ");

        // Column headers
        int colPrevRank = -1;
        for (int j = 0; j < groupSize; j++) {
          int colIdx = 2 + j;
          int r = devices[groupBase + j].memRank;
          if (r != colPrevRank) {
            table.DrawColBorder(colIdx);
            table.Set(0, colIdx, " Rank %02d ", r);
            colPrevRank = r;
          }
          table.Set(1, colIdx, " GPU %02d ", devices[groupBase + j].memIndex);
        }

        // Row headers and data
        int rowPrevRank = -1;
        for (int localI = 0; localI < groupSize; localI++) {
          int rowIdx = 2 + localI;
          int r = devices[groupBase + localI].memRank;
          if (r != rowPrevRank) {
            table.DrawRowBorder(rowIdx);
            table.Set(rowIdx, 0, " Rank %02d ", r);
            rowPrevRank = r;
          }
          table.Set(rowIdx, 1, " GPU %02d ", devices[groupBase + localI].memIndex);
          for (int localJ = 0; localJ < groupSize; localJ++) {
            if (groupBw[localI][localJ] >= 0)
              table.Set(rowIdx, 2 + localJ, " %.2f ", groupBw[localI][localJ]);
            else
              table.Set(rowIdx, 2 + localJ, " N/A ");
          }
        }
        table.PrintTable(ev.outputToCsv, ev.showBorders);
      }
    }
  }

  if (!Utils::RankDoesOutput()) return 0;

  if (Utils::HasDuplicateHostname()) {
    printf("[WARN] It is recommended to run TransferBench with one rank per host to avoid potential aliasing of executors\n");
  }

  return 0;
}

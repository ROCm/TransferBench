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

void StrideGenerate(std::vector<int>& list, int k) {
  int n = list.size();
  k = ((k % n) + n) % n;  // normalize to 0..n-1
  if (k == 0) return;

  int d = std::gcd(k, n);
  std::vector<int> out;
  out.reserve(n);

  for (int s = 0; s < d; s++) {
    for (int j = 0; j < n / d; j++) {
      out.push_back(list[(s + j * k) % n]);
    }
  }
  list = std::move(out);
}

int PodAllToAllPreset(EnvVars&           ev,
                      size_t      const  numBytesPerTransfer,
                      std::string const  presetName)
{
  enum
  {
    A2A_COPY       = 0,
    A2A_READ_ONLY  = 1,
    A2A_WRITE_ONLY = 2,
    A2A_CUSTOM     = 3,
  };
  // Do we want custom
  char a2aModeStr[4][20] = {"Copy", "Read-Only", "Write-Only", "Custom"};

  // Force single-stream mode for all-to-all benchmark
  ev.useSingleStream = 1;

  // Force to gfx unroll 2 unless explicitly set
  ev.gfxUnroll      = EnvVars::GetEnvVar("GFX_UNROLL", 2);

  int numRanks = TransferBench::GetNumRanks();
  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  // Collect env vars for this preset
  int a2aLocal      = EnvVars::GetEnvVar("A2A_LOCAL"      , 0);
  int memTypeIdx    = EnvVars::GetEnvVar("MEM_TYPE"       , 2);
  int numGpus       = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int numQueuePairs = EnvVars::GetEnvVar("NUM_QUEUE_PAIRS", 0);
//int numResults    = EnvVars::GetEnvVar("NUM_RESULTS"    , numRanks > 1 ? 1 : 0);
  int numSubExecs   = EnvVars::GetEnvVar("NUM_SUB_EXEC"   , 8);
  int showDetails   = EnvVars::GetEnvVar("SHOW_DETAILS"   , 0);
  int useDmaExec    = EnvVars::GetEnvVar("USE_DMA_EXEC"   , 0);
  int useFineGrain  = EnvVars::GetEnvVar("USE_FINE_GRAIN" , -999); // Deprecated
  int useRemoteRead = EnvVars::GetEnvVar("USE_REMOTE_READ", 0);
  int stride        = EnvVars::GetEnvVar("STRIDE"         , 1);
  int groupSize     = EnvVars::GetEnvVar("GROUP_SIZE"     , numDetectedGpus);

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

  // Deprecated env var check
  if (useFineGrain != -999) {
    memTypeIdx = useFineGrain ? 2 : 0;
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
//      if (numRanks > 1)
//        ev.Print("NUM_RESULTS"  , numResults   , "Showing top/bottom %d results", numResults);
      ev.Print("NUM_SUB_EXEC"   , numSubExecs  , "Using %d subexecutors/CUs per Transfer", numSubExecs);
//      ev.Print("SHOW_DETAILS"   , showDetails  , "%s full Test details", showDetails ? "Showing" : "Hiding");
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
//  if (numResults * 2 > numRanks) {
//    Utils::Print("[ERROR] Number of extrema results requested exceeds number of ranks.  NUM_RESULTS should be at most half the number of ranks\n");
//    return 1;
//  }

  if (groupSize % numGpus) {
    Utils::Print("[ERROR] Group size %d cannot evenly divide %d GPUs.\n", groupSize, numGpus);
    return 1;
  }

  Utils::Print("GPU-%s IntraPod All-To-All benchmark:\n", useDmaExec ? "DMA" : "GFX");
  Utils::Print("==============================\n");
  Utils::Print("[%lu bytes per Transfer] [%s:%d] [%d Read(s) %d Write(s)] [MemType:%s] [NIC QueuePairs:%d] [#Ranks:%d]\n",
               numBytesPerTransfer, useDmaExec ? "DMA" : "GFX", numSubExecs, numSrcs, numDsts,
               devMemTypeStr.c_str(), numQueuePairs, numRanks);

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  // Collect the number of GPU devices to use
  ExeType exeType = useDmaExec ? EXE_GPU_DMA : EXE_GPU_GFX;

  std::vector<std::map<std::pair<int, int>, int>> reIndex(numRanks);
  //std::vector<Transfer> transfers;

/////////////  
  // ?? should i launch all pods per test
  // ?? should i launch all a2a group at once?
  Utils::RankPodMap& rankToPod = Utils::GetRankPodMap();
  if (rankToPod.empty()) {
    Utils::Print("[ERROR] No pods detected. Set FORCE_SINGLE_POD=1 to treat all ranks as a single pod.\n");
    return 1;
  }
  for (auto const& [pod, ranks] : rankToPod) {
    // Add all devices in a pod
    int n = ranks.size() * numGpus;
    int numGroups = n / groupSize;
    std::vector<MemDevice> devices(n);
    std::vector<int> indices(n);
    for (int k = 0; k < n; k++) indices[k] = k;
    StrideGenerate(indices, stride);
    int idx = 0;
    for (int rank : ranks) {
      for (int devIdx = 0; devIdx < numGpus; devIdx++) {
        devices[indices[idx++]] = {memType, devIdx, rank};
      }
    }
    // Evenly divide devices into numGroups
    // reuse i ok here?
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
          transfer.srcs.push_back(devices[i]);
          transfer.dsts.push_back(devices[j]);
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

        // Create rings using NICs
        // should this ring be conform to reorder of group? And should it center on NIC?
        if (numQueuePairs > 0) {
          TransferBench::Transfer transfer;
          transfer.numBytes = numBytesPerTransfer;
          transfer.srcs.push_back(devices[i]);
          int next = group * groupSize + (i - group * groupSize + 1) % groupSize;
          transfer.dsts.push_back(devices[next]);
          // double check this
          transfer.exeDevice = {TransferBench::EXE_NIC_NEAREST,
                               (int32_t)devices[i].memIndex, (int32_t)devices[i].memRank};
          // to be fixed
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

      // Per-group table (PodPeerToPeer-style: devices in this group, ranks in group)
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
/////////////  

  // Only ranks that actually do output will compile results
  if (!Utils::RankDoesOutput()) return 0;
/*
{
  // Prepare table of results
  int numRows   = 2 + (numGpus + 1) * (1 + 2*numResults);
  int numCols   = 1 + (numGpus + (numQueuePairs > 1 ? 1 : 0) + 2) * (numResults > 0 ? 2 : 1);
  int precision = 2;
  Utils::TableHelper table(numRows, numCols, precision);

  // Header row
  int rowIdx = 0, colIdx = 0;
  table.DrawRowBorder(rowIdx);
  table.DrawColBorder(colIdx);
  table.Set(rowIdx, colIdx++, " SRC\\DST ");
  for (int gpuIdx = 0; gpuIdx < numGpus; gpuIdx++) {
    if (numResults > 0 || gpuIdx == 0) {
      table.DrawColBorder(colIdx);
    }
    table.Set(rowIdx, colIdx++, " GPU %02d ", gpuIdx);
    if (numResults > 0) {
      table.SetColAlignment(colIdx, Utils::TableHelper::ALIGN_CENTER);
      table.Set(rowIdx, colIdx++, "(Rnk)");
    }
  }
  table.DrawColBorder(colIdx);

  if (numQueuePairs > 0) {
    table.Set(rowIdx, colIdx++, " NIC ", numQueuePairs);
    if (numResults > 0) {
      table.SetColAlignment(colIdx, Utils::TableHelper::ALIGN_CENTER);
      table.Set(rowIdx, colIdx++, "(Rnk)");
    }
    table.DrawColBorder(colIdx);
  }

  table.Set(rowIdx, colIdx++, " STotal ");
  if (numResults > 0) {
    table.SetColAlignment(colIdx, Utils::TableHelper::ALIGN_CENTER);
    table.Set(rowIdx, colIdx++, "(Rnk)");
  }
  table.DrawColBorder(colIdx);

  table.Set(rowIdx, colIdx++, " Actual ");
  if (numResults > 0) {
    table.SetColAlignment(colIdx, Utils::TableHelper::ALIGN_CENTER);
    table.Set(rowIdx, colIdx++, "(Rnk)");
  }
  table.DrawColBorder(colIdx);
  rowIdx++;
  table.DrawRowBorder(rowIdx);

  // Header column
  for (int gpuIdx = 0; gpuIdx < numGpus; gpuIdx++) {
    // MAX results
    for (int i = 0; i < numResults; i++) {
      if (i == 0) table.Set(rowIdx, 0, " GPU %02d MAX ", gpuIdx);
      rowIdx++;
    }
    // Avg result
    table.Set(rowIdx++, 0, " GPU %02d%s ", gpuIdx, numResults > 0 ? " AVG" : "");
    // MIN results
    for (int i = numResults-1; i >= 0; i--) {
      if (i == 0) table.Set(rowIdx, 0, " GPU %02d MIN ", gpuIdx);
      rowIdx++;
    }
    if (numResults > 0 || gpuIdx == numGpus - 1) table.DrawRowBorder(rowIdx);
  }

  // RTotal
  for (int i = 0; i < numResults; i++) {
    if (i == 0) table.Set(rowIdx, 0, " RTotal MAX ");
    rowIdx++;
  }
  // Avg result
  table.Set(rowIdx++, 0, " RTotal%s ", numResults > 0 ? " AVG" : "");
  // MIN results
  for (int i = numResults-1; i >= 0; i--) {
    if (i == 0) table.Set(rowIdx, 0, " RTotal MIN ");
    rowIdx++;
  }
  table.DrawRowBorder(rowIdx);

  // Data cells
  std::vector<std::vector<double>> rowTotalBandwidth(numRanks, std::vector<double>(numGpus, 0.0));
  std::vector<std::vector<double>> colTotalBandwidth(numRanks, std::vector<double>(numGpus+3, 0.0));
  double totalBandwidthGpu = 0.0;

  for (int src = 0; src < numGpus; src++) {
    int rowBase = 1 + src * (2*numResults+1);

    std::vector<double> minBw(numRanks, std::numeric_limits<double>::max());
    int rowTransferCount = 0;

    for (int dst = 0; dst < numGpus; dst++) {
      int colBase = 1 + dst * (numResults ? 2 : 1);

      // Collect results for all ranks
      std::vector<std::pair<double, int>> bws;
      double average = 0.0;
      for (int rank = 0; rank < numRanks; rank++) {
        if (reIndex[rank].count(std::make_pair(src, dst))) {
          int const transferIdx = reIndex[rank][std::make_pair(src,dst)];
          double avgBw = results.tfrResults[transferIdx].avgBandwidthGbPerSec;
          average                      += avgBw;
          totalBandwidthGpu            += avgBw;
          rowTotalBandwidth[rank][src] += avgBw;
          colTotalBandwidth[rank][dst] += avgBw;
          minBw[rank] = std::min(minBw[rank], avgBw);
          bws.push_back(std::make_pair(avgBw, rank));
        }
      }
      if (bws.size() == 0) {
        table.Set(rowBase + numResults, colBase, " N/A ");
      } else {
        std::sort(bws.begin(), bws.end());
        average /= bws.size();
        rowTransferCount++;

        // MAX results
        for (int i = 0; i < numResults; i++) {
          table.Set(rowBase + i, colBase  , " %.2f ", bws[bws.size()-1-i].first);
          table.Set(rowBase + i, colBase+1, "%d ", bws[bws.size()-1-i].second);
        }

        // AVG results
        table.Set(rowBase + numResults, colBase, " %.2f ", average);

        // MIN results
        for (int i = 0; i < numResults; i++) {
          table.Set(rowBase + numResults + 1 + i, colBase   , " %.2f ", bws[numResults-1-i].first);
          table.Set(rowBase + numResults + 1 + i, colBase+1 , "%d ", bws[numResults-1-i].second);
        }
      }
    }

    // NIC results
    int colTotIdx = numGpus;
    if (numQueuePairs > 0) {
      int colBase = 1 + numGpus * (numResults ? 2 : 1);
      std::vector<std::pair<double, int>> bws;
      double average = 0.0;

      for (int rank = 0; rank < numRanks; rank++) {
        double avgBw = results.tfrResults[nicTransferIdx[rank][src]].avgBandwidthGbPerSec;
        average                            += avgBw;
        totalBandwidthGpu                  += avgBw;
        rowTotalBandwidth[rank][src]       += avgBw;
        colTotalBandwidth[rank][colTotIdx] += avgBw;
        minBw[rank] = std::min(minBw[rank], avgBw);
        bws.push_back(std::make_pair(avgBw, rank));
      }
      colTotIdx++;
      std::sort(bws.begin(), bws.end());
      average /= bws.size();
      rowTransferCount++;

      // MAX results
      for (int i = 0; i < numResults; i++) {
        table.Set(rowBase + i, colBase  , " %.2f ", bws[bws.size()-1-i].first);
        table.Set(rowBase + i, colBase+1,  "%d ", bws[bws.size()-1-i].second);
      }

      // AVG results
      table.Set(rowBase + numResults, colBase, " %.2f ", average);

      // MIN results
      for (int i = 0; i < numResults; i++) {
        table.Set(rowBase + numResults + 1 + i, colBase   , " %.2f ", bws[numResults-1-i].first);
        table.Set(rowBase + numResults + 1 + i, colBase+1 , "%d ", bws[numResults-1-i].second);
      }
    }

    // STotal
    {
      int colBase = 1 + (numGpus + (numQueuePairs ? 1 : 0)) * (numResults ? 2 : 1);
      std::vector<std::pair<double, int>> bws;
      double average = 0.0;
      for (int rank = 0; rank < numRanks; rank++) {
        double avgBw = rowTotalBandwidth[rank][src];
        bws.push_back(std::make_pair(avgBw, rank));
        colTotalBandwidth[rank][colTotIdx] += avgBw;
        average += avgBw;
      }
      colTotIdx++;
      std::sort(bws.begin(), bws.end());
      average /= bws.size();

      // MAX results
      for (int i = 0; i < numResults; i++) {
        table.Set(rowBase + i, colBase  , " %.2f ", bws[bws.size()-1-i].first);
        table.Set(rowBase + i, colBase+1, "%d ", bws[bws.size()-1-i].second);
      }

      // AVG results
      table.Set(rowBase + numResults, colBase, " %.2f ", average);

      // MIN results
      for (int i = 0; i < numResults; i++) {
        table.Set(rowBase + numResults + 1 + i, colBase   , " %.2f ", bws[numResults-1-i].first);
        table.Set(rowBase + numResults + 1 + i, colBase+1 , "%d ", bws[numResults-1-i].second);
      }
    }

    // Actual
    {
      int colBase = 1 + (numGpus + (numQueuePairs ? 1 : 0) + 1) * (numResults ? 2 : 1);
      std::vector<std::pair<double, int>> bws;
      double average = 0.0;
      for (int rank = 0; rank < numRanks; rank++) {
        double avgBw = rowTransferCount * minBw[rank];
        bws.push_back(std::make_pair(avgBw, rank));
        average += avgBw;
        colTotalBandwidth[rank][colTotIdx] += avgBw;
      }
      colTotIdx++;
      std::sort(bws.begin(), bws.end());
      average /= bws.size();

      // MAX results
      for (int i = 0; i < numResults; i++) {
        table.Set(rowBase + i, colBase  , " %.2f ", bws[bws.size()-1-i].first);
        table.Set(rowBase + i, colBase+1, "%d ", bws[bws.size()-1-i].second);
      }

      // AVG results
      table.Set(rowBase + numResults, colBase, " %.2f ", average);

      // MIN results
      for (int i = 0; i < numResults; i++) {
        table.Set(rowBase + numResults + 1 + i, colBase   , " %.2f ", bws[numResults-1-i].first);
        table.Set(rowBase + numResults + 1 + i, colBase+1 , "%d ", bws[numResults-1-i].second);
      }
    }
  }

  // RTotal
  int rowBase = 1 + (numGpus * (1 + 2 * numResults));
  for (int col = 0; col < (numGpus + (numQueuePairs ? 1 : 0) + 2); col++) {
    int colBase = 1 + col * (numResults ? 2 : 1);
    std::vector<std::pair<double, int>> bws;
    double average = 0.0;
    for (int rank = 0; rank < numRanks; rank++) {
      double avgBw = colTotalBandwidth[rank][col];
      bws.push_back(std::make_pair(avgBw, rank));
      average += avgBw;
    }
    std::sort(bws.begin(), bws.end());
    average /= bws.size();

    // MAX results
    for (int i = 0; i < numResults; i++) {
      table.Set(rowBase + i, colBase  , " %.2f ", bws[bws.size()-1-i].first);
      table.Set(rowBase + i, colBase+1, "%d ", bws[bws.size()-1-i].second);
    }

    // AVG results
    table.Set(rowBase + numResults, colBase, " %.2f ", average);

    // MIN results
    for (int i = 0; i < numResults; i++) {
      table.Set(rowBase + numResults + 1 + i, colBase   , " %.2f ", bws[numResults-1-i].first);
      table.Set(rowBase + numResults + 1 + i, colBase+1 , "%d ", bws[numResults-1-i].second);
    }
  }

  // Add CPU results
  table.Set(numRows - 1, numCols - 2, " CPU Timed: ");
  table.Set(numRows - 1, numCols - 1, " %.2f ", results.avgTotalBandwidthGbPerSec);
  for (int col = 0; col < numCols - 2; col++)
    table.SetCellBorder(numRows - 1, col, Utils::TableHelper::BORDER_TOP);
  table.SetCellBorder(numRows - 1, numCols - 2, Utils::TableHelper::BORDER_ALL);
  table.SetCellBorder(numRows - 1, numCols - 1, Utils::TableHelper::BORDER_ALL);

  table.PrintTable(ev.outputToCsv, ev.showBorders);
}
*/

/*
  Utils::Print("\n");
  Utils::Print("Average   bandwidth (GPU Timed): %8.3f GB/s\n", totalBandwidthGpu / transfers.size());
  Utils::Print("Aggregate bandwidth (GPU Timed): %8.3f GB/s\n", totalBandwidthGpu);
  Utils::Print("Aggregate bandwidth (CPU Timed): %8.3f GB/s\n", results.avgTotalBandwidthGbPerSec);
  Utils::PrintErrors(results.errResults);
*/

  if (Utils::HasDuplicateHostname()) {
    printf("[WARN] It is recommended to run TransferBench with one rank per host to avoid potential aliasing of executors\n");
  }

  if (useFineGrain != -999) {
    Utils::Print("[WARN] USE_FINE_GRAIN has been deprecated and replaced by MEM_TYPE\n");
    Utils::Print("[WARN] MEM_TYPE has been set to %d to correspond to previous use of USE_FINE_GRAIN=%d\n", memTypeIdx, useFineGrain);
  }

  return 0;
}

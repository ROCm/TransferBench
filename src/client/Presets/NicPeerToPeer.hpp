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

// Helper functions

int GetClosestDeviceToNic(MemType memType, int nicIdx, int rank) {
  return TransferBench::IsCpuMemType(memType) ?
         TransferBench::GetClosestCpuNumaToNic(nicIdx, rank) :
         TransferBench::GetClosestGpuToNic(nicIdx, rank);
}

int NicPeerToPeerPreset(EnvVars&          ev,
                        size_t      const numBytesPerTransfer,
                        std::string const presetName,
                        bool        const bytesSpecified)
{
  if (Utils::GetNumRankGroups() > 1) {
    Utils::Print("[ERROR] NIC p2p preset can only be run across ranks that are homogenous\n");
    Utils::Print("[ERROR] Run ./TransferBench without any args to display topology information\n");
    Utils::Print("[ERROR] TB_NIC_FILTER may also be used to limit NIC visibility\n");
    return 1;
  }

  int numRanks = TransferBench::GetNumRanks();
  int numNicsPerRank = TransferBench::GetNumExecutors(EXE_NIC);
  if (numNicsPerRank == 0) {
    Utils::Print("No NIC detected, NICs are required to run this preset\n");
    return 1;
  }

  // Collect env vars for this preset
  int numQueuePairs     = EnvVars::GetEnvVar("NUM_QUEUE_PAIRS", 1);
  int useRemoteRead     = EnvVars::GetEnvVar("USE_REMOTE_READ", 0);
  int showFullMatrix    = EnvVars::GetEnvVar("OUTPUT_FORMAT", 1);
  int srcCpu            = EnvVars::GetEnvVar("USE_CPU_SRC_MEM", 0);
  int dstCpu            = EnvVars::GetEnvVar("USE_CPU_DST_MEM", 0);
  int srcMemType        = EnvVars::GetEnvVar("SRC_MEM_TYPE", 2);
  int dstMemType        = EnvVars::GetEnvVar("DST_MEM_TYPE", 2);
  int nodeParallel      = EnvVars::GetEnvVar("PARALLEL_NODE", 1);
  int nicParLevel       = EnvVars::GetEnvVar("NIC_PARALLEL_LEVEL", numNicsPerRank);

  // Parse Memtype for src/dst
  MemType srcTypeActual = Utils::GetMemType(srcMemType, srcCpu);
  MemType dstTypeActual = Utils::GetMemType(dstMemType, dstCpu);
  std::string srcTypeStr = Utils::GetMemTypeStr(srcMemType, srcCpu);
  std::string dstTypeStr = Utils::GetMemTypeStr(dstMemType, dstCpu);

  // Display EnvVars
  if (Utils::RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
      if (!ev.outputToCsv) printf("[P2P Network Related]\n");
      ev.Print("NUM_NIC_SE",      numQueuePairs,  "Using %d queue pairs per Transfer", numQueuePairs);
      ev.Print("USE_REMOTE_READ", useRemoteRead,  "Using %s as executor", useRemoteRead ? "DST" : "SRC");
      ev.Print("OUTPUT_FORMAT",   showFullMatrix, "Printing results in %s format", showFullMatrix ? "full matrix" : "column");
      ev.Print("USE_CPU_SRC_MEM", srcCpu,         "Source memory is %s", srcCpu ? "CPU" : "GPU");
      ev.Print("USE_CPU_DST_MEM", dstCpu,         "Destination memory is %s", dstCpu ? "CPU" : "GPU");
      ev.Print("SRC_MEM_TYPE",    srcMemType,     "Using %s memory (%s)", srcTypeStr.c_str(), Utils::GetAllMemTypeStr(srcCpu).c_str());
      ev.Print("DST_MEM_TYPE",    dstMemType,     "Using %s memory (%s)", dstTypeStr.c_str(), Utils::GetAllMemTypeStr(dstCpu).c_str());
      ev.Print("PARALLEL_NODE",   nodeParallel,   "Executing p2p node pairs in parallel: %s", nodeParallel ? "yes" : "no");
      ev.Print("NIC_PARALLEL_LEVEL", nicParLevel, "Between a pair of nodes, %d pairs of NIC-NIC transfers executed in parallel", nicParLevel);
      printf("\n");
    }
  }

  // TODO: validate env vars

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  TransferBench::TestResults results;


  // Initialize output table
  Utils::Print("Unidirectional copy peak bandwidth GB/s (NIC RDMA Using Nearest Device)\n");

  int const numTotalNics = numNicsPerRank * numRanks;
  int numRows = showFullMatrix ? 3 + numTotalNics : 1 + numTotalNics * numTotalNics;
  int numCols = showFullMatrix ? numRows : 7;
  int precision = 2;
  Utils::TableHelper table(numRows, numCols, precision);
  // Device/Memory names for table
  std::vector<std::string> srcExes;
  std::vector<std::string> dstExes;
  std::vector<int>         srcMems;
  std::vector<int>         dstMems;

  // Query closest device to each NIC available, store device info to a map
  std::vector<double> avgBandwidth;

  // Create a round-robin schedule for all-to-all communication
  std::vector<std::vector<std::pair<int, int>>> schedule;
  std::vector<std::vector<std::pair<int, int>>> nicSchedule;

  Utils::RoundRobinSchedule(schedule, numRanks, nodeParallel);
  Utils::CombinationSchedule(nicSchedule, numNicsPerRank, nicParLevel);

  int totalTransfers = numRanks * numNicsPerRank * numRanks * numNicsPerRank;
  int counter = 0;
  double durationSec = 0;
  avgBandwidth.resize(totalTransfers);
  srcExes.resize(totalTransfers);
  dstExes.resize(totalTransfers);
  srcMems.resize(totalTransfers);
  dstMems.resize(totalTransfers);

  // Execute transfers: node-level rounds -> NIC-level rounds -> node pairs
  for (auto const& roundPairs : schedule) {
    for (auto const& nicRoundPairs : nicSchedule) {
      std::vector<Transfer> transfers;
      auto cpuStart = std::chrono::high_resolution_clock::now();

      for (auto const& nodePair : roundPairs) {
        int srcRank = nodePair.first;
        int dstRank = nodePair.second;

        for (auto const& nicPair : nicRoundPairs) {
          int srcNicIdx = nicPair.first;
          int dstNicIdx = nicPair.second;

          Transfer transfer;

          // Determine which GPU memory/CPU NUMA to use based on NIC proximity and its info
          int srcMemIndex = GetClosestDeviceToNic(srcTypeActual, srcNicIdx, srcRank);
          int dstMemIndex = GetClosestDeviceToNic(dstTypeActual, dstNicIdx, dstRank);

          if (srcMemIndex == -1 || dstMemIndex == -1) {
            Utils::Print("[ERROR] No proper GPU device can be found for transfer R%dN%d - R%dN%d\n",
                         srcRank, srcNicIdx, dstRank, dstNicIdx);
            return 1;
          }
          transfer.numBytes = numBytesPerTransfer;
          transfer.srcs.push_back({srcTypeActual, srcMemIndex, srcRank});
          transfer.dsts.push_back({dstTypeActual, dstMemIndex, dstRank});
          transfer.exeDevice = {EXE_NIC, (useRemoteRead ? dstNicIdx : srcNicIdx), (useRemoteRead ? dstRank : srcRank)};
          transfer.exeSubIndex = (useRemoteRead ? srcNicIdx : dstNicIdx);
          transfer.numSubExecs = numQueuePairs;

          transfers.push_back(transfer);
        }
      }

      if (!TransferBench::RunTransfers(cfg, transfers, results)) {
        for (auto const& err : results.errResults)
          Utils::Print("%s\n", err.errMsg.c_str());
        return 1;
      }

      counter += transfers.size();

      // Store results with correct indexing
      for (size_t i = 0; i < results.tfrResults.size(); i++) {
        int srcRank = transfers[i].srcs[0].memRank;
        int dstRank = transfers[i].dsts[0].memRank;

        auto srcExe = useRemoteRead ? results.tfrResults[i].exeDstDevice : results.tfrResults[i].exeDevice;
        auto dstExe = useRemoteRead ? results.tfrResults[i].exeDevice : results.tfrResults[i].exeDstDevice;
        int srcNicIdx = srcExe.exeIndex;
        int dstNicIdx = dstExe.exeIndex;

        // Calculate index in table-rendering order: srcRank x srcNicIdx x dstRank x dstNicIdx
        int idx = srcRank * (numNicsPerRank * numRanks * numNicsPerRank)
                + srcNicIdx * (numRanks * numNicsPerRank)
                + dstRank * numNicsPerRank
                + dstNicIdx;
        avgBandwidth[idx] = results.tfrResults[i].avgBandwidthGbPerSec;
        srcExes[idx] = TransferBench::GetExecutorName(srcExe);
        dstExes[idx] = TransferBench::GetExecutorName(dstExe);
        // TODO: add mem device info in transfer result?
        srcMems[idx] = transfers[i].srcs[0].memIndex;
        dstMems[idx] = transfers[i].dsts[0].memIndex;
      }

      auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
      durationSec += std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count();
      if (Utils::RankDoesOutput()) {
        fprintf(stderr, "Completed %d/%d pairs in %6.3fs, estimated remaining time %6.3fs.\n",
                counter, totalTransfers, durationSec, durationSec * (totalTransfers - counter) / counter);
      }
    }
  }

  // Draw table outlines
  table.DrawRowBorder(0);
  table.DrawColBorder(0);
  table.DrawColBorder(numCols);
  table.DrawRowBorder(numRows);

  // Rendering table
  if (showFullMatrix) {
    table.Set(0, 0, useRemoteRead ? "SRC\\DST+EXE " : "SRC+EXE\\DST ");
    table.DrawRowBorder(1);
    table.DrawColBorder(1);
    table.Set(1, 1, " NIC Device ");
    table.Set(2, 2, " Mem Device ");
    int rowIdx = 3;
    int entryIdx = 0;

    for (int rank = 0; rank < numRanks; rank++) {
      table.DrawRowBorder(rowIdx);
      table.DrawColBorder(rowIdx);
      table.Set(rowIdx, 0, " Rank %02d ", rank);
      table.Set(0, rowIdx, " Rank %02d ", rank);
      for (int nic = 0; nic < numNicsPerRank; nic++) {
        table.Set(rowIdx, 1, " %s ", srcExes[entryIdx].c_str());
        table.Set(rowIdx, 2, " %cPU %02d ", TransferBench::IsCpuMemType(srcTypeActual) ? 'C' : 'G', srcMems[entryIdx]);
        table.Set(1, rowIdx, " %s ", dstExes[rowIdx - 3].c_str());
        table.Set(2, rowIdx, " %cPU %02d ", TransferBench::IsCpuMemType(dstTypeActual) ? 'C' : 'G', dstMems[rowIdx - 3]);
        int colIdx = 3;
        for (int dstRank = 0; dstRank < numRanks; dstRank++) {
          for (int dstNic = 0; dstNic < numNicsPerRank; dstNic++) {
            table.Set(rowIdx, colIdx++ , " %.2f ", avgBandwidth[entryIdx++]);
          }
        }
        rowIdx++;
      }
    }
  } else {
    table.Set(0, 0, " SRC Rank ");
    table.Set(0, 1, " SRC NIC ");
    table.Set(0, 2, " SRC MEM ");
    table.Set(0, 3, " DST Rank ");
    table.Set(0, 4, " DST NIC ");
    table.Set(0, 5, " DST MEM ");
    table.Set(0, 6, " bw (GB/s) ");
    table.DrawColBorder(3);
    table.DrawColBorder(6);
    int rowIdx = 1;

    for (int src = 0; src < numRanks; src++) {
      for (int i = 0; i < numNicsPerRank; i++) {
        table.DrawRowBorder(rowIdx);
        for (int dst = 0; dst < numRanks; dst++) {
          for (int j = 0; j < numNicsPerRank; j++) {
            table.Set(rowIdx, 0, " Rank %02d ", src);
            table.Set(rowIdx, 1, " %s ", srcExes[rowIdx - 1].c_str());
            table.Set(rowIdx, 2, " %cPU %02d ", TransferBench::IsCpuMemType(srcTypeActual) ? 'C' : 'G', srcMems[rowIdx - 1]);
            table.Set(rowIdx, 3, " Rank %02d ", dst);
            table.Set(rowIdx, 4, " %s ", dstExes[rowIdx - 1].c_str());
            table.Set(rowIdx, 5, " %cPU %02d ", TransferBench::IsCpuMemType(dstTypeActual) ? 'C' : 'G', dstMems[rowIdx - 1]);
            table.Set(rowIdx, 6, " %.2f ", avgBandwidth[rowIdx - 1]);
            rowIdx++;
          }
        }
      }
    }
  }

  table.PrintTable(ev.outputToCsv, ev.showBorders);

  // Ranking fastest/slowest connection
  // TODO: expand length of the list via user passed in value
  Utils::TableHelper summaryTable(11, 6, precision);
  Utils::Print("Summary of top 10 fastest/slowest connection\n");

  summaryTable.Set(0, 0, " Fastest Bandwidth (GB/s) ");
  summaryTable.Set(0, 1, " Src ");
  summaryTable.Set(0, 2, " Dst ");
  summaryTable.Set(0, 3, " Slowest Bandwidth (GB/s) ");
  summaryTable.Set(0, 4, " Src ");
  summaryTable.Set(0, 5, " Dst ");

  summaryTable.DrawRowBorder(0);
  summaryTable.DrawRowBorder(1);
  summaryTable.DrawRowBorder(11);
  for (int i = 0; i <= 6; i++) summaryTable.DrawColBorder(i);

  std::vector<size_t> idx(avgBandwidth.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](size_t i1, size_t i2) {return avgBandwidth[i1] > avgBandwidth[i2];});
  for (int i = 0; i < 10; i++) {
    int index = idx[i];
    index /= numNicsPerRank;

    int dstRank = index % numRanks;
    index /= numRanks;

    index /= numNicsPerRank;

    int srcRank = index;

    summaryTable.Set(1 + i, 1, " R%02d:%s ", srcRank, srcExes[idx[i]].c_str());
    summaryTable.Set(1 + i, 2, " R%02d:%s ", dstRank, dstExes[idx[i]].c_str());
    summaryTable.Set(1 + i, 0, " %.2f ", avgBandwidth[idx[i]]);

    index = idx[idx.size() - 1 - i];
    index /= numNicsPerRank;

    index /= numRanks;

    index /= numNicsPerRank;

    srcRank = index;

    summaryTable.Set(1 + i, 4, " R%02d:%s ", srcRank, srcExes[idx[idx.size() - 1 - i]].c_str());
    summaryTable.Set(1 + i, 5, " R%02d:%s ", dstRank, dstExes[idx[idx.size() - 1 - i]].c_str());
    summaryTable.Set(1 + i, 3, " %.2f ", avgBandwidth[idx[idx.size() - 1 - i]]);
  }
  summaryTable.PrintTable(ev.outputToCsv, ev.showBorders);

  return 0;
}

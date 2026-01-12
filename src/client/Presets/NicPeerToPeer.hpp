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
MemType parseMemType(std::string const memTypeIdx) {
  bool isCpu = false;
  int  memType = 2;
  if (memTypeIdx.length() >= 1) {
    char firstChar = std::toupper(memTypeIdx[0]);
    if (firstChar == 'G' && firstChar == 'C') {
      Utils::Print("WARNING: Invalid MEM_POLICY first character '%c', using default 'G'\n", memTypeIdx[0]);
    }
    isCpu = firstChar == 'C';
  }
  
  if (memTypeIdx.length() >= 2) {
    if (std::isdigit(memTypeIdx[1])) {
      int level = memTypeIdx[1] - '0';
      if (level >= 0 && level <= 3) {
        memType = level;
      } else {
        Utils::Print("WARNING: Invalid MEM_POLICY level '%c', must be 0-3, using default 2\n", memTypeIdx[1]);
      }
    } else {
      Utils::Print("WARNING: Invalid MEM_POLICY second character '%c', using default 2\n", memTypeIdx[1]);
    }
  }

  return Utils::GetMemType(memType, isCpu);
}

int GetClosestDeviceToNic(MemType memType, int nicIdx, int rank) {
  return TransferBench::IsCpuMemType(memType) ?
         TransferBench::GetClosestCpuNumaToNic(nicIdx, rank) :
         TransferBench::GetClosestGpuToNic(nicIdx, rank);
}

int NicPeerToPeerPreset(EnvVars&           ev,
                        size_t      const  numBytesPerTransfer,
                        std::string const  presetName)
{
  int numRanks = TransferBench::GetNumRanks();

  int numDetectedNics = TransferBench::GetNumExecutors(EXE_NIC);

  // Collect env vars for this preset
  //int numCpuDevices  = EnvVars::GetEnvVar("NUM_CPU_DEVICES", numDetectedCpus);
  //int numGpuDevices  = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int numQueuePairs  = EnvVars::GetEnvVar("NUM_QUEUE_PAIRS", 1);
  int useRemoteRead  = EnvVars::GetEnvVar("USE_REMOTE_READ", 0);
  int showFullMatrix    = EnvVars::GetEnvVar("OUTPUT_FORMAT", 1);
  std::string nicFilter = EnvVars::GetEnvVar("NIC_FILTER", "");
  std::string srcMemIdx = EnvVars::GetEnvVar("SRC_MEM", "G2");
  std::string dstMemIdx = EnvVars::GetEnvVar("DST_MEM", "G2");
  int rr = EnvVars::GetEnvVar("FAST_EXE", 0);

  // Parse NIC_FILTER to build list of NIC indices to use
  std::vector<int> nicIndices;
  if (nicFilter.empty()) {
    // No filter specified, use all detected NICs
    for (int i = 0; i < numDetectedNics; i++) {
      nicIndices.push_back(i);
    }
  } else {
    // Parse comma-separated list of NIC indices or names
    std::istringstream ss(nicFilter);
    std::string token;
    while (std::getline(ss, token, ',')) {
      // Trim whitespace
      token.erase(0, token.find_first_not_of(" \t"));
      token.erase(token.find_last_not_of(" \t") + 1);

      // Check if token is a number (NIC index)
      bool isNumber = !token.empty() && std::all_of(token.begin(), token.end(), ::isdigit);

      if (isNumber) {
        int nicIdx = std::stoi(token);
        if (nicIdx >= 0 && nicIdx < numDetectedNics) {
          nicIndices.push_back(nicIdx);
        } else {
          Utils::Print("WARNING: NIC index %d out of range (0-%d), ignoring\n", nicIdx, numDetectedNics - 1);
        }
      } else {
        // Try to match by NIC name
        bool found = false;
        for (int nicIdx = 0; nicIdx < numDetectedNics; nicIdx++) {
          std::string nicName = TransferBench::GetExecutorName({EXE_NIC, nicIdx});
          if (nicName == token) {
            nicIndices.push_back(nicIdx);
            found = true;
            break;
          }
        }
        if (!found) {
          Utils::Print("WARNING: NIC '%s' not found, ignoring\n", token.c_str());
        }
      }
    }
  }

  // Parse Memtype for src/dst
  MemType srcTypeActual = parseMemType(srcMemIdx);
  MemType dstTypeActual = parseMemType(dstMemIdx);

  // Create a round-robin schedule for all-to-all communication
  std::vector<std::vector<std::pair<int, int>>> schedule;
  if (rr) {
    if (numRanks % 2 == 0) {
      // Even number of ranks: use round-robin tournament scheduling
      for (int round = 0; round < numRanks - 1; round++) {
        std::vector<std::pair<int, int>> roundPairs;
        for (int i = 0; i < numRanks / 2; i++) {
          int rank1 = i;
          int rank2 = numRanks - 1 - i;
          if (round > 0) {
            // Rotate all except the first rank
            if (rank1 > 0) rank1 = ((rank1 - 1 + round) % (numRanks - 1)) + 1;
            if (rank2 > 0) rank2 = ((rank2 - 1 + round) % (numRanks - 1)) + 1;
          }
          if (rank1 != rank2) {
            roundPairs.push_back({rank1, rank2});
            roundPairs.push_back({rank2, rank1});
          }
        }
        schedule.push_back(roundPairs);
      }
    } else {
      // Odd number of ranks: one rank sits out each round
      for (int round = 0; round < numRanks; round++) {
        std::vector<std::pair<int, int>> roundPairs;
        for (int i = 0; i < numRanks / 2; i++) {
          int rank1 = (round + i) % numRanks;
          int rank2 = (round + numRanks - 1 - i) % numRanks;
          if (rank1 != rank2) {
            roundPairs.push_back({rank1, rank2});
            roundPairs.push_back({rank2, rank1});
          }
        }
        schedule.push_back(roundPairs);
      }
    }
    // Finally, a round where every rank does loopback
    std::vector<std::pair<int, int>> selfRound;
    for (int rank = 0; rank < numRanks; rank++) {
      selfRound.push_back({rank, rank});
    }
    schedule.push_back(selfRound);
  }

  // Display EnvVars
  if (Utils::RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
      if (!ev.outputToCsv) printf("[P2P Network Related]\n");
      ev.Print("NUM_NIC_SE",      numQueuePairs,  "Using %d queue pairs per Transfer", numQueuePairs);
      ev.Print("USE_REMOTE_READ", useRemoteRead,  "Using %s as executor", useRemoteRead ? "DST" : "SRC");
      ev.Print("OUTPUT_FORMAT",   showFullMatrix, "Printing results in %s format", showFullMatrix ? "full matrix" : "column");
      ev.Print("NIC_FILTER",      nicFilter,      "Selecting %d NICs", nicFilter.size());
      // TODO: Display filtered NICs?
      // TODO: More detailed info about mem type?
      ev.Print("SRC_MEM",         srcMemIdx,      "Source memory type");
      ev.Print("DST_MEM",         dstMemIdx,      "Destination memory type");
      ev.Print("FAST_EXE",        rr,             "Executing p2p node pairs in parallel");
      printf("\n");
    }
  }

  // TODO: validate env vars
  // TODO: assert same RR schedule

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  TransferBench::TestResults results;

  // Calculate total IB devices per rank
  // TODO: assert same # of NIC all ranks
  int const numNicsPerRank = nicIndices.size();
  int const numTotalNics = numNicsPerRank * numRanks;

  // Initialize output table
  Utils::Print("Unidirectional copy peak bandwidth GB/s (Using Nearest NIC RDMA)\n");

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
  //std::vector<double> minBandwidth;
  //std::vector<double> maxBandwidth;
  //std::vector<double> stdDev;

  // Transfer starts
  if (rr) {
    // Pre-allocate result vectors for all transfer combinations
    int totalTransfers = numRanks * numNicsPerRank * numRanks * numNicsPerRank;
    avgBandwidth.resize(totalTransfers);
    srcExes.resize(totalTransfers);
    dstExes.resize(totalTransfers);
    srcMems.resize(totalTransfers);
    dstMems.resize(totalTransfers);
    for (auto const& roundPairs : schedule) {
      for (int srcNicIdx = 0; srcNicIdx < numNicsPerRank; srcNicIdx++) {
        for (int dstNicIdx = 0; dstNicIdx < numNicsPerRank; dstNicIdx++) {
          std::vector<Transfer> transfers;
          for (auto const& pair : roundPairs) {
            Transfer transfer;
            int srcRank = pair.first;
            int dstRank = pair.second;

            int srcNic = nicIndices[srcNicIdx];
            int dstNic = nicIndices[dstNicIdx];

            // Determine which GPU memory/CPU NUMA to use based on NIC proximity and its info
            int srcMemIndex = GetClosestDeviceToNic(srcTypeActual, srcNic, srcRank);
            int dstMemIndex = GetClosestDeviceToNic(dstTypeActual, dstNic, dstRank);

            // TODO: error msg
            if (srcMemIndex == -1 || dstMemIndex == -1) ;
            transfer.numBytes = numBytesPerTransfer;
            transfer.srcs.push_back({srcTypeActual, srcMemIndex, srcRank});
            transfer.dsts.push_back({dstTypeActual, dstMemIndex, dstRank});
            transfer.exeDevice = {EXE_NIC, (useRemoteRead ? dstMemIndex : srcMemIndex), (useRemoteRead ? dstRank : srcRank)};
            transfer.exeSubIndex = (useRemoteRead ? srcMemIndex : dstMemIndex);
            transfer.numSubExecs = numQueuePairs;

            transfers.push_back(transfer);
          }

          if (!TransferBench::RunTransfers(cfg, transfers, results)) {
            for (auto const& err : results.errResults)
              Utils::Print("%s\n", err.errMsg.c_str());
            return 1;
          }

          for (size_t i = 0; i < results.tfrResults.size(); i++) {
            int srcRank = transfers[i].srcs[0].memRank;
            int dstRank = transfers[i].dsts[0].memRank;

            // Calculate index in table-rendering order: srcRank x srcNicIdx x dstRank x dstNicIdx
            int idx = srcRank * (numNicsPerRank * numRanks * numNicsPerRank)
                    + srcNicIdx * (numRanks * numNicsPerRank)
                    + dstRank * numNicsPerRank
                    + dstNicIdx;

            avgBandwidth[idx] = results.tfrResults[i].avgBandwidthGbPerSec;
            srcExes[idx] = TransferBench::GetExecutorName(results.tfrResults[i].exeDevice);
            dstExes[idx] = TransferBench::GetExecutorName(results.tfrResults[i].exeDstDevice);
            // TODO: add mem device info in transfer result?
            srcMems[idx] = transfers[i].srcs[0].memIndex;
            dstMems[idx] = transfers[i].dsts[0].memIndex;

          }
        }
      }
    }
  } else {
    // Loop over all possible src+NIC/dst+NIC pairs across all ranks and collect P2P results
    for (int srcRank = 0; srcRank < numRanks; srcRank++) {
      for (int srcNicIdx = 0; srcNicIdx < numNicsPerRank; srcNicIdx++) {
        for (int dstRank = 0; dstRank < numRanks; dstRank++) {
          for (int dstNicIdx = 0; dstNicIdx < numNicsPerRank; dstNicIdx++) {
            std::vector<Transfer> transfers(1);

            int srcNic = nicIndices[srcNicIdx];
            int dstNic = nicIndices[dstNicIdx];

            // Determine which GPU memory/CPU NUMA to use based on NIC proximity and its info
            int srcMemIndex = GetClosestDeviceToNic(srcTypeActual, srcNic, srcRank);
            int dstMemIndex = GetClosestDeviceToNic(dstTypeActual, dstNic, dstRank);

            // TODO: error msg
            if (srcMemIndex == -1 || dstMemIndex == -1) ;
            transfers[0].numBytes = numBytesPerTransfer;
            transfers[0].srcs.push_back({srcTypeActual, srcMemIndex, srcRank});
            transfers[0].dsts.push_back({dstTypeActual, dstMemIndex, dstRank});
            transfers[0].exeDevice = {EXE_NIC, (useRemoteRead ? dstMemIndex : srcMemIndex), (useRemoteRead ? dstRank : srcRank)};
            transfers[0].exeSubIndex = (useRemoteRead ? srcMemIndex : dstMemIndex);
            transfers[0].numSubExecs = numQueuePairs;

            if (!TransferBench::RunTransfers(cfg, transfers, results)) {
              for (auto const& err : results.errResults)
                Utils::Print("%s\n", err.errMsg.c_str());
              return 1;
            }
            avgBandwidth.push_back(results.tfrResults[0].avgBandwidthGbPerSec);
            srcExes.push_back(TransferBench::GetExecutorName(results.tfrResults[0].exeDevice));
            dstExes.push_back(TransferBench::GetExecutorName(results.tfrResults[0].exeDstDevice));

            srcMems.push_back(srcMemIndex);
            dstMems.push_back(dstMemIndex);
          }
        }
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

  for (int i = 0; i <= 11; i++) summaryTable.DrawRowBorder(i);
  for (int i = 0; i <= 6; i++) summaryTable.DrawColBorder(i);

  std::vector<size_t> idx(avgBandwidth.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](size_t i1, size_t i2) {return avgBandwidth[i1] > avgBandwidth[i2];});
  for (int i = 0; i < 10; i++) {
    int index = idx[i];
    int dstNicIdx = index % numNicsPerRank;
    index /= numNicsPerRank;

    int dstRank = index % numRanks;
    index /= numRanks;

    int srcNicIdx = index % numNicsPerRank;
    index /= numNicsPerRank;

    int srcRank = index;

    summaryTable.Set(1 + i, 1, " R%02d:%s ", srcRank, srcExes[idx[i]].c_str());
    summaryTable.Set(1 + i, 2, " R%02d:%s ", dstRank, dstExes[idx[i]].c_str());
    summaryTable.Set(1 + i, 0, " %.2f ", avgBandwidth[idx[i]]);

    index = idx[idx.size() - 1 - i];
    dstNicIdx = index % numNicsPerRank;
    index /= numNicsPerRank;

    dstRank = index % numRanks;
    index /= numRanks;

    srcNicIdx = index % numNicsPerRank;
    index /= numNicsPerRank;

    srcRank = index;

    summaryTable.Set(1 + i, 4, " R%02d:%s ", srcRank, srcExes[idx[idx.size() - 1 - i]].c_str());
    summaryTable.Set(1 + i, 5, " R%02d:%s ", dstRank, dstExes[idx[idx.size() - 1 - i]].c_str());
    summaryTable.Set(1 + i, 3, " %.2f ", avgBandwidth[idx[idx.size() - 1 - i]]);
  }
  summaryTable.PrintTable(ev.outputToCsv, ev.showBorders);

  return 0;
}



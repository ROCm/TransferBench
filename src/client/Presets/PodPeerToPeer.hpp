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

int PodPeerToPeerPreset(EnvVars&           ev,
                        size_t      const  numBytesPerTransfer,
                        std::string const  presetName)
{
  if (Utils::GetNumRankGroups() > 1) {
    Utils::Print("[ERROR] Pod p2p preset can only be run across ranks that are homogenous\n");
    Utils::Print("[ERROR] All ranks currently have to be under the same physical and virtual pod\n");
    Utils::Print("[ERROR] Run ./TransferBench without any args to display topology information\n");
    return 1;
  }
  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  // Collect env vars for this preset
  int useDmaCopy     = EnvVars::GetEnvVar("USE_GPU_DMA",     0);
  int gpuMemTypeIdx  = EnvVars::GetEnvVar("GPU_MEM_TYPE",    0);
  int numGpuDevices  = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int numGpuSubExecs = EnvVars::GetEnvVar("NUM_GPU_SE",      useDmaCopy ? 1 : TransferBench::GetNumSubExecutors({EXE_GPU_GFX, 0}));
  int p2pMode        = EnvVars::GetEnvVar("P2P_MODE",        0);
  int parallelLevel  = EnvVars::GetEnvVar("PARALLEL_LVL",    0);
  int useFineGrain   = EnvVars::GetEnvVar("USE_FINE_GRAIN",  -999); // Deprecated
  int useRemoteRead  = EnvVars::GetEnvVar("USE_REMOTE_READ", 0);
  int showFullMatrix = EnvVars::GetEnvVar("OUTPUT_FORMAT", 1);

  MemType gpuMemType = Utils::GetGpuMemType(gpuMemTypeIdx);

  // Display environment variables

  if (Utils::RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
      int outputToCsv = ev.outputToCsv;
      if (!outputToCsv) printf("[P2P Related]\n");
      ev.Print("GPU_MEM_TYPE"   , gpuMemTypeIdx,  "Using %s (%s)", Utils::GetGpuMemTypeStr(gpuMemTypeIdx).c_str(), Utils::GetAllGpuMemTypeStr().c_str());
      ev.Print("NUM_GPU_DEVICES", numGpuDevices,  "Using %d GPUs", numGpuDevices);
      ev.Print("NUM_GPU_SE",      numGpuSubExecs, "Using %d GPU subexecutors/CUs per Transfer", numGpuSubExecs);
      ev.Print("P2P_MODE",        p2pMode,        "Running %s transfers", p2pMode == 0 ? "Uni + Bi" :
                                                                          p2pMode == 1 ? "Unidirectional"
                                                                                       : "Bidirectional");
      ev.Print("PARALLEL_LVL",    parallelLevel,  "Executing p2p in parallel level %d (0: no parallel, 1: node pairs in parallel)", parallelLevel);
      ev.Print("USE_GPU_DMA",     useDmaCopy,     "Using GPU-%s as GPU executor", useDmaCopy ? "DMA" : "GFX");
      ev.Print("USE_REMOTE_READ", useRemoteRead,  "Using %s as executor", useRemoteRead ? "DST" : "SRC");
      printf("\n");
    }
  }

  // Check for deprecated env vars
  if (useFineGrain != -999) {
    Utils::Print("[ERROR] USE_FINE_GRAIN has been deprecated and replaced by CPU_MEM_TYPE and GPU_MEM_TYPE\n");
    return 1;
  }

  char const separator = ev.outputToCsv ? ',' : ' ';
  Utils::Print("Bytes Per Direction%c%lu\n", separator, numBytesPerTransfer);

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  TransferBench::TestResults results;

  Utils::RankPodMap& rankToPod = Utils::GetRankPodMap();
  if (rankToPod.empty()) {
    Utils::Print("[ERROR] No pods detected. Set FORCE_SINGLE_POD=1 to treat all ranks as a single pod.\n");
    return 1;
  }
  for (auto const& [pod, ranks] : rankToPod) {
    // Add all devices in a pod
    int n = ranks.size() * numGpuDevices;

    std::vector<MemDevice> devices(n);
    int idx = 0;
    for (int rank : ranks) {
      for (int devIdx = 0; devIdx < numGpuDevices; devIdx++) {
        devices[idx++] = {gpuMemType, devIdx, rank};
      }
    }

    // Build reverse map: (memRank, memIndex) -> device index
    std::map<std::pair<int,int>, int> deviceLookup;
    for (int i = 0; i < n; i++)
      deviceLookup[{devices[i].memRank, devices[i].memIndex}] = i;

    ExeType const gpuExeType = useDmaCopy ? EXE_GPU_DMA : EXE_GPU_GFX;

    for (int isBidirectional = 0; isBidirectional <= 1; isBidirectional++) {
      if ((p2pMode == 1 && isBidirectional == 1) ||
          (p2pMode == 2 && isBidirectional == 0)) continue;

      Utils::Print("%sdirectional copy peak bandwidth GB/s [%s read / %s write] (GPU-Executor: %s)\n",
                   isBidirectional ? "Bi" : "Uni",
                   useRemoteRead ? "Remote" : "Local",
                   useRemoteRead ? "Local" : "Remote",
                   useDmaCopy    ? "DMA"   : "GFX");

      std::vector<double> avgBandwidth(n * n, 0.0);

      // Build rounds of transfers; all transfers in a round run in parallel
      std::vector<std::vector<std::pair<MemDevice, MemDevice>>> rounds;

      if (parallelLevel == 0) {
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            if (isBidirectional && i == j) continue;
            std::vector<std::pair<MemDevice, MemDevice>> pairs;
            pairs.push_back({devices[i], devices[j]});
            if (isBidirectional)
              pairs.push_back({devices[j], devices[i]});
            rounds.push_back(std::move(pairs));
          }
        }
      } else {
        // parallelLevel == 1: node pairs run concurrently, one device pair at a time per node pair
        std::vector<std::vector<std::pair<int, int>>> nodePairSchedule;
        RoundRobinSchedule(nodePairSchedule, (int)ranks.size(), 1);

        for (auto const& roundNodePairs : nodePairSchedule) {
          for (int srcDev = 0; srcDev < numGpuDevices; srcDev++) {
            for (int dstDev = 0; dstDev < numGpuDevices; dstDev++) {
              std::vector<std::pair<MemDevice, MemDevice>> pairs;
              for (auto const& [rankIdxA, rankIdxB] : roundNodePairs) {
                int const rA = ranks[rankIdxA];
                int const rB = ranks[rankIdxB];
                if (isBidirectional && rA == rB && srcDev == dstDev) continue;
                pairs.push_back({{gpuMemType, srcDev, rA}, {gpuMemType, dstDev, rB}});
                if (isBidirectional)
                  pairs.push_back({{gpuMemType, dstDev, rB}, {gpuMemType, srcDev, rA}});
              }
              if (!pairs.empty())
                rounds.push_back(std::move(pairs));
            }
          }
        }
      }

      // Execute rounds and collect results
      for (auto const& round : rounds) {
        std::vector<TransferBench::Transfer> transfers;
        for (auto const& [src, dst] : round) {
          TransferBench::Transfer transfer;
          transfer.numBytes = numBytesPerTransfer;
          transfer.srcs.push_back(src);
          transfer.dsts.push_back(dst);
          transfer.exeDevice = { gpuExeType,
                                useRemoteRead ? (int32_t)dst.memIndex : (int32_t)src.memIndex,
                                useRemoteRead ? (int32_t)dst.memRank  : (int32_t)src.memRank };
          transfer.exeSubIndex = -1;
          transfer.numSubExecs = numGpuSubExecs;
          transfers.push_back(transfer);
        }
        if (!TransferBench::RunTransfers(cfg, transfers, results)) {
          for (auto const& err : results.errResults)
            Utils::Print("%s\n", err.errMsg.c_str());
          return 1;
        }

        for (size_t k = 0; k < round.size(); k++) {
          auto const& [src, dst] = round[k];
          int i = deviceLookup[{src.memRank, src.memIndex}];
          int j = deviceLookup[{dst.memRank, dst.memIndex}];
          avgBandwidth[i * n + j] = results.tfrResults[k].avgBandwidthGbPerSec;
        }
      }

      // Output results
      int const podNumRanks = ranks.size();
      int const rowsPerSrc = isBidirectional ? 3 : 1;
      int const rowStride = isBidirectional ? rowsPerSrc + 1 : rowsPerSrc;
      int const numRows = showFullMatrix ? 2 + n * rowStride - (isBidirectional ? 1 : 0)
                                         : 1 + n * n * rowsPerSrc;
      int const numCols = showFullMatrix ? 2 + n : (isBidirectional ? 6 : 5);
      int const precision = 2;
      Utils::TableHelper table(numRows, numCols, precision);

      table.DrawRowBorder(0);
      table.DrawColBorder(0);
      table.DrawColBorder(numCols);
      table.DrawRowBorder(numRows);

      if (showFullMatrix) {
        if (isBidirectional)
          table.Set(0, 0, " SRC\\DST ");
        else
          table.Set(0, 0, useRemoteRead ? " SRC\\DST+EXE " : " SRC+EXE\\DST ");
        table.DrawRowBorder(1);
        table.DrawColBorder(1);
        table.Set(1, 1, " Mem Device ");

        int colPrevRank = -1;
        for (int j = 0; j < n; j++) {
          int colIdx = 2 + j;
          int r = devices[j].memRank;
          if (r != colPrevRank) {
            table.DrawColBorder(colIdx);
            table.Set(0, colIdx, " Rank %02d ", r);
            colPrevRank = r;
          }
          table.Set(1, colIdx, " GPU %02d ", devices[j].memIndex);
        }

        int rowPrevRank = -1;
        for (int i = 0; i < n; i++) {
          int r = devices[i].memRank;
          int baseRow = 2 + i * rowStride;
          if (r != rowPrevRank) {
            table.DrawRowBorder(baseRow);
            table.Set(baseRow, 0, " Rank %02d ", r);
            rowPrevRank = r;
          }

          for (int dir = 0; dir < rowsPerSrc; dir++) {
            int rowIdx = baseRow + dir;
            if (isBidirectional) {
              char const* arrow = (dir == 0) ? " ->" : (dir == 1) ? "<- " : "<->";
              table.Set(rowIdx, 1, " GPU %02d %s ", devices[i].memIndex, arrow);
            } else {
              table.Set(rowIdx, 1, " GPU %02d ", devices[i].memIndex);
            }

            for (int j = 0; j < n; j++) {
              double fwd = avgBandwidth[i * n + j];
              double rev = avgBandwidth[j * n + i];
              double val = (dir == 0) ? fwd : (dir == 1) ? rev : fwd + rev;
              if (val == 0.0)
                table.Set(rowIdx, 2 + j, " N/A ");
              else
                table.Set(rowIdx, 2 + j, " %.2f ", val);
            }
          }
        }
      } else {
        table.Set(0, 0, " SRC Rank ");
        table.Set(0, 1, " SRC MEM ");
        if (isBidirectional) {
          table.Set(0, 2, " Dir ");
          table.Set(0, 3, " DST Rank ");
          table.Set(0, 4, " DST MEM ");
          table.Set(0, 5, " bw (GB/s) ");
          table.DrawColBorder(3);
          table.DrawColBorder(5);
        } else {
          table.Set(0, 2, " DST Rank ");
          table.Set(0, 3, " DST MEM ");
          table.Set(0, 4, " bw (GB/s) ");
          table.DrawColBorder(2);
          table.DrawColBorder(4);
        }
        int rowIdx = 1;
        for (int i = 0; i < n; i++) {
          table.DrawRowBorder(rowIdx);
          for (int j = 0; j < n; j++) {
            for (int dir = 0; dir < rowsPerSrc; dir++) {
              double fwd = avgBandwidth[i * n + j];
              double rev = avgBandwidth[j * n + i];
              double val = (dir == 0) ? fwd : (dir == 1) ? rev : fwd + rev;
              if (isBidirectional) {
                char const* arrow = (dir == 0) ? " -> " : (dir == 1) ? " <- " : " <-> ";
                table.Set(rowIdx, 0, " Rank %02d ", devices[i].memRank);
                table.Set(rowIdx, 1, " GPU %02d ", devices[i].memIndex);
                table.Set(rowIdx, 2, arrow);
                table.Set(rowIdx, 3, " Rank %02d ", devices[j].memRank);
                table.Set(rowIdx, 4, " GPU %02d ", devices[j].memIndex);
                if (val == 0.0)
                  table.Set(rowIdx, 5, " N/A ");
                else
                  table.Set(rowIdx, 5, " %.2f ", val);
              } else {
                table.Set(rowIdx, 0, " Rank %02d ", devices[i].memRank);
                table.Set(rowIdx, 1, " GPU %02d ", devices[i].memIndex);
                table.Set(rowIdx, 2, " Rank %02d ", devices[j].memRank);
                table.Set(rowIdx, 3, " GPU %02d ", devices[j].memIndex);
                if (val == 0.0)
                  table.Set(rowIdx, 4, " N/A ");
                else
                  table.Set(rowIdx, 4, " %.2f ", val);
              }
              rowIdx++;
            }
          }
        }
      }
      table.PrintTable(ev.outputToCsv, ev.showBorders);
    }

  }
  return 0;
  
}

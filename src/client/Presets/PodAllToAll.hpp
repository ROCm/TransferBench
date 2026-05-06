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

  // When pod detection fails (e.g. amd-smi unavailable), the map is empty
  if (Utils::GetRankPerPodMap().empty()) {
    Utils::Print("[ERROR] No pods detected. Set TB_FORCE_SINGLE_POD=1 to treat all ranks as a single pod.\n");
    return ERR_FATAL;
  }
  // Restrict to single-pod runs; multi-pod support is not yet implemented
  if (Utils::GetRankPerPodMap().size() != 1) {
    Utils::Print("[ERROR] PodAllToAll preset currently requires all ranks to be in a single pod. Set TB_FORCE_SINGLE_POD=1 to treat all ranks as a single pod.\n");
    return ERR_FATAL;
  }

  // Collect env vars for this preset
  int a2aLocal      = EnvVars::GetEnvVar("A2A_LOCAL"      , 0);
  int memTypeIdx    = EnvVars::GetEnvVar("MEM_TYPE"       , 0);
  int numGpus       = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int numQueuePairs = EnvVars::GetEnvVar("NUM_QUEUE_PAIRS", 0);
  int numSubExecs   = EnvVars::GetEnvVar("NUM_SUB_EXEC"   , 8);
  int showDetails   = EnvVars::GetEnvVar("SHOW_DETAILS"   , 0);
  int useDmaExec    = EnvVars::GetEnvVar("USE_DMA_EXEC"   , 0);
  int useRemoteRead = EnvVars::GetEnvVar("USE_REMOTE_READ", 0);
  int groupStride   = EnvVars::GetEnvVar("GROUP_STRIDE"   , 1);
  int numGroups     = EnvVars::GetEnvVar("NUM_GROUPS"     , 1);

  // Check that all ranks have at least the number of GPUs requested
  // Warn if NIC configuration is slightly different from one another
  int numNics  = TransferBench::GetNumExecutors(EXE_NIC, 0);
  bool nicDifference = false;
  for (int rank = 0; rank < numRanks; rank++) {
    if (numGpus > TransferBench::GetNumExecutors(EXE_GPU_GFX, rank)) {
      Utils::Print("[ERROR] All-to-All preset requires each rank to have the same number of GPUs\n");
      return ERR_FATAL;
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
      return ERR_FATAL;
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
      ev.Print("GROUP_STRIDE"   , groupStride  , "Stride permutation on device list before splitting into groups");
      ev.Print("NUM_GROUPS"     , numGroups    , "Splitting each pod into %d group(s) for a2a", numGroups);
      printf("\n");
    }
  }
  // Validate env vars
  if (numGpus <= 0 || numGpus > numDetectedGpus) {
    Utils::Print("[ERROR] Cannot use %d GPUs.  Detected %d GPUs\n", numGpus, numDetectedGpus);
    return ERR_FATAL;
  }
  if (useDmaExec && (numSrcs != 1 || numDsts != 1)) {
    Utils::Print("[ERROR] DMA execution can only be used for copies (A2A_MODE=0)\n");
    return ERR_FATAL;
  }

  if (numGroups < 1) {
    Utils::Print("[ERROR] NUM_GROUPS must be >= 1 (got %d)\n", numGroups);
    return ERR_FATAL;
  }

  Utils::Print("GPU-%s IntraPod All-To-All benchmark:\n", useDmaExec ? "DMA" : "GFX");
  Utils::Print("==============================\n");
  Utils::Print("[%lu bytes per Transfer] [%s:%d] [%d Read(s) %d Write(s)] [MemType:%s] [NIC QueuePairs:%d] [#Ranks:%d]\n",
               numBytesPerTransfer, useDmaExec ? "DMA" : "GFX", numSubExecs, numSrcs, numDsts,
               devMemTypeStr.c_str(), numQueuePairs, numRanks);

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  ExeType exeType = useDmaExec ? EXE_GPU_DMA : EXE_GPU_GFX;

  Utils::RankPerPodMap& rankToPod = Utils::GetRankPerPodMap();
  for (auto const& [pod, ranks] : rankToPod) {
    int n = ranks.size() * numGpus;
    if (n % numGroups) {
      Utils::Print("[ERROR] NUM_GROUPS (%d) must divide pod device count (%d = %d ranks * %d gpus/rank)\n",
                   numGroups, n, (int)ranks.size(), numGpus);
      return ERR_FATAL;
    }
    int groupSize = n / numGroups;
    std::vector<MemDevice> devices(n);
    std::vector<int> indices(n);
    for (int k = 0; k < n; k++) indices[k] = k;
    Utils::StrideGenerate(indices, groupStride);
    int idx = 0;
    for (int rank : ranks) {
      for (int devIdx = 0; devIdx < numGpus; devIdx++) {
        devices[indices[idx++]] = {memType, devIdx, rank};
      }
    }

    // Build transfers for every group, then run once per pod so all groups share the same
    // timed iterations (traffic across groups is concurrent within RunTransfers).
    std::vector<Transfer> podTransfers;
    std::vector<size_t> groupTransferBase(numGroups);
    std::vector<std::vector<std::vector<int>>> groupReIndexes(numGroups);

    for (int group = 0; group < numGroups; group++) {
      groupTransferBase[group] = podTransfers.size();
      groupReIndexes[group].assign(groupSize, std::vector<int>(groupSize, -1));
      std::vector<std::vector<int>>& groupReIndex = groupReIndexes[group];

      for (int i = group * groupSize; i < (group + 1) * groupSize; i++) {
        for (int j = group * groupSize; j < (group + 1) * groupSize; j++) {
          if (i == j) {
            if (!a2aLocal) continue;
          }
          TransferBench::Transfer transfer;
          transfer.numBytes = numBytesPerTransfer;
          for (int x = 0; x < numSrcs; x++) transfer.srcs.push_back(devices[i]);
          // First dst is the remote peer (devices[j]); extra dsts are local (devices[i]) to stress-test src bandwidth
          if (numDsts) transfer.dsts.push_back(devices[j]);
          for (int x = 1; x < numDsts; x++) transfer.dsts.push_back(devices[i]);
          transfer.exeDevice = {exeType,
                               (int32_t)(useRemoteRead ? devices[j].memIndex : devices[i].memIndex),
                               (int32_t)(useRemoteRead ? devices[j].memRank : devices[i].memRank)};
          transfer.exeSubIndex = -1;
          transfer.numSubExecs = numSubExecs;
          int const localI = i - group * groupSize;
          int const localJ = j - group * groupSize;
          groupReIndex[localI][localJ] =
              (int)(podTransfers.size() - groupTransferBase[group]);
          podTransfers.push_back(transfer);
        }

        // NIC transfers are supplementary bandwidth; excluded from groupReIndex and bandwidth table
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
          podTransfers.push_back(transfer);
        }
      }
    }

    if (Utils::RankDoesOutput()) {
      for (int g = 0; g < numGroups; g++) {
        int const gb = g * groupSize;
        Utils::Print("A2A group %d:", g);
        std::vector<int> ord(groupSize);
        for (int i = 0; i < groupSize; i++) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) {
          MemDevice const& da = devices[gb + a];
          MemDevice const& db = devices[gb + b];
          if (da.memRank != db.memRank) return da.memRank < db.memRank;
          return da.memIndex < db.memIndex;
        });
        for (size_t si = 0; si < ord.size(); si++) {
          MemDevice const& d = devices[gb + ord[si]];
          Utils::Print("%s R%d:G%d", si ? "," : "", d.memRank, d.memIndex);
        }
        Utils::Print("\n");
      }
    }

    TransferBench::TestResults results;
    if (!TransferBench::RunTransfers(cfg, podTransfers, results)) {
      for (auto const& err : results.errResults)
        Utils::Print("%s\n", err.errMsg.c_str());
      return 1;
    }
    if (showDetails) {
      if (Utils::RankDoesOutput())
        Utils::Print("\n--- Pod AllToAll (all %d groups concurrent) ---\n", numGroups);
      Utils::PrintResults(ev, 1, podTransfers, results);
      Utils::Print("\n");
    }

    for (int group = 0; group < numGroups; group++) {
      std::vector<std::vector<int>> const& groupReIndex = groupReIndexes[group];
      size_t const tfrBase = groupTransferBase[group];

      // Per-group bandwidth table
      std::vector<std::vector<double>> groupBw(groupSize, std::vector<double>(groupSize, -1.0));
      for (int localI = 0; localI < groupSize; localI++) {
        for (int localJ = 0; localJ < groupSize; localJ++) {
          int const k = groupReIndex[localI][localJ];
          if (k >= 0)
            groupBw[localI][localJ] = results.tfrResults[tfrBase + k].avgBandwidthGbPerSec;
        }
      }
      if (Utils::RankDoesOutput()) {
        Utils::Print("\n--- Pod AllToAll Group %d ---\n", group);
        int const groupBase = group * groupSize;

        // Display order: group devices by MPI rank, then GPU index (stride only affects execution order).
        std::vector<int> order(groupSize);
        for (int i = 0; i < groupSize; i++) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
          MemDevice const& da = devices[groupBase + a];
          MemDevice const& db = devices[groupBase + b];
          if (da.memRank != db.memRank) return da.memRank < db.memRank;
          return da.memIndex < db.memIndex;
        });
        std::vector<int> colRanks;
        for (int slot : order) {
          int const r = devices[groupBase + slot].memRank;
          if (colRanks.empty() || colRanks.back() != r) colRanks.push_back(r);
        }
        std::vector<std::vector<int>> localsPerCol;
        localsPerCol.reserve(colRanks.size());
        for (int dr : colRanks) {
          std::vector<int> loc;
          for (int li = 0; li < groupSize; li++) {
            if (devices[groupBase + li].memRank == dr) loc.push_back(li);
          }
          std::sort(loc.begin(), loc.end(), [&](int a, int b) {
            return devices[groupBase + a].memIndex < devices[groupBase + b].memIndex;
          });
          localsPerCol.push_back(std::move(loc));
        }

        // Two trailing scalar columns (STotal, Actual) and a trailing RTotal row
        // matching the a2a/nica2a layouts.
        int const sTotalCol = 2 + (int)colRanks.size();
        int const actualCol = sTotalCol + 1;
        int const rTotalRow = 2 + groupSize;
        int const numRows = 2 + groupSize + 1;
        int const numCols = 2 + (int)colRanks.size() + 2;
        int const precision = 2;
        Utils::TableHelper table(numRows, numCols, precision);
        table.DrawRowBorder(0);
        table.DrawColBorder(0);
        table.DrawColBorder(numCols);
        table.DrawRowBorder(numRows);
        table.DrawRowBorder(rTotalRow);
        table.Set(0, 0, useRemoteRead ? " SRC\\DST+EXE " : " SRC+EXE\\DST ");
        table.DrawRowBorder(1);
        table.DrawColBorder(1);
        table.Set(1, 1, " Mem Device ");

        for (size_t c = 0; c < colRanks.size(); c++) {
          int const colIdx = 2 + (int)c;
          table.DrawColBorder(colIdx);
          table.Set(0, colIdx, " Rank %02d ", colRanks[c]);
          std::string gpuHdr;
          for (int li : localsPerCol[c]) {
            char t[24];
            snprintf(t, sizeof(t), "  GPU %02d", devices[groupBase + li].memIndex);
            gpuHdr += t;
          }
          gpuHdr += " ";
          table.Set(1, colIdx, "%s", gpuHdr.c_str());
          table.SetColAlignment((int)c + 2, Utils::TableHelper::ALIGN_LEFT);
        }

        // STotal / Actual column headers (centered, single-cell). Row 1 stays
        // blank for these cols since they are scalar columns, not GPU groups.
        table.DrawColBorder(sTotalCol);
        table.DrawColBorder(actualCol);
        table.Set(0, sTotalCol, " STotal ");
        table.SetCellAlignment(0, sTotalCol, Utils::TableHelper::ALIGN_CENTER);
        table.Set(0, actualCol, " Actual ");
        table.SetCellAlignment(0, actualCol, Utils::TableHelper::ALIGN_CENTER);
        table.Set(rTotalRow, 1, " RTotal ");
        table.SetCellAlignment(rTotalRow, 1, Utils::TableHelper::ALIGN_CENTER);

        // Per-(rank-col, dst-GPU-within-rank) running sums for the RTotal row.
        std::vector<std::vector<double>> colTotal(colRanks.size());
        for (size_t c = 0; c < colRanks.size(); c++)
          colTotal[c].assign(localsPerCol[c].size(), 0.0);
        double sTotalGrand = 0.0;
        double actualGrand = 0.0;

        int rowPrevRank = -1;
        for (int disp = 0; disp < groupSize; disp++) {
          int const localI = order[disp];
          int const rowIdx = 2 + disp;
          int const r = devices[groupBase + localI].memRank;
          if (r != rowPrevRank) {
            table.DrawRowBorder(rowIdx);
            table.Set(rowIdx, 0, " Rank %02d ", r);
            rowPrevRank = r;
          } else {
            table.Set(rowIdx, 0, " ");
          }
          table.Set(rowIdx, 1, " GPU %02d ", devices[groupBase + localI].memIndex);

          double rowSum   = 0.0;
          double rowMinBw = std::numeric_limits<double>::max();
          int    rowCount = 0;
          for (size_t c = 0; c < colRanks.size(); c++) {
            std::string cell;
            for (size_t k = 0; k < localsPerCol[c].size(); k++) {
              int const localJ = localsPerCol[c][k];
              char t[16];
              if (groupBw[localI][localJ] >= 0) {
                double const bw = groupBw[localI][localJ];
                snprintf(t, sizeof(t), " %7.2f", bw);
                rowSum         += bw;
                colTotal[c][k] += bw;
                rowMinBw        = std::min(rowMinBw, bw);
                rowCount++;
              } else {
                snprintf(t, sizeof(t), " %7s", "N/A");
              }
              cell += t;
            }
            cell += " ";
            int const colIdx = 2 + (int)c;
            table.Set(rowIdx, colIdx, "%s", cell.c_str());
            table.SetCellAlignment(rowIdx, colIdx, Utils::TableHelper::ALIGN_LEFT);
          }
          double const rowActual = (rowCount > 0) ? rowCount * rowMinBw : 0.0;
          table.Set(rowIdx, sTotalCol, " %.2f ", rowSum);
          table.Set(rowIdx, actualCol, " %.2f ", rowActual);
          sTotalGrand += rowSum;
          actualGrand += rowActual;
        }

        // RTotal row: per-dst-GPU column sums (packed per rank-col), plus grand
        // totals under STotal / Actual.
        for (size_t c = 0; c < colRanks.size(); c++) {
          std::string cell;
          for (size_t k = 0; k < localsPerCol[c].size(); k++) {
            char t[16];
            snprintf(t, sizeof(t), " %7.2f", colTotal[c][k]);
            cell += t;
          }
          cell += " ";
          int const colIdx = 2 + (int)c;
          table.Set(rTotalRow, colIdx, "%s", cell.c_str());
          table.SetCellAlignment(rTotalRow, colIdx, Utils::TableHelper::ALIGN_LEFT);
        }
        table.Set(rTotalRow, sTotalCol, " %.2f ", sTotalGrand);
        table.Set(rTotalRow, actualCol, " %.2f ", actualGrand);

        table.PrintTable(ev.outputToCsv, ev.showBorders);
      }
    }
  }

  if (!Utils::RankDoesOutput()) return 0;

  if (Utils::HasDuplicateHostname()) {
    printf("[WARN] It is recommended to run TransferBench with one rank per host to avoid potential aliasing of executors\n");
  }

  return ERR_NONE;
}

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

#include <cstring>
#include <limits>
#include <numeric>

int NicAllToAllPreset(EnvVars&                    ev,
                      size_t      const           numBytesPerTransfer,
                      std::string const           presetName,
                      [[maybe_unused]] bool const bytesSpecified)
{
  // Check for single homogenous group
  if (Utils::GetNumRankGroups() > 1) {
    Utils::Print("[ERROR] NIC all-to-all preset can only be run across ranks that are homogenous\n");
    Utils::Print("[ERROR] Run ./TransferBench without any args to display topology information\n");
    Utils::Print("[ERROR] TB_NIC_FILTER may also be used to limit NIC visibility to scale-out NICs\n");
    return ERR_FATAL;
  }

  int numRanks = TransferBench::GetNumRanks();
  int numNicsPerRank = TransferBench::GetNumExecutors(EXE_NIC);
  if (numNicsPerRank == 0) {
    Utils::Print("[ERROR] No NIC detected. This preset requires NIC executors.\n");
    return ERR_FATAL;
  }

  int useCpuMem = EnvVars::GetEnvVar("USE_CPU_MEM", 0);
  // Device count from topology: GFX executors, or CPU executors when USE_CPU_MEM (same pattern as NicRings).
  int numMemDevices = TransferBench::GetNumExecutors(useCpuMem ? EXE_CPU : EXE_GPU_GFX);
  if (numMemDevices == 0) {
    Utils::Print("[ERROR] No %s executors detected for NIC all-to-all.\n", useCpuMem ? "CPU" : "GPU GFX");
    return ERR_FATAL;
  }

  // Total NICs across all ranks (rank-major id: nicId = rank * numNicsPerRank + nic).
  int const N = numRanks * numNicsPerRank;

  int planeStride   = EnvVars::GetEnvVar("PLANE_STRIDE"   , 1);
  int numPlanes     = EnvVars::GetEnvVar("NUM_PLANES"     , 1);
  int groupStride   = EnvVars::GetEnvVar("GROUP_STRIDE"   , 1);
  int numGroups     = EnvVars::GetEnvVar("NUM_GROUPS"     , 1);
  int numQueuePairs = EnvVars::GetEnvVar("NUM_QUEUE_PAIRS", 1);
  int showDetails   = EnvVars::GetEnvVar("SHOW_DETAILS"   , 0);
  int useRdmaRead   = EnvVars::GetEnvVar("USE_RDMA_READ"  , 0);
  int memTypeIdx    = EnvVars::GetEnvVar("MEM_TYPE"       , 0);
  int a2aLocal      = EnvVars::GetEnvVar("A2A_LOCAL"      , 0);

  if (numPlanes < 1) {
    Utils::Print("[ERROR] NUM_PLANES must be >= 1 (got %d)\n", numPlanes);
    return ERR_FATAL;
  }
  if (N % numPlanes) {
    Utils::Print("[ERROR] NUM_PLANES (%d) must divide total NICs (%d = %d ranks * %d nics/rank).\n",
                 numPlanes, N, numRanks, numNicsPerRank);
    return ERR_FATAL;
  }
  int const planeSize = N / numPlanes;

  // NUM_GROUPS = groups per plane. Default 1 -> one group per plane (full a2a within each plane).
  if (numGroups < 1) {
    Utils::Print("[ERROR] NUM_GROUPS must be >= 1 (got %d)\n", numGroups);
    return ERR_FATAL;
  }
  if (planeSize % numGroups) {
    Utils::Print("[ERROR] NUM_GROUPS (%d) must divide plane size (%d = %d total NICs / %d planes).\n",
                 numGroups, planeSize, N, numPlanes);
    return ERR_FATAL;
  }
  int const groupSize = planeSize / numGroups;

  if (numQueuePairs < 1) {
    Utils::Print("[ERROR] NUM_QUEUE_PAIRS must be >= 1 (got %d)\n", numQueuePairs);
    return ERR_FATAL;
  }

  MemType memType        = Utils::GetMemType(memTypeIdx, useCpuMem);
  std::string memTypeStr = Utils::GetMemTypeStr(memTypeIdx, useCpuMem);

  if (Utils::RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
      if (!ev.outputToCsv) printf("[NIC A2A Related]\n");
      ev.Print("USE_CPU_MEM"         , useCpuMem     , "Using closest %s memory", useCpuMem ? "CPU" : "GPU");
      ev.Print("MEM_TYPE"            , memTypeIdx    , "Using %s memory (%s)", memTypeStr.c_str(), Utils::GetAllMemTypeStr(useCpuMem).c_str());
      ev.Print("PLANE_STRIDE"        , planeStride   , "Stride permutation on global NIC list before splitting into planes");
      ev.Print("NUM_PLANES"         , numPlanes    , "Splitting %d total NICs into %d plane(s) of %d NICs each", N, numPlanes, planeSize);
      ev.Print("GROUP_STRIDE"        , groupStride   , "Stride permutation within each plane before splitting into groups");
      ev.Print("NUM_GROUPS"          , numGroups     , "Splitting each plane into %d group(s) of %d NICs each", numGroups, groupSize);
      ev.Print("A2A_LOCAL"           , a2aLocal      , "%s self NIC endpoint transfers", a2aLocal ? "Include" : "Exclude");
      ev.Print("NUM_QUEUE_PAIRS"     , numQueuePairs , "Using %d queue pairs for NIC transfers", numQueuePairs);
      ev.Print("SHOW_DETAILS"        , showDetails   , "%s full Test details", showDetails ? "Showing" : "Hiding");
      ev.Print("USE_RDMA_READ"       , useRdmaRead   , "Performing RDMA %s", useRdmaRead ? "reads" : "writes");
      printf("\n");
    }
  }

  // For each rank/NIC, closest memory device (GPU or CPU NUMA) — several NICs may share the same device (same subgroup).
  std::vector<std::vector<int>> nicToMem(numRanks, std::vector<int>(numNicsPerRank, -1));
  for (int rank = 0; rank < numRanks; rank++) {
    for (int nic = 0; nic < numNicsPerRank; nic++) {
      int memIdx = useCpuMem ? TransferBench::GetClosestCpuNumaToNic(nic, rank)
                             : TransferBench::GetClosestGpuToNic(nic, rank);
      if (memIdx < 0) {
        Utils::Print("[ERROR] Failed to identify closest %s for Rank %d NIC %d\n",
                     useCpuMem ? "CPU NUMA node" : "GPU", rank, nic);
        return ERR_FATAL;
      }
      if (memIdx >= numMemDevices) {
        Utils::Print("[ERROR] Closest %s index %d for Rank %d NIC %d is out of range [0,%d)\n",
                     useCpuMem ? "CPU" : "GPU", memIdx, rank, nic, numMemDevices);
        return ERR_FATAL;
      }
      nicToMem[rank][nic] = memIdx;
    }
  }

  // Build planes: take the rank-major NIC list [0..N), permute by PLANE_STRIDE, chunk consecutively.
  // Within each plane: permute by GROUP_STRIDE, chunk consecutively into groups.
  // Each group runs an internal all-to-all; all groups (across all planes) run concurrently.
  std::vector<int> nicList(N);
  std::iota(nicList.begin(), nicList.end(), 0);
  Utils::StrideGenerate(nicList, planeStride);

  struct GroupInfo {
    int              planeIdx;
    int              groupIdx;
    std::vector<int> memberNicIds;   // global NIC ids in this group (post-stride order)
    size_t           transferStart;  // first index into `transfers`
    size_t           transferEnd;    // one past last index into `transfers`
  };
  std::vector<std::vector<int>> planeMembers(numPlanes);
  std::vector<GroupInfo>        allGroups;
  allGroups.reserve((size_t)numPlanes * numGroups);

  std::vector<Transfer> transfers;

  for (int p = 0; p < numPlanes; p++) {
    std::vector<int> planeNics(nicList.begin() + (size_t)p * planeSize,
                                nicList.begin() + (size_t)(p + 1) * planeSize);
    planeMembers[p] = planeNics;  // pre-group-stride membership for display
    Utils::StrideGenerate(planeNics, groupStride);

    for (int g = 0; g < numGroups; g++) {
      GroupInfo gi;
      gi.planeIdx = p;
      gi.groupIdx = g;
      gi.memberNicIds.assign(planeNics.begin() + (size_t)g * groupSize,
                             planeNics.begin() + (size_t)(g + 1) * groupSize);
      gi.transferStart = transfers.size();

      for (int srcId : gi.memberNicIds) {
        int const srcRank = srcId / numNicsPerRank;
        int const srcNic  = srcId % numNicsPerRank;
        int const srcMem  = nicToMem[srcRank][srcNic];
        for (int dstId : gi.memberNicIds) {
          int const dstRank = dstId / numNicsPerRank;
          int const dstNic  = dstId % numNicsPerRank;
          if (!a2aLocal && srcId == dstId) continue;

          int const dstMem = nicToMem[dstRank][dstNic];

          TransferBench::Transfer transfer;
          transfer.srcs.push_back({memType, srcMem, srcRank});
          transfer.dsts.push_back({memType, dstMem, dstRank});
          transfer.exeDevice   = {EXE_NIC, useRdmaRead ? dstNic : srcNic, useRdmaRead ? dstRank : srcRank};
          transfer.exeSubIndex = useRdmaRead ? srcNic : dstNic;
          transfer.numSubExecs = numQueuePairs;
          transfer.numBytes    = numBytesPerTransfer;
          transfers.push_back(transfer);
        }
      }
      gi.transferEnd = transfers.size();
      allGroups.push_back(std::move(gi));
    }
  }

  Utils::Print("NIC All-To-All benchmark\n");
  Utils::Print("========================\n");
  Utils::Print("[%lu bytes per Transfer] [Total Transfers: %lu] [MemType:%s] [NIC QueuePairs:%d] [#Ranks:%d]\n",
               numBytesPerTransfer, transfers.size(), memTypeStr.c_str(), numQueuePairs, numRanks);
  Utils::Print("Running %d parallel a2a group(s) each of %d devices.  All numbers in GB/s:\n", numGroups, groupSize);
  Utils::Print("%d total NICs (rank-major) split into %d plane(s) of %d NICs (PLANE_STRIDE=%d).\n",
               N, numPlanes, planeSize, planeStride);
  Utils::Print("Each plane split into %d group(s) of %d NICs (GROUP_STRIDE=%d).\n",
               numGroups, groupSize, groupStride);
  

  if (transfers.empty()) {
    Utils::Print("[WARN] No transfers were generated for this preset.\n");
    return 0;
  }

  // Print the plane / group breakdown up-front (before running) so the user can
  // see the rank-by-rank NIC layout that's about to be tested.
  Utils::Print("[Plane / Group breakdown]\n");
  {
    size_t groupCursor = 0;
    for (int p = 0; p < numPlanes; p++) {
      Utils::Print("Plane %02d (%d NICs):\n", p, (int)planeMembers[p].size());
      for (int g = 0; g < numGroups; g++, groupCursor++) {
        auto const& gi = allGroups[groupCursor];
        Utils::Print("  Group %02d (%d NICs): -> %lu transfers\n",
                     g, (int)gi.memberNicIds.size(),
                     gi.transferEnd - gi.transferStart);

        std::map<int, std::vector<int>> ranksToLocals;
        for (int id : gi.memberNicIds)
          ranksToLocals[id / numNicsPerRank].push_back(id % numNicsPerRank);
        for (auto& kv : ranksToLocals)
          std::sort(kv.second.begin(), kv.second.end());

        for (auto const& [rank, locals] : ranksToLocals) {
          std::string names;
          for (size_t k = 0; k < locals.size(); k++) {
            if (k) names += ", ";
            names += TransferBench::GetExecutorName({EXE_NIC, locals[k], rank});
          }
          Utils::Print("    Rank %02d: %s\n", rank, names.c_str());
        }
      }
    }
  }
  Utils::Print("\n");

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  TransferBench::TestResults results;
  if (!TransferBench::RunTransfers(cfg, transfers, results)) {
    for (auto const& err : results.errResults)
      Utils::Print("%s\n", err.errMsg.c_str());
    return ERR_FATAL;
  } else if (showDetails) {
    Utils::PrintResults(ev, 1, transfers, results);
    Utils::Print("\n");
  }

  if (!Utils::RankDoesOutput()) return 0;

  // Per-group SRC->DST bandwidth matrix (one table per group), styled to match
  // the PodAllToAll preset: outer columns/rows are ranks, with each rank's NICs
  // packed as fixed-width sub-cells inside one cell.
  size_t groupCursor = 0;
  for (int p = 0; p < numPlanes; p++) {
    for (int g = 0; g < numGroups; g++, groupCursor++) {
      auto const& gi = allGroups[groupCursor];

      // Rebuild (srcId,dstId) -> transfer index lookup using the same iteration
      // order used when transfers were generated above.
      std::map<std::pair<int,int>, size_t> pairIdx;
      {
        size_t idx = gi.transferStart;
        for (int srcId : gi.memberNicIds) {
          for (int dstId : gi.memberNicIds) {
            if (!a2aLocal && srcId == dstId) continue;
            pairIdx[std::make_pair(srcId, dstId)] = idx++;
          }
        }
      }

      // Group the member NICs by rank, sorted by local NIC index within each rank.
      std::map<int, std::vector<int>> ranksToLocals;
      for (int id : gi.memberNicIds) {
        ranksToLocals[id / numNicsPerRank].push_back(id % numNicsPerRank);
      }
      std::vector<int> rankOrder;
      rankOrder.reserve(ranksToLocals.size());
      for (auto& kv : ranksToLocals) {
        std::sort(kv.second.begin(), kv.second.end());
        rankOrder.push_back(kv.first);
      }

      int const M = (int)gi.memberNicIds.size();
      // Header layout: row 0 (and col 0) is logically a single "outer" row/col
      // but rendered as two TableHelper rows/cols with no border between them so
      // the rank label and the GPU/CPU backing-device sub-labels can sit
      // alongside each other.
      //   Rows 0,1 / Cols 0,1: doubled outer header block
      //     - Row 0: " Rank XX " in the first NIC col of each rank (others blank)
      //     - Row 1: " GPU XX " (or CPU) per src NIC col
      //     - Col 0: " Rank XX " on first row of each rank's group
      //     - Col 1: " GPU XX " (or CPU) per src NIC row
      //   Row 2 / Col 2: NIC sub-labels (the original "row 1 / col 1"), with
      //                  the SRC+EXE\DST corner at (2,2).
      //   Row 3+ / Col 3+: one cell per (src NIC, dst NIC) pair, with NIC cols
      //                    grouped by rank using selective column borders.
      //   Trailing 2 cols: STotal (per-src-row sum) and Actual (transferCount *
      //                    rowMinBw); trailing row: RTotal (per-dst-col sum).
      int const sTotalCol  = 3 + M;
      int const actualCol  = 3 + M + 1;
      int const rTotalRow  = 3 + M;
      int const numTblRows = 3 + M + 1;
      int const numTblCols = 3 + M + 2;
      int const precision  = 2;
      Utils::TableHelper tbl(numTblRows, numTblCols, precision);

      // Build the destination-NIC ordering used for data columns: same as the
      // source-NIC ordering (group by rank, sorted local NIC index within rank).
      // Also remember where each rank-group of columns starts so we can place
      // the row-0 "Rank XX" headers and rank-boundary col borders in the right
      // places.
      std::vector<int> dstNicIds;
      dstNicIds.reserve(M);
      std::vector<int> rankColStart;
      rankColStart.reserve(rankOrder.size());
      {
        int colCursor = 3;
        for (int rank : rankOrder) {
          rankColStart.push_back(colCursor);
          for (int nicLocal : ranksToLocals[rank]) {
            dstNicIds.push_back(rank * numNicsPerRank + nicLocal);
            colCursor++;
          }
        }
      }

      tbl.DrawRowBorder(0);
      tbl.DrawRowBorder(2);
      tbl.DrawRowBorder(3);
      tbl.DrawRowBorder(rTotalRow);
      tbl.DrawRowBorder(numTblRows);
      tbl.DrawColBorder(0);
      tbl.DrawColBorder(2);
      // Col borders only at rank-group boundaries (and at the right edge).
      // Col 3 is the start of the first rank-group, which also serves as the
      // separator between the NIC sub-label column and the data section.
      // The last rank boundary at col (3+M) doubles as the separator before
      // the trailing STotal column.
      {
        int colCursor = 3;
        tbl.DrawColBorder(colCursor);
        for (int rank : rankOrder) {
          colCursor += (int)ranksToLocals[rank].size();
          tbl.DrawColBorder(colCursor);
        }
      }
      tbl.DrawColBorder(actualCol);
      tbl.DrawColBorder(numTblCols);

      tbl.Set(0, 0, " Mem Device ");
      tbl.SetCellAlignment(0, 0, Utils::TableHelper::ALIGN_CENTER);
      tbl.Set(2, 2, useRdmaRead ? " SRC\\DST+EXE " : " SRC+EXE\\DST ");
      tbl.SetCellAlignment(2, 2, Utils::TableHelper::ALIGN_CENTER);
      tbl.Set(2, sTotalCol, " STotal ");
      tbl.SetCellAlignment(2, sTotalCol, Utils::TableHelper::ALIGN_CENTER);
      tbl.Set(2, actualCol, " Actual ");
      tbl.SetCellAlignment(2, actualCol, Utils::TableHelper::ALIGN_CENTER);
      tbl.Set(rTotalRow, 2, " RTotal ");
      tbl.SetCellAlignment(rTotalRow, 2, Utils::TableHelper::ALIGN_CENTER);

      char const* memDevPrefix = useCpuMem ? "CPU" : "GPU";

      Utils::Print("\n--- NIC AllToAll Plane %02d Group %02d (%d NICs) ---\n", p, g, M);

      // Row-0 outer rank header: place " Rank XX " in the first NIC col of each
      // rank-group; the other NIC cols in row 0 stay empty.
      for (size_t c = 0; c < rankOrder.size(); c++)
        tbl.Set(0, rankColStart[c], " Rank %02d ", rankOrder[c]);

      // Rows 1 and 2: per-NIC GPU/CPU sub-header and NIC name sub-header.
      for (int j = 0; j < M; j++) {
        int const dstId   = dstNicIds[j];
        int const dstRank = dstId / numNicsPerRank;
        int const dstNic  = dstId % numNicsPerRank;
        int const colIdx  = 3 + j;
        tbl.Set(1, colIdx, " %s %02d ", memDevPrefix, nicToMem[dstRank][dstNic]);
        tbl.Set(2, colIdx, " %s ",
                TransferBench::GetExecutorName({EXE_NIC, dstNic, dstRank}).c_str());
      }

      // Data rows (M rows, one per src NIC). The outer-row "Rank XX" label only
      // appears on the first row of each rank's group; per-row sub-labels carry
      // the GPU/CPU backing device and NIC name. Track per-row and per-col
      // running totals for STotal/RTotal/Actual.
      std::vector<double> colTotal(M, 0.0);
      double sTotalGrand = 0.0;
      double actualGrand = 0.0;
      int rowDisp = 0;
      for (int srcRank : rankOrder) {
        bool firstInRank = true;
        for (int srcLocal : ranksToLocals[srcRank]) {
          int const rowIdx = 3 + rowDisp;
          if (firstInRank) {
            tbl.DrawRowBorder(rowIdx);
            tbl.Set(rowIdx, 0, " Rank %02d ", srcRank);
            firstInRank = false;
          }
          tbl.Set(rowIdx, 1, " %s %02d ", memDevPrefix, nicToMem[srcRank][srcLocal]);
          tbl.Set(rowIdx, 2, " %s ",
                  TransferBench::GetExecutorName({EXE_NIC, srcLocal, srcRank}).c_str());

          int const srcId = srcRank * numNicsPerRank + srcLocal;
          double rowSum   = 0.0;
          double rowMinBw = std::numeric_limits<double>::max();
          int    rowCount = 0;
          for (int j = 0; j < M; j++) {
            int const dstId  = dstNicIds[j];
            int const colIdx = 3 + j;
            if (!a2aLocal && srcId == dstId) {
              tbl.Set(rowIdx, colIdx, " N/A ");
            } else {
              double const bw = results.tfrResults[pairIdx[std::make_pair(srcId, dstId)]]
                                  .avgBandwidthGbPerSec;
              tbl.Set(rowIdx, colIdx, " %.2f ", bw);
              rowSum      += bw;
              colTotal[j] += bw;
              rowMinBw     = std::min(rowMinBw, bw);
              rowCount++;
            }
          }
          double const rowActual = (rowCount > 0) ? rowCount * rowMinBw : 0.0;
          tbl.Set(rowIdx, sTotalCol, " %.2f ", rowSum);
          tbl.Set(rowIdx, actualCol, " %.2f ", rowActual);
          sTotalGrand += rowSum;
          actualGrand += rowActual;
          rowDisp++;
        }
      }

      // RTotal row: per-dst-col sums plus grand totals for STotal/Actual.
      for (int j = 0; j < M; j++)
        tbl.Set(rTotalRow, 3 + j, " %.2f ", colTotal[j]);
      tbl.Set(rTotalRow, sTotalCol, " %.2f ", sTotalGrand);
      tbl.Set(rTotalRow, actualCol, " %.2f ", actualGrand);

      tbl.PrintTable(ev.outputToCsv, ev.showBorders);
    }
  }
  Utils::Print("\n");

  Utils::Print("Aggregate bandwidth (CPU Timed): %8.3f GB/s\n", results.avgTotalBandwidthGbPerSec);
  Utils::PrintErrors(results.errResults);

  if (Utils::HasDuplicateHostname()) {
    printf("[WARN] It is recommended to run TransferBench with one rank per host to avoid potential aliasing of executors\n");
  }

  return 0;
}

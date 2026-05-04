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

  int numQueuePairs = EnvVars::GetEnvVar("NUM_QUEUE_PAIRS", 1);
  int showDetails   = EnvVars::GetEnvVar("SHOW_DETAILS", 0);
  int useRdmaRead   = EnvVars::GetEnvVar("USE_RDMA_READ", 0);
  int memTypeIdx    = EnvVars::GetEnvVar("MEM_TYPE", 0);
  int stride        = EnvVars::GetEnvVar("STRIDE", 1);

  // Compute orbit structure before reading GROUP_SIZE so its default can be stride-aware.
  // Stride orbits on devices (rank-major devLin = rank * numMemDevices + memIdx): same gcd structure as PodAllToAll's StrideGenerate,
  // but NIC A2A does not use the permuted slot order for GROUP_SIZE — subgroups follow natural order within each orbit.
  int const M         = numRanks * numMemDevices;
  int const kNorm     = ((stride % M) + M) % M;
  int const dCycles   = (kNorm == 0) ? 1 : std::gcd(kNorm, M);
  int const orbitSize = M / dCycles;

  int groupSize    = EnvVars::GetEnvVar("GROUP_SIZE", orbitSize);
  int noSameRank   = EnvVars::GetEnvVar("NIC_A2A_NO_SAME_RANK", 1);
  int numNicPlanes = EnvVars::GetEnvVar("NUM_NIC_PLANES", 1);

  if (numQueuePairs < 1) {
    Utils::Print("[ERROR] NUM_QUEUE_PAIRS must be >= 1 (got %d)\n", numQueuePairs);
    return ERR_FATAL;
  }
  if (groupSize < 1) {
    Utils::Print("[ERROR] GROUP_SIZE must be >= 1 (got %d)\n", groupSize);
    return ERR_FATAL;
  }

  bool scopeInter = false;
  {
    char const* scopeStr = getenv("NIC_A2A_SCOPE");
    if (scopeStr && scopeStr[0]) {
      if (!strcmp(scopeStr, "inter") || !strcmp(scopeStr, "INTER"))
        scopeInter = true;
      else if (strcmp(scopeStr, "intra") && strcmp(scopeStr, "INTRA")) {
        Utils::Print("[ERROR] NIC_A2A_SCOPE must be \"intra\" or \"inter\"\n");
        return ERR_FATAL;
      }
    }
  }

  MemType memType        = Utils::GetMemType(memTypeIdx, useCpuMem);
  std::string memTypeStr = Utils::GetMemTypeStr(memTypeIdx, useCpuMem);

  if (numNicPlanes < 1) {
    Utils::Print("[ERROR] NUM_NIC_PLANES must be >= 1\n");
    return ERR_FATAL;
  }

  // Same divisibility check as PodAllToAll (total devices = ranks × memory devices per rank).
  if (M % groupSize) {
    Utils::Print("[ERROR] Group size %d cannot evenly divide %d total devices from %d ranks.\n",
                 groupSize, M, numRanks);
    return ERR_FATAL;
  }

  // Within each stride orbit, partition by natural rank-major device index: orbit lists devLin = r, r+d, r+2d, ...
  // (r = devLin %% dCycles). Subgroup id = (index along that list) / GROUP_SIZE.
  if (orbitSize % groupSize != 0) {
    Utils::Print("[ERROR] GROUP_SIZE (%d) must divide stride-cycle size %d (devices M=%d, orbits=%d).\n",
                 groupSize, orbitSize, M, dCycles);
    Utils::Print("[ERROR] With STRIDE=%d there are %d disjoint cycles; use a GROUP_SIZE that divides each cycle's device count,\n",
                 stride, dCycles);
    Utils::Print("[ERROR] or use STRIDE=1 so the cycle size equals total devices (%d).\n", M);
    return ERR_FATAL;
  }

  std::vector<int> deviceSubgroup(M);
  for (int devLin = 0; devLin < M; devLin++) {
    int const r = devLin % dCycles;
    int const k = (devLin - r) / dCycles;  // 0 .. orbitSize-1 along natural order in this orbit
    deviceSubgroup[devLin] = k / groupSize;
  }

  if (Utils::RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
      if (!ev.outputToCsv) printf("[NIC A2A Related]\n");
      ev.Print("USE_CPU_MEM"         , useCpuMem     , "Using closest %s memory", useCpuMem ? "CPU" : "GPU");
      ev.Print("MEM_TYPE"            , memTypeIdx    , "Using %s memory (%s)", memTypeStr.c_str(), Utils::GetAllMemTypeStr(useCpuMem).c_str());
      ev.Print("STRIDE"              , stride        , "Reordering devices by taking %d steps", stride);
      ev.Print("GROUP_SIZE"          , groupSize     , "Dividing all devices into groups of %d for a2a", groupSize);
      ev.Print("NUM_NIC_PLANES"      , numNicPlanes  , "Number of planes on scale-out");
      if (scopeInter)
        ev.Print("NIC_A2A_SCOPE"     , "inter"       , "Between-group transfers only. Other value: intra");
      else
        ev.Print("NIC_A2A_SCOPE"     , "intra"       , "Within-group transfers only. Other value: inter");
      ev.Print("NIC_A2A_NO_SAME_RANK", noSameRank    , "%s transfers where src rank == dst rank", noSameRank ? "Excluding" : "Allowing");
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

  auto devLinOf = [&](int rank, int memIdx) -> int { return rank * numMemDevices + memIdx; };

  // NIC plane: independent of STRIDE over memory devices. Global rank-major order over NIC endpoints, round-robin into P planes.
  auto nicPlaneOf = [&](int rank, int nic) -> int {
    int const L = rank * numNicsPerRank + nic;
    return L % numNicPlanes;
  };

  std::vector<Transfer> transfers;

  auto const acceptPair = [&](int srcRank, int srcNic, int dstRank, int dstNic) -> bool {
    if (nicPlaneOf(srcRank, srcNic) != nicPlaneOf(dstRank, dstNic))
      return false;
    int srcDevLin = devLinOf(srcRank, nicToMem[srcRank][srcNic]);
    int dstDevLin = devLinOf(dstRank, nicToMem[dstRank][dstNic]);
    if ((srcDevLin % dCycles) != (dstDevLin % dCycles))
      return false;
    if (noSameRank && srcRank == dstRank)
      return false;
    if (scopeInter)
      return deviceSubgroup[srcDevLin] != deviceSubgroup[dstDevLin];
    return deviceSubgroup[srcDevLin] == deviceSubgroup[dstDevLin];
  };

  for (int srcRank = 0; srcRank < numRanks; srcRank++) {
    for (int srcNic = 0; srcNic < numNicsPerRank; srcNic++) {
      int srcMem = nicToMem[srcRank][srcNic];
      for (int dstRank = 0; dstRank < numRanks; dstRank++) {
        for (int dstNic = 0; dstNic < numNicsPerRank; dstNic++) {
          if (!acceptPair(srcRank, srcNic, dstRank, dstNic)) continue;

          int dstMem = nicToMem[dstRank][dstNic];

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
    }
  }

  Utils::Print("NIC All-To-All benchmark\n");
  Utils::Print("========================\n");
  Utils::Print("%s traffic over NIC executors. %d rank-major devices; STRIDE sets gcd-orbits; GROUP_SIZE chunks each orbit in natural order.\n",
               useCpuMem ? "CPU" : "GPU", M);
  Utils::Print("NICs map to devices via closest %s;\n", useCpuMem ? "CPU NUMA node" : "GPU");
  Utils::Print("NIC planes: %d , traffic only between NICs in the same plane. Stride: %d\n",
               numNicPlanes, stride);
  Utils::Print("Using closest %s per NIC endpoint and %s memory.\n",
               useCpuMem ? "CPU NUMA node" : "GPU", memTypeStr.c_str());
  Utils::Print("Visible NICs per rank: %d\n", numNicsPerRank);
  Utils::Print("%d queue pairs per NIC.  %lu bytes per Transfer.  All numbers are GB/s\n",
               numQueuePairs, numBytesPerTransfer);
  Utils::Print("Total transfers: %lu\n\n", transfers.size());

  if (transfers.empty()) {
    Utils::Print("[WARN] No transfers were generated for this preset.\n");
    return 0;
  }

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

  int numRows = 6 + numRanks;
  int numCols = 3 + numNicsPerRank;
  Utils::TableHelper table(numRows, numCols);

  table.Set(2, 0, " Rank ");
  table.Set(2, 1, " Name ");
  table.Set(1, numCols - 1, " TOTAL ");
  table.Set(2, numCols - 1, " (GB/s) ");
  table.SetColAlignment(1, Utils::TableHelper::ALIGN_LEFT);
  for (int rank = 0; rank < numRanks; rank++) {
    table.Set(3 + rank, 0, " %d ", rank);
    table.Set(3 + rank, 1, " %s ", TransferBench::GetHostname(rank).c_str());
  }
  table.Set(numRows - 3, 1, " MAX (GB/s) ");
  table.Set(numRows - 2, 1, " AVG (GB/s) ");
  table.Set(numRows - 1, 1, " MIN (GB/s) ");
  for (int row = numRows - 3; row < numRows; row++)
    table.SetCellAlignment(row, 1, Utils::TableHelper::ALIGN_RIGHT);
  table.DrawRowBorder(3);
  table.DrawRowBorder(numRows - 3);

  std::vector<std::vector<double>> bwByRankNic(numRanks, std::vector<double>(numNicsPerRank, 0.0));
  for (size_t i = 0; i < results.tfrResults.size(); i++) {
    int nicIdx  = results.tfrResults[i].exeDevice.exeIndex;
    int rankIdx = results.tfrResults[i].exeDevice.exeRank;
    // Guard against remapped NIC indices from RunTransfers executor resolution
    if (rankIdx < 0 || rankIdx >= numRanks || nicIdx < 0 || nicIdx >= numNicsPerRank) {
      Utils::Print("[WARN] Skipping result %zu: executor (rank=%d, nic=%d) outside expected range [0,%d)[0,%d)\n",
                   i, rankIdx, nicIdx, numRanks, numNicsPerRank);
      continue;
    }
    bwByRankNic[rankIdx][nicIdx] += results.tfrResults[i].avgBandwidthGbPerSec;
  }

  std::vector<bool> nicHasMixedMemMapping(numNicsPerRank, false);
  bool hasMixedMemMapping = false;
  for (int nic = 0; nic < numNicsPerRank; nic++) {
    int refMem = nicToMem[0][nic];
    for (int rank = 1; rank < numRanks; rank++) {
      if (nicToMem[rank][nic] != refMem) {
        nicHasMixedMemMapping[nic] = true;
        hasMixedMemMapping = true;
        break;
      }
    }
  }

  std::vector<double> rankTotal(numRanks, 0.0);
  int colIdx = 2;
  table.DrawColBorder(colIdx);
  for (int nic = 0; nic < numNicsPerRank; nic++) {
    table.Set(0, colIdx, " NIC %02d ", nic);
    if (nicHasMixedMemMapping[nic]) {
      table.Set(1, colIdx, " MIXED ");
    } else if (useCpuMem) {
      table.Set(1, colIdx, " CPU %02d ", nicToMem[0][nic]);
    } else {
      table.Set(1, colIdx, " GPU %02d ", nicToMem[0][nic]);
    }
    table.Set(2, colIdx, " %s ", TransferBench::GetExecutorName({EXE_NIC, nic}).c_str());

    double nicMin = std::numeric_limits<double>::max();
    double nicAvg = 0.0;
    double nicMax = std::numeric_limits<double>::lowest();
    for (int rank = 0; rank < numRanks; rank++) {
      double bw = bwByRankNic[rank][nic];
      table.Set(3 + rank, colIdx, " %.2f ", bw);
      nicMin = std::min(nicMin, bw);
      nicAvg += bw;
      nicMax = std::max(nicMax, bw);
      rankTotal[rank] += bw;
    }

    table.Set(numRows - 3, colIdx, " %.2f ", nicMax);
    table.Set(numRows - 2, colIdx, " %.2f ", nicAvg / numRanks);
    table.Set(numRows - 1, colIdx, " %.2f ", nicMin);
    colIdx++;
  }
  table.DrawColBorder(colIdx);

  double rankMin = std::numeric_limits<double>::max();
  double rankAvg = 0.0;
  double rankMax = std::numeric_limits<double>::lowest();
  for (int rank = 0; rank < numRanks; rank++) {
    table.Set(3 + rank, numCols - 1, " %.2f ", rankTotal[rank]);
    rankMin = std::min(rankMin, rankTotal[rank]);
    rankAvg += rankTotal[rank];
    rankMax = std::max(rankMax, rankTotal[rank]);
  }
  table.Set(numRows - 3, numCols - 1, " %.2f ", rankMax);
  table.Set(numRows - 2, numCols - 1, " %.2f ", rankAvg / numRanks);
  table.Set(numRows - 1, numCols - 1, " %.2f ", rankMin);

  table.PrintTable(ev.outputToCsv, ev.showBorders);
  Utils::Print("\n");
  if (hasMixedMemMapping) {
    Utils::Print("[WARN] NIC-to-%s mapping differs across ranks. 'MIXED' columns are detailed below.\n",
                 useCpuMem ? "CPU" : "GPU");

    int mapRows = 2 + numRanks;
    int mapCols = 2 + numNicsPerRank;
    Utils::TableHelper mapTable(mapRows, mapCols);
    mapTable.Set(0, 0, " Rank ");
    mapTable.Set(0, 1, " Name ");
    mapTable.SetColAlignment(1, Utils::TableHelper::ALIGN_LEFT);
    for (int nic = 0; nic < numNicsPerRank; nic++) {
      mapTable.Set(0, 2 + nic, " NIC %02d ", nic);
      mapTable.SetCellAlignment(0, 2 + nic, Utils::TableHelper::ALIGN_CENTER);
    }
    mapTable.DrawRowBorder(1);
    mapTable.DrawColBorder(2);

    for (int rank = 0; rank < numRanks; rank++) {
      int rowIdx = 1 + rank;
      mapTable.Set(rowIdx, 0, " %d ", rank);
      mapTable.Set(rowIdx, 1, " %s ", TransferBench::GetHostname(rank).c_str());
      for (int nic = 0; nic < numNicsPerRank; nic++) {
        mapTable.Set(rowIdx, 2 + nic, " %s %02d ", useCpuMem ? "CPU" : "GPU", nicToMem[rank][nic]);
      }
    }

    mapTable.PrintTable(ev.outputToCsv, ev.showBorders);
    Utils::Print("\n");
  }
  Utils::Print("Aggregate bandwidth (CPU Timed): %8.3f GB/s\n", results.avgTotalBandwidthGbPerSec);
  Utils::PrintErrors(results.errResults);

  if (Utils::HasDuplicateHostname()) {
    printf("[WARN] It is recommended to run TransferBench with one rank per host to avoid potential aliasing of executors\n");
  }

  return 0;
}

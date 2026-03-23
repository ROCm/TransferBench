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
#include <numeric>

int NicAllToAllPreset(EnvVars&           ev,
                      size_t      const  numBytesPerTransfer,
                      std::string const  presetName)
{
  // Check for single homogenous group
  if (Utils::GetNumRankGroups() > 1) {
    Utils::Print("[ERROR] NIC all-to-all preset can only be run across ranks that are homogenous\n");
    Utils::Print("[ERROR] Run ./TransferBench without any args to display topology information\n");
    Utils::Print("[ERROR] TB_NIC_FILTER may also be used to limit NIC visibility to scale-out NICs\n");
    return 1;
  }

  int numRanks = TransferBench::GetNumRanks();
  int numNicsPerRank = TransferBench::GetNumExecutors(EXE_NIC);
  if (numNicsPerRank == 0) {
    Utils::Print("[ERROR] No NIC detected. This preset requires NIC executors.\n");
    return 1;
  }

  // Device count from topology only (no NUM_GPU_DEVICES): same detected GFX count as PodAllToAll uses by default.
  int numGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  int numQueuePairs = EnvVars::GetEnvVar("NUM_QUEUE_PAIRS", 1);
  int showDetails   = EnvVars::GetEnvVar("SHOW_DETAILS", 0);
  int useRdmaRead   = EnvVars::GetEnvVar("USE_RDMA_READ", 0);
  int memTypeIdx    = EnvVars::GetEnvVar("MEM_TYPE", 0);
  int stride        = EnvVars::GetEnvVar("STRIDE"         , 1);
  int groupSize     = EnvVars::GetEnvVar("GROUP_SIZE"     , numRanks * numGpus);
  int noSameRank    = EnvVars::GetEnvVar("NIC_A2A_NO_SAME_RANK", 1);
  int numNicPlanes  = EnvVars::GetEnvVar("NUM_NIC_PLANES" , 1);

  bool scopeInter = false;
  {
    char const* scopeStr = getenv("NIC_A2A_SCOPE");
    if (scopeStr && scopeStr[0]) {
      if (!strcmp(scopeStr, "inter") || !strcmp(scopeStr, "INTER"))
        scopeInter = true;
      else if (strcmp(scopeStr, "intra") && strcmp(scopeStr, "INTRA")) {
        Utils::Print("[ERROR] NIC_A2A_SCOPE must be \"intra\" or \"inter\"\n");
        return 1;
      }
    }
  }

  MemType memType   = Utils::GetGpuMemType(memTypeIdx);
  std::string memTypeStr = Utils::GetGpuMemTypeStr(memTypeIdx);

  if (numNicPlanes < 1) {
    Utils::Print("[ERROR] NUM_NIC_PLANES must be >= 1\n");
    return 1;
  }

  // Same divisibility check as PodAllToAll (total devices = ranks × detected GPUs per rank).
  if (numRanks * numGpus % groupSize) {
    Utils::Print("[ERROR] Group size %d cannot evenly divide %d total devices from %d ranks.\n",
                 groupSize, numRanks * numGpus, numRanks);
    return 1;
  }

  int const M = numRanks * numGpus;

  // Stride orbits on devices (rank-major devLin = rank * numGpus + gpu): same gcd structure as PodAllToAll's StrideGenerate,
  // but NIC A2A does not use the permuted slot order for GROUP_SIZE — subgroups follow natural order within each orbit.
  int kNorm = ((stride % M) + M) % M;
  int dCycles;
  if (kNorm == 0)
    dCycles = 1;
  else
    dCycles = std::gcd(kNorm, M);

  int const orbitSize = M / dCycles;

  // Within each stride orbit, partition by natural rank-major device index: orbit lists devLin = r, r+d, r+2d, ...
  // (r = devLin %% dCycles). Subgroup id = (index along that list) / GROUP_SIZE.
  if (orbitSize % groupSize != 0) {
    Utils::Print("[ERROR] GROUP_SIZE (%d) must divide stride-cycle size %d (devices M=%d, orbits=%d).\n",
                 groupSize, orbitSize, M, dCycles);
    Utils::Print("[ERROR] With STRIDE=%d there are %d disjoint cycles; use a GROUP_SIZE that divides each cycle's device count,\n",
                 stride, dCycles);
    Utils::Print("[ERROR] or use STRIDE=1 so the cycle size equals total devices (%d).\n", M);
    return 1;
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
      ev.Print("MEM_TYPE"       , memTypeIdx   , "Using %s memory (%s)", memTypeStr.c_str(), Utils::GetAllGpuMemTypeStr().c_str());
      ev.Print("STRIDE"         , stride       , "Reordering devices by taking %d steps", stride);
      ev.Print("GROUP_SIZE"     , groupSize    , "Dividing all devices into groups of %d for a2a", groupSize);
      ev.Print("NUM_NIC_PLANES", numNicPlanes , "Number of planes on scale-out");
      if (scopeInter)
        ev.Print("NIC_A2A_SCOPE", "inter",
                 "Between-group transfers only. Other value: intra");
      else
        ev.Print("NIC_A2A_SCOPE", "intra",
                 "Within-group transfers only. Other value: inter");
      ev.Print("NIC_A2A_NO_SAME_RANK", noSameRank, "%s transfers where src rank == dst rank",
               noSameRank ? "Excluding" : "Allowing");
      ev.Print("NUM_QUEUE_PAIRS", numQueuePairs, "Using %d queue pairs for NIC transfers", numQueuePairs);
      ev.Print("SHOW_DETAILS"   , showDetails  , "%s full Test details", showDetails ? "Showing" : "Hiding");
      ev.Print("USE_RDMA_READ"  , useRdmaRead  , "Performing RDMA %s", useRdmaRead ? "reads" : "writes");
      printf("\n");
    }
  }

  // For each rank/NIC, closest GPU — several NICs may share the same GPU (same device / subgroup).
  std::vector<std::vector<int>> nicToGpu(numRanks, std::vector<int>(numNicsPerRank, -1));
  for (int rank = 0; rank < numRanks; rank++) {
    for (int nic = 0; nic < numNicsPerRank; nic++) {
      int gpu = TransferBench::GetClosestGpuToNic(nic, rank);
      if (gpu < 0) {
        Utils::Print("[ERROR] Failed to identify closest GPU for Rank %d NIC %d\n", rank, nic);
        return 1;
      }
      if (gpu >= numGpus) {
        Utils::Print("[ERROR] Closest GPU index %d for Rank %d NIC %d is out of detected GFX range [0,%d)\n",
                     gpu, rank, nic, numGpus);
        return 1;
      }
      nicToGpu[rank][nic] = gpu;
    }
  }

  auto devLinOf = [&](int rank, int gpuIdx) -> int { return rank * numGpus + gpuIdx; };

  // NIC plane: independent of GPU STRIDE. Global rank-major order over NIC endpoints, round-robin into P planes.
  auto nicPlaneOf = [&](int rank, int nic) -> int {
    int const L = rank * numNicsPerRank + nic;
    return L % numNicPlanes;
  };

  std::vector<Transfer> transfers;
  std::vector<int> srcRanks;
  std::vector<int> srcNics;
  size_t const maxPairs = (size_t)numNicsPerRank * numNicsPerRank * (size_t)numRanks * (size_t)numRanks;
  srcRanks.reserve(maxPairs);
  srcNics.reserve(maxPairs);

  auto const acceptPair = [&](int srcRank, int srcNic, int dstRank, int dstNic) -> bool {
    if (nicPlaneOf(srcRank, srcNic) != nicPlaneOf(dstRank, dstNic))
      return false;
    int srcDevLin = devLinOf(srcRank, nicToGpu[srcRank][srcNic]);
    int dstDevLin = devLinOf(dstRank, nicToGpu[dstRank][dstNic]);
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
      int srcGpu = nicToGpu[srcRank][srcNic];
      for (int dstRank = 0; dstRank < numRanks; dstRank++) {
        for (int dstNic = 0; dstNic < numNicsPerRank; dstNic++) {
          if (!acceptPair(srcRank, srcNic, dstRank, dstNic)) continue;

          int dstGpu = nicToGpu[dstRank][dstNic];

          TransferBench::Transfer transfer;
          transfer.srcs.push_back({memType, srcGpu, srcRank});
          transfer.dsts.push_back({memType, dstGpu, dstRank});
          transfer.exeDevice   = {EXE_NIC, useRdmaRead ? dstNic : srcNic, useRdmaRead ? dstRank : srcRank};
          transfer.exeSubIndex = useRdmaRead ? srcNic : dstNic;
          transfer.numSubExecs = numQueuePairs;
          transfer.numBytes    = numBytesPerTransfer;

          transfers.push_back(transfer);
          srcRanks.push_back(srcRank);
          srcNics.push_back(srcNic);
        }
      }
    }
  }

  Utils::Print("NIC All-To-All benchmark\n");
  Utils::Print("========================\n");
  Utils::Print("GPU traffic over NIC executors. %d rank-major devices; STRIDE sets gcd-orbits; GROUP_SIZE chunks each orbit in natural order.\n", M);
  Utils::Print("NICs map to devices via closest GPU;\n");
  Utils::Print("NIC planes: %d , traffic only between NICs in the same plane. Stride: %d\n",
               numNicPlanes, stride);
  Utils::Print("Using closest GPU per NIC endpoint and %s memory.\n", memTypeStr.c_str());
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
    return 1;
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
    bwByRankNic[srcRanks[i]][srcNics[i]] += results.tfrResults[i].avgBandwidthGbPerSec;
  }

  std::vector<double> rankTotal(numRanks, 0.0);
  int colIdx = 2;
  table.DrawColBorder(colIdx);
  for (int nic = 0; nic < numNicsPerRank; nic++) {
    table.Set(0, colIdx, " NIC %02d ", nic);
    table.Set(1, colIdx, " SRC GPU ");
    table.Set(2, colIdx, " %s ", TransferBench::GetExecutorName({EXE_NIC, nic}).c_str());

    double nicMin = std::numeric_limits<double>::max();
    double nicAvg = 0.0;
    double nicMax = std::numeric_limits<double>::min();
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
  double rankMax = std::numeric_limits<double>::min();
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
  Utils::Print("Aggregate bandwidth (CPU Timed): %8.3f GB/s\n", results.avgTotalBandwidthGbPerSec);
  Utils::PrintErrors(results.errResults);

  if (Utils::HasDuplicateHostname()) {
    printf("[WARN] It is recommended to run TransferBench with one rank per host to avoid potential aliasing of executors\n");
  }

  return 0;
}

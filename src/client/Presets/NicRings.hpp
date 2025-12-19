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

int NicRingsPreset(EnvVars&           ev,
                   size_t      const  numBytesPerTransfer,
                   std::string const  presetName)
{

  // Check for single homogenous group
  if (Utils::GetNumRankGroups() > 1) {
    Utils::Print("[ERROR] NIC-rings preset can only be run across ranks that are homogenous\n");
    Utils::Print("[ERROR] Run ./TransferBench without any args to display topology information\n");
    Utils::Print("[ERROR] NIC_FILTER may also be used to limit NIC visibility\n");
    return 1;
  }

  // Collect topology
  int numRanks = TransferBench::GetNumRanks();

  // Read in environment variables
  int numQueuePairs = EnvVars::GetEnvVar("NUM_QUEUE_PAIRS", 1);
  int showDetails   = EnvVars::GetEnvVar("SHOW_DETAILS"   , 0);
  int useCpuMem     = EnvVars::GetEnvVar("USE_CPU_MEM"    , 0);
  int memTypeIdx    = EnvVars::GetEnvVar("MEM_TYPE"       , 0);
  int useRdmaRead   = EnvVars::GetEnvVar("USE_RDMA_READ"  , 0);

  // Print off environment variables
  MemType memType        = Utils::GetMemType(memTypeIdx, useCpuMem);
  std::string memTypeStr = Utils::GetMemTypeStr(memTypeIdx, useCpuMem);

  if (Utils::RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
      if (!ev.outputToCsv) printf("[NIC-Rings Related]\n");
      ev.Print("NUM_QUEUE_PAIRS", numQueuePairs, "Using %d queue pairs for NIC transfers", numQueuePairs);
      ev.Print("SHOW_DETAILS"   , showDetails  , "%s full Test details", showDetails ? "Showing" : "Hiding");
      ev.Print("USE_CPU_MEM"    , useCpuMem    , "Using closest %s memory", useCpuMem ? "CPU" : "GPU");
      ev.Print("MEM_TYPE"       , memTypeIdx   , "Using %s memory (%s)", memTypeStr.c_str(), Utils::GetAllMemTypeStr(useCpuMem).c_str());
      if (numRanks > 1)
        ev.Print("USE_RDMA_READ", useRdmaRead  , "Performing RDMA %s", useRdmaRead ? "reads" : "writes");
      printf("\n");
    }
  }

  // Prepare list of transfers
  int numDevices = TransferBench::GetNumExecutors(useCpuMem ? EXE_CPU : EXE_GPU_GFX);
  std::vector<Transfer> transfers;
  int numRings = 0;
  for (int memIndex = 0; memIndex < numDevices; memIndex++) {
    std::vector<int> nicIndices;
    if (useCpuMem) {
      TransferBench::GetClosestNicsToCpu(nicIndices, memIndex);
    } else {
      TransferBench::GetClosestNicsToGpu(nicIndices, memIndex);
    }
    for (int nicIndex : nicIndices) {
      numRings++;
      for (int currRank = 0; currRank < numRanks; currRank++) {
        int nextRank = (currRank + 1) % numRanks;

        TransferBench::Transfer transfer;
        transfer.srcs.push_back({memType, memIndex, currRank});
        transfer.dsts.push_back({memType, memIndex, nextRank});
        transfer.exeDevice   = {EXE_NIC, nicIndex, useRdmaRead ? nextRank : currRank};
        transfer.exeSubIndex = nicIndex;
        transfer.numSubExecs = numQueuePairs;
        transfer.numBytes    = numBytesPerTransfer;

        transfers.push_back(transfer);
      }
    }
  }

  Utils::Print("NIC Rings benchmark\n");
  Utils::Print("==============================\n");
  Utils::Print("%d parallel RDMA-%s rings(s) using %s memory across %d ranks\n",
               numRings, useRdmaRead ? "read" : "write", memTypeStr.c_str(), numRanks);
  Utils::Print("%d queue pairs per NIC.  %lu bytes per Transfer.  All numbers are GB/s\n",
               numQueuePairs, numBytesPerTransfer);
  Utils::Print("\n");

  // Execute Transfers
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

  // Only ranks that actually do output will compile results
  if (!Utils::RankDoesOutput()) return 0;

  // Prepare table of results
  int numRows   = 6 + numRanks;
  int numCols   = 3 + numRings;
  Utils::TableHelper table(numRows, numCols);

  // Prepare headers
  table.Set(2, 0, " Rank ");
  table.Set(2, 1, " Name ");
  table.Set(1, numCols-1, " TOTAL ");
  table.Set(2, numCols-1, " (GB/s) ");
  table.SetColAlignment(1, Utils::TableHelper::ALIGN_LEFT);
  for (int rank = 0; rank < numRanks; rank++) {
    table.Set(3 + rank, 0, " %d ", rank);
    table.Set(3 + rank, 1, " %s ", TransferBench::GetHostname(rank).c_str());
  }

  table.Set(numRows-3, 1, " MAX (GB/s) ");
  table.Set(numRows-2, 1, " AVG (GB/s) ");
  table.Set(numRows-1, 1, " MIN (GB/s) ");
  for (int row = numRows-3; row < numRows; row++)
    table.SetCellAlignment(row, 1, Utils::TableHelper::ALIGN_RIGHT);
  table.DrawRowBorder(3);
  table.DrawRowBorder(numRows-3);

  int colIdx = 2;
  int transferIdx = 0;
  std::vector<double> rankTotal(numRanks, 0.0);
  for (int memIndex = 0; memIndex < numDevices; memIndex++) {
    std::vector<int> nicIndices;
    if (useCpuMem) {
      TransferBench::GetClosestNicsToCpu(nicIndices, memIndex);
      table.Set(0, colIdx, " CPU %02d ", memIndex);
    } else {
      TransferBench::GetClosestNicsToGpu(nicIndices, memIndex);
      table.Set(0, colIdx, " GPU %02d ", memIndex);
    }
    bool isFirst = true;
    for (int nicIndex : nicIndices) {
      if (isFirst) {
        isFirst = false;
        table.DrawColBorder(colIdx);
      }
      table.Set(1, colIdx, " NIC %02d ", nicIndex);
      table.Set(2, colIdx, " %s ", TransferBench::GetExecutorName({EXE_NIC, nicIndex}).c_str());

      double ringMin = std::numeric_limits<double>::max();
      double ringAvg = 0.0;
      double ringMax = std::numeric_limits<double>::min();

      for (int rank = 0; rank < numRanks; rank++) {
        double avgBw = results.tfrResults[transferIdx].avgBandwidthGbPerSec;
        table.Set(3 + rank, colIdx, " %.2f ", avgBw);
        ringMin = std::min(ringMin, avgBw);
        ringAvg += avgBw;
        ringMax = std::max(ringMax, avgBw);
        rankTotal[rank] += avgBw;
        transferIdx++;
      }

      table.Set(numRows-3, colIdx, " %.2f ", ringMax);
      table.Set(numRows-2, colIdx, " %.2f ", ringAvg / numRanks);
      table.Set(numRows-1, colIdx, " %.2f ", ringMin);
      colIdx++;
    }
    if (!isFirst) {
      table.DrawColBorder(colIdx);
    }
  }

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

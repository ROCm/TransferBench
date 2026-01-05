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

#pragma once

#include "TransferBench.hpp"
#include "Utilities.hpp"

static int RemappedCpuIndex(int origIdx)
{
  static std::vector<int> remappingCpu;

  // Build CPU remapping on first use
  // Skip numa nodes that are not configured
  if (remappingCpu.empty()) {
    for (int node = 0; node <= numa_max_node(); node++)
      if (numa_bitmask_isbitset(numa_get_mems_allowed(), node))
        remappingCpu.push_back(node);
  }
  return remappingCpu[origIdx];
}

static void PrintNicToGPUTopo(bool outputToCsv)
{
#ifdef NIC_EXEC_ENABLED
  printf(" NIC | Device Name | Active | PCIe Bus ID  | NUMA | Closest GPU(s) | GID Index | GID Descriptor\n");
  if(!outputToCsv)
    printf("-----+-------------+--------+--------------+------+----------------+-----------+-------------------\n");

  int numGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);
  auto const& ibvDeviceList = GetIbvDeviceList();
  for (int i = 0; i < ibvDeviceList.size(); i++) {

    std::string closestGpusStr = "";
    for (int j = 0; j < numGpus; j++) {
      if (TransferBench::GetClosestNicToGpu(j) == i) {
        if (closestGpusStr != "") closestGpusStr += ",";
        closestGpusStr += std::to_string(j);
      }
    }

    printf(" %-3d | %-11s | %-6s | %-12s | %-4d | %-14s | %-9s | %-20s\n",
           i, ibvDeviceList[i].name.c_str(),
           ibvDeviceList[i].hasActivePort ? "Yes" : "No",
           ibvDeviceList[i].busId.c_str(),
           ibvDeviceList[i].numaNode,
           closestGpusStr.c_str(),
           ibvDeviceList[i].isRoce && ibvDeviceList[i].hasActivePort ? std::to_string(ibvDeviceList[i].gidIndex).c_str() : "N/A",
           ibvDeviceList[i].isRoce && ibvDeviceList[i].hasActivePort ? ibvDeviceList[i].gidDescriptor.c_str() : "N/A"
          );
  }
  printf("\n");
#endif
}

void DisplaySingleRankTopology(bool outputToCsv)
{
  int numCpus = TransferBench::GetNumExecutors(EXE_CPU);
  int numGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);
  int numNics = TransferBench::GetNumExecutors(EXE_NIC);
  char sep = (outputToCsv ? ',' : '|');

  if (outputToCsv) {
    printf("NumCpus,%d\n", numCpus);
    printf("NumGpus,%d\n", numGpus);
    printf("NumNics,%d\n", numNics);
  } else {
    printf("\nDetected Topology:\n");
    printf("==================\n");
    printf("  %d configured CPU NUMA node(s) [%d total]\n", numCpus, numa_max_node() + 1);
    printf("  %d GPU device(s)\n", numGpus);
    printf("  %d Supported NIC device(s)\n", numNics);
  }

  // Print out detected CPU topology
  printf("\n            %c", sep);
  for (int j = 0; j < numCpus; j++)
    printf("NUMA %02d%c", j, sep);
  printf(" #Cpus %c Closest GPU(s)\n", sep);

  if (!outputToCsv) {
    printf("------------+");
    for (int j = 0; j <= numCpus; j++)
      printf("-------+");
    printf("---------------\n");
  }

  for (int i = 0; i < numCpus; i++) {
    int nodeI = RemappedCpuIndex(i);
    printf("NUMA %02d (%02d)%c", i, nodeI, sep);
    for (int j = 0; j < numCpus; j++) {
      int nodeJ = RemappedCpuIndex(j);
      int numaDist = numa_distance(nodeI, nodeJ);
      printf(" %5d %c", numaDist, sep);
    }

    int numCpuCores = 0;
    for (int j = 0; j < numa_num_configured_cpus(); j++)
      if (numa_node_of_cpu(j) == nodeI) numCpuCores++;
    printf(" %5d %c", numCpuCores, sep);

    for (int j = 0; j < numGpus; j++) {
      if (TransferBench::GetClosestCpuNumaToGpu(j) == nodeI) {
        printf(" %d", j);
      }
    }
    printf("\n");
  }
  printf("\n");

  // Print out detected NIC topology
  PrintNicToGPUTopo(outputToCsv);

  // Print out detected GPU topology
#if defined(__NVCC__)
  for (int i = 0; i < numGpus; i++) {
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, i));
    printf(" GPU %02d | %s\n", i, prop.name);
  }
  // No further topology detection done for NVIDIA platforms
  return;
#else
  // Print headers
  if (numGpus > 0) {
    if (!outputToCsv) {
      printf("        |");
      for (int j = 0; j < numGpus; j++) {
        hipDeviceProp_t prop;
        HIP_CALL(hipGetDeviceProperties(&prop, j));
        std::string fullName = prop.gcnArchName;
        std::string archName = fullName.substr(0, fullName.find(':'));
        printf(" %6s |", archName.c_str());
      }
      printf("\n");
    }

    printf("        %c", sep);
    for (int j = 0; j < numGpus; j++)
      printf(" GPU %02d %c", j, sep);
    printf(" PCIe Bus ID  %c #CUs %c NUMA %c #DMA %c #XCC %c NIC\n", sep, sep, sep, sep, sep);

    if (!outputToCsv) {
      for (int j = 0; j <= numGpus; j++)
        printf("--------+");
      printf("--------------+------+------+------+------+------\n");
    }

    // Loop over each GPU device
    for (int i = 0; i < numGpus; i++) {
      printf(" GPU %02d %c", i, sep);

      // Print off link information
      for (int j = 0; j < numGpus; j++) {
        if (i == j) {
          printf("    N/A %c", sep);
        } else {
          uint32_t linkType, hopCount;
          HIP_CALL(hipExtGetLinkTypeAndHopCount(i, j, &linkType, &hopCount));
          printf(" %s-%d %c",
                 linkType == HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT ? "  HT" :
                 linkType == HSA_AMD_LINK_INFO_TYPE_QPI            ? " QPI" :
                 linkType == HSA_AMD_LINK_INFO_TYPE_PCIE           ? "PCIE" :
                 linkType == HSA_AMD_LINK_INFO_TYPE_INFINBAND      ? "INFB" :
                 linkType == HSA_AMD_LINK_INFO_TYPE_XGMI           ? "XGMI" : "????",
                 hopCount, sep);
        }
      }

      char pciBusId[20];
      HIP_CALL(hipDeviceGetPCIBusId(pciBusId, 20, i));
      printf(" %-11s %c %-4d %c %-4d %c %-4d %c %-4d %c %-4d\n",
             pciBusId, sep,
             TransferBench::GetNumSubExecutors({EXE_GPU_GFX, i}), sep,
             TransferBench::GetClosestCpuNumaToGpu(i), sep,
             TransferBench::GetNumExecutorSubIndices({EXE_GPU_DMA, i}), sep,
             TransferBench::GetNumExecutorSubIndices({EXE_GPU_GFX, i}), sep,
             TransferBench::GetClosestNicToGpu(i));
    }
  }
#endif
}


void DisplayMultiRankTopology(bool outputToCsv, bool showBorders)
{
  Utils::RankGroupMap& groups = Utils::GetRankGroupMap();

  printf("%d rank(s) in %lu homogeneous group(s)\n", TransferBench::GetNumRanks(), groups.size());
  printf("\n");

  // Print off each group
  int groupNum = 1;

  for (auto const& group : groups) {
    Utils::GroupKey const& key    = group.first;
    std::vector<int> const& hosts = group.second;

    std::string              ppodId        = std::get<0>(key);
    int                      vpodId        = std::get<1>(key);
    std::vector<std::string> cpuNames      = std::get<2>(key);
    std::vector<int>         cpuSubExecs   = std::get<3>(key);
    std::vector<std::string> gpuNames      = std::get<4>(key);
    std::vector<int>         gpuSubExecs   = std::get<5>(key);
    std::vector<int>         gpuClosestCpu = std::get<6>(key);
    std::vector<std::string> nicNames      = std::get<7>(key);
    std::vector<int>         nicClosestCpu = std::get<8>(key);
    std::vector<int>         nicClosestGpu = std::get<9>(key);
    std::vector<int>         nicIsActive   = std::get<10>(key);

    int numRanks = hosts.size();
    int numCpus  = cpuNames.size();
    int numGpus  = gpuNames.size();
    int numNics  = nicNames.size();
    int numExecutors = numCpus + numGpus + numNics;
    int numActiveNics = 0;
    for (auto x : nicIsActive) numActiveNics += x;

    if (groupNum > 1) printf("\n");
    printf("Group %03d: %d rank(s) %d CPU(s) %d GPU(s) %d NIC(s) (%d active NICs)\n",
           groupNum++, numRanks, numCpus, numGpus, numNics, numActiveNics);

    // Determine size of table
    int numCols = 7;
    int numRows = 1 + std::max(numRanks, numExecutors);
    TransferBench::Utils::TableHelper table(numRows, numCols);

    // Table borders / alignment
    for (int col = 0; col <= numCols; col++) {
      table.DrawColBorder(col);
      table.SetColAlignment(col, TransferBench::Utils::TableHelper::ALIGN_LEFT);
    }
    table.DrawRowBorder(0);
    table.DrawRowBorder(1);
    table.DrawRowBorder(numRows);

    // Table header
    table.Set(0, 0, " Rank ");
    table.Set(0, 1, " Hostname ");
    table.Set(0, 2, " POD ");
    table.Set(0, 3, " VID ");
    table.Set(0, 4, " Executor ");
    table.Set(0, 5, " Executor Name ");
    table.Set(0, 6, " #SE ");

    // Fill in ranks / hosts
    for (int i = 0; i < numRanks; i++) {
      int rank = hosts[i];
      table.Set(1 + i, 0, " %04d ", rank);
      table.Set(1 + i, 1, " %s ", TransferBench::GetHostname(rank).c_str());
    }

    // Fill in PPOD and VPOD
    table.Set(1, 2, " %s ", ppodId.c_str());
    table.Set(1, 3, " %d ", vpodId);

    // Fill in Executor information
    int rowIdx = 1;
    for (int cpuIndex = 0; cpuIndex < numCpus; cpuIndex++) {
      table.Set(rowIdx, 4, " CPU %02d ", cpuIndex);
      table.Set(rowIdx, 5, " %s ",       cpuNames[cpuIndex].c_str());
      table.Set(rowIdx, 6, " %d ",       cpuSubExecs[cpuIndex]);
      rowIdx++;

      // Loop over each GPU closest to this CPU executor
      for (int gpuIndex = 0; gpuIndex < numGpus; gpuIndex++) {
        if (gpuClosestCpu[gpuIndex] != cpuIndex) continue;
        table.Set(rowIdx, 4, " - GPU %02d ", gpuIndex);
        table.Set(rowIdx, 5, " - %s ",         gpuNames[gpuIndex].c_str());
        table.Set(rowIdx, 6, " %d ",         gpuSubExecs[gpuIndex]);
        rowIdx++;

        //  Loop over each NIC closest to this GPU
        for (int nicIndex = 0; nicIndex < numNics; nicIndex++) {
          if (nicClosestGpu[nicIndex] != gpuIndex) continue;
          table.Set(rowIdx, 4, "   - NIC %02d ", nicIndex);
          table.Set(rowIdx, 5, "   - %s", nicNames[nicIndex].c_str());
          table.Set(rowIdx, 6, " %s ", nicIsActive[nicIndex] ? "ON" : "OFF");
          rowIdx++;
        }
      }

      // Loop over remaining NICs not associated with GPU but associated with this CPU
      for (int nicIndex = 0; nicIndex < numNics; nicIndex++) {
        if (nicClosestGpu[nicIndex] != -1 || nicClosestCpu[nicIndex] != cpuIndex) continue;
        table.Set(rowIdx, 4, " - NIC %02d ", nicIndex);
        table.Set(rowIdx, 5, " - %s ", nicNames[nicIndex].c_str());
        table.Set(rowIdx, 6, " %s ", nicIsActive[nicIndex] ? "ON" : "OFF");
        rowIdx++;
      }
    }
    table.PrintTable(outputToCsv, showBorders);
  }

  if (Utils::HasDuplicateHostname()) {
    printf("[WARN] It is recommended to run TransferBench with one rank per host to avoid potential aliasing of executors\n");
  }
}

void DisplayTopology(bool outputToCsv, bool showBorders)
{
  if (GetNumRanks() > 1)
    DisplayMultiRankTopology(outputToCsv, showBorders);
  else
    DisplaySingleRankTopology(outputToCsv);
}

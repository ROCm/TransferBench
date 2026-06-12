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
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <type_traits>
#include "EnvVars.hpp"
#include "TransferBench.hpp"

namespace TransferBench::Utils
{
  // Linear interpolation on sorted samples (same ordering as common empirical quantiles with (n-1) indexing).
  inline double PercentileDurationMsecFromSorted(std::vector<double> const& sortedAsc, int pct)
  {
    size_t const n = sortedAsc.size();
    if (n == 0)
      return 0.0;
    double const pos = (static_cast<double>(pct) / 100.0) * static_cast<double>(n - 1);
    size_t const lo = static_cast<size_t>(std::floor(pos));
    size_t const hi = static_cast<size_t>(std::ceil(pos));
    double const frac = pos - std::floor(pos);
    return sortedAsc[lo] * (1.0 - frac) + sortedAsc[hi] * frac;
  }

  // Helper class to help format tabular data / output to CSV
  class TableHelper
  {
  public:
    // Column alignment options
    enum {
      ALIGN_LEFT  = -1,
      ALIGN_CENTER = 0,
      ALIGN_RIGHT  = 1
    } AlignType;

    enum {
      BORDER_TOP   = 1,
      BORDER_BOT   = 2,
      BORDER_LEFT  = 4,
      BORDER_RIGHT = 8,
      BORDER_ALL   = 15,
    } BorderType;

    // Helper class to print off tabled data
    TableHelper(int numRows, int numCols, int precision = 2);

    // Set the value for a particular cell
    template <typename T>
    void Set(int rowIdx, int colIdx, T const& value);
    void Set(int rowIdx, int colIdx, const char* format, ...);

    // Set the alignment for a given cell
    void SetCellAlignment(int rowIdx, int colIdx, int8_t alignMode);
    // Set the alignment for all cells in a given column
    void SetColAlignment(int colIdx, int8_t alignMode);
    // Set the alignment for all cells in a given row
    void SetRowAlignment(int rowIdx, int8_t alignMode);

    // Set border around a cell
    void SetCellBorder(int rowIdx, int colIdx, int borderMask);
    // Draws a horizontal border on top of given row
    void DrawRowBorder(int rowIdx);
    // Draws a vertical border prior to given column
    void DrawColBorder(int colIdx);

    // Print the table
    void PrintTable(bool outputToCsv, bool drawBorders = true);

  private:
    int numRows;
    int numCols;
    int precision;
    std::vector<std::vector<std::string>> table;
    std::vector<std::vector<int8_t>> alignment;
    std::vector<int> colWidth;
    std::unordered_map<int, std::unordered_set<int>> rowBorders;
    std::unordered_map<int, std::unordered_set<int>> colBorders;
  };

  // Group information
  typedef std::tuple<
    int64_t,                       // Pod Index
    std::vector<std::string>,      // CPU Names
    std::vector<int>,              // CPU #Subexecutors
    std::vector<std::string>,      // GPU Names
    std::vector<int>,              // GPU #Subexecutors
    std::vector<int>,              // GPU Closest NUMA
    std::vector<std::string>,      // NIC Names
    std::vector<int>,              // NIC Closest NUMA
    std::vector<int>,              // NIC Closest GPU
    std::vector<int>               // NIC is active
    > GroupKey;

  typedef std::map<GroupKey, std::vector<int>> RankGroupMap;
  typedef std::map<int64_t, std::vector<int>> RankPerPodMap;

  // Get information about how ranks can be organized into homogenous groups
  RankGroupMap& GetRankGroupMap();

  // Return the number of homogenous groups of ranks
  int GetNumRankGroups();

  // Helper function for pod membership
  RankPerPodMap& GetRankPerPodMap();

  // Helper function to convert an ExeType to a string
  std::string ExeTypeToStr(ExeType exeType);

  // Helper function that converts MemDevices to a string
  std::string MemDevicesToStr(std::vector<MemDevice> const& memDevices);

  // Helper function to determine if current rank does output
  bool RankDoesOutput();

  // Helper function that only prints if current rank does output
  void Print(const char* format, ...);

  // Helper function to deal with ErrResults (exits on fatal error)
  void CheckForError(ErrResult const& error);

  // Helper function to deal with vector of ErrREsults (exits on fatal error)
  void PrintErrors(std::vector<ErrResult> const& errors);

  // Helper function to print TransferBench test results
  void PrintResults(EnvVars const& ev, int const testNum,
                    std::vector<Transfer> const& transfers,
                    TestResults const& results);

  // Returns true if more than one rank share the same hostname
  bool HasDuplicateHostname();

  // Helper function to map between integer index and memory types
  MemType GetCpuMemType(int memTypeIdx);
  MemType GetGpuMemType(int memTypeIdx);
  MemType GetMemType(int memTypeIdx, bool isCpu);

  // Helper function to map between integer index and memory type name
  std::string GetCpuMemTypeStr(int memTypeIdx);
  std::string GetGpuMemTypeStr(int memTypeIdx);
  std::string GetMemTypeStr(int memTypeIdx, bool isCpu);

  // Helper function to list all available options
  std::string GetAllCpuMemTypeStr();
  std::string GetAllGpuMemTypeStr();
  std::string GetAllMemTypeStr(bool isCpu);

  // Helper forwarders to allocation/deallocation functions
  // Returns true if error occurs
  bool AllocateMemory(MemDevice memDevice, size_t numBytes, void** memPtr);
  bool DeallocateMemory(MemType memType, void *memPtr, size_t const bytes);

  // Reorder elements of list by stepping through with stride k, wrapping around.
  // When gcd(k, n) > 1 the single cycle breaks into gcd(k, n) orbits which are
  // concatenated, so every element appears exactly once in the output.
  // The reordered list will be further separated into different groups.
  void StrideGenerate(std::vector<int>& list, int k);

  // Returns a schedule of round robin pairing of N elements, using Circle Method.
  // If parallel, each round contains N/2 pairs, otherwise serial.
  void RoundRobinSchedule(std::vector<std::vector<std::pair<int, int>>>& schedule,
                          int N, int parallel = 0);

  // Returns a schedule for ordered 2-combination of N elements
  // by pairing the list with its rotating self.
  // Each round contains n pairs, where 1 <= n <= N and N is divisible by n,
  // and an element cannot appear more than twice in a round.
  void CombinationSchedule(std::vector<std::vector<std::pair<int, int>>>& schedule,
                           int N, int n = 0);

  // Implementation details below
  //================================================================
  TableHelper::TableHelper(int numRows, int numCols, int precision) :
    numRows(numRows), numCols(numCols), precision(precision)
  {
    if (numRows < 0 || numCols < 0) {
      Print("[ERROR] Cannot create TableHelper of negative size\n");
      exit(1);
    }

    // Initialize internal data structures
    table.resize(numRows, std::vector<std::string>(numCols, ""));
    alignment.resize(numRows, std::vector<int8_t>(numCols, ALIGN_RIGHT));
    colWidth.resize(numCols, 0);
  }

  template <typename T>
  void TableHelper::Set(int rowIdx, int colIdx, T const& value)
  {
    if (0 <= rowIdx && rowIdx < numRows && 0 <= colIdx && colIdx < numCols) {
      std::stringstream ss;
      if constexpr (std::is_floating_point_v<T>) {
        ss << std::fixed << std::setprecision(precision) << value;
      } else {
        ss << value;
      }

      table[rowIdx][colIdx] = ss.str();
      colWidth[colIdx] = std::max(colWidth[colIdx], static_cast<int>(table[rowIdx][colIdx].size()));
    }
  }

  void TableHelper::Set(int rowIdx, int colIdx, const char* format, ...)
  {
    if (0 <= rowIdx && rowIdx < numRows && 0 <= colIdx && colIdx < numCols) {
      va_list args, args_copy;
      va_start(args, format);

      // Figure out size of the string
      va_copy(args_copy, args);
      int size = std::vsnprintf(nullptr, 0, format, args_copy);
      va_end(args_copy);

      table[rowIdx][colIdx].resize(size, '\0');
      std::vsnprintf(table[rowIdx][colIdx].data(), size + 1, format, args);
      va_end(args);

      colWidth[colIdx] = std::max(colWidth[colIdx], static_cast<int>(table[rowIdx][colIdx].size()));
    }
  }

  void TableHelper::SetCellAlignment(int rowIdx, int colIdx, int8_t alignMode)
  {
    if (0 <= rowIdx && rowIdx < numRows && 0 <= colIdx && colIdx < numCols && -1 <= alignMode && alignMode <= 1)
      alignment[rowIdx][colIdx] = alignMode;
  }

  void TableHelper::SetColAlignment(int colIdx, int8_t alignMode)
  {
    if (0 <= colIdx && colIdx < numCols && -1 <= alignMode && alignMode <= 1)
      for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
        alignment[rowIdx][colIdx] = alignMode;
  }

  void TableHelper::SetRowAlignment(int rowIdx, int8_t alignMode)
  {
    if (0 <= rowIdx && rowIdx < numRows && -1 <= alignMode && alignMode <= 1)
      for (int colIdx = 0; colIdx < numCols; colIdx++)
        alignment[rowIdx][colIdx] = alignMode;
  }

  void TableHelper::SetCellBorder(int rowIdx, int colIdx, int borderMask)
  {
    if (0 <= rowIdx && rowIdx < numRows && 0 <= colIdx && colIdx < numCols) {
      if (borderMask & BORDER_TOP)   rowBorders[rowIdx  ].insert(colIdx); else rowBorders[rowIdx  ].erase(colIdx);
      if (borderMask & BORDER_BOT)   rowBorders[rowIdx+1].insert(colIdx); else rowBorders[rowIdx+1].erase(colIdx);
      if (borderMask & BORDER_LEFT)  colBorders[colIdx  ].insert(rowIdx); else colBorders[colIdx  ].erase(rowIdx);
      if (borderMask & BORDER_RIGHT) colBorders[colIdx+1].insert(rowIdx); else colBorders[colIdx+1].erase(rowIdx);
    }
  }

  void TableHelper::DrawRowBorder(int rowIdx)
  {
    if (0 <= rowIdx && rowIdx <= numRows)
      for (int colIdx = 0; colIdx < numCols; colIdx++)
        rowBorders[rowIdx].insert(colIdx);
  }

  void TableHelper::DrawColBorder(int colIdx)
  {
    if (0 <= colIdx && colIdx <= numCols)
      for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
        colBorders[colIdx].insert(rowIdx);
  }

  void TableHelper::PrintTable(bool outputToCsv, bool drawBorders)
  {
    if (!RankDoesOutput()) return;

    std::string borders[16] =
      {" ", "│", "│", "│",
       "-", "┘", "┐", "┤",
       "-", "└", "┌", "├",
       "-", "┴", "┬", "┼"};

    int mask;
    for (int rowIdx = 0; rowIdx <= numRows; rowIdx++) {
      // Draw "top" border
      if (!outputToCsv && drawBorders && rowBorders[rowIdx].size() > 0) {
        for (int colIdx = 0; colIdx <= numCols; colIdx++) {
          mask = 0;
          if (colBorders[colIdx].count(rowIdx-1)) mask |= BORDER_TOP;
          if (colBorders[colIdx].count(rowIdx  )) mask |= BORDER_BOT;
          if (rowBorders[rowIdx].count(colIdx-1)) mask |= BORDER_LEFT;
          if (rowBorders[rowIdx].count(colIdx  )) mask |= BORDER_RIGHT;
          Print("%s", borders[mask].c_str());
          if (colIdx < numCols) {
            std::string ch = rowBorders[rowIdx].count(colIdx) ? "-" : " ";
            for (int i = 0; i < colWidth[colIdx]; i++) Print("%s", ch.c_str());
          }
        }
        Print("\n");
      }
      if (rowIdx == numRows) break;

      // Print off table data
      for (int colIdx = 0; colIdx <= numCols; colIdx++) {
        if (!outputToCsv)
          Print("%s", drawBorders && colBorders[colIdx].count(rowIdx) ? "│" : " ");
        if (colIdx == numCols) break;

        int gap = colWidth[colIdx] - table[rowIdx][colIdx].size();
        int lgap, rgap;
        switch (alignment[rowIdx][colIdx]) {
        case ALIGN_LEFT:   lgap = 0;     rgap = gap;        break;
        case ALIGN_CENTER: lgap = gap/2; rgap = gap - lgap; break;
        case ALIGN_RIGHT:  lgap = gap;   rgap = 0;          break;
        }
        for (int i = 0; i < lgap; i++) printf(" ");
        Print("%s", table[rowIdx][colIdx].c_str());
        for (int i = 0; i < rgap; i++) printf(" ");
        if (outputToCsv) Print(",");
      }
      Print("\n");
    }
  }

  RankGroupMap& GetRankGroupMap()
  {
    static RankGroupMap groups;
    static bool initialized = false;

    if (!initialized) {
      // Build GroupKey for each rank
      for (int rank = 0; rank < TransferBench::GetNumRanks(); rank++) {

        int64_t podId = TransferBench::GetPodIdx(rank);

        // CPU information
        int numCpus = TransferBench::GetNumExecutors(EXE_CPU, rank);
        std::vector<std::string> cpuNames;
        std::vector<int> cpuNumSubExecs;
        for (int exeIndex = 0; exeIndex < numCpus; exeIndex++) {
          ExeDevice exeDevice = {EXE_CPU, exeIndex, rank};
          cpuNames.push_back(TransferBench::GetExecutorName(exeDevice));
          cpuNumSubExecs.push_back(TransferBench::GetNumSubExecutors(exeDevice));
        }

        // GPU information
        int numGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX, rank);
        std::vector<std::string> gpuNames;
        std::vector<int> gpuNumSubExecs;
        std::vector<int> gpuClosestCpu;
        for (int exeIndex = 0; exeIndex < numGpus; exeIndex++) {
          ExeDevice exeDevice = {EXE_GPU_GFX, exeIndex, rank};
          gpuNames.push_back(TransferBench::GetExecutorName(exeDevice));
          gpuNumSubExecs.push_back(TransferBench::GetNumSubExecutors(exeDevice));
          gpuClosestCpu.push_back(TransferBench::GetClosestCpuNumaToGpu(exeIndex, rank));
        }

        // NIC information
        int numNics = TransferBench::GetNumExecutors(EXE_NIC, rank);

        std::vector<int> nicClosestGpu(numNics, -1);
        for (int gpuIndex = 0; gpuIndex < numGpus; gpuIndex++) {
          std::vector<int> nicIndices;
          TransferBench::GetClosestNicsToGpu(nicIndices, gpuIndex, rank);
          for (auto nicIndex : nicIndices) {
            nicClosestGpu[nicIndex] = gpuIndex;
          }
        }

        std::vector<std::string> nicNames;
        std::vector<int> nicClosestCpu;
        std::vector<int> nicIsActive;
        for (int exeIndex = 0; exeIndex < numNics; exeIndex++) {
          ExeDevice exeDevice = {EXE_NIC, exeIndex, rank};
          nicNames.push_back(TransferBench::GetExecutorName(exeDevice));
          nicClosestCpu.push_back(TransferBench::GetClosestCpuNumaToNic(exeIndex, rank));
          nicIsActive.push_back(TransferBench::NicIsActive(exeIndex, rank));
        }

        GroupKey key(podId,
                     cpuNames, cpuNumSubExecs,
                     gpuNames, gpuNumSubExecs, gpuClosestCpu,
                     nicNames, nicClosestCpu, nicClosestGpu, nicIsActive);

        groups[key].push_back(rank);
      }
      initialized = true;
    }
    return groups;
  }

  int GetNumRankGroups()
  {
    return GetRankGroupMap().size();
  }

  RankPerPodMap& GetRankPerPodMap()
  {
    static RankPerPodMap pods;
    static bool initialized = false;

    if (!initialized) {
      for (int rank = 0; rank < TransferBench::GetNumRanks(); rank++) {
        int64_t const podId = TransferBench::GetPodIdx(rank);
        if (podId == -1) continue;
        pods[podId].push_back(rank);
      }
      initialized = true;
    }
    return pods;
  }
  // Helper function to convert an ExeType to a string
  std::string ExeTypeToStr(ExeType exeType)
  {
    switch (exeType) {
    case EXE_CPU:           return "CPU";
    case EXE_GPU_GFX:       return "GPU";
    case EXE_GPU_DMA:       return "DMA";
    case EXE_NIC:           return "NIC";
    case EXE_NIC_NEAREST:   return "NIC";
    case EXE_GPU_BDMA:      return "BMA";
    case EXE_GPU_ASYNC_TENSOR: return "AT"; // async tensor kernel path
    case EXE_GPU_ASYNC_MEMOPS: return "AL"; // async load/store kernel path
    default:                return "N/A";
    }
  }

  // Helper function that converts MemDevices to a string
  std::string MemDevicesToStr(std::vector<MemDevice> const& memDevices)
  {
    if (memDevices.empty()) return "N";
    bool isMultiNode = TransferBench::GetNumRanks() > 1;
    std::stringstream ss;
    for (auto const& m : memDevices) {
      if (isMultiNode)
        ss << "R" << m.memRank;
      ss << TransferBench::MemTypeStr[m.memType] << m.memIndex;
    }
    return ss.str();
  }

  template <typename T>
  struct is_std_vector : std::false_type {};

  template <typename T, typename Alloc>
  struct is_std_vector<std::vector<T, Alloc>> : std::true_type {};

  // This function can be used to check if a value is identical across ranks
  template <typename T>
  bool IsUniform(const T& val) {
    if constexpr (is_std_vector<T>::value) {
      using Elem = typename T::value_type;
      static_assert(std::is_trivially_copyable_v<Elem>, "vector element must be trivially copyable");

      size_t size = val.size();
      size_t rootSize = size;
      System::Get().Broadcast(0, sizeof(rootSize), &rootSize);
      if (size != rootSize) return false;

      std::vector<Elem> ref = val;
      System::Get().Broadcast(0, rootSize * sizeof(Elem), ref.data());

      return (std::memcmp(ref.data(), val.data(), rootSize * sizeof(Elem)) == 0);
    } else {
      static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable");
      T ref = val;
      System::Get().Broadcast(0, sizeof(T), &ref);

      return (std::memcmp(&ref, &val, sizeof(T)) == 0);
    }
  }

  // Macro for use in presets that will return ERR_FATAL if a value is not uniform across ranks
#define IS_UNIFORM(val, name)                                                      \
  do {                                                                             \
    if (!Utils::IsUniform(val)) {                                                  \
      Utils::Print("[ERROR] %s must be uniform across all ranks\n", name);         \
      return ERR_FATAL;                                                            \
    }                                                                              \
  } while(0)

  // Helper function to determine if current rank does output
  bool RankDoesOutput()
  {
    return (TransferBench::GetCommMode() != TransferBench::COMM_MPI ||
            TransferBench::GetRank() == 0);
  }

  // Helper function that only prints if current rank does output
  void Print(const char* format, ...)
  {
    if (RankDoesOutput()) {
      va_list args;
      va_start(args, format);
      vprintf(format, args);
      va_end(args);
    }
  }

  // Helper function to deal with ErrResults (exits on fatal error)
  void CheckForError(ErrResult const& error)
  {
    switch (error.errType) {
    case ERR_NONE: return;
    case ERR_WARN:
      Print("[WARN] %s\n", error.errMsg.c_str());
      return;
    case ERR_FATAL:
      Print("[ERROR] %s\n", error.errMsg.c_str());
      exit(1);
    default:
      break;
    }
  }

  // Helper function to deal with vector of ErrREsults (exits on fatal error)
  void PrintErrors(std::vector<ErrResult> const& errors)
  {
    // When running in MPI mode, only the first rank produces output
    bool isFatal = false;
    for (auto const& err : errors) {
      Print("[%s] %s\n", err.errType == ERR_FATAL ? "ERROR" : "WARN", err.errMsg.c_str());
      isFatal |= (err.errType == ERR_FATAL);
    }
    if (isFatal) exit(1);
  }

  // Print TransferBench test results
  void PrintResults(EnvVars const& ev, int const testNum,
                    std::vector<Transfer> const& transfers,
                    TestResults const& results)
  {
    if (!RankDoesOutput()) return;

    if (!ev.outputToCsv) printf("Test %d:\n", testNum);

    bool isMultiRank = TransferBench::GetNumRanks() > 1;

    // Figure out table dimensions
    int numCols = 5, numRows = 1;
    size_t numTimedIterations = results.numTimedIterations;
    for (auto const& exeInfoPair : results.exeResults) {
      ExeResult const& exeResult = exeInfoPair.second;
      numRows += 1 + exeResult.transferIdx.size();
      if (!ev.showPercentiles.empty()) {
        numRows += static_cast<int>(ev.showPercentiles.size()) * static_cast<int>(exeResult.transferIdx.size());
      }
      if (ev.showIterations) {
        numRows += (numTimedIterations + 1) * exeResult.transferIdx.size();
      }
      if (ev.showIterations || !ev.showPercentiles.empty()) {
        for (int idx : exeResult.transferIdx) {
          TransferResult const& r = results.tfrResults[idx];
          if (r.perIterMsec.size() != numTimedIterations) {
            Print("[ERROR] Per iteration timing data unavailable: Expected %lu data points, but have %lu\n",
                  numTimedIterations, r.perIterMsec.size());
            exit(1);
          }
        }
      }
    }

    int showNumIterations = (ev.numIterations < 0) ? 1 : 0;

    TableHelper table(numRows+showNumIterations, numCols);
    for (int col = 1; col < numCols; col++)
      table.DrawColBorder(col);

    // Loop over each executor
    int rowIdx = 0;
    for (auto const& exeInfoPair : results.exeResults) {
      ExeDevice const& exeDevice = exeInfoPair.first;
      ExeResult const& exeResult = exeInfoPair.second;
      ExeType   const  exeType   = exeDevice.exeType;
      int32_t   const  exeIndex  = exeDevice.exeIndex;

      // Display Executor results
      table.DrawRowBorder(rowIdx);
      if (isMultiRank) {
        table.Set(rowIdx, 0, " Executor: Rank %d %3s %02d ", exeDevice.exeRank, ExeTypeToStr(exeType).c_str(), exeIndex);
        table.Set(rowIdx, 4, " %7.3f GB/s (sum) [%s]", exeResult.sumBandwidthGbPerSec, GetHostname(exeDevice.exeRank).c_str());
      } else {
        table.Set(rowIdx, 0, " Executor: %3s %02d ", ExeTypeToStr(exeType).c_str(), exeIndex);
        table.Set(rowIdx, 4, " %7.3f GB/s (sum)", exeResult.sumBandwidthGbPerSec);
      }
      table.Set(rowIdx, 1, "%8.3f GB/s " , exeResult.avgBandwidthGbPerSec);
      table.Set(rowIdx, 2, "%8.3f ms "   , exeResult.avgDurationMsec);
      table.Set(rowIdx, 3, "%12lu bytes ", exeResult.numBytes);
      table.SetCellAlignment(rowIdx, 4, TableHelper::ALIGN_LEFT);
      rowIdx++;
      table.DrawRowBorder(rowIdx);

      // Loop over the Transfers for this executor
      for (int idx : exeResult.transferIdx) {
        Transfer const& t = transfers[idx];
        TransferResult const& r = results.tfrResults[idx];

        table.Set(rowIdx, 0, "Transfer %-4d ", idx);
        table.Set(rowIdx, 1, "%8.3f GB/s "   , r.avgBandwidthGbPerSec);
        table.Set(rowIdx, 2, "%8.3f ms "     , r.avgDurationMsec);
        table.Set(rowIdx, 3, "%12lu bytes "  , r.numBytes);

        char exeSubIndexStr[32] = "";
        if (t.exeSubIndex != -1)
          sprintf(exeSubIndexStr, ".%d", t.exeSubIndex);

        if (isMultiRank) {
          table.Set(rowIdx, 4, " %s -> R%d%c%d%s:%d -> %s",
                    MemDevicesToStr(t.srcs).c_str(),
                    exeDevice.exeRank, ExeTypeStr[t.exeDevice.exeType], t.exeDevice.exeIndex,
                    exeSubIndexStr, t.numSubExecs,
                    MemDevicesToStr(t.dsts).c_str());
        } else {
          table.Set(rowIdx, 4, " %s -> %c%d%s:%d -> %s",
                    MemDevicesToStr(t.srcs).c_str(),
                    ExeTypeStr[t.exeDevice.exeType], t.exeDevice.exeIndex,
                    exeSubIndexStr, t.numSubExecs,
                    MemDevicesToStr(t.dsts).c_str());
        }
        table.SetCellAlignment(rowIdx, 4, TableHelper::ALIGN_LEFT);
        rowIdx++;

        // Show per-iteration timing information
        if (ev.showIterations) {

          // Compute standard deviation and track iterations by speed
          std::set<std::pair<double, int>> times;
          double stdDevTime = 0;
          double stdDevBw = 0;
          for (int i = 0; i < numTimedIterations; i++) {
            times.insert(std::make_pair(r.perIterMsec[i], i+1));
            double const varTime = fabs(r.avgDurationMsec - r.perIterMsec[i]);
            stdDevTime += varTime * varTime;

            double iterBandwidthGbs = (t.numBytes / 1.0E9) / r.perIterMsec[i] * 1000.0f;
            double const varBw = fabs(iterBandwidthGbs - r.avgBandwidthGbPerSec);
            stdDevBw += varBw * varBw;
          }
          stdDevTime = sqrt(stdDevTime / numTimedIterations);
          stdDevBw = sqrt(stdDevBw / numTimedIterations);

          // Loop over iterations (fastest to slowest)
          for (auto& time : times) {
            double iterDurationMsec = time.first;
            double iterBandwidthGbs = (t.numBytes / 1.0E9) / iterDurationMsec * 1000.0f;

            std::set<int> usedXccs;
            std::stringstream ss1;
            if (exeDevice.exeType == EXE_GPU_GFX) {
              if (time.second - 1 < r.perIterCUs.size()) {
                ss1 << " CUs: ";
                for (auto x : r.perIterCUs[time.second - 1]) {
                  ss1 << x.first << ":" << std::setfill('0') << std::setw(2) << x.second << " ";
                  usedXccs.insert(x.first);
                }
              }
            }

            std::stringstream ss2;
            if (!usedXccs.empty()) {
              ss2 << " XCCs:";
              for (auto x : usedXccs)
                ss2 << " "  << x;
            }

            table.Set(rowIdx, 0, "Iter %03d    ", time.second);
            table.Set(rowIdx, 1, "%8.3f GB/s ", iterBandwidthGbs);
            table.Set(rowIdx, 2, "%8.3f ms ", iterDurationMsec);
            table.Set(rowIdx, 3, ss1.str());
            table.Set(rowIdx, 4, ss2.str());
            rowIdx++;
          }

          table.Set(rowIdx, 0, "StandardDev ");
          table.Set(rowIdx, 1, "%8.3f GB/s ", stdDevBw);
          table.Set(rowIdx, 2, "%8.3f ms ", stdDevTime);
          rowIdx++;
          table.DrawRowBorder(rowIdx);
        }

        // Show percentiles
        if (!ev.showPercentiles.empty()) {
          std::vector<double> sortedDur = r.perIterMsec;
          std::sort(sortedDur.begin(), sortedDur.end());
          for (int pct : ev.showPercentiles) {
            double dur = PercentileDurationMsecFromSorted(sortedDur, pct);
            double bwGbs = dur > 0.0 ? (t.numBytes / 1.0E9) / dur * 1000.0 : 0.0;
            table.Set(rowIdx, 0, "p%d ", pct);
            table.Set(rowIdx, 1, "%8.3f GB/s ", bwGbs);
            table.Set(rowIdx, 2, "%8.3f ms ", dur);
            table.Set(rowIdx, 3, " ");
            table.Set(rowIdx, 4, " ");
            table.SetCellAlignment(rowIdx, 4, TableHelper::ALIGN_LEFT);
            rowIdx++;
          }
        }

      }
    }
    table.DrawRowBorder(rowIdx);
    table.Set(rowIdx, 0, "Aggregate (CPU) ");
    table.Set(rowIdx, 1, "%8.3f GB/s "      , results.avgTotalBandwidthGbPerSec);
    table.Set(rowIdx, 2, "%8.3f ms "        , results.avgTotalDurationMsec);
    table.Set(rowIdx, 3, "%12lu bytes "     , results.totalBytesTransferred);
    table.Set(rowIdx, 4, " Overhead %.3f ms", results.overheadMsec);
    table.SetCellAlignment(rowIdx, 4, TableHelper::ALIGN_LEFT);
    table.DrawRowBorder(rowIdx+1);

    if (showNumIterations) {
      rowIdx++;
      table.Set(rowIdx, 0, "# Iters Run:");
      table.Set(rowIdx, 1, "%lu ", numTimedIterations);
      table.SetCellAlignment(rowIdx, 1, TableHelper::ALIGN_LEFT);
      table.SetCellBorder(rowIdx, 0, 0);
      table.SetCellBorder(rowIdx, 1, 0);
      table.SetCellBorder(rowIdx, 2, 0);
      table.SetCellBorder(rowIdx, 3, 0);
      table.SetCellBorder(rowIdx, 4, 0);
      table.DrawRowBorder(rowIdx);
      table.DrawRowBorder(rowIdx+1);
    }
    table.PrintTable(ev.outputToCsv, ev.showBorders);
  }

  bool HasDuplicateHostname()
  {
    std::set<std::string> seenHosts;
    for (int rank = 0; rank < TransferBench::GetNumRanks(); rank++) {
      std::string hostname = TransferBench::GetHostname(rank);
      if (seenHosts.count(hostname)) return true;
      seenHosts.insert(hostname);
    }
    return false;
  }

  // Helper function to map between integer index and memory types
  MemType GetCpuMemType(int memTypeIdx)
  {
    switch (memTypeIdx) {
    case 0: return MEM_CPU;
    case 1: return MEM_CPU_COHERENT;
    case 2: return MEM_CPU_NONCOHERENT;
    case 3: return MEM_CPU_UNCACHED;
    case 4: return MEM_CPU_UNPINNED;
    default: return MEM_CPU;
    }
  }

  MemType GetGpuMemType(int memTypeIdx)
  {
    switch (memTypeIdx) {
    case 0: return MEM_GPU;
    case 1: return MEM_GPU_FINE;
    case 2: return MEM_GPU_UNCACHED;
    case 3: return MEM_MANAGED;
    default: return MEM_GPU;
    }
  }

  MemType GetMemType(int memTypeIdx, bool isCpu)
  {
    return isCpu ? GetCpuMemType(memTypeIdx) : GetGpuMemType(memTypeIdx);
  }

  // Helper function to map between integer index and memory type name
  std::string GetCpuMemTypeStr(int memTypeIdx)
  {
    switch (memTypeIdx) {
    case 0: return "default CPU";
    case 1: return "coherent CPU";
    case 2: return "non-coherent CPU";
    case 3: return "uncached CPU";
    case 4: return "unpinned CPU";
    default: return "default CPU";
    }
  }

  std::string GetGpuMemTypeStr(int memTypeIdx)
  {
    switch (memTypeIdx) {
    case 0: return "default GPU";
    case 1: return "fine-grained GPU";
    case 2: return "uncached GPU";
    case 3: return "managed";
    default: return "default GPU";
    }
  }

  std::string GetMemTypeStr(int memTypeIdx, bool isCpu)
  {
    return isCpu ? GetCpuMemTypeStr(memTypeIdx) : GetGpuMemTypeStr(memTypeIdx);
  }

  std::string GetAllCpuMemTypeStr()
  {
    return "0=default, 1=coherent, 2=non-coherent, 3=uncached, 4=unpinned";
  }
  std::string GetAllGpuMemTypeStr()
  {
    return "0=default, 1=fine-grained, 2=uncached, 3=managed";
  }
  std::string GetAllMemTypeStr(bool isCpu)
  {
    return isCpu ? GetAllCpuMemTypeStr() : GetAllGpuMemTypeStr();
  }

  bool AllocateMemory(MemDevice memDevice, size_t numBytes, void** memPtr)
  {
    return (TransferBench::AllocateMemory(memDevice, numBytes, memPtr).errType != TransferBench::ERR_NONE);
  }
  bool DeallocateMemory(MemType memType, void *memPtr, size_t const bytes)
  {
    return (TransferBench::DeallocateMemory(memType, memPtr, bytes).errType != TransferBench::ERR_NONE);
  }

  void StrideGenerate(std::vector<int>& list, int k)
  {
    int n = list.size();
    if (n == 0) return;
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

  void RoundRobinSchedule(std::vector<std::vector<std::pair<int, int>>>& schedule,
                          int N, int parallel)
  {
    if (N == 1) {
      schedule.push_back({{0, 0}});
      return;
    }
    // Generate standard round-robin tournament (maximum parallelism)
    std::vector<std::vector<std::pair<int, int>>> fullSchedule;

    // Pad odd number of ranks with a dummy round (N+1)
    int paddedN = N + N % 2;
    // Round-robin tournament scheduling
    for (int round = 0; round < paddedN - 1; round++) {
      std::vector<std::pair<int, int>> roundPairs;
      std::vector<std::pair<int, int>> roundPairsReversed;
      for (int i = 0; i < paddedN / 2; i++) {
        int item1 = i;
        int item2 = paddedN - 1 - i;
        if (round > 0) {
          // Rotate all except the first item
          if (item1 > 0) item1 = ((item1 - 1 + round) % (paddedN - 1)) + 1;
          if (item2 > 0) item2 = ((item2 - 1 + round) % (paddedN - 1)) + 1;
        }
        // Ignore dummy round, its partner sits out this round
        if (item1 < N && item2 < N) {
          roundPairs.push_back({item1, item2});
          roundPairsReversed.push_back({item2, item1});
        }
      }
      fullSchedule.push_back(roundPairs);
      fullSchedule.push_back(roundPairsReversed);
    }

    // A loopback round where all run in parallel
    std::vector<std::pair<int, int>> selfRound;
    for (int i = 0; i < N; i++) {
      selfRound.push_back({i, i});
    }
    fullSchedule.push_back(selfRound);

    if (parallel) {
      schedule = std::move(fullSchedule);
    } else {
      // Serialize each round if needed
      for (auto const& fullRound : fullSchedule) {
        for (auto const& match : fullRound) {
          std::vector<std::pair<int, int>> subRound;
          subRound.push_back({match.first, match.second});
          schedule.push_back(subRound);
        }
      }
    }
  }

  void CombinationSchedule(std::vector<std::vector<std::pair<int, int>>>& schedule,
                           int N, int n)
  {
    std::vector<std::vector<std::pair<int, int>>> fullSchedule;

    if (n <= 0) n = N;
    if (N <= 0 || n > N || N % n != 0) // Assuming balanced load for each round
    {
      n = 1;
      Print("[WARN] cannot create combination schedule, falling back to serial\n");
    }

    // Generate rounds of combination based on incrementing distance
    for (int i = 0; i < N; i++) {
      std::vector<std::pair<int, int>> round;
      for (int j = 0; j < N; j++) {
        round.push_back({j, (j + i) % N});
      }
      fullSchedule.push_back(round);
    }

    // Step 2: Split each full round into sub-rounds with at most n pairs
    for (auto const& fullRound : fullSchedule) {
      for (size_t start = 0; start < fullRound.size(); start += n) {
        std::vector<std::pair<int, int>> subRound;
        for (size_t i = start; i < start + n && i < fullRound.size(); i++) {
          subRound.push_back(fullRound[i]);
        }
        if (!subRound.empty()) {
          schedule.push_back(subRound);
        }
      }
    }
  }
};

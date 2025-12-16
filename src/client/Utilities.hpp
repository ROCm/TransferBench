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
#include <unordered_map>
#include <unordered_set>
#include "TransferBench.hpp"

namespace TransferBench::Utils
{
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
       "─", "┘", "┐", "┤",
       "─", "└", "┌", "├",
       "─", "┴", "┬", "┼"};

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
            std::string ch = rowBorders[rowIdx].count(colIdx) ? "─" : " ";
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

  // Helper function to convert an ExeType to a string
  std::string ExeTypeToStr(ExeType exeType)
  {
    switch (exeType) {
    case EXE_CPU:         return "CPU";
    case EXE_GPU_GFX:     return "GPU";
    case EXE_GPU_DMA:     return "DMA";
    case EXE_NIC:         return "NIC";
    case EXE_NIC_NEAREST: return "NIC";
    default:              return "N/A";
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
      if (ev.showIterations) {
        numRows += (numTimedIterations + 1);

        // Check that per-iteration information exists
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

    TableHelper table(numRows, numCols);
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
        table.Set(rowIdx, 1, "%8.3f GB/s "       , r.avgBandwidthGbPerSec);
        table.Set(rowIdx, 2, "%8.3f ms "         , r.avgDurationMsec);
        table.Set(rowIdx, 3, "%12lu bytes "      , r.numBytes);

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
      }
    }
    table.DrawRowBorder(rowIdx);
    table.Set(rowIdx, 0, "Aggregate (CPU) ");
    table.Set(rowIdx, 1, "%8.3f GB/s "      , results.avgTotalBandwidthGbPerSec);
    table.Set(rowIdx, 2, "%8.3f ms "        , results.avgTotalDurationMsec);
    table.Set(rowIdx, 3, "%12lu bytes "     , results.totalBytesTransferred);
    table.Set(rowIdx, 4, " Overhead %.3f ms", results.overheadMsec);
    table.SetCellAlignment(rowIdx, 4, TableHelper::ALIGN_LEFT);
    table.DrawRowBorder(rowIdx);

    table.PrintTable(ev.outputToCsv, ev.showBorders);
  }
};

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

#include "Presets.hpp"
#include "Topology.hpp"
#include <fstream>

void DisplayVersion();
void DisplayUsage(char const* cmdName);

using namespace TransferBench;
using namespace TransferBench::Utils;

size_t constexpr DEFAULT_BYTES_PER_TRANSFER = (1<<28);

int main(int argc, char **argv)
{
  // Collect environment variables
  EnvVars ev;

  // Display usage instructions and detected topology
  if (argc <= 1) {
    if (RankDoesOutput()) {
      if (!ev.outputToCsv) {
        DisplayVersion();
        DisplayUsage(argv[0]);
      }
      DisplayTopology(ev.outputToCsv, ev.showBorders);
    }
    exit(0);
  }

  // Determine number of bytes to run per Transfer
  size_t numBytesPerTransfer = argc > 2 ? atoll(argv[2]) : DEFAULT_BYTES_PER_TRANSFER;
  if (argc > 2) {
    // Adjust bytes if unit specified
    char units = argv[2][strlen(argv[2])-1];
    switch (units) {
    case 'G': case 'g': numBytesPerTransfer *= 1024;
    case 'M': case 'm': numBytesPerTransfer *= 1024;
    case 'K': case 'k': numBytesPerTransfer *= 1024;
    }
  }
  if (numBytesPerTransfer % 4) {
    Print("[ERROR] numBytesPerTransfer (%lu) must be a multiple of 4\n", numBytesPerTransfer);
    exit(1);
  }

  // Display TransferBench version and build configuration
  DisplayVersion();

  // Run preset benchmark if requested
  int retCode = 0;
  if (RunPreset(ev, numBytesPerTransfer, argc, argv, retCode, argv[0])) return retCode;

  // Read input from command line or configuration file
  bool isDryRun = !strcmp(argv[1], "dryrun");
  std::vector<std::string> lines;
  {
    std::string line;
    if (!strcmp(argv[1], "cmdline") || isDryRun) {
      for (int i = 3; i < argc; i++)
        line += std::string(argv[i]) + " ";
      lines.push_back(line);
    } else {
      std::ifstream cfgFile(argv[1]);
      if (!cfgFile.is_open()) {
        Print("[ERROR] Unable to open transfer configuration file: [%s]\n", argv[1]);
        exit(1);
     }
      while (std::getline(cfgFile, line))
        lines.push_back(line);
      cfgFile.close();
    }
  }

  ConfigOptions cfgOptions = ev.ToConfigOptions();
  TestResults results;
  std::vector<ErrResult> errors;

  // Dry run prints off transfers (and errors)
  if (isDryRun) {
    Print("Transfers to be executed (dry-run):\n");
    Print("================================================================================\n");
    std::vector<Transfer> transfers;
    CheckForError(ParseTransfers(lines[0], transfers));
    if (transfers.empty()) {
      Print("<none>\n");
    } else {
      bool isMultiNode = GetNumRanks() > 1;
      for (size_t i = 0; i < transfers.size(); i++) {
        Transfer const& t = transfers[i];
        Print("Transfer %5lu: (%s->", i, MemDevicesToStr(t.srcs).c_str());
        if (isMultiNode) Print("R%d", t.exeDevice.exeRank);
        Print("%c%d", ExeTypeStr[t.exeDevice.exeType], t.exeDevice.exeIndex);
        if (t.exeDevice.exeSlot) Print("%c", 'A' + t.exeDevice.exeSlot);
        if (t.exeSubIndex != -1) Print(".%d", t.exeSubIndex);
        if (t.exeSubSlot != 0) Print("%c", 'A' + t.exeSubSlot);
        Print("->%s)\n", MemDevicesToStr(t.dsts).c_str());
      }
    }
    return 0;
  }

  // Print environment variables and CSV header
  if (RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (ev.outputToCsv)
      Print("Test#,Transfer#,NumBytes,Src,Exe,Dst,CUs,BW(GB/s),Time(ms),SrcAddr,DstAddr\n");
  }

  // Process each line as a Test
  int testNum = 0;
  for (std::string const &line : lines) {
    // Check if line is a comment to be echoed to output (starts with ##)
    if (!ev.outputToCsv && line[0] == '#' && line[1] == '#') printf("%s\n", line.c_str());

    // Parse set of parallel Transfers to execute
    std::vector<Transfer> transfers;
    CheckForError(ParseTransfers(line, transfers));
    if (transfers.empty()) continue;

    // Check for variable sub-executors Transfers
    int numVariableTransfers = 0;
    int maxVarCount = 0;
    {
      std::map<ExeDevice, int> varTransferCount;
      for (auto const& t : transfers) {
        if (t.numSubExecs == 0) {
          if (t.exeDevice.exeType != EXE_GPU_GFX) {
            Print("[ERROR] Variable number of subexecutors is only supported on GFX executors\n");
            exit(1);
          }
          numVariableTransfers++;
          varTransferCount[t.exeDevice]++;
          maxVarCount = max(maxVarCount, varTransferCount[t.exeDevice]);
        }
      }
      if (numVariableTransfers > 0 && numVariableTransfers != transfers.size()) {
        Print("[ERROR] All or none of the Transfers in the Test must use variable number of Subexecutors\n");
        exit(1);
      }
    }

    // Track which transfers have already numBytes specified
    std::vector<bool> bytesSpecified(transfers.size());
    int hasUnspecified = false;
    for (int i = 0; i < transfers.size(); i++) {
      bytesSpecified[i] = (transfers[i].numBytes != 0);
      if (transfers[i].numBytes == 0) hasUnspecified = true;
    }

    // Run the specified numbers of bytes otherwise generate a range of values
    for (size_t bytes = (1<<10); bytes <= (1<<29); bytes *= 2) {
      size_t deltaBytes = std::max(1UL, bytes / ev.samplingFactor);
      size_t currBytes = (numBytesPerTransfer == 0) ? bytes : numBytesPerTransfer;
      do {
        for (int i = 0; i < transfers.size(); i++) {
          if (!bytesSpecified[i])
            transfers[i].numBytes = currBytes;
        }

        if (maxVarCount == 0) {
          if (RunTransfers(cfgOptions, transfers, results)) {
            PrintResults(ev, ++testNum, transfers, results);
          }
          if (RankDoesOutput()) {
            PrintErrors(results.errResults);
          }
        } else {
          // Variable subexecutors - Determine how many subexecutors to sweep up to
          int maxNumVarSubExec = ev.maxNumVarSubExec;
          if (maxNumVarSubExec == 0) {
            maxNumVarSubExec = GetNumSubExecutors({EXE_GPU_GFX, 0}) / maxVarCount;
          }

          TestResults bestResults;
          std::vector<Transfer> bestTransfers;
          for (int numSubExecs = ev.minNumVarSubExec; numSubExecs <= maxNumVarSubExec; numSubExecs++) {
            std::vector<Transfer> tempTransfers = transfers;
            for (auto& t : tempTransfers) {
              if (t.numSubExecs == 0) t.numSubExecs = numSubExecs;
            }

            TestResults tempResults;
            if (!RunTransfers(cfgOptions, tempTransfers, tempResults)) {
              PrintErrors(tempResults.errResults);
            } else {
              if (tempResults.avgTotalBandwidthGbPerSec > bestResults.avgTotalBandwidthGbPerSec) {
                bestResults = tempResults;
                bestTransfers = tempTransfers;
              }
            }
          }
          PrintResults(ev, ++testNum, bestTransfers, bestResults);
          PrintErrors(bestResults.errResults);
        }
        if (numBytesPerTransfer != 0 || !hasUnspecified) break;
        currBytes += deltaBytes;
      } while (currBytes < bytes * 2);
      if (numBytesPerTransfer != 0 || !hasUnspecified) break;
    }
  }
}

void DisplayVersion()
{
  bool nicSupport = false, mpiSupport = false;
#if NIC_EXEC_ENABLED
  nicSupport = true;
#endif
#if MPI_COMM_ENABLED
  mpiSupport = true;
#endif

  std::string support = "";
  if (mpiSupport && nicSupport) support = " (with MPI+NIC support)";
  else if (mpiSupport)          support = " (with MPI support)";
  else if (nicSupport)          support = " (with NIC support)";

  std::string multiNodeMode = "";
  switch (GetCommMode()) {
  case COMM_NONE:   multiNodeMode = " (Single-node mode)";       break;
  case COMM_SOCKET: multiNodeMode = " (Multi-node via sockets)"; break;
  case COMM_MPI:    multiNodeMode = " (Multi-node via MPI)";     break;
  }

  Print("TransferBench v%s.%s%s%s\n", VERSION, CLIENT_VERSION, support.c_str(), multiNodeMode.c_str());
  Print("=============================================================================================================\n");
}

void DisplayUsage(char const* cmdName)
{
  if (numa_available() == -1) {
    Print("[ERROR] NUMA library not supported. Check to see if libnuma has been installed on this system\n");
    exit(1);
  }

  DisplayBasicUsage(cmdName);
  Print("\n");
};

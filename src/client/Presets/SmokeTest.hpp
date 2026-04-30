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

#include <set>

namespace  {

#define NUM_SMOKE_TESTS 14
#define MAX_TRANSFER_STRLEN 128

// What to print on pass/fail/skip
const std::string pass = "*";
const std::string fail = "F";
const std::string skip = ".";

int RunTest(int                        testNum,
            std::set<int> const&       testsToRun,
            std::vector<size_t> const& sizeList,
            int                        numSubExecPerGpu,
            ConfigOptions&             cfg,
            MemType                    cpuMemType,
            MemType                    gpuMemType,
            size_t                     maxBytesPerSubExec,
            int                        totalGpus)
{
  int numFail = 0;

  // Collect some topology information
  int numRanks = TransferBench::GetNumRanks();

  std::vector<Transfer> transfers;
  std::vector<Transfer> allTransfers;
  TestResults results;
  char transferStr[MAX_TRANSFER_STRLEN] = {};


  // Different test categories
  bool isH2D       = (testNum == 1 || testNum ==  8);
  bool isD2H       = (testNum == 2 || testNum ==  9);
  bool isD2D_RW    = (testNum == 3 || testNum == 10);
  bool isD2D_RR    = (testNum == 4 || testNum == 11);
  bool isBroadcast = (testNum == 5 || testNum == 12);
  bool isGather    = (testNum == 6 || testNum == 13);
  bool isAllToAll  = (testNum == 7 || testNum == 14);

  // Determine executor type
  ExeType exeType;
  if      (1 <= testNum && testNum <= 7)  exeType = EXE_GPU_DMA;
  else if (8 <= testNum && testNum <= 14) exeType = EXE_GPU_GFX;
  else {
    Utils::Print("[ERROR] Unsupported test number %d\n", testNum);
    exit(1);
  }

  // Adjust number of subexecutors per transfer if performing multiple transfers
  int numSubExec = exeType == EXE_GPU_DMA ? 1 : numSubExecPerGpu;
  if (exeType == EXE_GPU_GFX && (isBroadcast || isGather || isAllToAll))
    numSubExec = std::max(1, numSubExecPerGpu / totalGpus);

  for (size_t numBytes : sizeList) {

    // Print skip symbol for skipped tests
    if (!testsToRun.count(testNum)) {
      Utils::Print("%s", skip.c_str()); fflush(stdout);
      continue;
    }
    if (exeType == EXE_GPU_GFX &&
        (numSubExec * cfg.data.blockBytes > numBytes ||
         numSubExec * maxBytesPerSubExec  < numBytes)) {
      Utils::Print("%s", skip.c_str()); fflush(stdout);
      continue;
    }
    // Skip test that require pod
    if (numRanks > 1 && Utils::GetRankPerPodMap().size() != 1 && !(isH2D || isD2H)) {
      Utils::Print("%s", skip.c_str()); fflush(stdout);
      continue;
    }

    bool allPass = true;
    allTransfers.clear();

    // Combine transfers from each GPU and run them all in parallel
    for (int rank = 0; allPass && rank < numRanks; rank++) {
      int numGpus = GetNumExecutors(exeType, rank);
      for (int gpuIdx = 0; allPass && gpuIdx < numGpus; gpuIdx++) {
        if (isH2D || isD2H) {
          // Copy to/from closest CPU NUMA node for this GPU
          int cpuIdx = GetClosestCpuNumaToGpu(gpuIdx, rank);
          snprintf(transferStr, MAX_TRANSFER_STRLEN, "-1 (R%d%c%d R%d%c%d R%d%c%d %d %lu)",
                   rank, MemTypeStr[isH2D ? cpuMemType : gpuMemType], isH2D ? cpuIdx : gpuIdx,
                   rank, ExeTypeStr[exeType], gpuIdx,
                   rank, MemTypeStr[isH2D ? gpuMemType : cpuMemType], isH2D ? gpuIdx : cpuIdx,
                   numSubExec, numBytes);
        } else if (isD2D_RW || isD2D_RR) {
          // Copy from this GPU to "next" GPU
          int dstRank = rank, dstGpuIdx = gpuIdx + 1;
          if (dstGpuIdx >= GetNumExecutors(exeType, dstRank)) {
            dstGpuIdx = 0;
            dstRank = (rank+1) % numRanks;
          }
          snprintf(transferStr, MAX_TRANSFER_STRLEN, "-1 (R%d%c%d R%d%c%d R%d%c%d %d %lu)",
                   rank, MemTypeStr[gpuMemType], gpuIdx,
                   isD2D_RW ? rank : dstRank, ExeTypeStr[exeType], isD2D_RW ? gpuIdx : dstGpuIdx,
                   dstRank, MemTypeStr[gpuMemType], dstGpuIdx,
                   numSubExec, numBytes);
        } else if (isBroadcast) {
          // Split up the number of CUs across all Transfers
          snprintf(transferStr, MAX_TRANSFER_STRLEN, "-1 (R%d%c%d R%d%c%d R*%c* %d %lu)",
                   rank, MemTypeStr[gpuMemType], gpuIdx,
                   rank, ExeTypeStr[exeType], gpuIdx,
                   MemTypeStr[gpuMemType],
                   numSubExec, numBytes);
        } else if (isGather) {
          // Split up the number of CUs across all Transfers
          snprintf(transferStr, MAX_TRANSFER_STRLEN, "-1 (R*%c* R%d%c%d R%d%c%d %d %lu)",
                   MemTypeStr[gpuMemType],
                   rank, ExeTypeStr[exeType], gpuIdx,
                   rank, MemTypeStr[gpuMemType], gpuIdx,
                   numSubExec, numBytes);
        } else if (isAllToAll) {
          // Split up the number of CUs across all Transfers
          snprintf(transferStr, MAX_TRANSFER_STRLEN, "-1 (R%d%c%d R%d%c%d R*%c* %d %lu)",
                   rank, MemTypeStr[gpuMemType], gpuIdx,
                   rank, ExeTypeStr[exeType], gpuIdx,
                   MemTypeStr[gpuMemType],
                   numSubExec, numBytes);
        }

        ErrResult err = ParseTransfers(transferStr, transfers);
        if (err.errType != ERR_NONE) {
          Utils::Print("[ERROR] Unexpected parsing error - %s.  This is a coding error\n", err.errMsg.c_str());
          exit(1);
        }

        if (isBroadcast || isGather) {
          if (!RunTransfers(cfg, transfers, results)) {
            allPass = false;
            break;
          }
        } else {
          allTransfers.insert(allTransfers.end(), transfers.begin(), transfers.end());
        }
      }
    }
    if (!(isBroadcast || isGather)) {
      if (!RunTransfers(cfg, allTransfers, results)) {
        allPass = false;
      }
    }
    Utils::Print("%s", allPass ? pass.c_str() : fail.c_str()); fflush(stdout);
    numFail += (allPass ? 0 : 1);
  }
  return numFail;
}

int SmokeTestPreset(EnvVars&          ev,
                    size_t      const numBytesPerTransfer,
                    std::string const presetName,
                    bool        const bytesSpecified)
{
  // Check for single pod
  if (Utils::GetRankPerPodMap().size() > 1) {
    Utils::Print("[ERROR] %s preset can only be run within a single pod\n", presetName.c_str());
    Utils::Print("[ERROR] Pod membership may be forced by setting TB_FORCE_SINGLE_POD=1\n");
    return ERR_FATAL;
  }

  // Collect topology and check that all GPUs have the same number of subExecutors
  int numRanks = TransferBench::GetNumRanks();
  int totalGpus = 0;
  int numSubExec = TransferBench::GetNumSubExecutors({EXE_GPU_GFX, 0, 0});
  for (int rank = 0; rank < numRanks; rank++) {
    int numGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX, rank);
    totalGpus += numGpus;
    for (int gpu = 0; gpu < numGpus; gpu++) {
      if (numSubExec != TransferBench::GetNumSubExecutors({EXE_GPU_GFX, gpu, rank})) {
        Utils::Print("[ERROR] %s preset can only be run on GPUs with the same number of subexecutors\n", presetName.c_str());
        return ERR_FATAL;
      }
    }
  }

  // Modify defaults unless they were set
  if (!getenv("ALWAYS_VALIDATE")) ev.alwaysValidate = 1;
  if (!getenv("NUM_ITERATIONS" )) ev.numIterations  = 2;
  if (!getenv("NUM_WARMUPS"    )) ev.numWarmups     = 0;

  // Collect env vars
  int                 cpuMemTypeIdx = EnvVars::GetEnvVar          ("CPU_MEM_TYPE",                  0);
  int                 gpuMemTypeIdx = EnvVars::GetEnvVar          ("GPU_MEM_TYPE",                  0);
  vector<int>         gfxSesList    = EnvVars::GetEnvVarArray     ("GFX_SE_LIST",      {1,numSubExec});
  std::string         seMaxBytesStr = EnvVars::GetEnvVar          ("SE_MAX_BYTES",             "128M");
  vector<std::string> sizeStrList   = EnvVars::GetEnvVarStrArray  ("SIZE_LIST",   {"1K","16M","256M"});
  vector<int>         testList      = EnvVars::GetEnvVarRangeArray("TEST_LIST",                    {});

  MemType cpuMemType = Utils::GetCpuMemType(cpuMemTypeIdx);
  MemType gpuMemType = Utils::GetGpuMemType(gpuMemTypeIdx);
  std::set<int> testsToRun(testList.begin(), testList.end());
  if (testList.empty()) {
    for (int testIdx = 1; testIdx <= NUM_SMOKE_TESTS; testIdx++)
      testsToRun.insert(testIdx);
  }

  vector<size_t> sizeList;
  if (bytesSpecified) {
    sizeList = {numBytesPerTransfer};
  } else {
    for (auto s : sizeStrList) {
      size_t val;
      if (sscanf(s.c_str(), "%lu", &val) == 1) {
        switch (s[s.size()-1]) {
        case 'G': case 'g': val *= 1024;
        case 'M': case 'm': val *= 1024;
        case 'K': case 'k': val *= 1024;
        }
        sizeList.push_back(val);
      }
    }
  }
  size_t seMaxBytes = 128 * 1024 * 1024;
  if (sscanf(seMaxBytesStr.c_str(), " %lu", &seMaxBytes) == 1) {
    switch (seMaxBytesStr[seMaxBytesStr.size()-1]) {
    case 'G': case 'g': seMaxBytes *= 1024;
    case 'M': case 'm': seMaxBytes *= 1024;
    case 'K': case 'k': seMaxBytes *= 1024;
    }
  }

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();

  // Print off environment variables
  if (Utils::RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
      if (!ev.outputToCsv) printf("[%s-preset Related]\n", presetName.c_str());
      ev.Print("CPU_MEM_TYPE", cpuMemTypeIdx,      "Using %s (%s)", Utils::GetCpuMemTypeStr(cpuMemTypeIdx).c_str(), Utils::GetAllCpuMemTypeStr().c_str());
      ev.Print("GFX_SE_LIST" , gfxSesList.size(),  "Testing GFX with subexecutor counts: %s", EnvVars::ToStr(gfxSesList).c_str());
      ev.Print("GPU_MEM_TYPE", gpuMemTypeIdx,      "Using %s (%s)", Utils::GetGpuMemTypeStr(gpuMemTypeIdx).c_str(), Utils::GetAllGpuMemTypeStr().c_str());
      ev.Print("SIZE_LIST"   , sizeStrList.size(), "Transfer sizes tested: %s", ev.GetStr(sizeStrList).c_str());
      ev.Print("SE_MAX_BYTES", seMaxBytesStr,      "Each SubExecutor can work on at most %lu bytes", seMaxBytes);
      ev.Print("TEST_LIST"   , testsToRun.size(),  testList.empty() ? "Running all tests" : "Running Tests: %s", ev.GetStr(testList).c_str());
      printf("\n");
    }
  }

  // Calculate cell-spacing / padding
  int numSizes  = sizeList.size();
  int colSize   = std::max(5, 2 + numSizes);
  int lPad1Size = (colSize -        3) / 2, rPad1Size = colSize - lPad1Size - 3;
  int lPad2Size = (colSize - numSizes) / 2, rPad2Size = colSize - lPad2Size - numSizes;

  std::string l1(lPad1Size, ' '), r1(rPad1Size, ' ');
  std::string l2(lPad2Size, ' '), r2(rPad2Size, ' ');

  int testsFailed = 0;
  auto test = [&](int x, int y) {
    Utils::Print("  %02d  |%s", x, l2.c_str());
    fflush(stdout);
    testsFailed += RunTest(x, testsToRun, sizeList, 1, cfg, cpuMemType, gpuMemType, seMaxBytes, totalGpus);
    Utils::Print("%s|  %02d  |", r2.c_str(), y);
    for (auto numSubExec : gfxSesList) {
      Utils::Print("%s", l2.c_str());
      fflush(stdout);
      testsFailed += RunTest(y, testsToRun, sizeList, numSubExec, cfg, cpuMemType, gpuMemType, seMaxBytes, totalGpus);
      Utils::Print("%s|", r2.c_str());
    }
    Utils::Print("\n");
    fflush(stdout);
  };

  Utils::Print("Running tests on %d GPUs total across %d rank(s)\n", totalGpus, numRanks);
  Utils::Print("Legend: %s=Pass %s=Skip %s=Fail\n", pass.c_str(), skip.c_str(), fail.c_str());

  // Print headers
  Utils::Print("                                    %s   %s       |", l1.c_str(), r1.c_str());
  for ([[maybe_unused]] auto numSubExec : gfxSesList)
    Utils::Print("%sGFX%s|", l1.c_str(), r1.c_str());
  Utils::Print("\n");
  Utils::Print("| Name                      | Test |%sDMA%s| Test |", l1.c_str(), r1.c_str());
  for (auto numSubExec : gfxSesList)
    Utils::Print("%s%03d%s|", l1.c_str(), numSubExec, r1.c_str());
  Utils::Print("\n");
  Utils::Print("|---------------------------|------|%s|------|", std::string(colSize, '-').c_str());
  for ([[maybe_unused]] auto numSubExec : gfxSesList)
    Utils::Print("%s|", std::string(colSize, '-').c_str());
  Utils::Print("\n");

  // Print table / Run Tests
  Utils::Print("| Copy (H2D)                |"); test(1, 8);
  Utils::Print("| Copy (D2H)                |"); test(2, 9);
  Utils::Print("| Copy (D2D) (Remote Write) |"); test(3,10);
  Utils::Print("| Copy (D2D) (Remote Read ) |"); test(4,11);
  Utils::Print("| Broadcast  (One to All)   |"); test(5,12);
  Utils::Print("| Gather     (All to One)   |"); test(6,13);
  Utils::Print("| All To All                |"); test(7,14);

  Utils::Print("|---------------------------|------|%s|------|", std::string(colSize, '-').c_str());
  for ([[maybe_unused]] auto numSubExec : gfxSesList)
    Utils::Print("%s|", std::string(colSize, '-').c_str());
  Utils::Print("\n\n");

  // Show summary
  if (testsFailed) {
    Utils::Print("[WARN] %d Tests FAILED\n", testsFailed);
  } else {
    Utils::Print("All tests passed\n");
  }
  if (numRanks > 1 && Utils::GetRankPerPodMap().size() != 1) {
    Utils::Print("[WARN] Copy (D2D) / Broadcast / Gather / AllToAll tests are skipped if ranks are not in same pod\n");
  }
  if (Utils::HasDuplicateHostname()) {
    Utils::Print("[WARN] It is recommended to run TransferBench with one rank per host to avoid potential aliasing of executors\n");
  }
  return testsFailed ? ERR_FATAL : ERR_NONE;
}

}

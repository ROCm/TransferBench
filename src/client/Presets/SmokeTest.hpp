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

#define NUM_SMOKE_TESTS 22
#define MAX_TRANSFER_STRLEN 128

// What to print on pass/fail/valFail/skip
const std::string pass    = "*";
const std::string fail    = "F";
const std::string valFail = "V";
const std::string skip    = ".";

// Executor override (set via TB_FORCE_EXEC=AUTO|DMA|GFX).
// Only affects GPU-executor tests (3-22); CPU-executor tests (1-2) always use EXE_CPU.
enum ForceExec { FORCE_AUTO, FORCE_DMA, FORCE_GFX };

// Distinguish validation failures from other fatal errors. The validation path in
// TransferBench.hpp emits "Unexpected mismatch" / "Unexpected output mismatch"
// prefixes; if those strings change upstream, V will silently downgrade to F.
inline bool IsValidationFailure(TestResults const& r) {
  for (auto const& e : r.errResults) {
    if (e.errType != ERR_FATAL) continue;
    if (e.errMsg.find("Unexpected mismatch")        != std::string::npos ||
        e.errMsg.find("Unexpected output mismatch") != std::string::npos)
      return true;
  }
  return false;
}

// Forward decl: tests 1 (H2D_RW) and 2 (D2H_RR) are CPU-executor tests dispatched
// from the top of RunTest.
int RunCpuTest(int                        testNum,
               std::set<int> const&       testsToRun,
               std::vector<size_t> const& sizeList,
               ConfigOptions&             cfg,
               MemType                    cpuMemType,
               MemType                    gpuMemType,
               bool                       isParallel,
               int                        targetGpu,
               int&                       numValFailOut);

int RunTest(int                        testNum,
            std::set<int> const&       testsToRun,
            std::vector<size_t> const& sizeList,
            int                        numSubExecPerGpu,
            ConfigOptions&             cfg,
            MemType                    cpuMemType,
            MemType                    gpuMemType,
            size_t                     maxBytesPerSubExec,
            bool                       isParallel,
            int                        targetGpu,
            int                        totalGpus,
            ForceExec                  forceExec,
            int&                       numValFailOut)
{
  // Route CPU-executor tests to a dedicated function. forceExec is ignored for
  // these (CPU tests cannot be coerced onto a GPU executor).
  if (testNum == 1 || testNum == 2)
    return RunCpuTest(testNum, testsToRun, sizeList, cfg, cpuMemType, gpuMemType,
                      isParallel, targetGpu, numValFailOut);

  int numFail = 0;

  // Collect some topology information
  int numRanks = TransferBench::GetNumRanks();

  std::vector<Transfer> transfers;
  std::vector<Transfer> allTransfers;
  TestResults results;
  char transferStr[MAX_TRANSFER_STRLEN] = {};


  static std::vector<pair<int, int>> gpuToDeviceMap;
  if (gpuToDeviceMap.empty()) {
    for (int r = 0; r < numRanks; r++) {
      int numGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX, r);
      for (int g = 0; g < numGpus; g++) {
        gpuToDeviceMap.push_back(std::make_pair(r, g));
      }
    }
  }
  int targetRank = gpuToDeviceMap[targetGpu].first;
  int targetIdx  = gpuToDeviceMap[targetGpu].second;

  // Different test categories (tests 1-2 routed to RunCpuTest above)
  bool isH2D_RR       = (testNum ==  3 || testNum == 13);
  bool isD2H_RW       = (testNum ==  4 || testNum == 14);
  bool isD2D_RW       = (testNum ==  5 || testNum == 15);
  bool isD2D_RR       = (testNum ==  6 || testNum == 16);
  bool isBroadcast_RW = (testNum ==  7 || testNum == 17);
  bool isBroadcast_RR = (testNum ==  8 || testNum == 18);
  bool isGather_RW    = (testNum ==  9 || testNum == 19);
  bool isGather_RR    = (testNum == 10 || testNum == 20);
  bool isAllToAll_RW  = (testNum == 11 || testNum == 21);
  bool isAllToAll_RR  = (testNum == 12 || testNum == 22);

  // Determine executor type (TB_FORCE_EXEC overrides the natural range mapping)
  ExeType exeType;
  if      (forceExec == FORCE_DMA)         exeType = EXE_GPU_DMA;
  else if (forceExec == FORCE_GFX)         exeType = EXE_GPU_GFX;
  else if (3  <= testNum && testNum <= 12) exeType = EXE_GPU_DMA;
  else if (13 <= testNum && testNum <= 22) exeType = EXE_GPU_GFX;
  else {
    Utils::Print("[ERROR] Unsupported test number %d\n", testNum);
    exit(1);
  }

  // Adjust number of subexecutors per transfer if performing multiple transfers
  int numSubExec = exeType == EXE_GPU_DMA ? 1 : numSubExecPerGpu;
  if (exeType == EXE_GPU_GFX && (isBroadcast_RW || isBroadcast_RR ||
                                 isGather_RW    || isGather_RR    ||
                                 isAllToAll_RW  || isAllToAll_RR))
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
    // Skip cross-pod tests except the two within-rank H2D/D2H GPU-executor variants
    if (numRanks > 1 && Utils::GetRankPerPodMap().size() != 1 && !(isH2D_RR || isD2H_RW)) {
      Utils::Print("%s", skip.c_str()); fflush(stdout);
      continue;
    }

    bool allPass = true;
    allTransfers.clear();

    // Combine transfers from each GPU and run them all in parallel (unless isParallel=false)
    for (int rank = 0; allPass && rank < numRanks; rank++) {
      if (!isParallel && rank != targetRank) continue;
      int numGpus = GetNumExecutors(exeType, rank);
      for (int gpuIdx = 0; allPass && gpuIdx < numGpus; gpuIdx++) {
        if (!isParallel && gpuIdx != targetIdx) continue;
        if (isH2D_RR || isD2H_RW) {
          // GPU-executor copy to/from closest CPU NUMA node for this GPU
          int cpuIdx = GetClosestCpuNumaToGpu(gpuIdx, rank);
          snprintf(transferStr, MAX_TRANSFER_STRLEN, "-1 (R%d%c%d R%d%c%d R%d%c%d %d %lu)",
                   rank, MemTypeStr[isH2D_RR ? cpuMemType : gpuMemType], isH2D_RR ? cpuIdx : gpuIdx,
                   rank, ExeTypeStr[exeType], gpuIdx,
                   rank, MemTypeStr[isH2D_RR ? gpuMemType : cpuMemType], isH2D_RR ? gpuIdx : cpuIdx,
                   numSubExec, numBytes);

          ErrResult err = ParseTransfers(transferStr, transfers);
          if (err.errType != ERR_NONE) {
            Utils::Print("[ERROR] Unexpected parsing error - %s.  This is a coding error\n", err.errMsg.c_str());
            exit(1);
          }
          allTransfers.insert(allTransfers.end(), transfers.begin(), transfers.end());

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

          ErrResult err = ParseTransfers(transferStr, transfers);
          if (err.errType != ERR_NONE) {
            Utils::Print("[ERROR] Unexpected parsing error - %s.  This is a coding error\n", err.errMsg.c_str());
            exit(1);
          }
          allTransfers.insert(allTransfers.end(), transfers.begin(), transfers.end());

        } else if (isBroadcast_RW) {
          // One transfer with R* dst, exec on src
          snprintf(transferStr, MAX_TRANSFER_STRLEN, "-1 (R%d%c%d R%d%c%d R*%c* %d %lu)",
                   rank, MemTypeStr[gpuMemType], gpuIdx,
                   rank, ExeTypeStr[exeType], gpuIdx,
                   MemTypeStr[gpuMemType],
                   numSubExec, numBytes);

          ErrResult err = ParseTransfers(transferStr, transfers);
          if (err.errType != ERR_NONE) {
            Utils::Print("[ERROR] Unexpected parsing error - %s.  This is a coding error\n", err.errMsg.c_str());
            exit(1);
          }
          // Inline launch (wildcard-based single-transfer-per-iteration)
          if (!RunTransfers(cfg, transfers, results)) { allPass = false; break; }

        } else if (isBroadcast_RR) {
          // 1 src (this GPU) -> N dsts; executor on EACH dst -> N transfers, one per dst
          for (int r2 = 0; r2 < numRanks; r2++) {
            int numDstGpus = GetNumExecutors(exeType, r2);
            for (int dstIdx = 0; dstIdx < numDstGpus; dstIdx++) {
              if (r2 == rank && dstIdx == gpuIdx) continue;
              snprintf(transferStr, MAX_TRANSFER_STRLEN, "-1 (R%d%c%d R%d%c%d R%d%c%d %d %lu)",
                       rank, MemTypeStr[gpuMemType], gpuIdx,
                       r2,   ExeTypeStr[exeType],    dstIdx,
                       r2,   MemTypeStr[gpuMemType], dstIdx,
                       numSubExec, numBytes);

              ErrResult err = ParseTransfers(transferStr, transfers);
              if (err.errType != ERR_NONE) {
                Utils::Print("[ERROR] Unexpected parsing error - %s.  This is a coding error\n", err.errMsg.c_str());
                exit(1);
              }
              allTransfers.insert(allTransfers.end(), transfers.begin(), transfers.end());
            }
          }

        } else if (isGather_RW) {
          // N srcs -> 1 dst (this GPU); executor on EACH src -> N transfers, one per src
          for (int r2 = 0; r2 < numRanks; r2++) {
            int numSrcGpus = GetNumExecutors(exeType, r2);
            for (int srcIdx = 0; srcIdx < numSrcGpus; srcIdx++) {
              if (r2 == rank && srcIdx == gpuIdx) continue;
              snprintf(transferStr, MAX_TRANSFER_STRLEN, "-1 (R%d%c%d R%d%c%d R%d%c%d %d %lu)",
                       r2,   MemTypeStr[gpuMemType], srcIdx,
                       r2,   ExeTypeStr[exeType],    srcIdx,
                       rank, MemTypeStr[gpuMemType], gpuIdx,
                       numSubExec, numBytes);

              ErrResult err = ParseTransfers(transferStr, transfers);
              if (err.errType != ERR_NONE) {
                Utils::Print("[ERROR] Unexpected parsing error - %s.  This is a coding error\n", err.errMsg.c_str());
                exit(1);
              }
              allTransfers.insert(allTransfers.end(), transfers.begin(), transfers.end());
            }
          }

        } else if (isGather_RR) {
          // One transfer with R* src, exec on dst
          snprintf(transferStr, MAX_TRANSFER_STRLEN, "-1 (R*%c* R%d%c%d R%d%c%d %d %lu)",
                   MemTypeStr[gpuMemType],
                   rank, ExeTypeStr[exeType], gpuIdx,
                   rank, MemTypeStr[gpuMemType], gpuIdx,
                   numSubExec, numBytes);

          ErrResult err = ParseTransfers(transferStr, transfers);
          if (err.errType != ERR_NONE) {
            Utils::Print("[ERROR] Unexpected parsing error - %s.  This is a coding error\n", err.errMsg.c_str());
            exit(1);
          }
          // Inline launch (wildcard-based single-transfer-per-iteration)
          if (!RunTransfers(cfg, transfers, results)) { allPass = false; break; }

        } else if (isAllToAll_RW) {
          // R* dst, exec on src -- one transfer per src, dst expanded by wildcard
          snprintf(transferStr, MAX_TRANSFER_STRLEN, "-1 (R%d%c%d R%d%c%d R*%c* %d %lu)",
                   rank, MemTypeStr[gpuMemType], gpuIdx,
                   rank, ExeTypeStr[exeType], gpuIdx,
                   MemTypeStr[gpuMemType],
                   numSubExec, numBytes);

          ErrResult err = ParseTransfers(transferStr, transfers);
          if (err.errType != ERR_NONE) {
            Utils::Print("[ERROR] Unexpected parsing error - %s.  This is a coding error\n", err.errMsg.c_str());
            exit(1);
          }
          allTransfers.insert(allTransfers.end(), transfers.begin(), transfers.end());

        } else if (isAllToAll_RR) {
          // Like AllToAll_RW but exec on dst -> one transfer per (this-src, each-dst)
          for (int r2 = 0; r2 < numRanks; r2++) {
            int numDstGpus = GetNumExecutors(exeType, r2);
            for (int dstIdx = 0; dstIdx < numDstGpus; dstIdx++) {
              if (r2 == rank && dstIdx == gpuIdx) continue;
              snprintf(transferStr, MAX_TRANSFER_STRLEN, "-1 (R%d%c%d R%d%c%d R%d%c%d %d %lu)",
                       rank, MemTypeStr[gpuMemType], gpuIdx,
                       r2,   ExeTypeStr[exeType],    dstIdx,
                       r2,   MemTypeStr[gpuMemType], dstIdx,
                       numSubExec, numBytes);

              ErrResult err = ParseTransfers(transferStr, transfers);
              if (err.errType != ERR_NONE) {
                Utils::Print("[ERROR] Unexpected parsing error - %s.  This is a coding error\n", err.errMsg.c_str());
                exit(1);
              }
              allTransfers.insert(allTransfers.end(), transfers.begin(), transfers.end());
            }
          }
        }
      }
    }
    // Inline-launch builders (Broadcast_RW, Gather_RR) already ran RunTransfers
    // per iteration; everything else launches the accumulated batch here.
    if (!(isBroadcast_RW || isGather_RR)) {
      if (!RunTransfers(cfg, allTransfers, results)) {
        allPass = false;
      }
    }
    bool valFailed = !allPass && IsValidationFailure(results);
    std::string const& sym = allPass   ? pass
                           : valFailed ? valFail
                           :             fail;
    Utils::Print("%s", sym.c_str()); fflush(stdout);
    numFail       += (allPass   ? 0 : 1);
    numValFailOut += (valFailed ? 1 : 0);
  }
  return numFail;
}

int RunCpuTest(int                        testNum,
               std::set<int> const&       testsToRun,
               std::vector<size_t> const& sizeList,
               ConfigOptions&             cfg,
               MemType                    cpuMemType,
               MemType                    gpuMemType,
               bool                       isParallel,
               int                        targetGpu,
               int&                       numValFailOut)
{
  int numFail = 0;
  int numRanks = TransferBench::GetNumRanks();

  std::vector<Transfer> transfers;
  std::vector<Transfer> allTransfers;
  TestResults results;
  char transferStr[MAX_TRANSFER_STRLEN] = {};

  static std::vector<pair<int, int>> gpuToDeviceMap;
  if (gpuToDeviceMap.empty()) {
    for (int r = 0; r < numRanks; r++) {
      int numGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX, r);
      for (int g = 0; g < numGpus; g++) {
        gpuToDeviceMap.push_back(std::make_pair(r, g));
      }
    }
  }
  int targetRank = gpuToDeviceMap[targetGpu].first;
  int targetIdx  = gpuToDeviceMap[targetGpu].second;

  // testNum == 1: H2D_RW (CPU writes to GPU)
  // testNum == 2: D2H_RR (CPU reads from GPU)
  bool isH2D_RW = (testNum == 1);

  for (size_t numBytes : sizeList) {
    if (!testsToRun.count(testNum)) {
      Utils::Print("%s", skip.c_str()); fflush(stdout);
      continue;
    }
    // No pod-skip needed: CPU<->GPU transfers are always within-rank.

    bool allPass = true;
    allTransfers.clear();

    for (int rank = 0; allPass && rank < numRanks; rank++) {
      if (!isParallel && rank != targetRank) continue;
      int numGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX, rank);
      for (int gpuIdx = 0; allPass && gpuIdx < numGpus; gpuIdx++) {
        if (!isParallel && gpuIdx != targetIdx) continue;
        int cpuIdx = GetClosestCpuNumaToGpu(gpuIdx, rank);
        snprintf(transferStr, MAX_TRANSFER_STRLEN, "-1 (R%d%c%d R%d%c%d R%d%c%d %d %lu)",
                 rank, MemTypeStr[isH2D_RW ? cpuMemType : gpuMemType], isH2D_RW ? cpuIdx : gpuIdx,
                 rank, ExeTypeStr[EXE_CPU],                            cpuIdx,
                 rank, MemTypeStr[isH2D_RW ? gpuMemType : cpuMemType], isH2D_RW ? gpuIdx : cpuIdx,
                 1, numBytes);

        ErrResult err = ParseTransfers(transferStr, transfers);
        if (err.errType != ERR_NONE) {
          Utils::Print("[ERROR] Unexpected parsing error - %s.  This is a coding error\n", err.errMsg.c_str());
          exit(1);
        }
        allTransfers.insert(allTransfers.end(), transfers.begin(), transfers.end());
      }
    }
    if (!RunTransfers(cfg, allTransfers, results)) allPass = false;

    bool valFailed = !allPass && IsValidationFailure(results);
    std::string const& sym = allPass   ? pass
                           : valFailed ? valFail
                           :             fail;
    Utils::Print("%s", sym.c_str()); fflush(stdout);
    numFail       += (allPass   ? 0 : 1);
    numValFailOut += (valFailed ? 1 : 0);
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
  ev.alwaysValidate = EnvVars::GetEnvVar("ALWAYS_VALIDATE", 1);
  ev.numIterations  = EnvVars::GetEnvVar("NUM_ITERATIONS",  2);
  ev.numWarmups     = EnvVars::GetEnvVar("NUM_WARMUPS",     0);

  // Collect env vars
  int                 cpuMemTypeIdx = EnvVars::GetEnvVar          ("CPU_MEM_TYPE",                  0);
  std::string         forceExecStr  = EnvVars::GetEnvVar          ("TB_FORCE_EXEC",            "AUTO");
  int                 gpuMemTypeIdx = EnvVars::GetEnvVar          ("GPU_MEM_TYPE",                  0);
  vector<int>         gfxSesList    = EnvVars::GetEnvVarArray     ("GFX_SE_LIST",      {1,numSubExec});
  int                 runParallel   = EnvVars::GetEnvVar          ("RUN_PARALLEL",                  1);
  std::string         seMaxBytesStr = EnvVars::GetEnvVar          ("SE_MAX_BYTES",             "128M");
  vector<std::string> sizeStrList   = EnvVars::GetEnvVarStrArray  ("SIZE_LIST",   {"1K","16M","256M"});
  vector<int>         testList      = EnvVars::GetEnvVarRangeArray("TEST_LIST",                    {});

  // Parse and validate TB_FORCE_EXEC
  ForceExec forceExec = FORCE_AUTO;
  if      (forceExecStr == "AUTO") forceExec = FORCE_AUTO;
  else if (forceExecStr == "DMA")  forceExec = FORCE_DMA;
  else if (forceExecStr == "GFX")  forceExec = FORCE_GFX;
  else {
    Utils::Print("[ERROR] TB_FORCE_EXEC must be one of AUTO|DMA|GFX (got '%s')\n", forceExecStr.c_str());
    return ERR_FATAL;
  }

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
      ev.Print("RUN_PARALLEL", runParallel,        "Running GPUs %s", runParallel ? "in parallel" : "serially");
      ev.Print("SIZE_LIST"   , sizeStrList.size(), "Transfer sizes tested: %s", ev.GetStr(sizeStrList).c_str());
      ev.Print("SE_MAX_BYTES", seMaxBytesStr,      "Each SubExecutor can work on at most %lu bytes", seMaxBytes);
      ev.Print("TB_FORCE_EXEC", forceExecStr,     "Force executor: %s (column header still reflects test category)", forceExecStr.c_str());
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

  int testsFailed    = 0;
  int testsValFailed = 0;
  int numCpuSubExec  = 1;  // CPU executors map 1:1 to host threads; passed for uniformity.

  auto ExecuteTests = [&](std::string label, int cpuTest, int dmaTest, int gfxTest, bool isParallel) {
    int numLines = isParallel ? 1 : totalGpus;

    for (int line = 0; line < numLines; line++) {
      Utils::Print("| %25s |", line ? "" : label.c_str());
      if (isParallel) {
        Utils::Print(" ALL |");
      } else {
        Utils::Print(" %3d |", line);
      }

      auto PrintTestCell = [&](int testNum) {
        if (line == 0) {
          if (testNum < 0) Utils::Print("  --  |");
          else             Utils::Print("  %02d  |", testNum);
        } else {
          Utils::Print("      |");
        }
      };

      auto RunOne = [&](int testNum, int numSubExecPerGpu) {
        if (testNum < 0) {
          // Blank result cell for rows that don't apply to this column group.
          Utils::Print("%s|", std::string(colSize, ' ').c_str());
          return;
        }
        Utils::Print("%s", l2.c_str());
        fflush(stdout);
        testsFailed += RunTest(testNum, testsToRun, sizeList, numSubExecPerGpu, cfg,
                               cpuMemType, gpuMemType, seMaxBytes, isParallel,
                               line, totalGpus,
                               forceExec, testsValFailed);
        Utils::Print("%s|", r2.c_str());
      };

      // CPU result group
      PrintTestCell(cpuTest);
      RunOne(cpuTest, numCpuSubExec);

      // DMA result group
      PrintTestCell(dmaTest);
      RunOne(dmaTest, 1);

      // GFX result group (one sub-cell per gfxSesList entry)
      PrintTestCell(gfxTest);
      for (auto numSubExec : gfxSesList) {
        RunOne(gfxTest, numSubExec);
      }

      Utils::Print("\n");
      fflush(stdout);
    }
    Utils::Print("|---------------------------|-----|------|%s|------|%s|------|",
                 std::string(colSize, '-').c_str(),
                 std::string(colSize, '-').c_str());
    for ([[maybe_unused]] auto numSubExec : gfxSesList)
      Utils::Print("%s|", std::string(colSize, '-').c_str());
    Utils::Print("\n");
  };

  Utils::Print("Running tests on %d GPUs total across %d rank(s)\n", totalGpus, numRanks);
  Utils::Print("Legend: %s=Pass %s=Skip %s=Fail %s=ValidationFail | Columns (natural mapping; see TB_FORCE_EXEC): CPU=EXE_CPU DMA=EXE_GPU_DMA GFX=EXE_GPU_GFX\n",
               pass.c_str(), skip.c_str(), fail.c_str(), valFail.c_str());

  // Print headers
  Utils::Print("                                          %sCPU%s       %sDMA%s       |",
               l1.c_str(), r1.c_str(), l1.c_str(), r1.c_str());
  for ([[maybe_unused]] auto numSubExec : gfxSesList)
    Utils::Print("%sGFX%s|", l1.c_str(), r1.c_str());
  Utils::Print("\n");
  Utils::Print("| Name                      | GPU | Test |%sCPU%s| Test |%sDMA%s| Test |",
               l1.c_str(), r1.c_str(), l1.c_str(), r1.c_str());
  for (auto numSubExec : gfxSesList)
    Utils::Print("%s%03d%s|", l1.c_str(), numSubExec, r1.c_str());
  Utils::Print("\n");
  Utils::Print("|---------------------------|-----|------|%s|------|%s|------|",
               std::string(colSize, '-').c_str(),
               std::string(colSize, '-').c_str());
  for ([[maybe_unused]] auto numSubExec : gfxSesList)
    Utils::Print("%s|", std::string(colSize, '-').c_str());
  Utils::Print("\n");

  // Print table / Run Tests
  ExecuteTests("Copy (H2D) (Remote Write)",  1, -1, -1, runParallel);
  ExecuteTests("Copy (H2D) (Remote Read )", -1,  3, 13, runParallel);
  ExecuteTests("Copy (D2H) (Remote Write)", -1,  4, 14, runParallel);
  ExecuteTests("Copy (D2H) (Remote Read )",  2, -1, -1, runParallel);
  ExecuteTests("Copy (D2D) (Remote Write)", -1,  5, 15, runParallel);
  ExecuteTests("Copy (D2D) (Remote Read )", -1,  6, 16, runParallel);
  ExecuteTests("Broadcast  (Remote Write)", -1,  7, 17, runParallel);
  ExecuteTests("Broadcast  (Remote Read )", -1,  8, 18, runParallel);
  ExecuteTests("Gather     (Remote Write)", -1,  9, 19, runParallel);
  ExecuteTests("Gather     (Remote Read )", -1, 10, 20, runParallel);
  ExecuteTests("All To All (Remote Write)", -1, 11, 21, true);
  ExecuteTests("All To All (Remote Read )", -1, 12, 22, true);

  // Show summary
  if (testsFailed) {
    Utils::Print("[WARN] %d Tests FAILED (%d validation)\n", testsFailed, testsValFailed);
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

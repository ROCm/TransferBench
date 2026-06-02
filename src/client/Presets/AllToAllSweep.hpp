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

#include "EnvVars.hpp"

int AllToAllSweepPreset(EnvVars&          ev,
                        size_t      const numBytesPerTransfer,
                        std::string const presetName,
                        bool        const bytesSpecified)
{
  if (TransferBench::GetNumRanks() > 1) {
    Utils::Print("[ERROR] All to All Sweep preset currently not supported for multi-node\n");
    return ERR_FATAL;
  }

  enum
  {
    A2A_COPY       = 0,
    A2A_READ_ONLY  = 1,
    A2A_WRITE_ONLY = 2,
    A2A_CUSTOM     = 3,
  };
  char a2aModeStr[4][20] = {"Copy", "Read-Only", "Write-Only", "Custom"};

  // Force single-stream mode for all-to-all benchmark
  ev.useSingleStream = 1;
  // Default to GPU-event timing for a2asweep (overridable via USE_HIP_EVENTS=0 for CPU wall-clock)
  ev.useHipEvents = EnvVars::GetEnvVar("USE_HIP_EVENTS", 1);

  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  // Collect env vars for this preset
  int a2aDirect     = EnvVars::GetEnvVar("A2A_DIRECT"     , 1);
  int a2aLocal      = EnvVars::GetEnvVar("A2A_LOCAL"      , 0);
  int memTypeIdx    = EnvVars::GetEnvVar("MEM_TYPE"       , 2);
  int numGpus       = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int showMinOnly   = EnvVars::GetEnvVar("SHOW_MIN_ONLY",   1);
  int useRemoteRead = EnvVars::GetEnvVar("USE_REMOTE_READ", 0);
  int useSpray      = EnvVars::GetEnvVar("USE_SPRAY",       0);
  int verbose       = EnvVars::GetEnvVar("VERBOSE",         0);

  std::vector<int> blockList  = EnvVars::GetEnvVarArray("BLOCKSIZES", {256,512,768,1024});
  std::vector<int> unrollList = EnvVars::GetEnvVarArray("UNROLLS", {1,2,3,4,6,8});
  std::vector<int> numSesList = EnvVars::GetEnvVarArray("NUM_SUB_EXECS", {4,8,12,16,24,32});

  // A2A_MODE may be 0,1,2 or else custom numSrcs:numDsts
  int numSrcs, numDsts;
  int a2aMode = 0;
  if (getenv("A2A_MODE") && sscanf(getenv("A2A_MODE"), "%d:%d", &numSrcs, &numDsts) == 2) {
    a2aMode = A2A_CUSTOM;
  } else {
    a2aMode = EnvVars::GetEnvVar("A2A_MODE", 0);
    if (a2aMode < 0 || a2aMode > 2) {
      printf("[ERROR] a2aMode must be between 0 and 2, or else numSrcs:numDsts\n");
      exit(1);
    }
    numSrcs = (a2aMode == A2A_WRITE_ONLY ? 0 : 1);
    numDsts = (a2aMode == A2A_READ_ONLY  ? 0 : 1);
  }

  MemType memType = Utils::GetGpuMemType(memTypeIdx);
  std::string devMemTypeStr = Utils::GetGpuMemTypeStr(memTypeIdx);

  // Print off environment variables
  ev.DisplayEnvVars();
  if (!ev.hideEnv) {
    if (!ev.outputToCsv) printf("[AllToAll Related]\n");
    ev.Print("A2A_DIRECT"     , a2aDirect        , a2aDirect ? "Only using direct links" : "Full all-to-all");
    ev.Print("A2A_LOCAL"      , a2aLocal         , "%s local transfers", a2aLocal ? "Include" : "Exclude");
    ev.Print("A2A_MODE"       , (a2aMode == A2A_CUSTOM) ?  std::to_string(numSrcs) + ":" + std::to_string(numDsts) : std::to_string(a2aMode),
                                (a2aMode == A2A_CUSTOM) ? (std::to_string(numSrcs) + " read(s) " +
                                                           std::to_string(numDsts) + " write(s)").c_str(): a2aModeStr[a2aMode]);
    ev.Print("BLOCKSIZES"     , blockList.size() , EnvVars::ToStr(blockList).c_str());
    ev.Print("MEM_TYPE"       , memTypeIdx   , "Using %s GPU memory (%s)", devMemTypeStr.c_str(), Utils::GetAllGpuMemTypeStr().c_str());
    ev.Print("NUM_GPU_DEVICES", numGpus          , "Using %d GPUs", numGpus);
    ev.Print("NUM_SUB_EXECS"  , numSesList.size(), EnvVars::ToStr(numSesList).c_str());
    ev.Print("SHOW_MIN_ONLY"  , showMinOnly      , showMinOnly ? "Showing only slowest GPU results" : "Showing slowest and fastest GPU results");
    ev.Print("UNROLLS"        , unrollList.size(), EnvVars::ToStr(unrollList).c_str());
    ev.Print("USE_REMOTE_READ", useRemoteRead    , "Using %s as executor", useRemoteRead ? "DST" : "SRC");
    ev.Print("USE_SPRAY"      , useSpray         , "%s per SubExecutor", useSpray ? "All targets" : "One target");
    ev.Print("VERBOSE"        , verbose          , verbose ? "Display test results" : "Display summary only");
    printf("\n");
  }

  // Validate env vars
  if (numGpus < 0 || numGpus > numDetectedGpus) {
    printf("[ERROR] Cannot use %d GPUs.  Detected %d GPUs\n", numGpus, numDetectedGpus);
    exit(1);
  }

  if (useSpray && numDsts > 1) {
    printf("[ERROR] Cannot use USE_SPRAY with multiple destination buffers\n");
    exit(1);
  }

  // Collect the number of GPU devices to use
  ExeType exeType = EXE_GPU_GFX;

  std::vector<Transfer> transfers;

  int targetCount = 0;
  if (!useSpray) {
    // Each SubExecutor will work on just one target
    for (int i = 0; i < numGpus; i++) {
      targetCount = 0;
      for (int j = 0; j < numGpus; j++) {
        // Check whether or not to execute this pair
        if (i == j) {
          if (!a2aLocal) continue;
        } else if (a2aDirect) {
#if !defined(__NVCC__)
          uint32_t linkType, hopCount;
          HIP_CALL(hipExtGetLinkTypeAndHopCount(i, j, &linkType, &hopCount));
          if (hopCount != 1) continue;
#endif
        }

        // Build Transfer and add it to list
        TransferBench::Transfer transfer;
        targetCount++;
        transfer.numBytes = numBytesPerTransfer;
        for (int x = 0; x < numSrcs; x++) transfer.srcs.push_back({memType, i});

        // When using multiple destinations, the additional destinations are "local"
        if (numDsts) transfer.dsts.push_back({memType, j});
        for (int x = 1; x < numDsts; x++) transfer.dsts.push_back({memType, i});
        transfer.exeDevice = {exeType, (useRemoteRead ? j : i)};
        transfer.exeSubIndex = -1;
        transfers.push_back(transfer);
      }
    }
  } else {
    // Each CU will work on all targets.
    // NOTE: targetCount ends up reflecting the last GPU's target count. This is correct for
    // symmetric topologies (all GPUs have equal peer counts), but may be inaccurate with
    // A2A_DIRECT on asymmetric hardware where different GPUs have different hop-1 peer counts.
    for (int i = 0; i < numGpus; i++) {
      TransferBench::Transfer transfer;
      transfer.numBytes = numBytesPerTransfer;
      transfer.exeDevice = {exeType, i};
      transfer.exeSubIndex = -1;
      targetCount = 0;
      for (int j = 0; j < numGpus; j++) {
        // Check whether or not to transfer to this GPU
        if (i == j) {
          if (!a2aLocal) continue;
        } else if (a2aDirect) {
#if !defined(__NVCC__)
          uint32_t linkType, hopCount;
          HIP_CALL(hipExtGetLinkTypeAndHopCount(i, j, &linkType, &hopCount));
          if (hopCount != 1) continue;
#endif
        }
        targetCount++;
        for (int x = 0; x < numSrcs; x++) transfer.srcs.push_back({memType, useRemoteRead ? j : i});

        if (numDsts) transfer.dsts.push_back({memType, j});
        for (int x = 1; x < numDsts; x++) transfer.dsts.push_back({memType, i});
      }
      transfers.push_back(transfer);
    }
  }

  Utils::Print("GPU-GFX All-To-All Sweep benchmark (%lu bytes, local=%s). All values are %s GB/s\n",
               numBytesPerTransfer,
               a2aLocal         ? "yes"                            : "no",
               ev.useHipEvents  ? "GPU-Event-Timed (min over GPUs)": "CPU-Timed");
  Utils::Print("=======================================================================================\n");
  if (transfers.size() == 0) {
    Utils::Print("[WARN] No transfers requested. Try adjusting A2A_DIRECT or A2A_LOCAL\n");
    return 0;
  }

  // Execute Transfers
  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();

  char sep = ev.outputToCsv ? ',' : ' ';

  double bestMinBw = 0.0;
  int bestBlock = -1, bestUnroll = -1, bestNumSes = -1;

  // Print header once
  Utils::Print(" BlkS %c UnR ", sep);
  for (int c : numSesList) {
    Utils::Print("%c  SE %03d", sep, c);
    if (ev.useHipEvents && !showMinOnly) {
      Utils::Print("%c SE%03dMx", sep, c);
    }
  }
  Utils::Print("\n");

  // Results keyed by (blockSize, numSes, unroll) for verbose output
  std::map<std::tuple<int,int,int>, TransferBench::TestResults> results;

  for (int blockSize : blockList) {
    cfg.gfx.blockSize = blockSize;

    for (int u : unrollList) {
      cfg.gfx.unrollFactor = u;
      Utils::Print("%5d %c %3d ", blockSize, sep, u);
      fflush(stdout);

      for (int c : numSesList) {
        for (auto& transfer : transfers) {
          transfer.numSubExecs = useSpray ? (c * targetCount) : c;
        }

        TransferBench::TestResults result;
        double minBw = 0.0, maxBw = 0.0;
        if (TransferBench::RunTransfers(cfg, transfers, result)) {
          if (!ev.useHipEvents) {
            minBw = result.avgTotalBandwidthGbPerSec;
            if (useSpray) {
              minBw *= targetCount;
            }
          } else {
            minBw = std::numeric_limits<double>::max();
            maxBw = std::numeric_limits<double>::lowest();
            for (auto const& exeResult : result.exeResults) {
              minBw = std::min(minBw, exeResult.second.avgBandwidthGbPerSec);
              maxBw = std::max(maxBw, exeResult.second.avgBandwidthGbPerSec);
            }
            if (useSpray) {
              minBw *= targetCount;
              maxBw *= targetCount;
            }
          }
          if (minBw > bestMinBw) {
            bestMinBw  = minBw;
            bestBlock  = blockSize;
            bestUnroll = u;
            bestNumSes = c;
          }
          if (verbose) {
            results[std::make_tuple(blockSize, c, u)] = result;
          }
        }
        Utils::Print("%c%8.2f", sep, minBw);
        if (ev.useHipEvents && !showMinOnly) {
          Utils::Print("%c%8.2f", sep, maxBw);
        }
        fflush(stdout);
      }
      Utils::Print("\n");
      fflush(stdout);
    }
  }
  Utils::Print("=======================================================================================\n");

  if (verbose) {
    int testNum = 0;
    for (int blockSize : blockList) {
      for (int c : numSesList) {
        for (int u : unrollList) {
          auto verboseTransfers = transfers;
          for (auto& t : verboseTransfers) {
            t.numSubExecs = useSpray ? (c * targetCount) : c;
          }
          Utils::Print("BlockSize: %d SubExecs: %d Unroll: %d\n", blockSize, c, u);
          Utils::PrintResults(ev, ++testNum, verboseTransfers, results[std::make_tuple(blockSize, c, u)]);
        }
      }
    }
  }

  // Print combination that produced highest bandwidth
  if (bestBlock != -1) {
    Utils::Print("Highest %s bandwidth found: %7.2f GB/s\n",
                 ev.useHipEvents ? "GPU-event-timed (min)" : "CPU-timed", bestMinBw);
    Utils::Print("          BlockSize  : %7d\n", bestBlock);
    Utils::Print("          Unroll     : %7d\n", bestUnroll);
    Utils::Print("          NumSubExec : %7d\n", bestNumSes);
  }

  return ERR_NONE;
}

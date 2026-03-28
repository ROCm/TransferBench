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

// Helper: run one side (sender or receiver) of PXN benchmark for a given node–pair round
static int RunPxnSide(EnvVars&                                     ev,
                      TransferBench::ConfigOptions const&           cfg,
                      size_t                                        numBytes,
                      int                                           numGpus,
                      int                                           numNicsPerRank,
                      MemType                                       memType,
                      std::string const&                            memTypeStr,
                      int                                           useDmaExec,
                      int                                           numSubExecs,
                      int                                           numQueuePairs,
                      int                                           pxnPattern,
                      int                                           pxnSrcGpu,
                      int                                           pxnDstGpu,
                      int                                           showDetails,
                      bool                                          isSender,
                      std::vector<std::vector<std::pair<int,int>>>& schedule)
{
  int numRanks  = TransferBench::GetNumRanks();
  ExeType gpuExeType = useDmaExec ? EXE_GPU_DMA : EXE_GPU_GFX;

  char const* sideName = isSender ? "Sender" : "Receiver";
  char const* patternName = (pxnPattern == 0) ? "Single-Path" :
                            (pxnPattern == 1) ? "Aggregated"  : "All-to-All";

  Utils::Print("PXN Benchmark (%s-Side, %s)\n", sideName, patternName);
  Utils::Print("========================================\n");
  Utils::Print("[%lu bytes per Transfer] [%s:%d] [%d QP] [MemType:%s] [#Ranks:%d]\n",
               numBytes, useDmaExec ? "DMA" : "GFX", numSubExecs,
               numQueuePairs, memTypeStr.c_str(), numRanks);
  Utils::Print("\n");

  // Accumulate per-round results for multi-node
  double totalXgmiDuration = 0.0;
  double totalNicDuration  = 0.0;
  double totalXgmiBw       = 0.0;
  double totalNicBw        = 0.0;
  int    numPairsCompleted  = 0;
  int    totalPairs         = 0;
  std::vector<TransferBench::ErrResult> allErrors;

  // Count total pairs (excluding self-pairs)
  for (auto const& roundPairs : schedule)
    for (auto const& pair : roundPairs)
      if (pair.first != pair.second) totalPairs++;

  auto cpuStart = std::chrono::high_resolution_clock::now();

  for (auto const& roundPairs : schedule) {
    for (auto const& nodePair : roundPairs) {
      int srcRank = nodePair.first;
      int dstRank = nodePair.second;

      // Skip self-pairs — PXN is inter-node only
      if (srcRank == dstRank) continue;

      int localRank  = isSender ? srcRank : dstRank;
      int remoteRank = isSender ? dstRank : srcRank;

      // Build xGMI staging transfers
      std::vector<Transfer> xgmiTransfers;
      // Build NIC transfers
      std::vector<Transfer> nicTransfers;

      if (pxnPattern == 0) {
        // Single-path mode
        int intermGpu = isSender ? pxnDstGpu : pxnSrcGpu;
        int srcGpu    = pxnSrcGpu;

        if (isSender) {
          // xGMI: src GPU -> intermediate GPU
          if (srcGpu != intermGpu) {
            Transfer t;
            t.numBytes    = numBytes;
            t.srcs.push_back({memType, srcGpu, localRank});
            t.dsts.push_back({memType, intermGpu, localRank});
            t.exeDevice   = {gpuExeType, srcGpu, localRank};
            t.exeSubIndex = -1;
            t.numSubExecs = numSubExecs;
            xgmiTransfers.push_back(t);
          }

          // NIC: intermediate GPU -> remote intermediate GPU
          Transfer nt;
          nt.numBytes    = numBytes;
          nt.srcs.push_back({memType, intermGpu, localRank});
          nt.dsts.push_back({memType, intermGpu, remoteRank});
          nt.exeDevice   = {EXE_NIC_NEAREST, intermGpu, localRank};
          nt.exeSubIndex = intermGpu;
          nt.numSubExecs = numQueuePairs;
          nicTransfers.push_back(nt);
        } else {
          // Receiver-side: NIC receive then xGMI fan-out
          Transfer nt;
          nt.numBytes    = numBytes;
          nt.srcs.push_back({memType, intermGpu, remoteRank});
          nt.dsts.push_back({memType, intermGpu, localRank});
          nt.exeDevice   = {EXE_NIC_NEAREST, intermGpu, localRank};
          nt.exeSubIndex = intermGpu;
          nt.numSubExecs = numQueuePairs;
          nicTransfers.push_back(nt);

          int dstGpu = pxnDstGpu;
          if (dstGpu != intermGpu) {
            Transfer t;
            t.numBytes    = numBytes;
            t.srcs.push_back({memType, intermGpu, localRank});
            t.dsts.push_back({memType, dstGpu, localRank});
            t.exeDevice   = {gpuExeType, intermGpu, localRank};
            t.exeSubIndex = -1;
            t.numSubExecs = numSubExecs;
            xgmiTransfers.push_back(t);
          }
        }
      } else if (pxnPattern == 1) {
        // Aggregated mode: fan-in to one intermediate GPU
        int intermGpu = pxnDstGpu;

        if (isSender) {
          for (int src = 0; src < numGpus; src++) {
            if (src == intermGpu) continue;
            Transfer t;
            t.numBytes    = numBytes;
            t.srcs.push_back({memType, src, localRank});
            t.dsts.push_back({memType, intermGpu, localRank});
            t.exeDevice   = {gpuExeType, src, localRank};
            t.exeSubIndex = -1;
            t.numSubExecs = numSubExecs;
            xgmiTransfers.push_back(t);
          }

          Transfer nt;
          nt.numBytes    = numBytes;
          nt.srcs.push_back({memType, intermGpu, localRank});
          nt.dsts.push_back({memType, intermGpu, remoteRank});
          nt.exeDevice   = {EXE_NIC_NEAREST, intermGpu, localRank};
          nt.exeSubIndex = intermGpu;
          nt.numSubExecs = numQueuePairs;
          nicTransfers.push_back(nt);
        } else {
          Transfer nt;
          nt.numBytes    = numBytes;
          nt.srcs.push_back({memType, intermGpu, remoteRank});
          nt.dsts.push_back({memType, intermGpu, localRank});
          nt.exeDevice   = {EXE_NIC_NEAREST, intermGpu, localRank};
          nt.exeSubIndex = intermGpu;
          nt.numSubExecs = numQueuePairs;
          nicTransfers.push_back(nt);

          for (int dst = 0; dst < numGpus; dst++) {
            if (dst == intermGpu) continue;
            Transfer t;
            t.numBytes    = numBytes;
            t.srcs.push_back({memType, intermGpu, localRank});
            t.dsts.push_back({memType, dst, localRank});
            t.exeDevice   = {gpuExeType, intermGpu, localRank};
            t.exeSubIndex = -1;
            t.numSubExecs = numSubExecs;
            xgmiTransfers.push_back(t);
          }
        }
      } else {
        // All-to-all mode: all rails active
        if (isSender) {
          // Phase 1: xGMI fan-in to ALL intermediate GPUs
          for (int interm = 0; interm < numGpus; interm++) {
            for (int src = 0; src < numGpus; src++) {
              if (src == interm) continue;
              Transfer t;
              t.numBytes    = numBytes;
              t.srcs.push_back({memType, src, localRank});
              t.dsts.push_back({memType, interm, localRank});
              t.exeDevice   = {gpuExeType, src, localRank};
              t.exeSubIndex = -1;
              t.numSubExecs = numSubExecs;
              xgmiTransfers.push_back(t);
            }
          }

          // Phase 2: NIC send on all rails
          for (int rail = 0; rail < numGpus; rail++) {
            Transfer nt;
            nt.numBytes    = numBytes;
            nt.srcs.push_back({memType, rail, localRank});
            nt.dsts.push_back({memType, rail, remoteRank});
            nt.exeDevice   = {EXE_NIC_NEAREST, rail, localRank};
            nt.exeSubIndex = rail;
            nt.numSubExecs = numQueuePairs;
            nicTransfers.push_back(nt);
          }
        } else {
          // Receiver-side: NIC receive on all rails
          for (int rail = 0; rail < numGpus; rail++) {
            Transfer nt;
            nt.numBytes    = numBytes;
            nt.srcs.push_back({memType, rail, remoteRank});
            nt.dsts.push_back({memType, rail, localRank});
            nt.exeDevice   = {EXE_NIC_NEAREST, rail, localRank};
            nt.exeSubIndex = rail;
            nt.numSubExecs = numQueuePairs;
            nicTransfers.push_back(nt);
          }

          // Phase 2: xGMI fan-out from all intermediate GPUs
          for (int interm = 0; interm < numGpus; interm++) {
            for (int dst = 0; dst < numGpus; dst++) {
              if (dst == interm) continue;
              Transfer t;
              t.numBytes    = numBytes;
              t.srcs.push_back({memType, interm, localRank});
              t.dsts.push_back({memType, dst, localRank});
              t.exeDevice   = {gpuExeType, interm, localRank};
              t.exeSubIndex = -1;
              t.numSubExecs = numSubExecs;
              xgmiTransfers.push_back(t);
            }
          }
        }
      }

      // Determine phase order: sender = xGMI first, receiver = NIC first
      auto& phase1Transfers = isSender ? xgmiTransfers : nicTransfers;
      auto& phase2Transfers = isSender ? nicTransfers  : xgmiTransfers;

      // Execute Phase 1
      TransferBench::TestResults phase1Results;
      if (!phase1Transfers.empty()) {
        if (!TransferBench::RunTransfers(cfg, phase1Transfers, phase1Results)) {
          for (auto const& err : phase1Results.errResults)
            Utils::Print("%s\n", err.errMsg.c_str());
          return 1;
        }
        if (showDetails) {
          Utils::PrintResults(ev, 1, phase1Transfers, phase1Results);
          Utils::Print("\n");
        }
      }

      // Execute Phase 2
      TransferBench::TestResults phase2Results;
      if (!phase2Transfers.empty()) {
        if (!TransferBench::RunTransfers(cfg, phase2Transfers, phase2Results)) {
          for (auto const& err : phase2Results.errResults)
            Utils::Print("%s\n", err.errMsg.c_str());
          return 1;
        }
        if (showDetails) {
          Utils::PrintResults(ev, 1, phase2Transfers, phase2Results);
          Utils::Print("\n");
        }
      }

      // Accumulate non-fatal errors
      if (!phase1Transfers.empty())
        allErrors.insert(allErrors.end(), phase1Results.errResults.begin(), phase1Results.errResults.end());
      if (!phase2Transfers.empty())
        allErrors.insert(allErrors.end(), phase2Results.errResults.begin(), phase2Results.errResults.end());

      // Collect timing
      double xgmiDur = isSender ? phase1Results.avgTotalDurationMsec
                                : phase2Results.avgTotalDurationMsec;
      double nicDur  = isSender ? phase2Results.avgTotalDurationMsec
                                : phase1Results.avgTotalDurationMsec;
      double xgmiBw  = isSender ? phase1Results.avgTotalBandwidthGbPerSec
                                : phase2Results.avgTotalBandwidthGbPerSec;
      double nicBw   = isSender ? phase2Results.avgTotalBandwidthGbPerSec
                                : phase1Results.avgTotalBandwidthGbPerSec;

      if (phase1Transfers.empty()) {
        if (isSender) { xgmiDur = 0; xgmiBw = 0; }
        else          { nicDur  = 0; nicBw  = 0; }
      }
      if (phase2Transfers.empty()) {
        if (isSender) { nicDur  = 0; nicBw  = 0; }
        else          { xgmiDur = 0; xgmiBw = 0; }
      }

      totalXgmiDuration += xgmiDur;
      totalNicDuration  += nicDur;
      totalXgmiBw       += xgmiBw;
      totalNicBw        += nicBw;
      numPairsCompleted++;

      auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
      double durationSec = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count();
      if (Utils::RankDoesOutput()) {
        fprintf(stderr, "Completed %d/%d node pairs in %6.3fs\n",
                numPairsCompleted, totalPairs, durationSec);
      }
    }
  }

  // Gate output to the designated output rank
  if (!Utils::RankDoesOutput()) return 0;

  if (numPairsCompleted == 0) {
    Utils::Print("[WARN] No node pairs were tested\n");
    return 0;
  }

  // Print summary
  double avgXgmiDuration = totalXgmiDuration / numPairsCompleted;
  double avgNicDuration  = totalNicDuration  / numPairsCompleted;
  double avgXgmiBw       = totalXgmiBw       / numPairsCompleted;
  double avgNicBw        = totalNicBw        / numPairsCompleted;
  double totalPxnTime    = avgXgmiDuration + avgNicDuration;

  int numRails = (pxnPattern == 0) ? 1 : (pxnPattern == 1) ? 1 : numGpus;
  double effectivePxnBwPerRail = (totalPxnTime > 0)
    ? (numBytes / 1e9) / (totalPxnTime / 1e3)
    : 0.0;
  double aggregatePxnBw = effectivePxnBwPerRail * numRails;

  char const* phase1Label = isSender ? "xGMI Staging" : "NIC Receive";
  char const* phase2Label = isSender ? "NIC Send"     : "xGMI Fan-out";
  double phase1Dur = isSender ? avgXgmiDuration : avgNicDuration;
  double phase2Dur = isSender ? avgNicDuration  : avgXgmiDuration;
  double phase1Bw  = isSender ? avgXgmiBw       : avgNicBw;
  double phase2Bw  = isSender ? avgNicBw        : avgXgmiBw;

  Utils::Print("\nPhase 1 - %s:\n", phase1Label);
  Utils::Print("  Avg duration:  %8.3f ms\n", phase1Dur);
  Utils::Print("  Avg bandwidth: %8.3f GB/s\n", phase1Bw);

  Utils::Print("\nPhase 2 - %s:\n", phase2Label);
  Utils::Print("  Avg duration:  %8.3f ms\n", phase2Dur);
  Utils::Print("  Avg bandwidth: %8.3f GB/s\n", phase2Bw);

  Utils::Print("\nEffective PXN Summary:\n");
  Utils::Print("  Total PXN time:          %8.3f ms\n", totalPxnTime);
  Utils::Print("  Effective PXN BW:        %8.3f GB/s per rail\n", effectivePxnBwPerRail);
  if (numRails > 1)
    Utils::Print("  Aggregate PXN BW:        %8.3f GB/s (%d rails)\n", aggregatePxnBw, numRails);
  Utils::Print("  Bottleneck:              %s\n",
               phase1Bw < phase2Bw ? phase1Label : phase2Label);

  Utils::PrintErrors(allErrors);
  Utils::Print("\n");

  return 0;
}

int PxnPreset(EnvVars&          ev,
              size_t      const numBytesPerTransfer,
              std::string const presetName,
              [[maybe_unused]] bool const bytesSpecified)
{
  // Check for homogeneous rank groups
  if (Utils::GetNumRankGroups() > 1) {
    Utils::Print("[ERROR] PXN preset can only be run across ranks that are homogenous\n");
    Utils::Print("[ERROR] Run ./TransferBench without any args to display topology information\n");
    Utils::Print("[ERROR] TB_NIC_FILTER may also be used to limit NIC visibility\n");
    return 1;
  }

  int numRanks = TransferBench::GetNumRanks();
  if (numRanks < 2) {
    Utils::Print("[ERROR] PXN preset requires at least 2 ranks (multi-node)\n");
    Utils::Print("[ERROR] Use mpirun -np 2 or TB_NUM_RANKS=2 to run multi-node\n");
    return 1;
  }

  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);
  int numNicsPerRank  = TransferBench::GetNumExecutors(EXE_NIC);
  if (numNicsPerRank == 0) {
    Utils::Print("[ERROR] No NICs detected. NICs are required to run PXN preset\n");
    Utils::Print("[ERROR] Build with -DENABLE_NIC_EXEC=ON\n");
    return 1;
  }

  // PXN should be run with 1 rank per host — warn early if multiple ranks share a hostname
  if (Utils::HasDuplicateHostname()) {
    Utils::Print("[WARN] Multiple ranks detected on the same host.\n");
    Utils::Print("[WARN] PXN preset is designed for 1 rank per node (each rank owns all GPUs on the node).\n");
    Utils::Print("[WARN] Consider running with: mpirun -np <num_nodes> --map-by node TransferBench pxn\n");
  }

  // Default to 16MB if bytes not specified
  size_t numBytes = bytesSpecified ? numBytesPerTransfer : (16 * 1024 * 1024);

  // Collect env vars for this preset
  int pxnSide       = EnvVars::GetEnvVar("PXN_SIDE"       , 0);
  int pxnPattern    = EnvVars::GetEnvVar("PXN_PATTERN"    , 2);
  int pxnSrcGpu     = EnvVars::GetEnvVar("PXN_SRC_GPU"    , 0);
  int pxnDstGpu     = EnvVars::GetEnvVar("PXN_DST_GPU"    , 1);
  int numGpus       = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int numQueuePairs = EnvVars::GetEnvVar("NUM_QUEUE_PAIRS", 1);
  int memTypeIdx    = EnvVars::GetEnvVar("MEM_TYPE"       , 2);
  int useDmaExec    = EnvVars::GetEnvVar("USE_DMA_EXEC"   , 0);
  int numSubExecs   = EnvVars::GetEnvVar("NUM_SUB_EXEC"   , 8);
  int showDetails   = EnvVars::GetEnvVar("SHOW_DETAILS"   , 0);
  int nodeParallel  = EnvVars::GetEnvVar("PARALLEL_NODE"  , 1);

  // Parse memory type
  MemType memType = Utils::GetGpuMemType(memTypeIdx);
  std::string devMemTypeStr = Utils::GetGpuMemTypeStr(memTypeIdx);

  // Display env vars
  if (Utils::RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
      if (!ev.outputToCsv) printf("[PXN Related]\n");
      ev.Print("PXN_SIDE"       , pxnSide      , "Benchmarking %s PXN",
               pxnSide == 0 ? "sender-side" : pxnSide == 1 ? "receiver-side" : "both sides");
      ev.Print("PXN_PATTERN"    , pxnPattern   , "Running %s pattern",
               pxnPattern == 0 ? "single-path" : pxnPattern == 1 ? "aggregated" : "all-to-all");
      if (pxnPattern <= 1) {
        ev.Print("PXN_SRC_GPU"  , pxnSrcGpu    , "Source GPU index");
        ev.Print("PXN_DST_GPU"  , pxnDstGpu    , "Destination / intermediate GPU index");
      }
      ev.Print("NUM_GPU_DEVICES", numGpus      , "Using %d GPUs per rank", numGpus);
      ev.Print("NUM_QUEUE_PAIRS", numQueuePairs, "Using %d queue pairs for NIC transfers", numQueuePairs);
      ev.Print("MEM_TYPE"       , memTypeIdx   , "Using %s GPU memory (%s)", devMemTypeStr.c_str(), Utils::GetAllGpuMemTypeStr().c_str());
      ev.Print("USE_DMA_EXEC"   , useDmaExec   , "Using %s executor for xGMI staging", useDmaExec ? "DMA" : "GFX");
      ev.Print("NUM_SUB_EXEC"   , numSubExecs  , "Using %d subexecutors/CUs per Transfer", numSubExecs);
      ev.Print("SHOW_DETAILS"   , showDetails  , "%s full Transfer details", showDetails ? "Showing" : "Hiding");
      ev.Print("PARALLEL_NODE"  , nodeParallel , "Executing node pairs %s", nodeParallel ? "in parallel" : "serially");
      printf("\n");
    }
  }

  // Validate env vars
  if (numGpus < 1 || numGpus > numDetectedGpus) {
    Utils::Print("[ERROR] Cannot use %d GPUs. Detected %d GPUs\n", numGpus, numDetectedGpus);
    return 1;
  }
  if (pxnSide < 0 || pxnSide > 2) {
    Utils::Print("[ERROR] PXN_SIDE must be 0 (sender), 1 (receiver), or 2 (both)\n");
    return 1;
  }
  if (pxnPattern < 0 || pxnPattern > 2) {
    Utils::Print("[ERROR] PXN_PATTERN must be 0 (single), 1 (aggregated), or 2 (all-to-all)\n");
    return 1;
  }
  if (pxnPattern <= 1) {
    if (pxnSrcGpu < 0 || pxnSrcGpu >= numGpus) {
      Utils::Print("[ERROR] PXN_SRC_GPU %d out of range [0, %d)\n", pxnSrcGpu, numGpus);
      return 1;
    }
    if (pxnDstGpu < 0 || pxnDstGpu >= numGpus) {
      Utils::Print("[ERROR] PXN_DST_GPU %d out of range [0, %d)\n", pxnDstGpu, numGpus);
      return 1;
    }
  }

  // Topology warnings
  if (numGpus != numNicsPerRank) {
    Utils::Print("[WARN] GPU count (%d) != NIC count (%d). Non-standard rail topology\n",
                 numGpus, numNicsPerRank);
  }

  // Validate xGMI links between GPU pairs (sample check)
#if !defined(__NVCC__)
  if (pxnPattern <= 1 && pxnSrcGpu != pxnDstGpu) {
    uint32_t linkType, hopCount;
    HIP_CALL(hipExtGetLinkTypeAndHopCount(pxnSrcGpu, pxnDstGpu, &linkType, &hopCount));
    if (linkType != HSA_AMD_LINK_INFO_TYPE_XGMI) {
      Utils::Print("[WARN] GPU %d to GPU %d is not connected via xGMI (linkType=%d, hops=%d)\n",
                   pxnSrcGpu, pxnDstGpu, linkType, hopCount);
    }
  }
#endif

  // Build round-robin schedule for node pairs
  std::vector<std::vector<std::pair<int, int>>> schedule;
  RoundRobinSchedule(schedule, numRanks, nodeParallel);

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();

  // Run sender-side PXN
  if (pxnSide == 0 || pxnSide == 2) {
    int rc = RunPxnSide(ev, cfg, numBytes, numGpus, numNicsPerRank,
                        memType, devMemTypeStr, useDmaExec, numSubExecs,
                        numQueuePairs, pxnPattern, pxnSrcGpu, pxnDstGpu,
                        showDetails, true, schedule);
    if (rc != 0) return rc;
  }

  // Run receiver-side PXN
  if (pxnSide == 1 || pxnSide == 2) {
    int rc = RunPxnSide(ev, cfg, numBytes, numGpus, numNicsPerRank,
                        memType, devMemTypeStr, useDmaExec, numSubExecs,
                        numQueuePairs, pxnPattern, pxnSrcGpu, pxnDstGpu,
                        showDetails, false, schedule);
    if (rc != 0) return rc;
  }

  if (Utils::HasDuplicateHostname()) {
    printf("[WARN] It is recommended to run TransferBench with one rank per host to avoid potential aliasing of executors\n");
  }

  return 0;
}

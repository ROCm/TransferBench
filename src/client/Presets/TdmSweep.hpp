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

// TdmSweepPreset - sweeps every knob that affects Tensor-Data-Mover (TDM) copy
// performance for a given TDM Transfer and reports the best-performing
// combination, mirroring the structure of the "gfxsweep" preset.
//
// The TDM executor (EXE_GPU_TDM, letter 'T' in a Transfer expr) stages HBM->LDS
// ->HBM without touching cache. Its performance is governed by:
//   * threadblock size   (TDM_BLOCK_SIZE : # threads/block, multiple of 32)
//   * LDS staging window  (TDM_LDS_BYTES : bytes/block, 0 = device max)
//   * threadblock order   (TDM_BLOCK_ORDER: 0=sequential 1=interleaved 2=random)
//   * number of subExecs  (threadblocks/WGPs engaged, NUM_SUB_EXECS)
// This preset sweeps the cartesian product of all four and finds the peak.
int TdmSweepPreset(EnvVars&          ev,
                   size_t      const numBytesPerTransfer,
                   std::string const presetName,
                   bool        const bytesSpecified)
{
  enum TimingMode
  {
    TimingModeAuto = -1,
    TimingModeCpu  =  0,
    TimingModeHip  =  1,
    TimingModeGpu  =  2
  };

  // Verify the hardware can actually run TDM copies before sweeping anything.
  int const numTdmGpus = TransferBench::GetNumExecutors(EXE_GPU_TDM);
  if (numTdmGpus <= 0 || !tdm::IsTdmCopySupported(0)) {
    Utils::Print("[WARN] TDM executor is not supported on this device "
                 "(requires TDM-capable hardware: gfx1250 or NVIDIA sm_90+). Terminating tdmsweep.\n");
    return ERR_FATAL;
  }

  // Collect environment variables for this preset
  vector<int> blockList     = EnvVars::GetEnvVarArray("BLOCKSIZES",   {64,128,256,512,1024});
  vector<int> blockOrders   = EnvVars::GetEnvVarArray("BLOCK_ORDERS",                    {0});
  vector<int> ldsList       = EnvVars::GetEnvVarArray("LDS_BYTES",  {8192,16384,32768,65536,0});
  vector<int> numSesList    = EnvVars::GetEnvVarArray("NUM_SUB_EXECS",       {2,4,8,16,32,64});
  int         numTransfers  = EnvVars::GetEnvVar(     "NUM_TRANSFERS",                     1);
  int         timingMode    = EnvVars::GetEnvVar(     "TIMING_MODE",          TimingModeAuto);
  std::string transferStr   = EnvVars::GetEnvVar(     "TDM_TRANSFER",          "G0->T0->G0");

  // Print off relevant environment variables
  if (Utils::RankDoesOutput()) {
    if (!ev.hideEnv) {
      ev.DisplayEnvVars();
      if (!ev.outputToCsv)
        Utils::Print("[TDM Sweep Related]\n");
      ev.Print("BLOCKSIZES",    blockList.size(),   EnvVars::ToStr(blockList).c_str());
      ev.Print("BLOCK_ORDERS",  blockOrders.size(), "%s (0=sequential 1=interleaved 2=random)", EnvVars::ToStr(blockOrders).c_str());
      ev.Print("LDS_BYTES",     ldsList.size(),     "%s (0 = device max LDS per block)", EnvVars::ToStr(ldsList).c_str());
      ev.Print("NUM_SUB_EXECS", numSesList.size(),  EnvVars::ToStr(numSesList).c_str());
      ev.Print("NUM_TRANSFERS", numTransfers,       "Number of Transfers specified in TDM_TRANSFER");
      ev.Print("TIMING_MODE",   timingMode,         "-1=auto 0=Aggregate CPU, 1=Executor Time, 2=Transfer Time");
      ev.Print("TDM_TRANSFER",  transferStr,        "TDM Transfer to sweep (see config file format)");
      Utils::Print("\n");
    }
  }

  if (timingMode < TimingModeAuto || timingMode > TimingModeGpu) {
    Utils::Print("TIMING_MODE value is invalid (%d)\n", timingMode);
    return ERR_FATAL;
  }

  if (numSesList.empty()) {
    Utils::Print("NUM_SUB_EXECS should not be empty\n");
    return ERR_FATAL;
  }

  // TDM block size must be a positive multiple of 32
  for (int bs : blockList) {
    if (bs <= 0 || bs % 32) {
      Utils::Print("[ERROR] BLOCKSIZES value %d is invalid (TDM block size must be a positive multiple of 32)\n", bs);
      return ERR_FATAL;
    }
  }
  for (int lds : ldsList) {
    if (lds < 0) {
      Utils::Print("[ERROR] LDS_BYTES value %d is invalid (must be >= 0; 0 = device max)\n", lds);
      return ERR_FATAL;
    }
  }
  for (int bo : blockOrders) {
    if (bo < 0 || bo > 2) {
      Utils::Print("[ERROR] BLOCK_ORDERS value %d is invalid (must be 0, 1, or 2)\n", bo);
      return ERR_FATAL;
    }
  }

  std::vector<Transfer> transfers;
  Utils::CheckForError(ParseTransfers(std::to_string(numTransfers) + " 1 " + transferStr, transfers));
  if (transfers.size() == 0) {
    Utils::Print("[WARN] No valid Transfers found in TDM_TRANSFER\n");
    return 0;
  }

  // Automatically pick timing method
  if (timingMode == TimingModeAuto) {
    // Use Transfer timing if there is only one Transfer
    if (transfers.size() == 1) timingMode = TimingModeGpu;
    // Use Executor timing if there is only one executor
    else {
      bool singleExecutor = true;
      for (size_t i = 1; i < transfers.size(); i++) {
        if (transfers[i].exeDevice   <  transfers[0].exeDevice   ||
            transfers[0].exeDevice   <  transfers[i].exeDevice   ||
            transfers[i].exeSubIndex != transfers[0].exeSubIndex ||
            transfers[i].exeSubSlot  != transfers[0].exeSubSlot) {
          singleExecutor = false;
          break;
        }
      }
      timingMode = singleExecutor ? TimingModeHip : TimingModeCpu;
    }
  }
  if (timingMode < 0 || timingMode > 2) {
    Utils::Print("[ERROR] Invalid timing mode %d\n", timingMode);
    return ERR_FATAL;
  }

  // Print out the Transfers being run
  Utils::Print("TDM sweep: (%lu bytes per Transfer). All values are %s-timed GB/s\n", numBytesPerTransfer,
               timingMode == TimingModeCpu ? "Aggregate-CPU" :
               timingMode == TimingModeHip ? "HIP-event"     :
                                             "GPU wallclock");
  Utils::Print("=======================================================================================\n");

  bool isMultiNode = GetNumRanks() > 1;
  for (size_t i = 0; i < transfers.size(); i++) {
    Transfer& t = transfers[i];
    Utils::Print("Transfer %5lu: (%s->", i, Utils::MemDevicesToStr(t.srcs).c_str());
    if (isMultiNode)         Utils::Print("R%d", t.exeDevice.exeRank);
    Utils::Print("%c%d", ExeTypeStr[t.exeDevice.exeType], t.exeDevice.exeIndex);
    if (t.exeDevice.exeSlot) Utils::Print("%c", 'A' + t.exeDevice.exeSlot);
    if (t.exeSubIndex != -1) Utils::Print(".%d", t.exeSubIndex);
    if (t.exeSubSlot != 0)   Utils::Print("%c", 'A' + t.exeSubSlot);
    Utils::Print("->%s)\n",  Utils::MemDevicesToStr(t.dsts).c_str());

    if (t.exeDevice.exeType != EXE_GPU_TDM) {
      Utils::Print("[ERROR] tdmsweep preset only works on Transfers that are using the TDM executor "
                   "(use the 'T' executor, e.g. TDM_TRANSFER=\"G0->T0->G1\")\n");
      return ERR_FATAL;
    }
    t.numBytes = numBytesPerTransfer;
  }

  Utils::Print("=======================================================================================\n");

  ConfigOptions cfg = ev.ToConfigOptions();

  // Print header
  char sep = ev.outputToCsv ? ',' : ' ';
  Utils::Print(" BlkO %c  BlkS  %c  LDSBytes ", sep, sep);
  for  (int numSubExec : numSesList)
    Utils::Print("%c  SE %03d", sep, numSubExec);
  Utils::Print("\n");

  int bestSe = -1;
  double overallBestBw = 0;
  vector<double> bestBw(numSesList.size(), 0.0);
  // best[s] = {blockOrder, blockSize, ldsBytes, numSubExec}
  vector<vector<int>> best(numSesList.size(), vector<int>(4));

  // Loop over all combinations
  for (int blockOrder : blockOrders) {          cfg.tdm.blockOrder = blockOrder;
    for (int blockSize : blockList) {           cfg.tdm.blockSize  = blockSize;
      for (int ldsBytes : ldsList) {            cfg.tdm.ldsBytes   = ldsBytes;
        Utils::Print("  %1d   %c  %4d  %c  %8d ", blockOrder, sep, blockSize, sep, ldsBytes);
        fflush(stdout);
        for (auto s = 0; s < numSesList.size(); s++) {
          int numSubExec = numSesList[s];
          for (Transfer& t : transfers) t.numSubExecs = numSubExec;

          TestResults result;
          // A given combination may be rejected by the library (e.g. LDS window
          // larger than the device max). Treat that as a skipped cell (N/A) and
          // keep sweeping instead of aborting the whole matrix.
          if (RunTransfers(cfg, transfers, result)) {
            double bw = 0.0;
            switch (timingMode) {
            case 0: bw = result.avgTotalBandwidthGbPerSec; break;
            case 1:
              for (auto const& e : result.exeResults) {
                bw = std::max(bw, e.second.avgBandwidthGbPerSec);
              }
              break;
            case 2: default:
              for (auto const& t : result.tfrResults) {
                bw = std::max(bw, t.avgBandwidthGbPerSec);
              }
              break;
            }

            if (bw > bestBw[s]) {
              bestBw[s] = bw;
              best[s] = {blockOrder, blockSize, ldsBytes, numSubExec};
              if (bw > overallBestBw) {
                overallBestBw = bw;
                bestSe = s;
              }
            }
            Utils::Print("%c%8.2f", sep, bw);
          } else {
            Utils::Print("%c%8s", sep, "N/A");
          }
          fflush(stdout);
        }
        Utils::Print("\n");
        fflush(stdout);
      }
    }
  }

  Utils::Print(" BlkO %c  BlkS  %c  LDSBytes ", sep, sep);
  for (auto s = 0; s < numSesList.size(); s++) {
    Utils::Print("%c%8.2f", sep, bestBw[s]);
  }
  Utils::Print("\n");

  if (bestSe == -1) {
    Utils::Print("[WARN] No transfers executed successfully - check sweep parameters and TDM support\n");
    return ERR_FATAL;
  }

  // Print combination that produced highest bandwidth
  Utils::Print("=======================================================================================\n");
  Utils::Print("Highest bandwidth found: %7.2f GB/s (%s-timed)\n", overallBestBw,
               timingMode == TimingModeCpu ? "Aggregate-CPU" :
               timingMode == TimingModeHip ? "HIP-event"     :
                                             "GPU wallclock");
  Utils::Print("          BlockOrder   : %7d  [TDM_BLOCK_ORDER=%d]\n", best[bestSe][0], best[bestSe][0]);
  Utils::Print("          BlockSize    : %7d  [TDM_BLOCK_SIZE=%d]\n",  best[bestSe][1], best[bestSe][1]);
  Utils::Print("          LDS Bytes    : %7d  [TDM_LDS_BYTES=%d]\n",   best[bestSe][2], best[bestSe][2]);
  Utils::Print("          NumSubExec   : %7d\n", best[bestSe][3]);
  Utils::Print("Command to run best result:\n");
  Utils::Print("TDM_BLOCK_ORDER=%d TDM_BLOCK_SIZE=%d TDM_LDS_BYTES=%d ./TransferBench cmdline %lu \"%d %d %s\"\n",
               best[bestSe][0], best[bestSe][1], best[bestSe][2],
               numBytesPerTransfer, numTransfers, best[bestSe][3], transferStr.c_str());
  return ERR_NONE;
}

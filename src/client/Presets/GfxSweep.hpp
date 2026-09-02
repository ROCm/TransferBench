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

int GfxSweepPreset(EnvVars&          ev,
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

  // Collect environment variables for this preset
  vector<int> blockList      = EnvVars::GetEnvVarArray("BLOCKSIZES",   {256,512,768,1024});
  std::string transferStr    = EnvVars::GetEnvVar(     "GFX_TRANSFER", "R0G0->R0G0->R0G0");
  vector<int> kernelList     = EnvVars::GetEnvVarArray("KERNELS",                     {0});
  vector<int> numSesList     = EnvVars::GetEnvVarArray("NUM_SUB_EXECS",    {4,8,16,32,64});
  int         numTransfers   = EnvVars::GetEnvVar(     "NUM_TRANSFERS",                 1);
  vector<int> temporalList   = EnvVars::GetEnvVarArray("TEMPORAL_MODES",              {0});
  int         timingMode     = EnvVars::GetEnvVar(     "TIMING_MODE",      TimingModeAuto);
  vector<int> unrollList     = EnvVars::GetEnvVarArray("UNROLLS",            {1,2,4,8,16});
  vector<int> waveOrderList  = EnvVars::GetEnvVarArray("WAVE_ORDERS",                 {0});
  vector<int> wordSizeList   = EnvVars::GetEnvVarArray("WORDSIZES",                   {4});

  // Print off relevant environment variables
  if (Utils::RankDoesOutput()) {
    if (!ev.hideEnv) {
      ev.DisplayEnvVars();
      if (!ev.outputToCsv)
        Utils::Print("[GFX Sweep Related]\n");
      ev.Print("BLOCKSIZES",     blockList.size(),     EnvVars::ToStr(blockList).c_str());
      ev.Print("KERNELS",        kernelList.size(),    EnvVars::ToStr(kernelList).c_str());
      ev.Print("NUM_TRANSFERS",  numTransfers,         "Number of Transfers specified in GFX_TRANSFER");
      ev.Print("NUM_SUB_EXECS",  numSesList.size(),    EnvVars::ToStr(numSesList).c_str());
      ev.Print("TEMPORAL_MODES", temporalList.size(),  EnvVars::ToStr(temporalList).c_str());
      ev.Print("TIMING_MODE",    timingMode,           "-1=auto 0=Aggregate CPU, 1=Executor Time, 2=Transfer Time");
      ev.Print("UNROLLS",        unrollList.size(),    EnvVars::ToStr(unrollList).c_str());
      ev.Print("WAVE_ORDERS",    waveOrderList.size(), EnvVars::ToStr(waveOrderList).c_str());
      ev.Print("WORDSIZES",      wordSizeList.size(),  EnvVars::ToStr(wordSizeList).c_str());
      ev.Print("GFX_TRANSFER",   transferStr,          "GFX Transfer to sweep (see config file format)");
      Utils::Print("\n");
    }
  }

  if (timingMode < TimingModeAuto || timingMode > TimingModeGpu) {
    Utils::Print("TIMING_MODE value is invalid (%d)\n", timingMode);
    return ERR_FATAL;
  }

  if (numSesList.empty()){
    Utils::Print("NUM_SUB_EXECS should not be empty\n");
    return ERR_FATAL;
  }

  std::vector<Transfer> transfers;
  Utils::CheckForError(ParseTransfers(std::to_string(numTransfers) + " 1 " + transferStr, transfers));
  if (transfers.size() == 0) {
    Utils::Print("[WARN] No valid Transfers found in GFX_TRANSFER\n");
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
  Utils::Print("GFX sweep: (%lu bytes per Transfer). All values are %s-timed GB/s\n", numBytesPerTransfer,
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

    if (t.exeDevice.exeType != EXE_GPU_GFX) {
      Utils::Print("[ERROR] gfxsweep preset only works on Transfers that are using GFX executor\n");
      return ERR_FATAL;
    }
    t.numBytes = numBytesPerTransfer;
  }

  Utils::Print("=======================================================================================\n");

  ConfigOptions cfg = ev.ToConfigOptions();

  // Print header
  char sep = ev.outputToCsv ? ',' : ' ';
  Utils::Print(" WvO %c WSz %c TpM %c BlkS %c UnR %c KrN ", sep, sep, sep, sep, sep);
  for  (int numSubExec : numSesList)
    Utils::Print("%c  SE %03d", sep, numSubExec);
  Utils::Print("\n");

  int bestSe = -1;
  double overallBestBw = 0;
  vector<double> bestBw(numSesList.size(), 0.0);
  vector<vector<int>> best(numSesList.size(), vector<int>(7));

  // Loop over all combinations
  for (int waveOrder : waveOrderList) {         cfg.gfx.waveOrder    = waveOrder;
    for (int wordSize : wordSizeList) {         cfg.gfx.wordSize     = wordSize;
      for (int temporalMode : temporalList) {   cfg.gfx.temporalMode = temporalMode;
        for (int blockSize : blockList) {       cfg.gfx.blockSize    = blockSize;
          for (int unroll : unrollList) {       cfg.gfx.unrollFactor = unroll;
            for (int kernelIdx : kernelList) {  cfg.gfx.gfxKernel    = kernelIdx;
              Utils::Print("  %1d  %c  %1d  %c  %1d  %c %4d %c %2d  %c  %1d  ",
                           waveOrder, sep, wordSize, sep,  temporalMode, sep,
                           blockSize, sep, unroll, sep, kernelIdx, sep);
              fflush(stdout);
              for (auto s = 0; s < numSesList.size(); s++) {
                int numSubExec = numSesList[s];
                for (Transfer& t : transfers) t.numSubExecs = numSubExec;

                TestResults result;
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
                    best[s] = {waveOrder, wordSize, temporalMode, blockSize, unroll, kernelIdx, numSubExec};
                    if (bw > overallBestBw) {
                      overallBestBw = bw;
                      bestSe = s;
                    }
                  }
                  Utils::Print("%c%8.2f", sep, bw);
                  fflush(stdout);
                } else {
                  Utils::Print("\n");
                  Utils::PrintErrors(result.errResults);
                  return ERR_FATAL;
                }
              }
              Utils::Print("\n");
              fflush(stdout);
            }
          }
        }
      }
    }
  }

  Utils::Print(" WvO %c WSz %c TpM %c BlkS %c UnR %c KrN ", sep, sep, sep, sep, sep);
  for (auto s = 0; s < numSesList.size(); s++) {
    Utils::Print("%c%8.2f", sep, bestBw[s]);
  }
  Utils::Print("\n");

  if (bestSe == -1) {
    Utils::Print("[WARN] No transfers executed - make sure sweep parameters lists are not empty\n");
    return ERR_FATAL;
  }

  // Print combination that produced highest bandwidth
  Utils::Print("=======================================================================================\n");
  Utils::Print("Highest bandwidth found: %7.2f GB/s (%s-timed)\n", overallBestBw,
               timingMode == TimingModeCpu ? "Aggregate-CPU" :
               timingMode == TimingModeHip ? "HIP-event"     :
                                             "GPU wallclock");
  Utils::Print("          WaveOrder    : %7d  [GFX_WAVE_ORDER=%d]\n", best[bestSe][0], best[bestSe][0]);
  Utils::Print("          WordSize     : %7d  [GFX_WORD_SIZE=%d]\n",  best[bestSe][1], best[bestSe][1]);
  Utils::Print("          Temporal Mode: %7d  [GFX_TEMPORAL=%d]\n",   best[bestSe][2], best[bestSe][2]);
  Utils::Print("          BlockSize    : %7d  [GFX_BLOCK_SIZE=%d]\n", best[bestSe][3], best[bestSe][3]);
  Utils::Print("          Unroll       : %7d  [GFX_UNROLL=%d]\n",     best[bestSe][4], best[bestSe][4]);
  Utils::Print("          Kernel       : %7d  [GFX_KERNEL=%d]\n"    , best[bestSe][5], best[bestSe][5]);
  Utils::Print("          NumSubExec   : %7d\n", best[bestSe][6]);
  Utils::Print("Command to run best result:\n");
  Utils::Print("GFX_WAVE_ORDER=%d GFX_WORD_SIZE=%d GFX_TEMPORAL=%d GFX_BLOCK_SIZE=%d "
               "GFX_UNROLL=%d GFX_KERNEL=%d ./TransferBench cmdline %lu \"%d %d %s\"\n",
               best[bestSe][0], best[bestSe][1], best[bestSe][2], best[bestSe][3],
               best[bestSe][4], best[bestSe][5], numBytesPerTransfer, numTransfers, best[bestSe][6], transferStr.c_str());
  return ERR_NONE;
}

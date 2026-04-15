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
  // Collect environment variables for this preset
  vector<int> blockList     = EnvVars::GetEnvVarArray("BLOCKSIZES",   {256,512,768,1024});
  std::string transferStr   = EnvVars::GetEnvVar(     "GFX_TRANSFER", "R0G0->R0G0->R0G0");
  vector<int> numSesList    = EnvVars::GetEnvVarArray("NUM_SUB_EXECS",    {4,8,16,32,64});
  vector<int> temporalList  = EnvVars::GetEnvVarArray("TEMPORAL_MODES",              {0});
  vector<int> unrollList    = EnvVars::GetEnvVarArray("UNROLLS",               {1,2,4,8});
  vector<int> waveOrderList = EnvVars::GetEnvVarArray("WAVE_ORDERS",                 {0});
  vector<int> wordSizeList  = EnvVars::GetEnvVarArray("WORDSIZES",                   {4});
  int         verbose       = EnvVars::GetEnvVar(     "VERBOSE",                       0);

  // Print off relevant environment variables
  if (Utils::RankDoesOutput()) {
    if (!ev.hideEnv) {
      ev.DisplayEnvVars();
      if (!ev.outputToCsv)
        Utils::Print("[GFX Sweep Related]\n");
      ev.Print("BLOCKSIZES",     blockList.size(),     EnvVars::ToStr(blockList).c_str());
      ev.Print("GFX_TRANSFER",   transferStr,          "GFX Transfer to sweep (see config file format)");
      ev.Print("NUM_SUB_EXECS",  numSesList.size(),    EnvVars::ToStr(numSesList).c_str());
      ev.Print("TEMPORAL_MODES", temporalList.size(),  EnvVars::ToStr(temporalList).c_str());
      ev.Print("UNROLLS",        unrollList.size(),    EnvVars::ToStr(unrollList).c_str());
      ev.Print("WAVE_ORDERS",    waveOrderList.size(), EnvVars::ToStr(waveOrderList).c_str());
      ev.Print("WORDSIZES",      wordSizeList.size(),  EnvVars::ToStr(wordSizeList).c_str());
      ev.Print("VERBOSE",        verbose,              verbose ? "Display test results" : "Display summary only");
      Utils::Print("\n");
    }
  }

  std::vector<Transfer> transfers;
  Utils::CheckForError(ParseTransfers("1 1 " + transferStr, transfers));

  // Print out the Transfers being ru
  Utils::Print("GFX sweep: (%lu bytes per Transfer). All values are CPU-timed GB/s\n", numBytesPerTransfer);
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
      return 1;
    }
    t.numBytes = numBytesPerTransfer;
  }

  Utils::Print("=======================================================================================\n");

  ConfigOptions cfg = ev.ToConfigOptions();

  // Print header
  char sep = ev.outputToCsv ? ',' : ' ';
  Utils::Print(" WvO %c WSz %c TpM %c BlkS %c UnR ", sep, sep, sep, sep);
  for  (int numSubExec : numSesList)
    Utils::Print("%c SE %03d", sep, numSubExec);
  Utils::Print("\n");

  double bestBw = 0.0;
  vector<int> best(6);

  // Loop over all combinations
  for (int waveOrder : waveOrderList) {         cfg.gfx.waveOrder    = waveOrder;
    for (int wordSize : wordSizeList) {         cfg.gfx.wordSize     = wordSize;
      for (int temporalMode : temporalList) {   cfg.gfx.temporalMode = temporalMode;
        for (int blockSize : blockList) {       cfg.gfx.blockSize    = blockSize;
          for (int unroll : unrollList) {       cfg.gfx.unrollFactor = unroll;
            Utils::Print("  %d  %c  %d  %c  %d  %c %4d %c %3d ",
                         waveOrder, sep, wordSize, sep, temporalMode, sep, blockSize, sep, unroll, sep);

            for (int numSubExec : numSesList) {
              for (Transfer& t : transfers) t.numSubExecs = numSubExec;

              TestResults result;
              if (RunTransfers(cfg, transfers, result)) {
                double bw = result.avgTotalBandwidthGbPerSec;
                if (bw > bestBw) {
                  bestBw = bw;
                  best = {waveOrder, wordSize, temporalMode, blockSize, unroll, numSubExec};
                }
                Utils::Print("%c%7.2f", sep, bw);
                fflush(stdout);
              } else {
                Utils::Print("\n");
                Utils::PrintErrors(result.errResults);
                return 1;
              }
            }
            Utils::Print("\n");
          }
        }
      }
    }
  }

  // Print combination that produced highest bandwidth
  Utils::Print("=======================================================================================\n");
  Utils::Print("Highest bandwidth found: %7.2f GB/s (CPU-timed)\n", bestBw);
  Utils::Print("          WaveOrder    : %7d\n", best[0]);
  Utils::Print("          WordSize     : %7d\n", best[1]);
  Utils::Print("          Temporal Mode: %7d\n", best[2]);
  Utils::Print("          BlockSize    : %7d\n", best[3]);
  Utils::Print("          Unroll       : %7d\n", best[4]);
  Utils::Print("          NumSubExec   : %7d\n", best[5]);

  return 0;
}

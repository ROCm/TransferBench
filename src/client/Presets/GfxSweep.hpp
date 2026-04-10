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

namespace {

bool LooksLikeFullTransferLine(std::string const& spec)
{
  size_t i = 0;
  while (i < spec.size() && isspace(static_cast<unsigned char>(spec[i])))
    ++i;
  if (i >= spec.size())
    return false;
  if (spec[i] == '-')
    return i + 1 < spec.size() && isdigit(static_cast<unsigned char>(spec[i + 1]));
  return isdigit(static_cast<unsigned char>(spec[i])) != 0;
}

}  // namespace

int GfxSweepPreset(EnvVars&          ev,
                   size_t      const numBytesPerTransfer,
                   std::string const presetName,
                   bool const bytesSpecified)
{
  int showMinOnly   = EnvVars::GetEnvVar("SHOW_MIN_ONLY", 1);
  int verbose       = EnvVars::GetEnvVar("VERBOSE", 0);
  std::vector<int> blockList    = EnvVars::GetEnvVarArray("BLOCKSIZES", {256});
  std::vector<int> unrollList   = EnvVars::GetEnvVarArray("UNROLLS", {1, 2, 3, 4, 6, 8});
  std::vector<int> numSesList   = EnvVars::GetEnvVarArray("NUM_SUB_EXECS", {4, 8, 12, 16, 24, 32});
  std::vector<int> wordSizeList = EnvVars::GetEnvVarArray("WORDSIZES", {4});
  std::vector<int> temporalList = EnvVars::GetEnvVarArray("TEMPORAL_MODES", {0});
  std::vector<int> waveOrderList = EnvVars::GetEnvVarArray("WAVE_ORDERS", {0});

  std::string const spec = EnvVars::GetEnvVar("GFX_SWEEP_TRANSFER",
                                               TransferBench::GetNumRanks() > 1 ? "R0G0->R0G0->R0G0" : "G0->G0->G0");
  std::string const line = LooksLikeFullTransferLine(spec) ? spec : (std::string("1 1 ") + spec);

  std::vector<TransferBench::Transfer> transfers;
  TransferBench::Utils::CheckForError(TransferBench::ParseTransfers(line, transfers));

  if (transfers.size() != 1) {
    if (TransferBench::GetNumRanks() > 1 && transfers.size() > 1) {
      TransferBench::Utils::Print(
          "[WARN] gfxsweep: In Multinode setting, omitted rank fields on SRC/DST/EXE are filled per rank, "
          "and transfers without ranks specified will expand to multiple parallel copy per node. "
          "gfxsweep expects exactly one entry here and forbids such entries; for a local sweep use a single rank (`-np 1`), "
          "or adjust GFX_SWEEP_TRANSFER / rank syntax so expansion yields one transfer.\n");
    }
    TransferBench::Utils::Print(
        "[ERROR] gfxsweep expects exactly one transfer after parsing (got %zu). "
        "Set GFX_SWEEP_TRANSFER to a single SRC EXE DST triplet or one basic/advanced line that expands to one transfer.\n",
        transfers.size());
    return 1;
  }

  if (transfers[0].exeDevice.exeType != TransferBench::EXE_GPU_GFX) {
    TransferBench::Utils::Print(
        "[ERROR] gfxsweep requires a GPU GFX (G) executor; parsed executor type is not GFX.\n");
    return 1;
  }

  transfers[0].numBytes = numBytesPerTransfer;

  if (TransferBench::Utils::RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
      if (!ev.outputToCsv)
        TransferBench::Utils::Print("[GfxSweep Related]\n");
      ev.Print("GFX_SWEEP_TRANSFER", spec, "Transfer spec (see config file format)");
      ev.Print("BLOCKSIZES", blockList.size(), EnvVars::ToStr(blockList).c_str());
      ev.Print("NUM_SUB_EXECS", numSesList.size(), EnvVars::ToStr(numSesList).c_str());
      ev.Print("WORDSIZES", wordSizeList.size(), EnvVars::ToStr(wordSizeList).c_str());
      ev.Print("TEMPORAL_MODES", temporalList.size(), EnvVars::ToStr(temporalList).c_str());
      ev.Print("WAVE_ORDERS", waveOrderList.size(), EnvVars::ToStr(waveOrderList).c_str());
      ev.Print("SHOW_MIN_ONLY", showMinOnly, showMinOnly ? "Showing only slowest sub-executor aggregate" : "Showing slowest and fastest");
      ev.Print("UNROLLS", unrollList.size(), EnvVars::ToStr(unrollList).c_str());
      ev.Print("VERBOSE", verbose, verbose ? "Display test results" : "Display summary only");
      TransferBench::Utils::Print("\n");
    }
  }

  TransferBench::Utils::Print("GFX sweep (single transfer):\n");
  TransferBench::Utils::Print("============================\n");
  TransferBench::Utils::Print("- Parsed line: %s\n", line.c_str());
  TransferBench::Utils::Print("- %lu bytes per transfer\n", static_cast<unsigned long>(numBytesPerTransfer));

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();

  using GfxSweepKey = std::tuple<int, int, int, int, int, int>;  // block, wordSize, temporal, waveOrder, subExecs, unroll
  std::map<GfxSweepKey, TransferBench::TestResults> results;

  for (int blockSize : blockList) {
    ev.gfxBlockSize = cfg.gfx.blockSize = blockSize;

    for (int wordSize : wordSizeList) {
      ev.gfxWordSize = cfg.gfx.wordSize = wordSize;

      for (int temporalMode : temporalList) {
        ev.gfxTemporal = cfg.gfx.temporalMode = temporalMode;

        for (int waveOrder : waveOrderList) {
          ev.gfxWaveOrder = cfg.gfx.waveOrder = waveOrder;

          TransferBench::Utils::Print("Blocksize: %d  WORD_SIZE: %d  TEMPORAL: %d  WAVE_ORDER: %d\n",
                                      blockSize, wordSize, temporalMode, waveOrder);

          TransferBench::Utils::Print("#CUs\\Unroll");
          for (int u : unrollList) {
            TransferBench::Utils::Print("  %d(Min) ", u);
            if (!showMinOnly)
              TransferBench::Utils::Print("  %d(Max) ", u);
          }
          TransferBench::Utils::Print("\n");

          for (int c : numSesList) {
            TransferBench::Utils::Print("   %5d   ", c);
            fflush(stdout);
            for (int u : unrollList) {
              ev.gfxUnroll = cfg.gfx.unrollFactor = u;
              transfers[0].numSubExecs = c;

              double minBandwidth = std::numeric_limits<double>::max();
              double maxBandwidth = std::numeric_limits<double>::min();
              TransferBench::TestResults result;
              GfxSweepKey const key = std::make_tuple(blockSize, wordSize, temporalMode, waveOrder, c, u);
              if (TransferBench::RunTransfers(cfg, transfers, result)) {
                for (auto const& exeResult : result.exeResults) {
                  minBandwidth = std::min(minBandwidth, exeResult.second.avgBandwidthGbPerSec);
                  maxBandwidth = std::max(maxBandwidth, exeResult.second.avgBandwidthGbPerSec);
                }
                results[key] = result;
              } else {
                minBandwidth = 0.0;
              }
              TransferBench::Utils::Print(" %7.2f ", minBandwidth);
              if (!showMinOnly)
                TransferBench::Utils::Print(" %7.2f ", maxBandwidth);
              fflush(stdout);
            }
            TransferBench::Utils::Print("\n");
            fflush(stdout);
          }

          if (verbose) {
            int testNum = 0;
            for (int c : numSesList) {
              for (int u : unrollList) {
                GfxSweepKey const key = std::make_tuple(blockSize, wordSize, temporalMode, waveOrder, c, u);
                TransferBench::Utils::Print(
                    "Blocksize: %d  WORD_SIZE: %d  TEMPORAL: %d  WAVE_ORDER: %d  SubExecs: %d  Unroll: %d\n",
                    blockSize, wordSize, temporalMode, waveOrder, c, u);
                transfers[0].numSubExecs = c;
                auto const resultIt = results.find(key);
                if (resultIt != results.end()) {
                  TransferBench::Utils::PrintResults(ev, ++testNum, transfers, resultIt->second);
                } else {
                  ++testNum;
                  TransferBench::Utils::Print("No results available for this sweep point (transfer run failed).\n");
                }
              }
            }
          }
        }
      }
    }
  }

  return 0;
}

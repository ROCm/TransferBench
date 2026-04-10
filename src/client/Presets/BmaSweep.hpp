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

int BmaSweepPreset(EnvVars&          ev,
                   size_t      const numBytesPerTransfer,
                   std::string const presetName,
                   bool        const bytesSpecified)
{
  if (TransferBench::GetNumRanks() > 1) {
    Utils::Print("[ERROR] BMA sweep preset currently not supported for multi-node\n");
    return 1;
  }

#ifndef BMA_EXEC_ENABLED
  Utils::Print("[ERROR] BMA executor requires ROCm 7.0 or newer\n");
  return 1;
#endif

  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  // Collect env vars for this preset
  int         exeIndex      = EnvVars::GetEnvVar("EXE_INDEX"         ,               0);
  int         localCopy     = EnvVars::GetEnvVar("LOCAL_COPY"        ,               0);
  int         gpuMemTypeIdx = EnvVars::GetEnvVar("GPU_MEM_TYPE"      ,               0);
  int         numGpuDevices = EnvVars::GetEnvVar("NUM_GPU_DEVICES"   , numDetectedGpus);
  vector<int> numSesList    = EnvVars::GetEnvVarArray("NUM_SUB_EXECS",       {1,2,4,8});

  MemType gpuMemType = Utils::GetGpuMemType(gpuMemTypeIdx);

  // Display environment variables
  if (Utils::RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
      int outputToCsv = ev.outputToCsv;
      if (!outputToCsv) printf("[BMA Sweep Related]\n");
      ev.Print("EXE_INDEX"      , exeIndex,          "Executing on GPU %d", exeIndex);
      ev.Print("LOCAL_COPY"     , localCopy,         "%s local copy to GPU %d", localCopy ? "Including" : "Excluding", exeIndex);
      ev.Print("GPU_MEM_TYPE"   , gpuMemTypeIdx,     "Using %s (%s)", Utils::GetGpuMemTypeStr(gpuMemTypeIdx).c_str(), Utils::GetAllGpuMemTypeStr().c_str());
      ev.Print("NUM_GPU_DEVICES", numGpuDevices,     "Using %d GPUs", numGpuDevices);
      ev.Print("NUM_SUB_EXECS"  , numSesList.size(), EnvVars::ToStr(numSesList).c_str());
      printf("\n");
    }
  }

  if (exeIndex < 0 || exeIndex >= numGpuDevices) {
    Utils::Print("EXE_INDEX must be between 0 and %d inclusively\n", numGpuDevices - 1);
    return 1;
  }

  int numTransfers = numGpuDevices - 1 + (localCopy ? 1 : 0);

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  TransferBench::TestResults results;

  // Prepare table of results
  int minPow2Exp = 12;
  int maxPow2Exp = 30;
  int numRows    = (maxPow2Exp - minPow2Exp + 1) + 1;
  int numCols    = 2 + numSesList.size();

  Utils::TableHelper table(numRows, numCols);

  Utils::Print("Performing %d simultaneous DMA Transfers from GPU %0 to other GPUs\n", numTransfers, exeIndex);

  // Prepare headers
  table.Set(0, 0, " Bytes ");
  table.Set(0, 1, " DMA ");
  for (int i = 0; i < numSesList.size(); i++) {
    table.Set(0, 2+i, " BMA (%d) ", numSesList[i]);
  }
  table.DrawRowBorder(0);
  table.DrawRowBorder(1);
  table.DrawRowBorder(numRows);
  table.DrawColBorder(0);
  table.DrawColBorder(1);
  table.DrawColBorder(2);
  table.DrawColBorder(numCols);

  if (!ev.outputToCsv){
    Utils::Print("Executing: ");
    fflush(stdout);
  };

  for (size_t numBytes = 1ULL<<minPow2Exp, currRow=1; numBytes <= (1ULL<<maxPow2Exp); numBytes<<=1, currRow++) {
    if (!ev.outputToCsv) {
      Utils::Print(".");
      fflush(stdout);
    }

    table.Set(currRow, 0, " %lu ", numBytes);
    std::vector<Transfer> transfers(1);

    Transfer& t = transfers[0];
    t.numBytes = numBytes;
    t.srcs     = {{gpuMemType, exeIndex}};
    t.dsts.clear();
    for (int i = 0; i < numGpuDevices; i++) {
      if (i == exeIndex && localCopy == 0) continue;
      t.dsts.push_back({gpuMemType, i});
    }

    // DMA executor first
    t.exeDevice = {EXE_GPU_DMA, exeIndex};
    t.numSubExecs = 1;

    if (!TransferBench::RunTransfers(cfg, transfers, results)) {
      for (auto const& err : results.errResults)
        Utils::Print("%s\n", err.errMsg.c_str());
      return 1;
    }

    table.Set(currRow, 1, " %6.2f ", numTransfers * results.tfrResults[0].avgBandwidthGbPerSec);

    // BMA executor next
    t.exeDevice = {EXE_GPU_BDMA, exeIndex};
    for (int i = 0; i < numSesList.size(); i++) {
      t.numSubExecs = numSesList[i];

      if (!TransferBench::RunTransfers(cfg, transfers, results)) {
        for (auto const& err : results.errResults)
          Utils::Print("%s\n", err.errMsg.c_str());
        return 1;
      }

      table.Set(currRow, 2+i, " %6.2f ", numTransfers * results.tfrResults[0].avgBandwidthGbPerSec);
    }
  }

  if (!ev.outputToCsv) {
    Utils::Print("\n");
  }
  table.PrintTable(ev.outputToCsv, ev.showBorders);
  Utils::Print("Reported numbers are all GB/s, normalized for per Transfer for %d Transfers\n", numTransfers);

  return 0;
}

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
int SchmooPreset(EnvVars&          ev,
                 size_t      const numBytesPerTransfer,
                 std::string const presetName,
                 bool        const bytesSpecified)
{
  if (TransferBench::GetNumRanks() > 1) {
    Utils::Print("[ERROR] Schmoo preset currently not supported for multi-node\n");
    return ERR_FATAL;
  }

  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  if (numDetectedGpus < 2) {
    printf("[ERROR] Schmoo benchmark requires at least 2 GPUs\n");
    return ERR_FATAL;
  }

  // Collect env vars for this preset
  int gpuMemTypeIdx = EnvVars::GetEnvVar("GPU_MEM_TYPE",    0);
  int localIdx      = EnvVars::GetEnvVar("LOCAL_IDX",      0);
  int remoteIdx     = EnvVars::GetEnvVar("REMOTE_IDX",     1);
  int sweepMax      = EnvVars::GetEnvVar("SWEEP_MAX",      32);
  int sweepMin      = EnvVars::GetEnvVar("SWEEP_MIN",      1);

  MemType gpuMemType = Utils::GetGpuMemType(gpuMemTypeIdx);

  // Display environment variables
  ev.DisplayEnvVars();
  if (!ev.hideEnv) {
    int outputToCsv = ev.outputToCsv;
    if (!outputToCsv) printf("[Schmoo Related]\n");
    ev.Print("GPU_MEM_TYPE"   , gpuMemTypeIdx,  "Using %s (%s)", Utils::GetGpuMemTypeStr(gpuMemTypeIdx).c_str(), Utils::GetAllGpuMemTypeStr().c_str());
    ev.Print("LOCAL_IDX",      localIdx,     "Local GPU index");
    ev.Print("REMOTE_IDX",     remoteIdx,    "Remote GPU index");
    ev.Print("SWEEP_MAX",      sweepMax,     "Max number of subExecutors to use");
    ev.Print("SWEEP_MIN",      sweepMin,     "Min number of subExecutors to use");
    printf("\n");
  }

  // Validate env vars
  if (localIdx >= numDetectedGpus || remoteIdx >= numDetectedGpus) {
    printf("[ERROR] Cannot execute schmoo test with local GPU device %d, remote GPU device %d\n", localIdx, remoteIdx);
    return ERR_FATAL;
  }

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  TransferBench::TestResults results;

  char memChar = MemTypeStr[gpuMemType];
  printf("Bytes to transfer: %lu Local GPU: %d Remote GPU: %d\n", numBytesPerTransfer, localIdx, remoteIdx);
  printf("       | Local Read  | Local Write | Local Copy  | Remote Read | Remote Write| Remote Copy |\n");
  printf("  #CUs |%c%02d->G%02d->N00|N00->G%02d->%c%02d|%c%02d->G%02d->%c%02d|%c%02d->G%02d->N00|N00->G%02d->%c%02d|%c%02d->G%02d->%c%02d|\n",
         memChar, localIdx, localIdx,
         localIdx, memChar, localIdx,
         memChar, localIdx, localIdx, memChar, localIdx,
         memChar, remoteIdx, localIdx,
         localIdx, memChar, remoteIdx,
         memChar, localIdx, localIdx, memChar, remoteIdx);
  printf("|------|-------------|-------------|-------------|-------------|-------------|-------------|\n");

  std::vector<Transfer> transfers(1);
  Transfer& t   = transfers[0];
  t.exeDevice   = {EXE_GPU_GFX, localIdx};
  t.exeSubIndex = -1;
  t.numBytes    = numBytesPerTransfer;

  for (int numCUs = sweepMin; numCUs <= sweepMax; numCUs++) {
    t.numSubExecs = numCUs;

    // Local Read
    t.srcs = {{gpuMemType, localIdx}};
    t.dsts = {};
    if (!TransferBench::RunTransfers(cfg, transfers, results)) {
      Utils::PrintErrors(results.errResults);
      return ERR_FATAL;
    }
    double const localRead = results.tfrResults[0].avgBandwidthGbPerSec;

    // Local Write
    t.srcs = {};
    t.dsts = {{gpuMemType, localIdx}};
    if (!TransferBench::RunTransfers(cfg, transfers, results)) {
      Utils::PrintErrors(results.errResults);
      return ERR_FATAL;
    }
    double const localWrite = results.tfrResults[0].avgBandwidthGbPerSec;

    // Local Copy
    t.srcs = {{gpuMemType, localIdx}};
    t.dsts = {{gpuMemType, localIdx}};
    if (!TransferBench::RunTransfers(cfg, transfers, results)) {
      Utils::PrintErrors(results.errResults);
      return ERR_FATAL;
    }
    double const localCopy = results.tfrResults[0].avgBandwidthGbPerSec;

    // Remote Read
    t.srcs = {{gpuMemType, remoteIdx}};
    t.dsts = {};
    if (!TransferBench::RunTransfers(cfg, transfers, results)) {
      Utils::PrintErrors(results.errResults);
      return ERR_FATAL;
    }
    double const remoteRead = results.tfrResults[0].avgBandwidthGbPerSec;

    // Remote Write
    t.srcs = {};
    t.dsts = {{gpuMemType, remoteIdx}};
    if (!TransferBench::RunTransfers(cfg, transfers, results)) {
      Utils::PrintErrors(results.errResults);
      return ERR_FATAL;
    }
    double const remoteWrite = results.tfrResults[0].avgBandwidthGbPerSec;

    // Remote Copy
    t.srcs = {{gpuMemType, localIdx}};
    t.dsts = {{gpuMemType, remoteIdx}};
    if (!TransferBench::RunTransfers(cfg, transfers, results)) {
      Utils::PrintErrors(results.errResults);
      return ERR_FATAL;
    }
    double const remoteCopy = results.tfrResults[0].avgBandwidthGbPerSec;

    printf("   %3d   %11.3f   %11.3f   %11.3f   %11.3f   %11.3f   %11.3f  \n",
           numCUs, localRead, localWrite, localCopy, remoteRead, remoteWrite, remoteCopy);
  }
  return ERR_NONE;
}

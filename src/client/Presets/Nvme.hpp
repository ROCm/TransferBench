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
#include <sys/stat.h>

int NvmePreset(EnvVars&          ev,
               size_t      const numBytesPerTransfer,
               std::string const presetName,
               bool        const bytesSpecified)
{
  if (TransferBench::GetNumRanks() > 1) {
    Utils::Print("[ERROR] nvme preset currently not supported for multi-node\n");
    return ERR_FATAL;
  }

#if defined(__NVCC__)
  Utils::Print("[ERROR] nvme preset requires the hipFile executor, which is not supported on NVIDIA hardware\n");
  return ERR_FATAL;
#endif

  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);
  if (numDetectedGpus == 0) {
    Utils::Print("[ERROR] No GPU detected, GPUs are required to run this preset\n");
    return ERR_FATAL;
  }

  int numStoragePaths = TransferBench::GetNumStoragePaths();
  if (numStoragePaths == 0) {
    Utils::Print("[ERROR] nvme preset requires at least one storage path, but TB_STORAGE_PATHS is not set\n");
    Utils::Print("        Set TB_STORAGE_PATHS to a comma-separated list of files (typically on an NVMe-backed filesystem)\n");
    Utils::Print("        and/or raw block devices.  Storage memory SN refers to the N-th entry (starting from 0).  For example:\n");
    Utils::Print("          TB_STORAGE_PATHS=/mnt/nvme0/tb.dat,/mnt/nvme1/tb.dat ./TransferBench nvme 1G\n");
    Utils::Print("        Regular files are created/grown as needed and are left in place afterwards\n");
    Utils::Print("        Files that already hold data are only read from, unless TB_STORAGE_ALLOW_OVERWRITE=1 is set\n");
    Utils::Print("        Block devices (e.g. /dev/nvme1n1) are only read from, unless TB_STORAGE_ALLOW_BLOCK_WRITE=1 is set\n");
    return ERR_FATAL;
  }
  if (!TransferBench::IsStoragePathListValid()) {
    Utils::Print("[ERROR] TB_STORAGE_PATHS has an empty entry ahead of a path, which would renumber the storage indices\n");
    Utils::Print("        that follow it.  Remove the empty entry\n");
    return ERR_FATAL;
  }

  if (numBytesPerTransfer == 0) {
    Utils::Print("[ERROR] nvme preset requires a non-zero number of bytes per Transfer\n");
    return ERR_FATAL;
  }

  // Collect env vars for this preset
  int numFileSubExecs = EnvVars::GetEnvVar("NUM_FILE_SE",     4);
  int numGpuDevices   = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int nvmeMode        = EnvVars::GetEnvVar("NVME_MODE",       0);
  int nvmePaired      = EnvVars::GetEnvVar("NVME_PAIRED",     0);

  if (numGpuDevices < 1 || numGpuDevices > numDetectedGpus) {
    Utils::Print("[ERROR] NUM_GPU_DEVICES must be between 1 and %d (instead of %d)\n", numDetectedGpus, numGpuDevices);
    return ERR_FATAL;
  }
  if (numFileSubExecs < 1) {
    Utils::Print("[ERROR] NUM_FILE_SE must be positive (instead of %d)\n", numFileSubExecs);
    return ERR_FATAL;
  }
  if (nvmeMode < 0 || nvmeMode > 2) {
    Utils::Print("[ERROR] NVME_MODE must be 0 (write + read), 1 (write only) or 2 (read only) (instead of %d)\n", nvmeMode);
    return ERR_FATAL;
  }

  // Display environment variables
  if (Utils::RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
      if (!ev.outputToCsv) printf("[NVMe Related]\n");
      ev.Print("NUM_FILE_SE",     numFileSubExecs, "Using %d I/O thread(s) per Transfer", numFileSubExecs);
      ev.Print("NUM_GPU_DEVICES", numGpuDevices,   "Using %d GPUs", numGpuDevices);
      ev.Print("NVME_MODE",       nvmeMode,        "Running %s transfers", nvmeMode == 0 ? "write + read" :
                                                                          nvmeMode == 1 ? "write-only"
                                                                                        : "read-only");
      ev.Print("NVME_PAIRED",     nvmePaired,      "%s", nvmePaired ? "Each GPU uses one storage path (GPU index modulo # paths)"
                                                                    : "Each GPU uses every storage path");
      printf("\n");
    }
  }

  if (!TransferBench::IsHipFileAvailable()) {
    Utils::Print("[ERROR] hipFile executor is not available (TransferBench was built without hipFile support)\n");
    Utils::Print("        Rebuild with the hipFile development package (amdrocm-hipfile-dev<version>) installed\n");
    return ERR_FATAL;
  }

  // Raw block devices and files that already hold data are only written to if explicitly allowed
  // (same rule as enforced by backend)
  char const* allowBlockWriteStr = getenv("TB_STORAGE_ALLOW_BLOCK_WRITE");
  char const* allowOverwriteStr  = getenv("TB_STORAGE_ALLOW_OVERWRITE");
  bool const  allowBlockWrite    = (allowBlockWriteStr && !strcmp(allowBlockWriteStr, "1"));
  bool const  allowOverwrite     = (allowOverwriteStr  && !strcmp(allowOverwriteStr,  "1"));
  std::vector<bool> isReadOnlyStorage(numStoragePaths);
  for (int i = 0; i < numStoragePaths; i++) {
    std::string const path = TransferBench::GetStoragePath(i);
    struct stat st;
    bool const exists        = (stat(path.c_str(), &st) == 0);
    bool const isBlockDevice = (exists && S_ISBLK(st.st_mode));
    bool const holdsData     = (exists && S_ISREG(st.st_mode) && st.st_size > 0);
    isReadOnlyStorage[i] = isBlockDevice ? !allowBlockWrite : (holdsData && !allowOverwrite);
    Utils::Print("Storage %d (S%d): %s (%s)\n", i, i, path.c_str(),
                 isBlockDevice ? (allowBlockWrite ? "block device, contents will be overwritten"
                                                  : "block device, read-only unless TB_STORAGE_ALLOW_BLOCK_WRITE=1") :
                 holdsData     ? (allowOverwrite  ? "existing file, contents will be overwritten"
                                                  : "existing file, read-only unless TB_STORAGE_ALLOW_OVERWRITE=1")
                               : "file");
  }
  if (ev.alwaysValidate < 0)
    Utils::Print("[WARN] Data validation is disabled (ALWAYS_VALIDATE < 0)\n");

  char const separator = ev.outputToCsv ? ',' : ' ';
  Utils::Print("Bytes Per Transfer%c%lu\n", separator, numBytesPerTransfer);

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  TransferBench::TestResults results;

  // Each row is one (GPU, storage) combination
  int const numStoragePerGpu = nvmePaired ? 1 : numStoragePaths;
  int const numRows = 1 + numGpuDevices * numStoragePerGpu;
  int const numCols = 7;
  Utils::TableHelper table(numRows, numCols);

  table.Set(0, 0, " GPU ");
  table.Set(0, 1, " Storage ");
  table.Set(0, 2, " Write (GB/s) ");
  table.Set(0, 3, " Write path ");
  table.Set(0, 4, " Read (GB/s) ");
  table.Set(0, 5, " Read path ");
  table.Set(0, 6, " Result ");
  table.DrawRowBorder(0);
  table.DrawRowBorder(1);
  table.DrawRowBorder(numRows);
  for (int col = 0; col <= numCols; col++)
    table.DrawColBorder(col);

  int numRun = 0, numFailed = 0, numUnvalidated = 0;
  std::set<std::string> shownWarnings;

  int rowIdx = 1;
  for (int gpuIdx = 0; gpuIdx < numGpuDevices; gpuIdx++) {
    for (int i = 0; i < numStoragePerGpu; i++) {
      int  const storageIdx = nvmePaired ? (gpuIdx % numStoragePaths) : i;
      bool const isReadOnly = isReadOnlyStorage[storageIdx];

      table.Set(rowIdx, 0, " GPU %02d ", gpuIdx);
      table.Set(rowIdx, 1, " S%d ",    storageIdx);

      bool rowFailed = false, rowValidated = true, rowRan = false;

      // Write (GPU -> storage), then read (storage -> GPU)
      for (int isRead = 0; isRead <= 1; isRead++) {
        int const bwCol = isRead ? 4 : 2;
        if ((nvmeMode == 1 && isRead) || (nvmeMode == 2 && !isRead) || (!isRead && isReadOnly)) {
          table.Set(rowIdx, bwCol,     " N/A ");
          table.Set(rowIdx, bwCol + 1, " N/A ");
          continue;
        }

        MemDevice const gpuMem     = {MEM_GPU,     gpuIdx};
        MemDevice const storageMem = {MEM_STORAGE, storageIdx};

        std::vector<Transfer> transfers(1);
        transfers[0].numBytes    = numBytesPerTransfer;
        transfers[0].srcs.push_back(isRead ? storageMem : gpuMem);
        transfers[0].dsts.push_back(isRead ? gpuMem : storageMem);
        transfers[0].exeDevice   = {EXE_GPU_HIPFILE, gpuIdx};
        transfers[0].exeSubIndex = -1;
        transfers[0].numSubExecs = numFileSubExecs;

        numRun++;
        rowRan = true;
        bool const success = TransferBench::RunTransfers(cfg, transfers, results);
        for (auto const& err : results.errResults) {
          if (err.errType == ERR_FATAL) {
            Utils::Print("[ERROR] (%s->%s->%s) %s\n",
                         Utils::MemDevicesToStr(transfers[0].srcs).c_str(),
                         Utils::ExeDeviceToStr(transfers[0].exeDevice).c_str(),
                         Utils::MemDevicesToStr(transfers[0].dsts).c_str(), err.errMsg.c_str());
          } else if (shownWarnings.insert(err.errMsg).second) {
            Utils::Print("[WARN] %s\n", err.errMsg.c_str());
          }
        }

        if (!success) {
          numFailed++;
          rowFailed = true;
          table.Set(rowIdx, bwCol,     " FAIL ");
          table.Set(rowIdx, bwCol + 1, " N/A ");
          continue;
        }

        // Existing contents of read-only storage cannot be validated
        if (ev.alwaysValidate < 0 || (isRead && isReadOnly)) {
          numUnvalidated++;
          rowValidated = false;
        }
        table.Set(rowIdx, bwCol,     " %.3f ", results.tfrResults[0].avgBandwidthGbPerSec);
        table.Set(rowIdx, bwCol + 1, " %s ",   TransferBench::IoPathTypeStr[results.tfrResults[0].ioPath]);
      }

      table.Set(rowIdx, 6, " %s ", !rowRan       ? "N/A"  :
                                   rowFailed     ? "FAIL" :
                                   !rowValidated ? "UNVALIDATED" : "PASS");
      rowIdx++;
    }
  }

  Utils::Print("GPU <-> storage bandwidth GB/s (hipFile %s, %d I/O thread(s) per Transfer)\n",
               TransferBench::GetHipFileVersion().c_str(), numFileSubExecs);
  table.PrintTable(ev.outputToCsv, ev.showBorders);

  if (numRun == 0) {
    Utils::Print("[ERROR] No Transfers were run (see NVME_MODE, TB_STORAGE_ALLOW_OVERWRITE and TB_STORAGE_ALLOW_BLOCK_WRITE)\n");
    return ERR_FATAL;
  }
  if (numFailed) {
    Utils::Print("[ERROR] %d of %d Transfers FAILED\n", numFailed, numRun);
    return ERR_FATAL;
  }
  if (numUnvalidated)
    Utils::Print("%d Transfers completed (%d without data validation)\n", numRun, numUnvalidated);
  else
    Utils::Print("All %d Transfers passed\n", numRun);
  return ERR_NONE;
}

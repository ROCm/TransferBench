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

// Pingpong latency presets, which differ only in which pairs of GPU executors run together:
//   p2p_latency     - Every pair runs by itself, one pair at a time
//   one2all_latency - One PING executor runs against all PONG executors at once, one PING executor at a time
//   a2a_latency     - Every pair runs at once
//   ring_latency    - Each GPU pings its neighbor; every hop in the ring(s) runs at once
int LatencyPreset(EnvVars&          ev,
                  size_t      const /*numBytesPerTransfer*/,
                  std::string const presetName,
                  [[maybe_unused]] bool const bytesSpecified)
{
  bool const isP2p     = (presetName == "p2p_latency");
  bool const isOne2All = (presetName == "one2all_latency");
  bool const isA2a     = (presetName == "a2a_latency");
  bool const isRing    = (presetName == "ring_latency");

  if (!Utils::AllRanksHaveSameGpuCount()) {
    Utils::Print("[ERROR] %s preset requires all ranks to have the same number of GPUs\n", presetName.c_str());
    Utils::Print("[ERROR] Run ./TransferBench without any args to display topology information\n");
    return ERR_FATAL;
  }

  int const numRanks        = TransferBench::GetNumRanks();
  int const numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  // Collect env vars for this preset
  int a2aLocal      = isA2a ? EnvVars::GetEnvVar("A2A_LOCAL", 0) : 0;
  int memTypeIdx    = EnvVars::GetEnvVar("GPU_MEM_TYPE"   , 0);
  int numGpuDevices = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int numLaps       = EnvVars::GetEnvVar("NUM_LAPS"       , 1000);
  int useRemoteRead = EnvVars::GetEnvVar("USE_REMOTE_READ", 0);
  int stride        = isRing ? EnvVars::GetEnvVar("STRIDE", 1) : 1;
  int ringSize      = isRing ? EnvVars::GetEnvVar("RING_SIZE", numRanks * numGpuDevices) : 0;

  MemType     const memType    = Utils::GetGpuMemType(memTypeIdx);
  std::string const memTypeStr = Utils::GetGpuMemTypeStr(memTypeIdx);

  // Display environment variables
  if (Utils::RankDoesOutput()) {
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
      if (!ev.outputToCsv) printf("[Latency Related]\n");
      if (isA2a)
        ev.Print("A2A_LOCAL",     a2aLocal,      "%s pairs with PING and PONG on the same GPU", a2aLocal ? "Include" : "Exclude");
      ev.Print("GPU_MEM_TYPE",    memTypeIdx,    "Using %s memory for flags (%s)", memTypeStr.c_str(), Utils::GetAllGpuMemTypeStr().c_str());
      ev.Print("NUM_GPU_DEVICES", numGpuDevices, "Using %d GPUs%s", numGpuDevices, numRanks > 1 ? " per rank" : "");
      ev.Print("NUM_LAPS",        numLaps,       "Timing %d round trips per iteration", numLaps);
      if (isRing) {
        ev.Print("RING_SIZE",     ringSize,      "Building rings of size %d", ringSize);
        ev.Print("STRIDE",        stride,        "Reordering devices by taking %d steps", stride);
      }
      ev.Print("USE_REMOTE_READ", useRemoteRead, "%s", useRemoteRead ? "Executors write to their own memory and poll their partner's"
                                                                     : "Executors write to their partner's memory and poll their own");
      printf("\n");
    }
  }

  // Check that input parameters are uniform across all ranks
  IS_UNIFORM(a2aLocal,      "A2A_LOCAL");
  IS_UNIFORM(memTypeIdx,    "GPU_MEM_TYPE");
  IS_UNIFORM(numGpuDevices, "NUM_GPU_DEVICES");
  IS_UNIFORM(numLaps,       "NUM_LAPS");
  if (isRing) {
    IS_UNIFORM(ringSize,    "RING_SIZE");
    IS_UNIFORM(stride,      "STRIDE");
  }
  IS_UNIFORM(useRemoteRead, "USE_REMOTE_READ");

  // Validate env vars
  if (numGpuDevices < 1 || numGpuDevices > numDetectedGpus) {
    Utils::Print("[ERROR] NUM_GPU_DEVICES must be between 1 and %d (got %d)\n", numDetectedGpus, numGpuDevices);
    return ERR_FATAL;
  }
  if (numLaps < 1) {
    Utils::Print("[ERROR] NUM_LAPS must be positive (got %d)\n", numLaps);
    return ERR_FATAL;
  }

  // Executors are ordered rank-major so that each rank's GPUs stay together in the matrix
  std::vector<ExeDevice> exes;
  for (int rank = 0; rank < numRanks; rank++)
    for (int gpu = 0; gpu < numGpuDevices; gpu++)
      exes.push_back({EXE_GPU_GFX, gpu, rank});
  int const numExes = (int)exes.size();

  // Ring hops follow the same ordering as the rings preset: devices are listed rank-major,
  // reordered by STRIDE, then split into rings of RING_SIZE.  GPU i pings the next GPU in its ring.
  std::vector<int> ringOrder;
  if (isRing) {
    if (ringSize <= 1) {
      if (numExes < 2)
        Utils::Print("[ERROR] %s requires at least 2 GPUs\n", presetName.c_str());
      else
        Utils::Print("[ERROR] RING_SIZE must be greater than 1 (got %d)\n", ringSize);
      return ERR_FATAL;
    }
    if (numExes % ringSize) {
      Utils::Print("[ERROR] Ring size %d must evenly divide the total number of GPUs %d\n", ringSize, numExes);
      return ERR_FATAL;
    }
    ringOrder.resize(numExes);
    for (int i = 0; i < numExes; i++) ringOrder[i] = i;
    Utils::StrideGenerate(ringOrder, stride);
  }

  // Select the (PING, PONG) pairs to measure.  Cross-rank pairs exchange flags through fabric
  // handles, which requires pod support and both ranks to be in the same pod
  auto canPair = [&](int i, int j) {
    bool ok = (exes[i].exeRank == exes[j].exeRank);
#ifdef POD_COMM_ENABLED
    ok |= TransferBench::IsSamePod(exes[j].exeRank, exes[i].exeRank);
#endif
    return ok;
  };

  std::vector<std::vector<bool>> isMeasured(numExes, std::vector<bool>(numExes, false));
  int numPairs = 0, numSkipped = 0;
  if (isRing) {
    int const numRings = numExes / ringSize;
    for (int ringIdx = 0; ringIdx < numRings; ringIdx++) {
      int const ringBase = ringIdx * ringSize;
      for (int hop = 0; hop < ringSize; hop++) {
        int const src = ringOrder[ringBase + hop];
        int const dst = ringOrder[ringBase + (hop + 1) % ringSize];
        if (!canPair(src, dst)) {
          numSkipped++;
          continue;
        }
        isMeasured[src][dst] = true;
        numPairs++;
      }
    }
  } else {
    for (int i = 0; i < numExes; i++) {
      for (int j = 0; j < numExes; j++) {
        if (i == j && !isP2p && !a2aLocal) continue;
        if (!canPair(i, j)) {
          numSkipped++;
          continue;
        }
        isMeasured[i][j] = true;
        numPairs++;
      }
    }
  }

  if (numSkipped > 0)
    Utils::Print("[WARN] %d cross-rank pair(s) are shown as N/A: this requires pod communication support and ranks in the same pod\n",
                 numSkipped);
  if (numPairs == 0) {
    Utils::Print("[WARN] No pairs to measure. %s requires at least 2 GPUs%s\n",
                 presetName.c_str(), isA2a ? " (or A2A_LOCAL=1)" : "");
    return ERR_NONE;
  }

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  TransferBench::TestResults results;

  // Round-trip latency per lap in microseconds, negative when not measured
  std::vector<std::vector<double>> latencyUs(numExes, std::vector<double>(numExes, -1.0));

  // Runs the given (PING, PONG) pairs in parallel
  auto runPairs = [&](std::vector<std::pair<int, int>> const& pairs) {
    std::vector<Transfer> transfers;
    for (auto const& [i, j] : pairs) {
      MemDevice const pingMem = {memType, exes[i].exeIndex, exes[i].exeRank};
      MemDevice const pongMem = {memType, exes[j].exeIndex, exes[j].exeRank};

      Transfer t;
      t.numBytes      = 8;  // Only sizes the flag seed allocation; pingpong always exchanges 1-byte flags
      t.numSubExecs   = 1;
      t.numLaps       = numLaps;
      t.srcs          = {{MEM_NULL, 0}, {MEM_NULL, 0}};
      t.exeDevice     = exes[i];
      t.exeDevicePong = exes[j];
      // Each half writes to its own DST, which the partner half polls
      if (useRemoteRead) t.dsts = {pingMem, pongMem};
      else               t.dsts = {pongMem, pingMem};
      transfers.push_back(t);
    }

    if (!TransferBench::RunTransfers(cfg, transfers, results)) {
      for (auto const& err : results.errResults)
        Utils::Print("[%s] %s\n", err.errType == ERR_FATAL ? "ERROR" : "WARN", err.errMsg.c_str());
      return;
    }

    for (size_t k = 0; k < pairs.size(); k++)
      latencyUs[pairs[k].first][pairs[k].second] = results.tfrResults[k].avgDurationMsec * 1000.0;
  };

  char const sep   = ev.outputToCsv ? ',' : ' ';
  int  const width = numRanks > 1 ? 12 : 10;
  auto exeStr = [&](ExeDevice const& exe) {
    char buf[32];
    if (numRanks > 1) snprintf(buf, sizeof(buf), "R%d GPU %02d", exe.exeRank, exe.exeIndex);
    else              snprintf(buf, sizeof(buf), "GPU %02d", exe.exeIndex);
    return std::string(buf);
  };

  char modeBuf[192];
  if (isRing) {
    int const numRings = numExes / ringSize;
    if (stride == 1)
      snprintf(modeBuf, sizeof(modeBuf), "%d parallel ring%s of %d, each GPU pinging its neighbor",
               numRings, numRings == 1 ? "" : "s", ringSize);
    else
      snprintf(modeBuf, sizeof(modeBuf), "%d parallel ring%s of %d (stride %d), each GPU pinging its neighbor",
               numRings, numRings == 1 ? "" : "s", ringSize, stride);
  } else if (isP2p) {
    snprintf(modeBuf, sizeof(modeBuf), "each pair run by itself");
  } else if (isOne2All) {
    snprintf(modeBuf, sizeof(modeBuf), "each PING executor run against all PONG executors in parallel");
  } else {
    snprintf(modeBuf, sizeof(modeBuf), "all pairs run in parallel");
  }
  Utils::Print("Pingpong round-trip latency per lap (us), %s\n", modeBuf);
  if (isRing) {
    int const numRings = numExes / ringSize;
    for (int ringIdx = 0; ringIdx < numRings; ringIdx++) {
      int const ringBase = ringIdx * ringSize;
      Utils::Print("Ring %02d:", ringIdx);
      for (int hop = 0; hop < ringSize; hop++)
        Utils::Print(" %s ->", exeStr(exes[ringOrder[ringBase + hop]]).c_str());
      Utils::Print(" %s\n", exeStr(exes[ringOrder[ringBase]]).c_str());
    }
  }
  Utils::Print("[%d laps] [%s memory flags] [%s]\n", numLaps, memTypeStr.c_str(),
               useRemoteRead ? "local write / remote poll" : "remote write / local poll");

  // PING executors are rows, PONG executors are columns
  Utils::Print("%*s", width, "PING\\PONG");
  for (int j = 0; j < numExes; j++)
    Utils::Print("%c%*s", sep, width, exeStr(exes[j]).c_str());
  Utils::Print("\n");

  if (isA2a || isRing) {
    std::vector<std::pair<int, int>> pairs;
    for (int i = 0; i < numExes; i++)
      for (int j = 0; j < numExes; j++)
        if (isMeasured[i][j]) pairs.push_back({i, j});
    runPairs(pairs);
  }

  for (int i = 0; i < numExes; i++) {
    if (isP2p) {
      for (int j = 0; j < numExes; j++)
        if (isMeasured[i][j]) runPairs({{i, j}});
    } else if (isOne2All) {
      std::vector<std::pair<int, int>> pairs;
      for (int j = 0; j < numExes; j++)
        if (isMeasured[i][j]) pairs.push_back({i, j});
      if (!pairs.empty()) runPairs(pairs);
    }

    Utils::Print("%*s", width, exeStr(exes[i]).c_str());
    for (int j = 0; j < numExes; j++) {
      if (latencyUs[i][j] < 0)
        Utils::Print("%c%*s", sep, width, "N/A");
      else
        Utils::Print("%c%*.3f", sep, width, latencyUs[i][j]);
    }
    Utils::Print("\n");
  }

  // Summarize latencies between distinct executors
  int    count = 0;
  double sumUs = 0.0;
  std::pair<int, int> minPair, maxPair;
  for (int i = 0; i < numExes; i++) {
    for (int j = 0; j < numExes; j++) {
      if (i == j || latencyUs[i][j] < 0) continue;
      if (count == 0 || latencyUs[i][j] < latencyUs[minPair.first][minPair.second]) minPair = {i, j};
      if (count == 0 || latencyUs[i][j] > latencyUs[maxPair.first][maxPair.second]) maxPair = {i, j};
      sumUs += latencyUs[i][j];
      count++;
    }
  }
  if (count > 0) {
    Utils::Print("\n");
    Utils::Print("Average latency: %8.3f us over %d pairs\n", sumUs / count, count);
    Utils::Print("Minimum latency: %8.3f us (PING %s, PONG %s)\n", latencyUs[minPair.first][minPair.second],
                 exeStr(exes[minPair.first]).c_str(), exeStr(exes[minPair.second]).c_str());
    Utils::Print("Maximum latency: %8.3f us (PING %s, PONG %s)\n", latencyUs[maxPair.first][maxPair.second],
                 exeStr(exes[maxPair.first]).c_str(), exeStr(exes[maxPair.second]).c_str());
  }

  if (numRanks > 1 && Utils::HasDuplicateHostname()) {
    Utils::Print("[WARN] It is recommended to run TransferBench with one rank per host to avoid potential aliasing of executors\n");
  }
  return ERR_NONE;
}

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

#pragma once
#include <map>
#include <vector>

// EnvVars is available to all presets
#include "EnvVars.hpp"
#include "Utilities.hpp"

#include "AllToAll.hpp"
#include "AllToAllN.hpp"
#include "AllToAllSweep.hpp"
#include "BmaSweep.hpp"
#include "GfxSweep.hpp"
#include "HbmBandwidth.hpp"
#include "HealthCheck.hpp"
#include "NicRings.hpp"
#include "NicPeerToPeer.hpp"
#include "OneToAll.hpp"
#include "PeerToPeer.hpp"
#include "PodAllToAll.hpp"
#include "PodPeerToPeer.hpp"
#include "Scaling.hpp"
#include "Schmoo.hpp"
#include "SmokeTest.hpp"
#include "Sweep.hpp"
#include "WallClock.hpp"

typedef int (*PresetFunc)(EnvVars&          ev,
                          size_t      const numBytesPerTransfer,
                          std::string const presetName,
                          [[maybe_unused]] bool const bytesSpecified);

struct PresetInfo
{
  PresetFunc   func;
  bool         multiRankCompatible;
  std::string  description;
  std::string  details;
};

std::map<std::string, PresetInfo> presetFuncMap =
{
  {"a2a",         {AllToAllPreset,      true,  "Tests parallel transfers between all pairs of GPU devices",
                   "Runs dense all-to-all copies across all visible GPUs (and ranks when present)."}},
  {"a2a_n",       {AllToAllRdmaPreset,  false, "Tests parallel transfers between all pairs of GPU devices using Nearest NIC RDMA transfers",
                   "Exercises nearest-NIC RDMA path for all GPU pairs (single rank only)."}},
  {"a2asweep",    {AllToAllSweepPreset, false, "Test GFX-based all-to-all transfers swept across different CU and GFX unroll counts",
                   "Sweeps CU and unroll settings to tune GFX all-to-all behavior."}},
  {"bmasweep",    {BmaSweepPreset,      false, "Test and compare batched DMA executor for multi destination copies",
                   "Compares batched DMA strategies for fan-out copy patterns."}},
  {"gfxsweep",    {GfxSweepPreset,      true,  "Sweep over various GFX kernel options for a given GFX Transfer",
                   "Sweeps GFX kernel parameters and reports best-performing combinations."}},
  {"hbm",         {HbmBandwidthPreset,  true,  "Tests HBM bandwidth",
                   "Measures sustained HBM read/write/copy behavior per GPU."}},
  {"healthcheck", {HealthCheckPreset,   false, "Simple bandwidth health check (MI300X series only)",
                   "Quick functional and bandwidth sanity test for supported MI300X setups."}},
  {"nicrings",    {NicRingsPreset,      true,  "Tests NIC rings created across identical NIC indices across ranks",
                   "Builds rank-wise NIC rings and measures collective ring bandwidth."}},
  {"nicp2p",      {NicPeerToPeerPreset, true,  "Multi-node peer-to-peer RDMA transfer test between all NICs",
                   "Runs exhaustive NIC-to-NIC RDMA throughput checks across ranks."}},
  {"one2all",     {OneToAllPreset,      false, "Test all subsets of parallel transfers from one GPU to all others",
                   "Evaluates one-source to many-destination transfer combinations."}},
  {"p2p"   ,      {PeerToPeerPreset,    false, "Peer-to-peer device memory bandwidth test",
                   "Benchmarks direct GPU-to-GPU memory transfer throughput."}},
  {"poda2a",      {PodAllToAllPreset,   true,  "All-to-all transfers between subgroups of ranks within a pod",
                   "Runs all-to-all over pod-scoped rank groups using detected pod membership."}},
  {"podp2p",      {PodPeerToPeerPreset, true,  "Peer-to-peer transfers test among ranks within a pod",
                   "Benchmarks pod-local peer transfer patterns across participating ranks."}},
  {"rsweep",      {SweepPreset,         false, "Randomly sweep through sets of Transfers",
                   "Randomized transfer sweep for broad spot-checking of transfer combinations."}},
  {"scaling",     {ScalingPreset,       false, "Run scaling test from one GPU to other devices",
                   "Measures scaling as destination count grows from a source GPU."}},
  {"schmoo",      {SchmooPreset,        false, "Scaling tests for local/remote read/write/copy",
                   "Runs schmoo-style sweeps over size and transfer type combinations."}},
  {"smoketest",   {SmokeTestPreset,     true,  "Simple correctness smoke-test",
                   "Fast correctness and sanity checks before running longer benchmarks."}},
  {"sweep",       {SweepPreset,         false, "Ordered sweep through sets of Transfers",
                   "Deterministic ordered sweep through predefined transfer combinations."}},
  {"wallclock",   {WallClockPreset,     true,  "Tests wallclock consistency across XCCs within a GPU",
                   "Checks GPU wallclock consistency and timing alignment across XCCs."}},
};

void DisplayBasicUsage(char const* cmdName)
{
  printf("Usage: %s config <N>\n", cmdName);
  printf("  config: Either:\n");
  printf("          - Filename of config file containing Transfers to execute (see example.cfg for format)\n");
  printf("          - Name of preset config (run '%s presets' to list available presets)\n", cmdName);
  printf("          - 'cmdline' followed by one transfer expression\n");
  printf("          - 'dryrun' followed by one transfer expression (prints parsed transfers only)\n");
  printf("  N     : (Optional) Number of bytes to copy per Transfer.\n");
  printf("          If not specified, defaults to 268435456 bytes. Must be a multiple of 4 bytes\n");
  printf("          If 0 is specified, a range of Ns will be benchmarked\n");
  printf("          May append a suffix ('K', 'M', 'G') for kilobytes / megabytes / gigabytes\n");
}

void DisplayTbEnvVarUsage()
{
  printf("\nInternal TB_* environment variables:\n");
  printf("====================================\n");
  printf(" TB_RANK            - Rank of this process (0-based, socket communicator)\n");
  printf(" TB_NUM_RANKS       - Total number of ranks (socket communicator)\n");
  printf(" TB_MASTER_ADDR     - Rank 0 IP/hostname for socket communicator\n");
  printf(" TB_MASTER_PORT     - Rank 0 port for socket communicator (default: 29500)\n");
  printf(" TB_SINGLE_LOG      - In socket mode, only rank 0 logs when set\n");
  printf(" TB_VERBOSE         - Enables additional internal logging\n");
  printf(" TB_DUMP_CFG_FILE   - Writes executed transfers to a config file\n");
  printf(" TB_DUMP_LINES      - Dumps randomized input-line statistics for FILL_COMPRESS setup\n");
  printf(" TB_NIC_FILTER      - Regex filter to limit NIC visibility for NIC executors\n");
  printf(" TB_FORCE_SINGLE_POD- Forces all ranks into one pod (skips pod query)\n");
  printf(" TB_WALLCLOCK_RATE  - Overrides queried GPU wallclock rate if needed\n");
  printf(" TB_PAUSE           - Pauses startup for debugger attachment\n");
}

void DisplayPresets()
{
  printf("\nAvailable Presets:\n");
  printf("======================================================================================================================\n");
  printf(" %-12s | %-18s | %-56s\n", "Preset", "Multi-rank", "What it does");
  printf("======================================================================================================================\n");
  for (auto const& x : presetFuncMap) {
    printf(" %-12s | %-18s | %-56s\n",
           x.first.c_str(),
           x.second.multiRankCompatible ? "Yes (see notes)" : "No",
           x.second.details.c_str());
  }
  printf(" %-12s | %-18s | %-56s\n", "help", "N/A", "Shows usage details, public env vars, and internal TB_* env vars");
  printf(" %-12s | %-18s | %-56s\n", "presets", "N/A", "Shows this preset table with compatibility and descriptions");
  printf("======================================================================================================================\n");
}

void DisplayHelp(char const* cmdName)
{
  DisplayBasicUsage(cmdName);
  printf("\n");
  EnvVars::DisplayUsage();
  DisplayTbEnvVarUsage();
  printf("\n");
  printf("Run '%s presets' for preset compatibility/details.\n", cmdName);
}

int RunPreset(EnvVars&       ev,
              size_t   const numBytesPerTransfer,
              int      const argc,
              char**   const argv,
              int&           retCode,
              char const*    cmdName)
{
  std::string preset = (argc > 1 ? argv[1] : "");
  bool bytesSpecified = (argc > 2);
  if (preset == "help") {
    DisplayHelp(cmdName);
    retCode = 0;
    return 1;
  }
  if (preset == "presets") {
    DisplayPresets();
    retCode = 0;
    return 1;
  }
  if (presetFuncMap.count(preset)) {
    retCode = (presetFuncMap[preset].func)(ev, numBytesPerTransfer, preset, bytesSpecified);
    return 1;
  }
  return 0;
}

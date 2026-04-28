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
#include "EnvVarsList.hpp"
#include "GfxSweep.hpp"
#include "HbmBandwidth.hpp"
#include "HealthCheck.hpp"
#include "Help.hpp"
#include "NicRings.hpp"
#include "NicPeerToPeer.hpp"
#include "OneToAll.hpp"
#include "PeerToPeer.hpp"
#include "PodAllToAll.hpp"
#include "PodPeerToPeer.hpp"
#include "PodRing.hpp"
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
  std::string  description;
};

std::map<std::string, PresetInfo> presetFuncMap =
{
  {"a2a",         {AllToAllPreset,      "Tests parallel transfers between all pairs of GPU devices"}},
  {"a2a_n",       {AllToAllRdmaPreset,  "Tests parallel transfers between all pairs of GPU devices using Nearest NIC RDMA transfers"}},
  {"a2asweep",    {AllToAllSweepPreset, "Test GFX-based all-to-all transfers swept across different CU and GFX unroll counts"}},
  {"bmasweep",    {BmaSweepPreset,      "Test and compare batched DMA executor for multi destination copies"}},
  {"envvars",     {EnvVarsPreset,       "Show list of environment variables that can be used to modify behavior"}},
  {"gfxsweep",    {GfxSweepPreset,      "Sweep over various GFX kernel options for a given GFX Transfer"}},
  {"hbm",         {HbmBandwidthPreset,  "Tests HBM bandwidth"}},
  {"healthcheck", {HealthCheckPreset,   "Simple bandwidth health check (MI300X series only)"}},
  {"help",        {HelpPreset,          "Shows example usage details"}},
  {"nicrings",    {NicRingsPreset,      "Tests NIC rings created across identical NIC indices across ranks"}},
  {"nicp2p",      {NicPeerToPeerPreset, "Multi-node peer-to-peer RDMA transfer test between all NICs"}},
  {"one2all",     {OneToAllPreset,      "Test all subsets of parallel transfers from one GPU to all others"}},
  {"p2p"   ,      {PeerToPeerPreset,    "Peer-to-peer device memory bandwidth test"}},
  {"poda2a",      {PodAllToAllPreset,   "All-to-all transfers between subgroups of ranks within a pod"}},
  {"podp2p",      {PodPeerToPeerPreset, "Peer-to-peer transfers test among ranks within a pod"}},
  {"podring",     {PodRingPreset,      "Ring transfers within subgroups of ranks in a pod"}},
  {"rsweep",      {SweepPreset,         "Randomly sweep through sets of Transfers"}},
  {"scaling",     {ScalingPreset,       "Run scaling test from one GPU to other devices"}},
  {"schmoo",      {SchmooPreset,        "Scaling tests for local/remote read/write/copy"}},
  {"smoketest",   {SmokeTestPreset,     "Simple correctness smoke-test"}},
  {"sweep",       {SweepPreset,         "Ordered sweep through sets of Transfers"}},
  {"wallclock",   {WallClockPreset,     "Tests wallclock consistency across XCCs within a GPU"}},
};

void DisplayPresets()
{
  if (!Utils::RankDoesOutput()) return;
  printf(" %-12s | %-56s\n", "Preset", "Description");
  printf("=============================================================================================================\n");
  for (auto const& x : presetFuncMap) {
    printf(" %-12s | %-56s\n",
           x.first.c_str(),
           x.second.description.c_str());
  }
  printf("=============================================================================================================\n");
}

int RunPreset(EnvVars&       ev,
              size_t   const numBytesPerTransfer,
              int      const argc,
              char**   const argv,
              int&           retCode)
{
  std::string preset = (argc > 1 ? argv[1] : "");
  bool bytesSpecified = (argc > 2);

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

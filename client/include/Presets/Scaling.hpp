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

#ifndef SCALING_PRESET_HPP
#define SCALING_PRESET_HPP

#include "EnvVars.hpp"
#include "Utilities.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>


int ScalingPreset(EnvVars& ev,
                  size_t const numBytesPerTransfer,
                  [[maybe_unused]] std::string const presetName)
{
    if (TransferBench::GetNumRanks() > 1) {
        Utils::Print("[ERROR] Scaling preset currently not supported for multi-node\n");
        return 1;
    }

    int numDetectedCpus = TransferBench::GetNumExecutors(EXE_CPU);
    int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

    // Collect env vars for this preset
    int localIdx      = EnvVars::GetEnvVar("LOCAL_IDX", 0);
    int numCpuDevices = EnvVars::GetEnvVar("NUM_CPU_DEVICES", numDetectedCpus);
    int numGpuDevices = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
    int sweepMax      = EnvVars::GetEnvVar("SWEEP_MAX", 32);
    int sweepMin      = EnvVars::GetEnvVar("SWEEP_MIN", 1);
    int useFineGrain  = EnvVars::GetEnvVar("USE_FINE_GRAIN", 0);

    // Display environment variables
    ev.DisplayEnvVars();
    if (!ev.hideEnv) {
        int outputToCsv = ev.outputToCsv;
        if (!outputToCsv) {
            printf("[Schmoo Related]\n");
        }
        ev.Print("LOCAL_IDX", localIdx, "Local GPU index");
        ev.Print("SWEEP_MAX", sweepMax, "Max number of subExecutors to use");
        ev.Print("SWEEP_MIN", sweepMin, "Min number of subExecutors to use");
        printf("\n");
    }

    // Validate env vars
    if (localIdx >= numDetectedGpus) {
        printf("[ERROR] Cannot execute scaling test with local GPU device %d\n", localIdx);
        return 1;
    }

    TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
    TransferBench::TestResults results;

    char separator = (ev.outputToCsv ? ',' : ' ');
    int numDevices = numCpuDevices + numGpuDevices;

    printf("GPU-GFX Scaling benchmark:\n");
    printf("==========================\n");
    printf("- Copying %lu bytes from GPU %d to other devices\n", numBytesPerTransfer, localIdx);
    printf("- All numbers reported as GB/sec\n\n");
    printf("NumCUs");
    for (int i = 0; i < numDevices; i++) {
        printf("%c  %s%02d     ",
               separator,
               i < numCpuDevices ? "CPU" : "GPU",
               i < numCpuDevices ? i : i - numCpuDevices);
    }
    printf("\n");

    std::vector<std::pair<double, int>> bestResult(numDevices);

    MemType memType = useFineGrain ? MEM_GPU_FINE : MEM_GPU;

    std::vector<Transfer> transfers(1);
    Transfer& t   = transfers[0];
    t.exeDevice   = {EXE_GPU_GFX, localIdx};
    t.exeSubIndex = -1;
    t.numBytes    = numBytesPerTransfer;
    t.srcs        = {{memType, localIdx}};

    for (int numSubExec = sweepMin; numSubExec <= sweepMax; numSubExec++) {
        t.numSubExecs = numSubExec;
        printf("%4d  ", numSubExec);

        for (int i = 0; i < numDevices; i++) {
            t.dsts = {
                {i < numCpuDevices ? MEM_CPU : MEM_GPU, i < numCpuDevices ? i : i - numCpuDevices}};
            if (!RunTransfers(cfg, transfers, results)) {
                Utils::PrintErrors(results.errResults);
                return 1;
            }
            double bw = results.tfrResults[0].avgBandwidthGbPerSec;
            printf("%c%7.2f     ", separator, bw);

            if (bw > bestResult[i].first) {
                bestResult[i].first  = bw;
                bestResult[i].second = numSubExec;
            }
        }
        printf("\n");
    }

    printf(" Best ");
    for (int i = 0; i < numDevices; i++) {
        printf("%c%7.2f(%3d)", separator, bestResult[i].first, bestResult[i].second);
    }
    printf("\n");
    return 0;
}


#endif    // SCALING_PRESET_HPP

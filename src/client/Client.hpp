/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#define CLIENT_VERSION "1.54"

#include <iostream>
#include "TransferBench.hpp"
#include "EnvVars.hpp"

size_t const DEFAULT_BYTES_PER_TRANSFER = (1<<26);
char const ExeTypeName[4][4] = {"CPU", "GPU", "DMA", "IBV"};

void DisplayTopology(bool outputToCsv);
void DisplayUsage(char const* cmdName);
void PrintResults(EnvVars const& ev, int const testNum,
                  std::vector<Transfer> const& transfers,
                  TransferBench::TestResults const& results);
std::string MemDevicesToStr(std::vector<MemDevice> const& memDevices);
void CheckForError(ErrResult const& error);
void PrintErrors(std::vector<ErrResult> const& errors);

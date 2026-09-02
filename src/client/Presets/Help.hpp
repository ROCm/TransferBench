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

int HelpPreset([[maybe_unused]] EnvVars&          ev,
               [[maybe_unused]] size_t      const numBytesPerTransfer,
               [[maybe_unused]] std::string const presetName,
               [[maybe_unused]] bool        const bytesSpecified)
{
  if (!Utils::RankDoesOutput()) return 0;

  printf("# ConfigFile Format:\n");
  printf("# ==================\n");
  printf("# A Transfer is defined as a single operation where an Executor reads and adds together\n");
  printf("# values from Source (SRC) memory locations, then writes the sum to destination (DST) memory locations.\n");
  printf("# This simplifies to a simple copy operation when dealing with single SRC/DST.\n");
  printf("#\n");
  printf("#                SRC 0                DST 0\n");
  printf("#                SRC 1 -> Executor -> DST 1\n");
  printf("#                SRC X                DST Y\n");
  printf("\n");
  printf("# Six Executors are supported by TransferBench\n");
  printf("#   Executor:              SubExecutor:\n");
  printf("#   1) CPU                 CPU thread\n");
  printf("#   2) GPU                 GPU threadblock/Compute Unit (CU)\n");
  printf("#   3) DMA                 N/A.                                 (Must have single SRC, at least one DST)\n");
  printf("#   4) NIC                 Queue Pair\n");
  printf("#   5) Batched-DMA         Batch item                           (Must have single SRC, at least one DST)\n");
  printf("#   6) TDM                 GPU threadblock/Compute Unit (CU)    (Requires hardware support: AMD gfx1250 or NVIDIA sm_90+)\n");

  printf("\n");
  printf("# Each single line in the configuration file defines a set of Transfers (a Test) to run in parallel\n");
  printf("\n");
  printf("# There are two ways to specify a Test:\n");
  printf("\n");
  printf("# 1) Basic\n");
  printf("#    The basic specification assumes the same number of SubExecutors (SE) used per Transfer\n");
  printf("#    A positive number of Transfers is specified followed by that number of triplets describing each Transfer\n");
  printf("\n");
  printf("#    #Transfers #SEs (srcMem1->Executor1->dstMem1) ... (srcMemL->ExecutorL->dstMemL)\n");
  printf("\n");
  printf("# 2) Advanced\n");
  printf("#    A negative number of Transfers is specified, followed by quintuplets describing each Transfer\n");
  printf("#    A non-zero number of bytes specified will override any provided value\n");
  printf("#    -#Transfers (srcMem1->Executor1->dstMem1 #SEs1 Bytes1) ... (srcMemL->ExecutorL->dstMemL #SEsL BytesL)\n");
  printf("\n");
  printf("# Argument Details:\n");
  printf("#   #Transfers:   Number of Transfers to be run in parallel\n");
  printf("#   #SEs      :   Number of SubExecutors to use (CPU threads/ GPU threadblocks)\n");
  printf("#   srcMemL   :   Source memory locations (Where the data is to be read from)\n");
  printf("#   Executor  :   Executor is specified by a character indicating type, followed by device index (0-indexed)\n");
  printf("#                 - C:    CPU-executed          (Indexed from 0 to # NUMA nodes - 1)\n");
  printf("#                 - G:    GPU-executed          (Indexed from 0 to # GPUs - 1)\n");
  printf("#                 - D:    DMA-executor          (Indexed from 0 to # GPUs - 1)\n");
  printf("#                 - B:    Batched-DMA-executor  (Indexed from 0 to # GPUs - 1)\n");
  printf("#                 - I#.#: NIC executor          (Indexed from 0 to # NICs - 1)\n");
  printf("#                 - N#.#: Nearest NIC executor  (Indexed from 0 to # GPUs - 1)\n");
  printf("#                 - T:    TDM-executor          (Indexed from 0 to # GPUs - 1)\n");
  printf("#   dstMemL   :   Destination memory locations (Where the data is to be written to)\n");
  printf("#   bytesL    :   Number of bytes to copy (0 means use command-line specified size)\n");
  printf("#                 Must be a multiple of 4 and may be suffixed with ('K','M', or 'G')\n");
  printf("#\n");
  printf("#                 Memory locations are specified by one or more (device character / device index) pairs\n");
  printf("#                 Character indicating memory type followed by device index (0-indexed)\n");
  printf("#                 Supported memory locations are:\n");
  printf("#                 - C:    Pinned host memory              (on NUMA node, indexed from 0 to [# NUMA nodes-1])\n");
  printf("#                 - P:    Pinned host memory              (on NUMA node, indexed by closest GPU [# GPUs -1])\n");
  printf("#                 - B:    Coherent pinned host memory     (on NUMA node, indexed from 0 to [# NUMA nodes-1])\n");
  printf("#                 - D:    Non-coherent pinned host memory (on NUMA node, indexed from 0 to [# NUMA nodes-1])\n");
  printf("#                 - K:    Uncached pinned host memory     (on NUMA node, indexed from 0 to [# NUMA nodes-1])\n");
  printf("#                 - H:    Unpinned host memory            (on NUMA node, indexed from 0 to [# NUMA nodes-1])\n");
  printf("#                 - G:    Global device memory            (on GPU device indexed from 0 to [# GPUs - 1])\n");
  printf("#                 - F:    Fine-grain device memory        (on GPU device indexed from 0 to [# GPUs - 1])\n");
  printf("#                 - U:    Uncached device memory          (on GPU device indexed from 0 to [# GPUs - 1])\n");
  printf("#                 - N:    Null memory                     (index ignored)\n");
  printf("\n");
  printf("\n");
  printf("# Examples:\n");
  printf("# 1 4 (G0->G0->G1)                   Uses 4 CUs on GPU0 to copy from GPU0 to GPU1\n");
  printf("# 1 4 (C1->G2->G0)                   Uses 4 CUs on GPU2 to copy from CPU1 to GPU0\n");
  printf("# 2 4 G0->G0->G1 G1->G1->G0          Copies from GPU0 to GPU1, and GPU1 to GPU0, each with 4 SEs\n");
  printf("# -2 (G0 G0 G1 4 1M) (G1 G1 G0 2 2M) Copies 1Mb from GPU0 to GPU1 with 4 SEs, and 2Mb from GPU1 to GPU0 with 2 SEs\n");
  printf("# 1 2 (F0->I0.2->F1)                 Uses 2 QPs to transfer data from GPU0 via NIC0 to GPU1 via NIC2\n");
  printf("# 1 1 (F0->N0.1->F1)                 Uses 1 QP to transfer data from GPU0 via GPU0's closest NIC to GPU1 via GPU1's closest NIC\n");
  printf("# -2 (G0->N0.1->G1 2 128M) (G1->N1.0->G0 1 256M) Uses Nearest NIC executor to copy 128Mb from GPU0 to GPU1 with 2 QPs,\n");
  printf("#                                                and 256Mb from GPU1 to GPU0 with 1 QP\n");
  printf("# Round brackets and arrows' ->' may be included for human clarity, but will be ignored and are unnecessary\n");
  printf("# Lines starting with # will be ignored. Lines starting with ## will be echoed to output\n");
  printf("\n");
  printf("## Single GPU-executed Transfer between GPUs 0 and 1 using 4 CUs\n");
  printf("1 4 (G0->G0->G1)\n");
  printf("\n");
  printf("## Single DMA executed Transfer between GPUs 0 and 1\n");
  printf("1 1 (G0->D0->G1)\n");
  printf("\n");
  printf("## Copy 1Mb from GPU0 to GPU1 with 4 CUs, and 2Mb from GPU1 to GPU0 with 8 CUs\n");
  printf("-2 (G0->G0->G1 4 1M) (G1->G1->G0 8 2M)\n");
  printf("\n");
  printf("## \"Memset\" by GPU 0 to GPU 0 memory\n");
  printf("1 32 (N0->G0->G0)\n");
  printf("\n");
  printf("## \"Read-only\" by CPU 0\n");
  printf("1 4 (C0->C0->N0)\n");
  printf("\n");
  printf("## Broadcast from GPU 0 to GPU 0 and GPU 1\n");
  printf("1 16 (G0->G0->G0G1)\n");
  return 0;
}

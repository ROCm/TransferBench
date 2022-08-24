/*
Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef ENVVARS_HPP
#define ENVVARS_HPP

#include <algorithm>
#include <random>
#include <time.h>
#define TB_VERSION "1.04"

extern char const MemTypeStr[];

enum ConfigModeEnum
{
  CFG_FILE  = 0,
  CFG_P2P   = 1,
  CFG_SWEEP = 2
};

// This class manages environment variable that affect TransferBench
class EnvVars
{
public:
  // Default configuration values
  int const DEFAULT_NUM_WARMUPS          =  1;
  int const DEFAULT_NUM_ITERATIONS       = 10;
  int const DEFAULT_SAMPLING_FACTOR      =  1;
  int const DEFAULT_NUM_CPU_PER_TRANSFER =  4;

  int const DEFAULT_SWEEP_SRC_IS_EXE  = 0;
  std::string const DEFAULT_SWEEP_SRC = "CG";
  std::string const DEFAULT_SWEEP_EXE = "CG";
  std::string const DEFAULT_SWEEP_DST = "CG";
  int const DEFAULT_SWEEP_MIN         = 1;
  int const DEFAULT_SWEEP_MAX         = 24;
  int const DEFAULT_SWEEP_TEST_LIMIT  = 0;
  int const DEFAULT_SWEEP_TIME_LIMIT  = 0;

  // Environment variables
  int blockBytes;        // Each CU, except the last, gets a multiple of this many bytes to copy
  int byteOffset;        // Byte-offset for memory allocations
  int numCpuDevices;     // Number of CPU devices to use (defaults to # NUMA nodes detected)
  int numCpuPerTransfer; // Number of CPU child threads to use per CPU Transfer
  int numGpuDevices;     // Number of GPU devices to use (defaults to # HIP devices detected)
  int numIterations;     // Number of timed iterations to perform.  If negative, run for -numIterations seconds instead
  int numWarmups;        // Number of un-timed warmup iterations to perform
  int outputToCsv;       // Output in CSV format
  int samplingFactor;    // Affects how many different values of N are generated (when N set to 0)
  int sharedMemBytes;    // Amount of shared memory to use per threadblock
  int useHipCall;        // Use hipMemcpy/hipMemset instead of custom shader kernels
  int useInteractive;    // Pause for user-input before starting transfer loop
  int useMemset;         // Perform a memset instead of a copy (ignores source memory)
  int usePcieIndexing;   // Base GPU indexing on PCIe address instead of HIP device
  int useSingleStream;   // Use a single stream per device instead of per Tink. Can not be used with USE_HIP_CALL

  std::vector<float> fillPattern; // Pattern of floats used to fill source data

  // Environment variables only for Sweep-preset
  int sweepSrcIsExe;     // Non-zero if executor should always be the same as source
  int sweepMin;          // Min number of simultaneous Transfers to be executed per test
  int sweepMax;          // Max number of simulatneous Transfers to be executed per test
  int sweepTestLimit;    // Max number of tests to run during sweep (0 = no limit)
  int sweepTimeLimit;    // Max number of seconds to run sweep for  (0 = no limit)
  int sweepXgmiMin;      // Min number of XGMI hops for Transfers
  int sweepXgmiMax;      // Max number of XGMI hops for Transfers (-1 = no limit)
  int sweepSeed;         // Random seed to use
  int sweepRandBytes;    // Whether or not to use random number of bytes per Transfer
  std::string sweepSrc;  // Set of src memory types to be swept
  std::string sweepExe;  // Set of executors to be swept
  std::string sweepDst;  // Set of dst memory types to be swept

  // Used to track current configuration mode
  ConfigModeEnum configMode;

  // Random generator
  std::default_random_engine *generator;

  // Constructor that collects values
  EnvVars()
  {
    int maxSharedMemBytes = 0;
    hipDeviceGetAttribute(&maxSharedMemBytes,
                          hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, 0);

    int numDetectedCpus = numa_num_configured_nodes();
    int numDetectedGpus;
    hipGetDeviceCount(&numGpuDevices);

    blockBytes        = GetEnvVar("BLOCK_BYTES"         , 256);
    byteOffset        = GetEnvVar("BYTE_OFFSET"         , 0);
    numCpuDevices     = GetEnvVar("NUM_CPU_DEVICES"     , numDetectedCpus);
    numCpuPerTransfer = GetEnvVar("NUM_CPU_PER_TRANSFER", DEFAULT_NUM_CPU_PER_TRANSFER);
    numGpuDevices     = GetEnvVar("NUM_GPU_DEVICES"     , numDetectedGpus);
    numIterations     = GetEnvVar("NUM_ITERATIONS"      , DEFAULT_NUM_ITERATIONS);
    numWarmups        = GetEnvVar("NUM_WARMUPS"         , DEFAULT_NUM_WARMUPS);
    outputToCsv       = GetEnvVar("OUTPUT_TO_CSV"       , 0);
    samplingFactor    = GetEnvVar("SAMPLING_FACTOR"     , DEFAULT_SAMPLING_FACTOR);
    sharedMemBytes    = GetEnvVar("SHARED_MEM_BYTES"    , maxSharedMemBytes / 2 + 1);
    useHipCall        = GetEnvVar("USE_HIP_CALL"        , 0);
    useInteractive    = GetEnvVar("USE_INTERACTIVE"     , 0);
    useMemset         = GetEnvVar("USE_MEMSET"          , 0);
    usePcieIndexing   = GetEnvVar("USE_PCIE_INDEX"      , 0);
    useSingleStream   = GetEnvVar("USE_SINGLE_STREAM"   , 0);

    sweepSrcIsExe     = GetEnvVar("SWEEP_SRC_IS_EXE"    , DEFAULT_SWEEP_SRC_IS_EXE);
    sweepMin          = GetEnvVar("SWEEP_MIN"           , DEFAULT_SWEEP_MIN);
    sweepMax          = GetEnvVar("SWEEP_MAX"           , DEFAULT_SWEEP_MAX);
    sweepSrc          = GetEnvVar("SWEEP_SRC"           , DEFAULT_SWEEP_SRC);
    sweepExe          = GetEnvVar("SWEEP_EXE"           , DEFAULT_SWEEP_EXE);
    sweepDst          = GetEnvVar("SWEEP_DST"           , DEFAULT_SWEEP_DST);
    sweepTestLimit    = GetEnvVar("SWEEP_TEST_LIMIT"    , DEFAULT_SWEEP_TEST_LIMIT);
    sweepTimeLimit    = GetEnvVar("SWEEP_TIME_LIMIT"    , DEFAULT_SWEEP_TIME_LIMIT);
    sweepXgmiMin      = GetEnvVar("SWEEP_XGMI_MIN"      , 0);
    sweepXgmiMax      = GetEnvVar("SWEEP_XGMI_MAX"      , -1);
    sweepRandBytes    = GetEnvVar("SWEEP_RAND_BYTES"    , 0);

    // Determine random seed

    char *sweepSeedStr = getenv("SWEEP_SEED");
    sweepSeed = (sweepSeedStr != NULL ? atoi(sweepSeedStr) : time(NULL));
    generator = new std::default_random_engine(sweepSeed);

    // Check for fill pattern
    char* pattern = getenv("FILL_PATTERN");
    if (pattern != NULL)
    {
      int patternLen = strlen(pattern);
      if (patternLen % 2)
      {
        printf("[ERROR] FILL_PATTERN must contain an even-number of hex digits\n");
        exit(1);
      }

      // Read in bytes
      std::vector<unsigned char> bytes;
      unsigned char val = 0;
      for (int i = 0; i < patternLen; i++)
      {
        if ('0' <= pattern[i] && pattern[i] <= '9')
          val += (pattern[i] - '0');
        else if ('A' <= pattern[i] && pattern[i] <= 'F')
          val += (pattern[i] - 'A' + 10);
        else if ('a' <= pattern[i] && pattern[i] <= 'f')
          val += (pattern[i] - 'a' + 10);
        else
        {
          printf("[ERROR] FILL_PATTERN must contain an even-number of hex digits (0-9'/a-f/A-F).  (not %c)\n", pattern[i]);
          exit(1);
        }

        if (i % 2 == 0)
          val <<= 4;
        else
        {
          bytes.push_back(val);
          val = 0;
        }
      }

      // Reverse bytes (input is assumed to be given in big-endian)
      std::reverse(bytes.begin(), bytes.end());

      // Figure out how many copies of the pattern are necessary to fill a 4-byte float properly
      int copies;
      switch (patternLen % 8)
      {
      case 0:  copies = 1; break;
      case 4:  copies = 2; break;
      default: copies = 4; break;
      }

      // Fill floats
      int numFloats = copies * patternLen / 8;
      fillPattern.resize(numFloats);
      unsigned char* rawData = (unsigned char*) fillPattern.data();
      for (int i = 0; i < numFloats * 4; i++)
        rawData[i] = bytes[i % bytes.size()];
    }
    else fillPattern.clear();

    // Perform some basic validation
    if (numCpuDevices > numDetectedCpus)
    {
      printf("[ERROR] Number of CPUs to use (%d) cannot exceed number of detected CPUs (%d)\n", numCpuDevices, numDetectedCpus);
      exit(1);
    }
    if (numGpuDevices > numDetectedGpus)
    {
      printf("[ERROR] Number of GPUs to use (%d) cannot exceed number of detected GPUs (%d)\n", numGpuDevices, numDetectedGpus);
      exit(1);
    }
    if (byteOffset % sizeof(float))
    {
      printf("[ERROR] BYTE_OFFSET must be set to multiple of %lu\n", sizeof(float));
      exit(1);
    }
    if (numWarmups < 0)
    {
      printf("[ERROR] NUM_WARMUPS must be set to a non-negative number\n");
      exit(1);
    }
    if (samplingFactor < 1)
    {
      printf("[ERROR] SAMPLING_FACTOR must be greater or equal to 1\n");
      exit(1);
    }
    if (numCpuPerTransfer < 1)
    {
      printf("[ERROR] NUM_CPU_PER_TRANSFER must be greater or equal to 1\n");
      exit(1);
    }
    if (sharedMemBytes < 0 || sharedMemBytes > maxSharedMemBytes)
    {
      printf("[ERROR] SHARED_MEM_BYTES must be between 0 and %d\n", maxSharedMemBytes);
      exit(1);
    }
    if (blockBytes <= 0 || blockBytes % 4)
    {
      printf("[ERROR] BLOCK_BYTES must be a positive multiple of 4\n");
      exit(1);
    }
    if (useSingleStream && useHipCall)
    {
      printf("[ERROR] Single stream mode cannot be used with HIP calls\n");
      exit(1);
    }

    for (auto ch : sweepSrc)
    {
      if (!strchr(MemTypeStr, ch))
      {
        printf("[ERROR] Unrecognized memory type '%c' specified for sweep source\n", ch);
        exit(1);
      }
      if (strchr(sweepSrc.c_str(), ch) != strrchr(sweepSrc.c_str(), ch))
      {
        printf("[ERROR] Duplicate memory type '%c' specified for sweep source\n", ch);
        exit(1);
      }
    }

    for (auto ch : sweepDst)
    {
      if (!strchr(MemTypeStr, ch))
      {
        printf("[ERROR] Unrecognized memory type '%c' specified for sweep destination\n", ch);
        exit(1);
      }
      if (strchr(sweepDst.c_str(), ch) != strrchr(sweepDst.c_str(), ch))
      {
        printf("[ERROR] Duplicate memory type '%c' specified for sweep destination\n", ch);
        exit(1);
      }
    }

    char const* permittedExecutors = "CG";
    for (auto ch : sweepExe)
    {
      if (!strchr(permittedExecutors, ch))
      {
        printf("[ERROR] Unrecognized executor type '%c' specified for sweep executor\n", ch);
        exit(1);
      }
      if (strchr(sweepExe.c_str(), ch) != strrchr(sweepExe.c_str(), ch))
      {
        printf("[ERROR] Duplicate executor type '%c' specified for sweep executor\n", ch);
        exit(1);
      }
    }
  }

  // Display info on the env vars that can be used
  static void DisplayUsage()
  {
    printf("Environment variables:\n");
    printf("======================\n");
    printf(" BLOCK_BYTES=B          - Each CU (except the last) receives a multiple of BLOCK_BYTES to copy\n");
    printf(" BYTE_OFFSET            - Initial byte-offset for memory allocations.  Must be multiple of 4. Defaults to 0\n");
    printf(" FILL_PATTERN=STR       - Fill input buffer with pattern specified in hex digits (0-9,a-f,A-F).  Must be even number of digits, (byte-level big-endian)\n");
    printf(" NUM_CPU_DEVICES=X      - Restrict number of CPUs to X.  May not be greater than # detected NUMA nodes\n");
    printf(" NUM_CPU_PER_TRANSFER=C - Use C threads per Transfer for CPU-executed copies\n");
    printf(" NUM_GPU_DEVICES=X      - Restrict number of GCPUs to X.  May not be greater than # detected HIP devices\n");
    printf(" NUM_ITERATIONS=I       - Perform I timed iteration(s) per test\n");
    printf(" NUM_WARMUPS=W          - Perform W untimed warmup iteration(s) per test\n");
    printf(" OUTPUT_TO_CSV          - Outputs to CSV format if set\n");
    printf(" SAMPLING_FACTOR=F      - Add F samples (when possible) between powers of 2 when auto-generating data sizes\n");
    printf(" SHARED_MEM_BYTES=X     - Use X shared mem bytes per threadblock, potentially to avoid multiple threadblocks per CU\n");
    printf(" USE_HIP_CALL           - Use hipMemcpy/hipMemset instead of custom shader kernels for GPU-executed copies\n");
    printf(" USE_INTERACTIVE        - Pause for user-input before starting transfer loop\n");
    printf(" USE_MEMSET             - Perform a memset instead of a copy (ignores source memory)\n");
    printf(" USE_PCIE_INDEX         - Index GPUs by PCIe address-ordering instead of HIP-provided indexing\n");
    printf(" USE_SINGLE_STREAM      - Use single stream per device instead of per Transfer.  Cannot be used with USE_HIP_CALL\n");
  }

  // Display env var settings
  void DisplayEnvVars() const
  {
    if (!outputToCsv)
    {
      printf("Run configuration (TransferBench v%s)\n", TB_VERSION);
      printf("=====================================================\n");
      printf("%-20s = %12d : Each CU gets a multiple of %d bytes to copy\n", "BLOCK_BYTES", blockBytes, blockBytes);
      printf("%-20s = %12d : Using byte offset of %d\n", "BYTE_OFFSET", byteOffset, byteOffset);
      printf("%-20s = %12s : ", "FILL_PATTERN", getenv("FILL_PATTERN") ? "(specified)" : "(unset)");
      if (fillPattern.size())
        printf("Pattern: %s", getenv("FILL_PATTERN"));
      else
        printf("Pseudo-random: (Element i = i modulo 383 + 31)");
      printf("\n");
      printf("%-20s = %12d : Using %d CPU devices\n" , "NUM_CPU_DEVICES", numCpuDevices, numCpuDevices);
      printf("%-20s = %12d : Using %d CPU thread(s) per CPU-executed Transfer\n", "NUM_CPU_PER_TRANSFER", numCpuPerTransfer, numCpuPerTransfer);
      printf("%-20s = %12d : Using %d GPU devices\n", "NUM_GPU_DEVICES", numGpuDevices, numGpuDevices);
      printf("%-20s = %12d : Running %d %s per Test\n", "NUM_ITERATIONS", numIterations,
             numIterations > 0 ? numIterations : -numIterations,
             numIterations > 0 ? "timed iteration(s)" : "second(s)");
      printf("%-20s = %12d : Running %d warmup iteration(s) per Test\n", "NUM_WARMUPS", numWarmups, numWarmups);
      printf("%-20s = %12d : Output to %s\n", "OUTPUT_TO_CSV", outputToCsv,
             outputToCsv ? "CSV" : "console");
      printf("%-20s = %12s : Using %d shared mem per threadblock\n", "SHARED_MEM_BYTES",
             getenv("SHARED_MEM_BYTES") ? "(specified)" : "(unset)", sharedMemBytes);
      printf("%-20s = %12d : Using %s for GPU-executed copies\n", "USE_HIP_CALL", useHipCall,
             useHipCall ? "HIP functions" : "custom kernels");
      if (useHipCall && !useMemset)
      {
        char* env = getenv("HSA_ENABLE_SDMA");
        printf("%-20s = %12s : %s\n", "HSA_ENABLE_SDMA", env,
               (env && !strcmp(env, "0")) ? "Using blit kernels for hipMemcpy" : "Using DMA copy engines");
      }
      printf("%-20s = %12d : Running in %s mode\n", "USE_INTERACTIVE", useInteractive,
             useInteractive ? "interactive" : "non-interactive");
      printf("%-20s = %12d : Performing %s\n", "USE_MEMSET", useMemset,
             useMemset ? "memset" : "memcopy");
      printf("%-20s = %12d : Using %s-based GPU indexing\n", "USE_PCIE_INDEX",
             usePcieIndexing, (usePcieIndexing ? "PCIe" : "HIP"));
      printf("%-20s = %12d : Using single stream per %s\n", "USE_SINGLE_STREAM",
             useSingleStream, (useSingleStream ? "device" : "Transfer"));
      printf("\n");
    }
    else
    {
      printf("EnvVar,Value,Description,(TransferBench v%s)\n", TB_VERSION);
      printf("BLOCK_BYTES,%d,Each CU gets a multiple of %d bytes to copy\n", blockBytes, blockBytes);
      printf("BYTE_OFFSET,%d,Using byte offset of %d\n", byteOffset, byteOffset);
      printf("FILL_PATTERN,%s,", getenv("FILL_PATTERN") ? "(specified)" : "(unset)");
      if (fillPattern.size())
        printf("Pattern: %s", getenv("FILL_PATTERN"));
      else
        printf("Pseudo-random: (Element i = i modulo 383 + 31)");
      printf("\n");
      printf("NUM_CPU_DEVICES,%d,Using %d CPU devices\n" , numCpuDevices, numCpuDevices);
      printf("NUM_CPU_PER_TRANSFER,%d,Using %d CPU thread(s) per CPU-executed Transfer\n", numCpuPerTransfer, numCpuPerTransfer);
      printf("NUM_GPU_DEVICES,%d,Using %d GPU devices\n", numGpuDevices, numGpuDevices);
      printf("NUM_ITERATIONS,%d,Running %d %s per Test\n", numIterations,
             numIterations > 0 ? numIterations : -numIterations,
             numIterations > 0 ? "timed iteration(s)" : "second(s)");
      printf("NUM_WARMUPS,%d,Running %d warmup iteration(s) per Test\n", numWarmups, numWarmups);
      printf("SHARED_MEM_BYTES,%d,Using %d shared mem per threadblock\n", sharedMemBytes, sharedMemBytes);
      printf("USE_HIP_CALL,%d,Using %s for GPU-executed copies\n", useHipCall, useHipCall ? "HIP functions" : "custom kernels");
      printf("USE_MEMSET,%d,Performing %s\n", useMemset, useMemset ? "memset" : "memcopy");
      printf("USE_PCIE_INDEX,%d,Using %s-based GPU indexing\n", usePcieIndexing, (usePcieIndexing ? "PCIe" : "HIP"));
      printf("USE_SINGLE_STREAM,%d,Using single stream per %s\n", useSingleStream, (useSingleStream ? "device" : "Transfer"));
    }
  };

  // Display env var settings
  void DisplaySweepEnvVars() const
  {
    if (!outputToCsv)
    {
      printf("Sweep configuration (TransferBench v%s)\n", TB_VERSION);
      printf("=====================================================\n");
      printf("%-20s = %12d : Random seed\n", "SWEEP_SEED", sweepSeed);
      printf("%-20s = %12s : Source Memory Types to sweep\n", "SWEEP_SRC", sweepSrc.c_str());
      printf("%-20s = %12s : Executor Types to sweep\n", "SWEEP_EXE", sweepExe.c_str());
      printf("%-20s = %12s : Destination Memory Types to sweep\n", "SWEEP_DST", sweepDst.c_str());
      printf("%-20s = %12d : Transfer executor %s Transfer source\n", "SWEEP_SRC_IS_EXE", sweepSrcIsExe, sweepSrcIsExe ? "must match" : "may have any");
      printf("%-20s = %12d : Min simultaneous Transfers\n", "SWEEP_MIN", sweepMin);
      printf("%-20s = %12d : Max simultaneous Transfers              (0 = no limit)\n", "SWEEP_MAX", sweepMax);
      printf("%-20s = %12d : Max number of tests to run during sweep (0 = no limit)\n", "SWEEP_TEST_LIMIT", sweepTestLimit);
      printf("%-20s = %12d : Max number of seconds to run sweep for  (0 = no limit)\n", "SWEEP_TIME_LIMIT", sweepTimeLimit);
      printf("%-20s = %12d : Min number of XGMI hops for Transfers\n", "SWEEP_XGMI_MIN", sweepXgmiMin);
      printf("%-20s = %12d : Max number of XGMI hops for Transfers (-1 = no limit)\n", "SWEEP_XGMI_MAX", sweepXgmiMax);
      printf("%-20s = %12d : Using %s number of bytes per Transfer\n", "SWEEP_RAND_BYTES", sweepRandBytes, sweepRandBytes ? "random" : "constant");
      printf("%-20s = %12d : Using %d CPU devices\n" , "NUM_CPU_DEVICES", numCpuDevices, numCpuDevices);
      printf("%-20s = %12d : Using %d CPU thread(s) per CPU-executed Transfer\n", "NUM_CPU_PER_TRANSFER", numCpuPerTransfer, numCpuPerTransfer);
      printf("%-20s = %12d : Using %d GPU devices\n", "NUM_GPU_DEVICES", numGpuDevices, numGpuDevices);
      printf("%-20s = %12d : Each CU gets a multiple of %d bytes to copy\n", "BLOCK_BYTES", blockBytes, blockBytes);
      printf("%-20s = %12d : Using byte offset of %d\n", "BYTE_OFFSET", byteOffset, byteOffset);
      printf("%-20s = %12s : ", "FILL_PATTERN", getenv("FILL_PATTERN") ? "(specified)" : "(unset)");
      if (fillPattern.size())
        printf("Pattern: %s", getenv("FILL_PATTERN"));
      else
        printf("Pseudo-random: (Element i = i modulo 383 + 31)");
      printf("\n");
      printf("%-20s = %12d : Running %d %s per Test\n", "NUM_ITERATIONS", numIterations,
             numIterations > 0 ? numIterations : -numIterations,
             numIterations > 0 ? "timed iteration(s)" : "second(s)");
      printf("%-20s = %12d : Running %d warmup iteration(s) per Test\n", "NUM_WARMUPS", numWarmups, numWarmups);
      printf("%-20s = %12d : Output to %s\n", "OUTPUT_TO_CSV", outputToCsv,
             outputToCsv ? "CSV" : "console");
      printf("%-20s = %12s : Using %d shared mem per threadblock\n", "SHARED_MEM_BYTES",
             getenv("SHARED_MEM_BYTES") ? "(specified)" : "(unset)", sharedMemBytes);
      printf("%-20s = %12d : Using %s for GPU-executed copies\n", "USE_HIP_CALL", useHipCall,
             useHipCall ? "HIP functions" : "custom kernels");
      if (useHipCall && !useMemset)
      {
        char* env = getenv("HSA_ENABLE_SDMA");
        printf("%-20s = %12s : %s\n", "HSA_ENABLE_SDMA", env,
               (env && !strcmp(env, "0")) ? "Using blit kernels for hipMemcpy" : "Using DMA copy engines");
      }
      printf("%-20s = %12d : Using %s-based GPU indexing\n", "USE_PCIE_INDEX",
             usePcieIndexing, (usePcieIndexing ? "PCIe" : "HIP"));
      printf("%-20s = %12d : Using single stream per %s\n", "USE_SINGLE_STREAM",
             useSingleStream, (useSingleStream ? "device" : "Transfer"));
      printf("\n");
    }
    else
    {
      printf("EnvVar,Value,Description,(TransferBench v%s)\n", TB_VERSION);
      printf("SWEEP_SRC,%s,Source Memory Types to sweep\n", sweepSrc.c_str());
      printf("SWEEP_EXE,%s,Executor Types to sweep\n", sweepExe.c_str());
      printf("SWEEP_DST,%s,Destination Memory Types to sweep\n", sweepDst.c_str());
      printf("SWEEP_SRC_IS_EXE,%d, Transfer executor %s Transfer source\n", sweepSrcIsExe, sweepSrcIsExe ? "must match" : "may have any");
      printf("SWEEP_SEED,%d,Random seed\n", sweepSeed);
      printf("SWEEP_MIN,%d,Min simultaneous Transfers\n", sweepMin);
      printf("SWEEP_MAX,%d,Max simultaneous Transfers (0 = no limit)\n", sweepMax);
      printf("SWEEP_TEST_LIMIT,%d,Max number of tests to run during sweep (0 = no limit)\n", sweepTestLimit);
      printf("SWEEP_TIME_LIMIT,%d,Max number of seconds to run sweep for (0 = no limit)\n", sweepTimeLimit);
      printf("SWEEP_XGMI_MIN,%d,Min number of XGMI hops for Transfers\n", sweepXgmiMin);
      printf("SWEEP_XGMI_MAX,%d,Max number of XGMI hops for Transfers (-1 = no limit)\n", sweepXgmiMax);
      printf("SWEEP_RAND_BYTES,%d,Using %s number of bytes per Transfer\n", sweepRandBytes, sweepRandBytes ? "random" : "constant");
      printf("NUM_CPU_DEVICES,%d,Using %d CPU devices\n" , numCpuDevices, numCpuDevices);
      printf("NUM_CPU_PER_TRANSFER,%d,Using %d CPU thread(s) per CPU-executed Transfer\n", numCpuPerTransfer, numCpuPerTransfer);
      printf("NUM_GPU_DEVICES,%d,Using %d GPU devices\n", numGpuDevices, numGpuDevices);
      printf("BLOCK_BYTES,%d,Each CU gets a multiple of %d bytes to copy\n", blockBytes, blockBytes);
      printf("BYTE_OFFSET,%d,Using byte offset of %d\n", byteOffset, byteOffset);
      printf("FILL_PATTERN,%s,", getenv("FILL_PATTERN") ? "(specified)" : "(unset)");
      if (fillPattern.size())
        printf("Pattern: %s", getenv("FILL_PATTERN"));
      else
        printf("Pseudo-random: (Element i = i modulo 383 + 31)");
      printf("\n");
      printf("NUM_ITERATIONS,%d,Running %d %s per Test\n", numIterations,
             numIterations > 0 ? numIterations : -numIterations,
             numIterations > 0 ? "timed iteration(s)" : "second(s)");
      printf("NUM_WARMUPS,%d,Running %d warmup iteration(s) per Test\n", numWarmups, numWarmups);
      printf("SHARED_MEM_BYTES,%d,Using %d shared mem per threadblock\n", sharedMemBytes, sharedMemBytes);
      printf("USE_HIP_CALL,%d,Using %s for GPU-executed copies\n", useHipCall, useHipCall ? "HIP functions" : "custom kernels");
      printf("USE_PCIE_INDEX,%d,Using %s-based GPU indexing\n", usePcieIndexing, (usePcieIndexing ? "PCIe" : "HIP"));
      printf("USE_SINGLE_STREAM,%d,Using single stream per %s\n", useSingleStream, (useSingleStream ? "device" : "Transfer"));
    }
  };

  // Helper function that gets parses environment variable or sets to default value
  static int GetEnvVar(std::string const& varname, int defaultValue)
  {
    if (getenv(varname.c_str()))
      return atoi(getenv(varname.c_str()));
    return defaultValue;
  }

  static std::string GetEnvVar(std::string const& varname, std::string const& defaultValue)
  {
    if (getenv(varname.c_str()))
      return getenv(varname.c_str());
    return defaultValue;
  }
};

#endif

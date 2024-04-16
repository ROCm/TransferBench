/*
Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.

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
#include "Compatibility.hpp"
#include "Kernels.hpp"

#define TB_VERSION "1.50"

extern char const MemTypeStr[];
extern char const ExeTypeStr[];

enum ConfigModeEnum
{
  CFG_FILE   = 0,
  CFG_P2P    = 1,
  CFG_SWEEP  = 2,
  CFG_SCALE  = 3,
  CFG_A2A    = 4,
  CFG_SCHMOO = 5,
  CFG_RWRITE = 6
};

enum BlockOrderEnum
{
  ORDER_SEQUENTIAL  = 0,
  ORDER_INTERLEAVED = 1,
  ORDER_RANDOM      = 2
};

// This class manages environment variable that affect TransferBench
class EnvVars
{
public:
  // Default configuration values
  int const DEFAULT_NUM_WARMUPS       =  3;
  int const DEFAULT_NUM_ITERATIONS    = 10;
  int const DEFAULT_SAMPLING_FACTOR   =  1;

  // Peer-to-peer Benchmark preset defaults
  int const DEFAULT_P2P_NUM_CPU_SE    = 4;

  // Sweep-preset defaults
  std::string const DEFAULT_SWEEP_SRC = "CG";
  std::string const DEFAULT_SWEEP_EXE = "CDG";
  std::string const DEFAULT_SWEEP_DST = "CG";
  int const DEFAULT_SWEEP_MIN         = 1;
  int const DEFAULT_SWEEP_MAX         = 24;
  int const DEFAULT_SWEEP_TEST_LIMIT  = 0;
  int const DEFAULT_SWEEP_TIME_LIMIT  = 0;

  // Environment variables
  int alwaysValidate;    // Validate after each iteration instead of once after all iterations
  int blockBytes;        // Each subexecutor, except the last, gets a multiple of this many bytes to copy
  int blockOrder;        // How blocks are ordered in single-stream mode (0=Sequential, 1=Interleaved, 2=Random)
  int byteOffset;        // Byte-offset for memory allocations
  int continueOnError;   // Continue tests even after mismatch detected
  int gfxBlockSize;      // Size of each threadblock (must be multiple of 64)
  int gfxSingleTeam;     // Team all subExecutors across the data array
  int gfxUnroll;         // GFX-kernel unroll factor
  int gfxWaveOrder;      // GFX-kernel wavefront ordering
  int hideEnv;           // Skip printing environment variable
  int numCpuDevices;     // Number of CPU devices to use (defaults to # NUMA nodes detected)
  int numGpuDevices;     // Number of GPU devices to use (defaults to # HIP devices detected)
  int numIterations;     // Number of timed iterations to perform.  If negative, run for -numIterations seconds instead
  int numWarmups;        // Number of un-timed warmup iterations to perform
  int outputToCsv;       // Output in CSV format
  int samplingFactor;    // Affects how many different values of N are generated (when N set to 0)
  int sharedMemBytes;    // Amount of shared memory to use per threadblock
  int showIterations;    // Show per-iteration timing info
  int useInteractive;    // Pause for user-input before starting transfer loop
  int usePcieIndexing;   // Base GPU indexing on PCIe address instead of HIP device
  int usePrepSrcKernel;  // Use GPU kernel to prepare source data instead of copy (can't be used with fillPattern)
  int useSingleStream;   // Use a single stream per GPU GFX executor instead of stream per Transfer
  int useXccFilter;      // Use XCC filtering (experimental)
  int validateDirect;    // Validate GPU destination memory directly instead of staging GPU memory on host

  std::vector<float> fillPattern; // Pattern of floats used to fill source data
  std::vector<uint32_t> cuMask;   // Bit-vector representing the CU mask
  std::vector<std::vector<int>> prefXccTable;

  // Environment variables only for P2P preset
  int numCpuSubExecs;    // Number of CPU subexecttors to use
  int numGpuSubExecs;    // Number of GPU subexecutors to use
  int p2pMode;           // Both = 0, Unidirectional = 1, Bidirectional = 2
  int useDmaCopy;        // Use DMA copy instead of GPU copy
  int useRemoteRead;     // Use destination memory type as executor instead of source memory type
  int useFineGrain;      // Use fine-grained memory

  // Environment variables only for Sweep-preset
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

  // Enviroment variables only for A2A preset
  int a2aDirect;         // Only execute on links that are directly connected
  int a2aMode;           // Perform 0=copy, 1=read only, 2 = write only

  // Developer features
  int enableDebug;       // Enable debug output
  int gpuMaxHwQueues;    // Tracks GPU_MAX_HW_QUEUES environment variable

  // Used to track current configuration mode
  ConfigModeEnum configMode;

  // Random generator
  std::default_random_engine *generator;

  // Track how many CPUs are available per NUMA node
  std::vector<int> numCpusPerNuma;

  std::vector<int> wallClockPerDeviceMhz;

  std::vector<std::set<int>> xccIdsPerDevice;

  // Constructor that collects values
  EnvVars()
  {
    int maxSharedMemBytes = 0;
    HIP_CALL(hipDeviceGetAttribute(&maxSharedMemBytes,
                                   hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, 0));
#if !defined(__NVCC__)
    int defaultSharedMemBytes = maxSharedMemBytes / 2 + 1;
#else
    int defaultSharedMemBytes = 0;
#endif

    int numDeviceCUs = 0;
    HIP_CALL(hipDeviceGetAttribute(&numDeviceCUs, hipDeviceAttributeMultiprocessorCount, 0));

    int numDetectedCpus = numa_num_configured_nodes();
    int numDetectedGpus;
    HIP_CALL(hipGetDeviceCount(&numDetectedGpus));

    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, 0));
    std::string fullName = prop.gcnArchName;
    std::string archName = fullName.substr(0, fullName.find(':'));

    // Different hardware pick different GPU kernels
    // This performance difference is generally only noticable when executing fewer CUs
    int defaultGfxUnroll = 4;
    if      (archName == "gfx906") defaultGfxUnroll = 8;
    else if (archName == "gfx90a") defaultGfxUnroll = 8;
    else if (archName == "gfx940") defaultGfxUnroll = 6;
    else if (archName == "gfx941") defaultGfxUnroll = 6;
    else if (archName == "gfx942") defaultGfxUnroll = 4;

    alwaysValidate    = GetEnvVar("ALWAYS_VALIDATE"     , 0);
    blockBytes        = GetEnvVar("BLOCK_BYTES"         , 256);
    blockOrder        = GetEnvVar("BLOCK_ORDER"         , 0);
    byteOffset        = GetEnvVar("BYTE_OFFSET"         , 0);
    continueOnError   = GetEnvVar("CONTINUE_ON_ERROR"   , 0);
    gfxBlockSize      = GetEnvVar("GFX_BLOCK_SIZE"      , 256);
    gfxSingleTeam     = GetEnvVar("GFX_SINGLE_TEAM"     , 1);
    gfxUnroll         = GetEnvVar("GFX_UNROLL"          , defaultGfxUnroll);
    gfxWaveOrder      = GetEnvVar("GFX_WAVE_ORDER"      , 0);
    hideEnv           = GetEnvVar("HIDE_ENV"            , 0);
    numCpuDevices     = GetEnvVar("NUM_CPU_DEVICES"     , numDetectedCpus);
    numGpuDevices     = GetEnvVar("NUM_GPU_DEVICES"     , numDetectedGpus);
    numIterations     = GetEnvVar("NUM_ITERATIONS"      , DEFAULT_NUM_ITERATIONS);
    numWarmups        = GetEnvVar("NUM_WARMUPS"         , DEFAULT_NUM_WARMUPS);
    outputToCsv       = GetEnvVar("OUTPUT_TO_CSV"       , 0);
    samplingFactor    = GetEnvVar("SAMPLING_FACTOR"     , DEFAULT_SAMPLING_FACTOR);
    sharedMemBytes    = GetEnvVar("SHARED_MEM_BYTES"    , defaultSharedMemBytes);
    showIterations    = GetEnvVar("SHOW_ITERATIONS"     , 0);
    useInteractive    = GetEnvVar("USE_INTERACTIVE"     , 0);
    usePcieIndexing   = GetEnvVar("USE_PCIE_INDEX"      , 0);
    usePrepSrcKernel  = GetEnvVar("USE_PREP_KERNEL"     , 0);
    useSingleStream   = GetEnvVar("USE_SINGLE_STREAM"   , 1);
    useXccFilter      = GetEnvVar("USE_XCC_FILTER"      , 0);
    validateDirect    = GetEnvVar("VALIDATE_DIRECT"     , 0);
    enableDebug       = GetEnvVar("DEBUG"               , 0);
    gpuMaxHwQueues    = GetEnvVar("GPU_MAX_HW_QUEUES"   , 4);

    // P2P Benchmark related
    useDmaCopy        = GetEnvVar("USE_GPU_DMA"         , 0); // Needed for numGpuSubExec

    numCpuSubExecs    = GetEnvVar("NUM_CPU_SE"          , DEFAULT_P2P_NUM_CPU_SE);
    numGpuSubExecs    = GetEnvVar("NUM_GPU_SE"          , useDmaCopy ? 1 : numDeviceCUs);
    p2pMode           = GetEnvVar("P2P_MODE"            , 0);
    useRemoteRead     = GetEnvVar("USE_REMOTE_READ"     , 0);
    useFineGrain      = GetEnvVar("USE_FINE_GRAIN"      , 0);

    // Sweep related
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

    // A2A Benchmark related
    a2aDirect         = GetEnvVar("A2A_DIRECT"          , 1);
    a2aMode           = GetEnvVar("A2A_MODE"            , 0);

    // Determine random seed
    char *sweepSeedStr = getenv("SWEEP_SEED");
    sweepSeed = (sweepSeedStr != NULL ? atoi(sweepSeedStr) : time(NULL));
    generator = new std::default_random_engine(sweepSeed);

    // Check for fill pattern
    char* pattern = getenv("FILL_PATTERN");
    if (pattern != NULL)
    {
      if (usePrepSrcKernel)
      {
        printf("[ERROR] Unable to use FILL_PATTERN and USE_PREP_KERNEL together\n");
        exit(1);
      }

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

    // Check for CU mask
    cuMask.clear();
    char* cuMaskStr = getenv("CU_MASK");
    if (cuMaskStr != NULL)
    {
#if defined(__NVCC__)
      printf("[WARN] CU_MASK is not supported in CUDA\n");
#else
      std::vector<std::pair<int, int>> ranges;
      int maxCU = 0;
      char* token = strtok(cuMaskStr, ",");
      while (token)
      {
        int start, end;
        if (sscanf(token, "%d-%d", &start, &end) == 2)
        {
          ranges.push_back(std::make_pair(std::min(start, end), std::max(start, end)));
          maxCU = std::max(maxCU, std::max(start, end));
        }
        else if (sscanf(token, "%d", &start) == 1)
        {
          ranges.push_back(std::make_pair(start, start));
          maxCU = std::max(maxCU, start);
        }
        else
        {
          printf("[ERROR] Unrecognized token [%s]\n", token);
          exit(1);
        }
        token = strtok(NULL, ",");
      }
      cuMask.resize(maxCU / 32 + 1, 0);

      for (auto range : ranges)
      {
        for (int i = range.first; i <= range.second; i++)
        {
          cuMask[i / 32] |= (1 << (i % 32));
        }
      }
#endif
    }

    // Figure out number of xccs per device
    int maxNumXccs = 64;
    xccIdsPerDevice.resize(numGpuDevices);
    for (int i = 0; i < numGpuDevices; i++)
    {
      int* data;
      HIP_CALL(hipSetDevice(i));
      HIP_CALL(hipHostMalloc((void**)&data, maxNumXccs * sizeof(int)));
      CollectXccIdsKernel<<<maxNumXccs, 1>>>(data);
      HIP_CALL(hipDeviceSynchronize());

      xccIdsPerDevice[i].clear();
      for (int j = 0; j < maxNumXccs; j++)
        xccIdsPerDevice[i].insert(data[j]);

      HIP_CALL(hipHostFree(data));
    }

    // Parse preferred XCC table (if provided
    prefXccTable.resize(numGpuDevices);
    for (int i = 0; i < numGpuDevices; i++)
    {
      prefXccTable[i].resize(numGpuDevices, -1);
    }

    char* prefXccStr = getenv("XCC_PREF_TABLE");
    if (prefXccStr)
    {
      char* token = strtok(prefXccStr, ",");
      int tokenCount = 0;
      while (token)
      {
        int xccId;
        if (sscanf(token, "%d", &xccId) == 1)
        {
          int src = tokenCount / numGpuDevices;
          int dst = tokenCount % numGpuDevices;
          if (xccIdsPerDevice[src].count(xccId) == 0)
          {
            printf("[ERROR] GPU %d does not contain XCC %d\n", src, xccId);
            exit(1);
          }
          prefXccTable[src][dst] = xccId;

          tokenCount++;
          if (tokenCount == (numGpuDevices * numGpuDevices)) break;
        }
        else
        {
          printf("[ERROR] Unrecognized token [%s]\n", token);
          exit(1);
        }
        token = strtok(NULL, ",");
      }
    }

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
    if (gfxBlockSize % 64)
    {
      printf("[ERROR] GFX_BLOCK_SIZE (%d) must be a multiple of 64\n", gfxBlockSize);
      exit(1);
    }
    if (gfxBlockSize > MAX_BLOCKSIZE)
    {
      printf("[ERROR] BLOCK_SIZE (%d) must be less than %d\n", gfxBlockSize, MAX_BLOCKSIZE);
      exit(1);
    }
    if (byteOffset % sizeof(float))
    {
      printf("[ERROR] BYTE_OFFSET must be set to multiple of %lu\n", sizeof(float));
      exit(1);
    }
    if (blockOrder < 0 || blockOrder > 2)
    {
      printf("[ERROR] BLOCK_ORDER must be 0 (Sequential), 1 (Interleaved), or 2 (Random)\n");
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
    if (numGpuSubExecs <= 0)
    {
      printf("[ERROR] NUM_GPU_SE must be greater than 0\n");
      exit(1);
    }

    if (numCpuSubExecs <= 0)
    {
      printf("[ERROR] NUM_CPU_SE must be greater than 0\n");
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

    for (auto ch : sweepExe)
    {
      if (!strchr(ExeTypeStr, ch))
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

    if (a2aMode < 0 || a2aMode > 2)
    {
      printf("[ERROR] a2aMode must be between 0 and 2\n");
      exit(1);
    }

    if (gfxUnroll < 1 || gfxUnroll > MAX_UNROLL)
    {
      printf("[ERROR] GFX kernel unroll factor must be between 1 and %d (Not %d)\n", MAX_UNROLL, gfxUnroll);
      exit(1);
    }

    if (gfxWaveOrder < 0 || gfxWaveOrder >= 6)
    {
      printf("[ERROR] GFX wave order must be between 0 and 5\n");
      exit(1);
    }

    // Determine how many CPUs exit per NUMA node (to avoid executing on NUMA without CPUs)
    numCpusPerNuma.resize(numDetectedCpus);
    int const totalCpus = numa_num_configured_cpus();
    for (int i = 0; i < totalCpus; i++)
      numCpusPerNuma[numa_node_of_cpu(i)]++;

    // Build array of wall clock rates per GPU device
    wallClockPerDeviceMhz.resize(numDetectedGpus);
    for (int i = 0; i < numDetectedGpus; i++)
    {
#if defined(__NVCC__)
      wallClockPerDeviceMhz[i] = 1000000;
#else
      hipDeviceProp_t prop;
      HIP_CALL(hipGetDeviceProperties(&prop, i));
      int value = 25000;
      std::string fullName = prop.gcnArchName;
      std::string archName = fullName.substr(0, fullName.find(':'));
      if (archName == "gfx940" || archName == "gfx941" || archName == "gfx942")
        wallClockPerDeviceMhz[i] = 100000;
      else
        wallClockPerDeviceMhz[i] = 25000;
#endif
    }

    // Check for deprecated env vars
    if (getenv("USE_HIP_CALL"))
    {
      printf("[WARN] USE_HIP_CALL has been deprecated.  Please use DMA executor 'D' or set USE_GPU_DMA for P2P-Benchmark preset\n");
      exit(1);
    }

    if (getenv("GPU_KERNEL"))
    {
      printf("[WARN] GPU_KERNEL has been deprecated and replaced by GFX_KERNEL and GFX_UNROLL\n");
      exit(1);
    }

    char* enableSdma = getenv("HSA_ENABLE_SDMA");
    if (enableSdma && !strcmp(enableSdma, "0"))
    {
      printf("[WARN] DMA functionality disabled due to environment variable HSA_ENABLE_SDMA=0.  Copies will fallback to blit kernels\n");
    }
  }

  // Display info on the env vars that can be used
  static void DisplayUsage()
  {
    printf("Environment variables:\n");
    printf("======================\n");
    printf(" ALWAYS_VALIDATE        - Validate after each iteration instead of once after all iterations\n");
    printf(" BLOCK_SIZE             - # of threads per threadblock (Must be multiple of 64). Defaults to 256\n");
    printf(" BLOCK_BYTES            - Each CU (except the last) receives a multiple of BLOCK_BYTES to copy\n");
    printf(" BLOCK_ORDER            - Threadblock ordering in single-stream mode (0=Serial, 1=Interleaved, 2=Random)\n");
    printf(" BYTE_OFFSET            - Initial byte-offset for memory allocations.  Must be multiple of 4. Defaults to 0\n");
    printf(" CONTINUE_ON_ERROR      - Continue tests even after mismatch detected\n");
    printf(" CU_MASK                - CU mask for streams specified in hex digits (0-0,a-f,A-F)\n");
    printf(" FILL_PATTERN=STR       - Fill input buffer with pattern specified in hex digits (0-9,a-f,A-F).  Must be even number of digits, (byte-level big-endian)\n");
    printf(" GFX_UNROLL             - Unroll factor for GFX kernel (0=auto), must be less than %d\n", MAX_UNROLL);
    printf(" GFX_SINGLE_TEAM        - Have subexecutors work together on full array instead of working on individual disjoint subarrays\n");
    printf(" GFX_WAVE_ORDER         - Stride pattern for GFX kernel (0=UWC,1=UCW,2=WUC,3=WCU,4=CUW,5=CWU)\n");
    printf(" HIDE_ENV               - Hide environment variable value listing\n");
    printf(" NUM_CPU_DEVICES=X      - Restrict number of CPUs to X.  May not be greater than # detected NUMA nodes\n");
    printf(" NUM_GPU_DEVICES=X      - Restrict number of GPUs to X.  May not be greater than # detected HIP devices\n");
    printf(" NUM_ITERATIONS=I       - Perform I timed iteration(s) per test\n");
    printf(" NUM_WARMUPS=W          - Perform W untimed warmup iteration(s) per test\n");
    printf(" OUTPUT_TO_CSV          - Outputs to CSV format if set\n");
    printf(" SAMPLING_FACTOR=F      - Add F samples (when possible) between powers of 2 when auto-generating data sizes\n");
    printf(" SHARED_MEM_BYTES=X     - Use X shared mem bytes per threadblock, potentially to avoid multiple threadblocks per CU\n");
    printf(" SHOW_ITERATIONS        - Show per-iteration timing info\n");
    printf(" USE_INTERACTIVE        - Pause for user-input before starting transfer loop\n");
    printf(" USE_PCIE_INDEX         - Index GPUs by PCIe address-ordering instead of HIP-provided indexing\n");
    printf(" USE_PREP_KERNEL        - Use GPU kernel to initialize source data array pattern\n");
    printf(" USE_SINGLE_STREAM      - Use a single stream per GPU GFX executor instead of stream per Transfer\n");
    printf(" USE_XCC_FILTER         - Use XCC filtering (experimental)\n");
    printf(" VALIDATE_DIRECT        - Validate GPU destination memory directly instead of staging GPU memory on host\n");
  }

  // Helper macro to switch between CSV and terminal output
#define PRINT_EV(NAME, VALUE, DESCRIPTION)                              \
  printf("%-20s%s%12d%s%s\n", NAME, outputToCsv ? "," : " = ", VALUE, outputToCsv ? "," : " : ",  (DESCRIPTION).c_str())

#define PRINT_ES(NAME, VALUE, DESCRIPTION)                           \
  printf("%-20s%s%12s%s%s\n", NAME, outputToCsv ? "," : " = ", VALUE, outputToCsv ? "," : " : ",  (DESCRIPTION).c_str())

  // Display env var settings
  void DisplayEnvVars() const
  {
    if (!outputToCsv)
    {
      printf("TransferBench v%s\n", TB_VERSION);
      printf("===============================================================\n");
      if (!hideEnv) printf("[Common]                              (Suppress by setting HIDE_ENV=1)\n");
    }
    else if (!hideEnv)
      printf("EnvVar,Value,Description,(TransferBench v%s)\n", TB_VERSION);
    if (hideEnv) return;

    PRINT_EV("ALWAYS_VALIDATE", alwaysValidate,
             std::string("Validating after ") + (alwaysValidate ? "each iteration" : "all iterations"));
    PRINT_EV("BLOCK_BYTES", blockBytes,
             std::string("Each CU gets a multiple of " + std::to_string(blockBytes) + " bytes to copy"));
    PRINT_EV("BLOCK_ORDER", blockOrder,
             std::string("Transfer blocks order: " + std::string((blockOrder == 0 ? "Sequential"  :
                                                                  blockOrder == 1 ? "Interleaved" :
                                                                                    "Random"))));
    PRINT_EV("BYTE_OFFSET", byteOffset,
             std::string("Using byte offset of " + std::to_string(byteOffset)));
    PRINT_EV("CONTINUE_ON_ERROR", continueOnError,
             std::string(continueOnError ? "Continue on mismatch error" : "Stop after first error"));
    PRINT_EV("CU_MASK", getenv("CU_MASK") ? 1 : 0,
             (cuMask.size() ? GetCuMaskDesc() : "All"));
    PRINT_EV("FILL_PATTERN", getenv("FILL_PATTERN") ? 1 : 0,
             (fillPattern.size() ? std::string(getenv("FILL_PATTERN")) : PrepSrcValueString()));
    PRINT_EV("GFX_BLOCK_SIZE", gfxBlockSize,
             std::string("Threadblock size of " + std::to_string(gfxBlockSize)));
    PRINT_EV("GFX_SINGLE_TEAM", gfxSingleTeam,
             (gfxSingleTeam ? std::string("Combining CUs to work across entire data array") :
                              std::string("Each CUs operates on its own disjoint subarray")));
    PRINT_EV("GFX_UNROLL", gfxUnroll,
             std::string("Using GFX unroll factor of ") + std::to_string(gfxUnroll));
    PRINT_EV("GFX_WAVE_ORDER", gfxWaveOrder,
             std::string("Using GFX wave ordering of ") + std::string((gfxWaveOrder == 0 ? "Unroll,Wavefront,CU" :
                                                                       gfxWaveOrder == 1 ? "Unroll,CU,Wavefront" :
                                                                       gfxWaveOrder == 2 ? "Wavefront,Unroll,CU" :
                                                                       gfxWaveOrder == 3 ? "Wavefront,CU,Unroll" :
                                                                       gfxWaveOrder == 4 ? "CU,Unroll,Wavefront" :
                                                                                           "CU,Wavefront,Unroll")));
    PRINT_EV("NUM_CPU_DEVICES", numCpuDevices,
             std::string("Using ") + std::to_string(numCpuDevices) + " CPU devices");
    PRINT_EV("NUM_GPU_DEVICES", numGpuDevices,
             std::string("Using ") + std::to_string(numGpuDevices) + " GPU devices");
    PRINT_EV("NUM_ITERATIONS", numIterations,
             std::string("Running ") + std::to_string(numIterations > 0 ? numIterations : -numIterations) + " "
             + (numIterations > 0 ? " timed iteration(s)" : "seconds(s) per Test"));
    PRINT_EV("NUM_WARMUPS", numWarmups,
             std::string("Running " + std::to_string(numWarmups) + " warmup iteration(s) per Test"));
    PRINT_EV("SHARED_MEM_BYTES", sharedMemBytes,
             std::string("Using " + std::to_string(sharedMemBytes) + " shared mem per threadblock"));
    PRINT_EV("SHOW_ITERATIONS", showIterations,
             std::string(showIterations ? "Showing" : "Hiding") + " per-iteration timing");
    PRINT_EV("USE_INTERACTIVE", useInteractive,
             std::string("Running in ") + (useInteractive ? "interactive" : "non-interactive") + " mode");
    PRINT_EV("USE_PCIE_INDEX", usePcieIndexing,
             std::string("Use ") + (usePcieIndexing ? "PCIe" : "HIP") + " GPU device indexing");
    PRINT_EV("USE_PREP_KERNEL", usePrepSrcKernel,
             std::string("Using ") + (usePrepSrcKernel ? "GPU kernels" : "hipMemcpy") + " to initialize source data");
    PRINT_EV("USE_SINGLE_STREAM", useSingleStream,
             std::string("Using single stream per ") + (useSingleStream ? "device" : "Transfer"));
    PRINT_EV("USE_XCC_FILTER", useXccFilter,
             std::string("XCC filtering ") + (useXccFilter ? "enabled" : "disabled"));
    if (useXccFilter)
    {
      printf("%36s: Preferred XCC Table (XCC_PREF_TABLE)\n", "");
      printf("%36s:         ", "");
      for (int i = 0; i < numGpuDevices; i++) printf(" %3d", i); printf(" (#XCCs)\n");
      for (int i = 0; i < numGpuDevices; i++)
      {
        printf("%36s: GPU %3d ", "", i);
        for (int j = 0; j < numGpuDevices; j++)
          printf(" %3d", prefXccTable[i][j]);
        printf(" %3lu\n", xccIdsPerDevice[i].size());
      }
    }
    PRINT_EV("VALIDATE_DIRECT", validateDirect,
             std::string("Validate GPU destination memory ") + (validateDirect ? "directly" : "via CPU staging buffer"));
    printf("\n");

    if (blockOrder != ORDER_SEQUENTIAL && !useSingleStream)
      printf("[WARN] BLOCK_ORDER is ignored if USE_SINGLE_STREAM is not enabled\n");
  };

  // Display env var for P2P Benchmark preset
  void DisplayP2PBenchmarkEnvVars() const
  {
    DisplayEnvVars();

    if (hideEnv) return;

    if (!outputToCsv)
      printf("[P2P Related]\n");

    PRINT_EV("NUM_CPU_SE", numCpuSubExecs,
             std::string("Using ") + std::to_string(numCpuSubExecs) + " CPU subexecutors");
    PRINT_EV("NUM_GPU_SE", numGpuSubExecs,
             std::string("Using ") + std::to_string(numGpuSubExecs) + " GPU subexecutors");
    PRINT_EV("P2P_MODE", p2pMode,
             std::string("Running ") + (p2pMode == 1 ? "Unidirectional" :
                                        p2pMode == 2 ? "Bidirectional"  :
                                                       "Unidirectional + Bidirectional"));
    PRINT_EV("USE_FINE_GRAIN", useFineGrain,
             std::string("Using ") + (useFineGrain ? "fine" : "coarse") + "-grained memory");

    PRINT_EV("USE_GPU_DMA", useDmaCopy,
             std::string("Using GPU-") + (useDmaCopy ? "DMA" : "GFX") + " as GPU executor");
    PRINT_EV("USE_REMOTE_READ", useRemoteRead,
             std::string("Using ") + (useRemoteRead ? "DST" : "SRC") + " as executor");
    printf("\n");
  }

  // Display env var settings
  void DisplaySweepEnvVars() const
  {
    DisplayEnvVars();
    if (hideEnv) return;

    if (!outputToCsv)
      printf("[Sweep Related]\n");
    PRINT_ES("SWEEP_DST", sweepDst.c_str(),
             std::string("Destination Memory Types to sweep"));
    PRINT_ES("SWEEP_EXE", sweepExe.c_str(),
             std::string("Executor Types to sweep"));
    PRINT_EV("SWEEP_MAX", sweepMax,
             std::string("Max simultaneous transfers (0 = no limit)"));
    PRINT_EV("SWEEP_MIN", sweepMin,
             std::string("Min simultaenous transfers"));
    PRINT_EV("SWEEP_RAND_BYTES", sweepRandBytes,
             std::string("Using ") + (sweepRandBytes ? "random" : "constant") + " number of bytes per Transfer");
    PRINT_EV("SWEEP_SEED", sweepSeed,
             std::string("Random seed set to ") + std::to_string(sweepSeed));
    PRINT_ES("SWEEP_SRC", sweepSrc.c_str(),
             std::string("Source Memory Types to sweep"));
    PRINT_EV("SWEEP_TEST_LIMIT", sweepTestLimit,
             std::string("Max number of tests to run during sweep (0 = no limit)"));
    PRINT_EV("SWEEP_TIME_LIMIT", sweepTimeLimit,
             std::string("Max number of seconds to run sweep for  (0 = no limit)"));
    PRINT_EV("SWEEP_XGMI_MAX", sweepXgmiMax,
             std::string("Max number of XGMI hops for Transfers (-1 = no limit)"));
    PRINT_EV("SWEEP_XGMI_MIN", sweepXgmiMin,
             std::string("Min number of XGMI hops for Transfers"));
    printf("\n");
  }

  void DisplayA2AEnvVars() const
  {
    DisplayEnvVars();
    if (hideEnv) return;
    if (!outputToCsv)
      printf("[AllToAll Related]\n");
    PRINT_EV("A2A_DIRECT", a2aDirect,
             std::string(a2aDirect ? "Only using direct links" : "Full all-to-all"));
    PRINT_EV("A2A_MODE", a2aMode,
             std::string(a2aMode == 0 ? "Perform copy" :
                         a2aMode == 1 ? "Perform read-only" :
                                        "Perform write-only"));
    PRINT_EV("USE_FINE_GRAIN", useFineGrain,
             std::string("Using ") + (useFineGrain ? "fine" : "coarse") + "-grained memory");
    PRINT_EV("USE_REMOTE_READ", useRemoteRead,
             std::string("Using ") + (useRemoteRead ? "DST" : "SRC") + " as executor");

    printf("\n");
  }

  void DisplaySchmooEnvVars() const
  {
    DisplayEnvVars();
    if (hideEnv) return;
    if (!outputToCsv)
      printf("[Schmoo Related]\n");
    PRINT_EV("USE_FINE_GRAIN", useFineGrain,
             std::string("Using ") + (useFineGrain ? "fine" : "coarse") + "-grained memory");
  }

  void DisplayRemoteWriteEnvVars() const
  {
    DisplayEnvVars();
    if (hideEnv) return;
    if (!outputToCsv)
      printf("[Remote-Write Related]\n");
    PRINT_EV("USE_FINE_GRAIN", useFineGrain,
             std::string("Using ") + (useFineGrain ? "fine" : "coarse") + "-grained memory");
    PRINT_EV("USE_REMOTE_READ", useRemoteRead,
             std::string("Performing remote ") + (useRemoteRead ? "reads" : "writes"));
    printf("\n");
  }

  void DisplayParallelCopyEnvVars() const
  {
    DisplayEnvVars();
    if (hideEnv) return;
    if (!outputToCsv)
      printf("[Parallel-copy Related]\n");
    PRINT_EV("USE_FINE_GRAIN", useFineGrain,
             std::string("Using ") + (useFineGrain ? "fine" : "coarse") + "-grained memory");
    PRINT_EV("USE_GPU_DMA", useDmaCopy,
             std::string("Using GPU-") + (useDmaCopy ? "DMA" : "GFX") + " as GPU executor");
    printf("\n");
  }

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

  std::string GetCuMaskDesc() const
  {
    std::vector<std::pair<int, int>> runs;

    bool inRun = false;
    std::pair<int, int> curr;
    int used = 0;
    for (int i = 0; i < cuMask.size(); i++)
    {
      for (int j = 0; j < 32; j++)
      {
        if (cuMask[i] & (1 << j))
        {
          used++;
          if (!inRun)
          {
            inRun = true;
            curr.first = i * 32 + j;
          }
        }
        else
        {
          if (inRun)
          {
            inRun = false;
            curr.second = i * 32 + j - 1;
            runs.push_back(curr);
          }
        }
      }
    }
    if (inRun)
      curr.second = cuMask.size() * 32 - 1;

    std::string result = "CUs used: (" + std::to_string(used) + ") ";
    for (int i = 0; i < runs.size(); i++)
    {
      if (i) result += ",";
      if (runs[i].first == runs[i].second) result += std::to_string(runs[i].first);
      else result += std::to_string(runs[i].first) + "-" + std::to_string(runs[i].second);
    }
    return result;
  }
};

#endif

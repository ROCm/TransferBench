/*
Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.

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

#define PackedFloat_t   float4
#define MAX_BLOCKSIZE   512
#define FLOATS_PER_PACK (sizeof(PackedFloat_t) / sizeof(float))
#define MEMSET_CHAR     75
#define MEMSET_VAL      13323083.0f


#if defined(__NVCC__)
#define warpSize 32
#endif

#define MAX_WAVEGROUPS  MAX_BLOCKSIZE / warpSize
#define MAX_UNROLL      8
#define NUM_WAVEORDERS  6

// Each subExecutor is provided with subarrays to work on
#define MAX_SRCS 16
#define MAX_DSTS 16
struct SubExecParam
{
  // Inputs
  size_t    N;                                  // Number of floats this subExecutor works on
  int       numSrcs;                            // Number of source arrays
  int       numDsts;                            // Number of destination arrays
  float*    src[MAX_SRCS];                      // Source array pointers
  float*    dst[MAX_DSTS];                      // Destination array pointers
  int32_t   preferredXccId;                     // XCC ID to execute on

  // Prepared
  int       teamSize;                           // Index of this sub executor amongst team
  int       teamIdx;                            // Size of team this sub executor is part of

  // Outputs
  long long startCycle;                         // Start timestamp for in-kernel timing (GPU-GFX executor)
  long long stopCycle;                          // Stop  timestamp for in-kernel timing (GPU-GFX executor)
  uint32_t  hwId;                               // Hardware ID
  uint32_t  xccId;                              // XCC ID
};

// Macro for collecting HW_REG_HW_ID
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1200__) || defined(__gfx1201__)
#define GetHwId(hwId) \
  hwId = 0
#elif defined(__NVCC__)
#define GetHwId(hwId) \
  asm("mov.u32 %0, %smid;" : "=r"(hwId) )
#else
#define GetHwId(hwId) \
  asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s" (hwId));
#endif

// Macro for collecting HW_REG_XCC_ID
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define GetXccId(val) \
  asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s" (val));
#else
#define GetXccId(val) \
  val = 0
#endif

void CpuReduceKernel(SubExecParam const& p)
{
  int const& numSrcs = p.numSrcs;
  int const& numDsts = p.numDsts;

  if (numSrcs == 0)
  {
    for (int i = 0; i < numDsts; ++i)
      memset(p.dst[i], MEMSET_CHAR, p.N * sizeof(float));
  }
  else if (numSrcs == 1)
  {
    float const* __restrict__ src = p.src[0];
    if (numDsts == 0)
    {
      float sum = 0.0;
      for (int j = 0; j < p.N; j++)
        sum += p.src[0][j];

      // Add a dummy check to ensure the read is not optimized out
      if (sum != sum)
      {
        printf("[ERROR] Nan detected\n");
      }
    }
    else
    {
      for (int i = 0; i < numDsts; ++i)
      {
        memcpy(p.dst[i], src, p.N * sizeof(float));
      }
    }
  }
  else
  {
    float sum = 0.0f;
    for (int j = 0; j < p.N; j++)
    {
      sum = p.src[0][j];
      for (int i = 1; i < numSrcs; i++) sum += p.src[i][j];
      for (int i = 0; i < numDsts; i++) p.dst[i][j] = sum;
    }
  }
}

std::string PrepSrcValueString()
{
  return "Element i = ((i * 517) modulo 383 + 31) * (srcBufferIdx + 1)";
}

__host__ __device__ float PrepSrcValue(int srcBufferIdx, size_t idx)
{
  return (((idx % 383) * 517) % 383 + 31) * (srcBufferIdx + 1);
}

__global__ void CollectXccIdsKernel(int* xccIds)
{
  int xccId;
  GetXccId(xccId);
  xccIds[blockIdx.x] = xccId;
}

// GPU kernel to prepare src buffer data
__global__ void
PrepSrcDataKernel(float* ptr, size_t N, int srcBufferIdx)
{
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < N;
       idx += blockDim.x * gridDim.x)
  {
    ptr[idx] = PrepSrcValue(srcBufferIdx, idx);
  }
}

__device__ int64_t GetTimestamp()
{
#if defined(__NVCC__)
  int64_t result;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(result));
  return result;
#else
  return wall_clock64();
#endif
}

// Helper function for memset
template <typename T> __device__ __forceinline__ T      MemsetVal();
template <>           __device__ __forceinline__ float  MemsetVal(){ return MEMSET_VAL; };
template <>           __device__ __forceinline__ float4 MemsetVal(){ return make_float4(MEMSET_VAL, MEMSET_VAL, MEMSET_VAL, MEMSET_VAL); }

template <int BLOCKSIZE, int UNROLL>
__global__ void __launch_bounds__(BLOCKSIZE)
  GpuReduceKernel(SubExecParam* params, int waveOrder, int numSubIterations)
{
  int64_t startCycle;
  if (threadIdx.x == 0) startCycle = GetTimestamp();

  SubExecParam& p = params[blockIdx.y];

  // (Experimental) Filter by XCC if desired
#if !defined(__NVCC__)
  int32_t xccId;
  GetXccId(xccId);
  if (p.preferredXccId != -1 && xccId != p.preferredXccId) return;
#endif

  // Collect data information
  int32_t const  numSrcs  = p.numSrcs;
  int32_t const  numDsts  = p.numDsts;
  float4  const* __restrict__ srcFloat4[MAX_SRCS];
  float4*        __restrict__ dstFloat4[MAX_DSTS];
  for (int i = 0; i < numSrcs; i++) srcFloat4[i] = (float4*)p.src[i];
  for (int i = 0; i < numDsts; i++) dstFloat4[i] = (float4*)p.dst[i];

  // Operate on wavefront granularity
  int32_t const nTeams   = p.teamSize;             // Number of threadblocks working together on this subarray
  int32_t const teamIdx  = p.teamIdx;              // Index of this threadblock within the team
  int32_t const nWaves   = BLOCKSIZE   / warpSize; // Number of wavefronts within this threadblock
  int32_t const waveIdx  = threadIdx.x / warpSize; // Index of this wavefront within the threadblock
  int32_t const tIdx     = threadIdx.x % warpSize; // Thread index within wavefront

  size_t  const numFloat4 = p.N / 4;

  int32_t teamStride, waveStride, unrlStride, teamStride2, waveStride2;
  switch (waveOrder)
  {
  case 0: /* U,W,C */ unrlStride = 1; waveStride = UNROLL; teamStride = UNROLL * nWaves;  teamStride2 = nWaves; waveStride2 = 1     ; break;
  case 1: /* U,C,W */ unrlStride = 1; teamStride = UNROLL; waveStride = UNROLL * nTeams;  teamStride2 = 1;      waveStride2 = nTeams; break;
  case 2: /* W,U,C */ waveStride = 1; unrlStride = nWaves; teamStride = nWaves * UNROLL;  teamStride2 = nWaves; waveStride2 = 1     ; break;
  case 3: /* W,C,U */ waveStride = 1; teamStride = nWaves; unrlStride = nWaves * nTeams;  teamStride2 = nWaves; waveStride2 = 1     ; break;
  case 4: /* C,U,W */ teamStride = 1; unrlStride = nTeams; waveStride = nTeams * UNROLL;  teamStride2 = 1;      waveStride2 = nTeams; break;
  case 5: /* C,W,U */ teamStride = 1; waveStride = nTeams; unrlStride = nTeams * nWaves;  teamStride2 = 1;      waveStride2 = nTeams; break;
  }

  int subIterations = 0;
  while (1) {
    // First loop: Each wavefront in the team works on UNROLL float4s per thread
    size_t const loop1Stride = nTeams * nWaves * UNROLL * warpSize;
    size_t const loop1Limit  = numFloat4 / loop1Stride * loop1Stride;
    {
      float4 val[UNROLL];
      if (numSrcs == 0) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++)
          val[u] = MemsetVal<float4>();
      }

      for (size_t idx = (teamIdx * teamStride + waveIdx * waveStride) * warpSize + tIdx; idx < loop1Limit; idx += loop1Stride)
      {
        // Read sources into memory and accumulate in registers
        if (numSrcs)
        {
          for (int u = 0; u < UNROLL; u++)
            val[u] = srcFloat4[0][idx + u * unrlStride * warpSize];
          for (int s = 1; s < numSrcs; s++)
            for (int u = 0; u < UNROLL; u++)
              val[u] += srcFloat4[s][idx + u * unrlStride * warpSize];
        }

        // Write accumulation to all outputs
        for (int d = 0; d < numDsts; d++)
        {
          #pragma unroll
          for (int u = 0; u < UNROLL; u++)
            dstFloat4[d][idx + u * unrlStride * warpSize] = val[u];
        }
      }
    }

    // Second loop: Deal with remaining float4s
    {
      if (loop1Limit < numFloat4)
      {
        float4 val;
        if (numSrcs == 0) val = MemsetVal<float4>();

        size_t const loop2Stride = nTeams * nWaves * warpSize;
        for (size_t idx = loop1Limit + (teamIdx * teamStride2 + waveIdx * waveStride2) * warpSize + tIdx; idx < numFloat4; idx += loop2Stride)
        {
          if (numSrcs)
          {
            val = srcFloat4[0][idx];
            for (int s = 1; s < numSrcs; s++)
              val += srcFloat4[s][idx];
          }

          for (int d = 0; d < numDsts; d++)
            dstFloat4[d][idx] = val;
        }
      }
    }

    // Third loop; Deal with remaining floats
    {
      if (numFloat4 * 4 < p.N)
      {
        float val;
        if (numSrcs == 0) val = MemsetVal<float>();

        size_t const loop3Stride = nTeams * nWaves * warpSize;
        for( size_t idx = numFloat4 * 4 + (teamIdx * teamStride2 + waveIdx * waveStride2) * warpSize + tIdx; idx < p.N; idx += loop3Stride)
        {
          if (numSrcs)
          {
            val = p.src[0][idx];
            for (int s = 1; s < numSrcs; s++)
              val += p.src[s][idx];
          }

          for (int d = 0; d < numDsts; d++)
            p.dst[d][idx] = val;
        }
      }
    }

    if (++subIterations == numSubIterations) break;
  }

  // Wait for all threads to finish
  __syncthreads();
  if (threadIdx.x == 0)
  {
    __threadfence_system();
    p.stopCycle  = GetTimestamp();
    p.startCycle = startCycle;
    GetHwId(p.hwId);
    GetXccId(p.xccId);
  }
}

typedef void (*GpuKernelFuncPtr)(SubExecParam*, int, int);

#define GPU_KERNEL_UNROLL_DECL(BLOCKSIZE) \
  {GpuReduceKernel<BLOCKSIZE, 1>,  \
   GpuReduceKernel<BLOCKSIZE, 2>,  \
   GpuReduceKernel<BLOCKSIZE, 3>,  \
   GpuReduceKernel<BLOCKSIZE, 4>,  \
   GpuReduceKernel<BLOCKSIZE, 5>,  \
   GpuReduceKernel<BLOCKSIZE, 6>,  \
   GpuReduceKernel<BLOCKSIZE, 7>,  \
   GpuReduceKernel<BLOCKSIZE, 8>}

GpuKernelFuncPtr GpuKernelTable[MAX_WAVEGROUPS][MAX_UNROLL] =
{
  GPU_KERNEL_UNROLL_DECL(64),
  GPU_KERNEL_UNROLL_DECL(128),
  GPU_KERNEL_UNROLL_DECL(192),
  GPU_KERNEL_UNROLL_DECL(256),
  GPU_KERNEL_UNROLL_DECL(320),
  GPU_KERNEL_UNROLL_DECL(384),
  GPU_KERNEL_UNROLL_DECL(448),
  GPU_KERNEL_UNROLL_DECL(512)
};

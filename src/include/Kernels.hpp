/*
Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.

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


#define MAX_WAVEGROUPS  MAX_BLOCKSIZE / warpSize
#define MAX_UNROLL      16
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
  uint32_t  preferredXccId;                     // XCC ID to execute on

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
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__NVCC__)
#define __trace_hwreg() \
  p.hwId = 0
#else
#define __trace_hwreg() \
  asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s" (p.hwId));
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
    for (int i = 0; i < numDsts; ++i)
    {
      memcpy(p.dst[i], src, p.N * sizeof(float));
    }
  }
  else
  {
    for (int j = 0; j < p.N; j++)
    {
      float sum = p.src[0][j];
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

// Helper function for memset
template <typename T> __device__ __forceinline__ T      MemsetVal();
template <>           __device__ __forceinline__ float  MemsetVal(){ return MEMSET_VAL; };
template <>           __device__ __forceinline__ float4 MemsetVal(){ return make_float4(MEMSET_VAL, MEMSET_VAL, MEMSET_VAL, MEMSET_VAL); }

// GPU copy kernel 0: 3 loops: unroll float 4, float4s, floats
template <int LOOP1_UNROLL>
__global__ void __launch_bounds__(MAX_BLOCKSIZE)
GpuReduceKernel1(SubExecParam* params)
{
  int64_t startCycle;
  if (threadIdx.x == 0) startCycle = wall_clock64();

  SubExecParam& p = params[blockIdx.y];

  // Filter by XCC if desired
  int xccId;
  GetXccId(xccId);
  if (p.preferredXccId != -1 && xccId != p.preferredXccId) return;

  // Operate on wavefront granularity
  int const numSrcs  = p.numSrcs;
  int const numDsts  = p.numDsts;
  int const waveId   = threadIdx.x / warpSize; // Wavefront number
  int const threadId = threadIdx.x % warpSize; // Thread index within wavefront

  // 1st loop - each wavefront operates on LOOP1_UNROLL x FLOATS_PER_PACK per thread per iteration
  // Determine the number of packed floats processed by the first loop
  size_t       Nrem        = p.N;
  size_t const loop1Npack  = (Nrem / (FLOATS_PER_PACK * LOOP1_UNROLL * warpSize)) * (LOOP1_UNROLL * warpSize);
  size_t const loop1Nelem  = loop1Npack * FLOATS_PER_PACK;
  size_t const loop1Inc    = blockDim.x * LOOP1_UNROLL;
  size_t       loop1Offset = waveId * LOOP1_UNROLL * warpSize + threadId;

  while (loop1Offset < loop1Npack)
  {
    PackedFloat_t vals[LOOP1_UNROLL] = {};

    if (numSrcs == 0)
    {
      #pragma unroll
      for (int u = 0; u < LOOP1_UNROLL; ++u) vals[u] = MemsetVal<float4>();
    }
    else
    {
      for (int i = 0; i < numSrcs; ++i)
      {
        PackedFloat_t const* __restrict__ packedSrc = (PackedFloat_t const*)(p.src[i]) + loop1Offset;
        #pragma unroll
        for (int u = 0; u < LOOP1_UNROLL; ++u)
          vals[u] += *(packedSrc + u * warpSize);
      }
    }

    for (int i = 0; i < numDsts; ++i)
    {
      PackedFloat_t* __restrict__ packedDst = (PackedFloat_t*)(p.dst[i]) + loop1Offset;
      #pragma unroll
      for (int u = 0; u < LOOP1_UNROLL; ++u) *(packedDst + u * warpSize) = vals[u];
    }
    loop1Offset += loop1Inc;
  }
  Nrem -= loop1Nelem;

  if (Nrem > 0)
  {
    // 2nd loop - Each thread operates on FLOATS_PER_PACK per iteration
    // NOTE: Using int32_t due to smaller size requirements
    int32_t const loop2Npack  = Nrem / FLOATS_PER_PACK;
    int32_t const loop2Nelem  = loop2Npack * FLOATS_PER_PACK;
    int32_t const loop2Inc    = blockDim.x;
    int32_t       loop2Offset = threadIdx.x;

    while (loop2Offset < loop2Npack)
    {
      PackedFloat_t val;
      if (numSrcs == 0)
      {
        val = MemsetVal<float4>();
      }
      else
      {
        val = {};
        for (int i = 0; i < numSrcs; ++i)
        {
          PackedFloat_t const* __restrict__ packedSrc = (PackedFloat_t const*)(p.src[i] + loop1Nelem) + loop2Offset;
          val += *packedSrc;
        }
      }

      for (int i = 0; i < numDsts; ++i)
      {
        PackedFloat_t* __restrict__ packedDst = (PackedFloat_t*)(p.dst[i] + loop1Nelem) + loop2Offset;
        *packedDst = val;
      }
      loop2Offset += loop2Inc;
    }
    Nrem -= loop2Nelem;

    // Deal with leftovers less than FLOATS_PER_PACK)
    if (threadIdx.x < Nrem)
    {
      int offset = loop1Nelem + loop2Nelem + threadIdx.x;
      float val = 0;
      if (numSrcs == 0)
      {
        val = MEMSET_VAL;
      }
      else
      {
        for (int i = 0; i < numSrcs; ++i)
          val += p.src[i][offset];
      }

      for (int i = 0; i < numDsts; ++i)
        p.dst[i][offset] = val;
    }
  }

  __syncthreads();
  if (threadIdx.x == 0)
  {
    __threadfence_system();
    p.stopCycle  = wall_clock64();
    p.startCycle = startCycle;
    p.xccId      = xccId;
    __trace_hwreg();
  }
}

template <typename FLOAT_TYPE, int UNROLL_FACTOR>
__device__ size_t GpuReduceFuncImpl2(SubExecParam const &p, size_t const offset, size_t const N)
{
  int    constexpr numFloatsPerPack = sizeof(FLOAT_TYPE) / sizeof(float); // Number of floats handled at a time per thread
  size_t constexpr loopPackInc      = blockDim.x * UNROLL_FACTOR;
  size_t constexpr numPacksPerWave  = warpSize * UNROLL_FACTOR;
  int    const     waveId           = threadIdx.x / warpSize;            // Wavefront number
  int    const     threadId         = threadIdx.x % warpSize;            // Thread index within wavefront
  int    const     numSrcs          = p.numSrcs;
  int    const     numDsts          = p.numDsts;
  size_t const     numPacksDone     = (numFloatsPerPack == 1 && UNROLL_FACTOR == 1) ? N : (N / (FLOATS_PER_PACK * numPacksPerWave)) * numPacksPerWave;
  size_t const     numFloatsLeft    = N - numPacksDone * numFloatsPerPack;
  size_t           loopPackOffset   = waveId * numPacksPerWave + threadId;

  while (loopPackOffset < numPacksDone)
  {
    FLOAT_TYPE vals[UNROLL_FACTOR];

    if (numSrcs == 0)
    {
      #pragma unroll UNROLL_FACTOR
      for (int u = 0; u < UNROLL_FACTOR; ++u) vals[u] = MemsetVal<FLOAT_TYPE>();
    }
    else
    {
      FLOAT_TYPE const* __restrict__ src0Ptr = ((FLOAT_TYPE const*)(p.src[0] + offset)) + loopPackOffset;
      #pragma unroll UNROLL_FACTOR
      for (int u = 0; u < UNROLL_FACTOR; ++u)
        vals[u] = *(src0Ptr + u * warpSize);

      for (int i = 1; i < numSrcs; ++i)
      {
        FLOAT_TYPE const* __restrict__ srcPtr = ((FLOAT_TYPE const*)(p.src[i] + offset)) + loopPackOffset;

        #pragma unroll UNROLL_FACTOR
        for (int u = 0; u < UNROLL_FACTOR; ++u)
          vals[u] += *(srcPtr + u * warpSize);
      }
    }

    for (int i = 0; i < numDsts; ++i)
    {
      FLOAT_TYPE* __restrict__ dstPtr = (FLOAT_TYPE*)(p.dst[i + offset]) + loopPackOffset;
      #pragma unroll UNROLL_FACTOR
      for (int u = 0; u < UNROLL_FACTOR; ++u)
        *(dstPtr + u * warpSize) = vals[u];
    }
    loopPackOffset += loopPackInc;
  }

  return numFloatsLeft;
}

template <typename FLOAT_TYPE, int UNROLL_FACTOR>
__device__ size_t GpuReduceFuncImpl(SubExecParam const &p, size_t const offset, size_t const N)
{
  // Each thread in the block works on UNROLL_FACTOR FLOAT_TYPEs during each iteration of the loop
  int    constexpr numFloatsPerRead      = sizeof(FLOAT_TYPE) / sizeof(float);
  size_t const     numFloatsPerInnerLoop = blockDim.x * numFloatsPerRead;
  size_t const     numFloatsPerOuterLoop = numFloatsPerInnerLoop * UNROLL_FACTOR;
  size_t const     numFloatsLeft         = (numFloatsPerRead == 1 && UNROLL_FACTOR == 1) ? 0 : N % numFloatsPerOuterLoop;
  size_t const     numFloatsDone         = N - numFloatsLeft;
  int    const     numSrcs               = p.numSrcs;
  int    const     numDsts               = p.numDsts;

  for (size_t idx = threadIdx.x * numFloatsPerRead; idx < numFloatsDone; idx += numFloatsPerOuterLoop)
  {
    FLOAT_TYPE tmp[UNROLL_FACTOR];

    if (numSrcs == 0)
    {
        #pragma unroll UNROLL_FACTOR
        for (int u = 0; u < UNROLL_FACTOR; ++u)
          tmp[u] = MemsetVal<FLOAT_TYPE>();
    }
    else
    {
      #pragma unroll UNROLL_FACTOR
      for (int u = 0; u < UNROLL_FACTOR; ++u)
        tmp[u] = *((FLOAT_TYPE*)(&p.src[0][offset + idx + u * numFloatsPerInnerLoop]));

      for (int i = 1; i < numSrcs; ++i)
      {
        #pragma unroll UNROLL_FACTOR
        for (int u = 0; u < UNROLL_FACTOR; ++u)
          tmp[u] += *((FLOAT_TYPE*)(&p.src[i][offset + idx + u * numFloatsPerInnerLoop]));
      }
    }

    for (int i = 0; i < numDsts; ++i)
    {
      for (int u = 0; u < UNROLL_FACTOR; ++u)
      {
        *((FLOAT_TYPE*)(&p.dst[i][offset + idx + u * numFloatsPerInnerLoop])) = tmp[u];
      }
    }
  }
  return numFloatsLeft;
}

template <typename FLOAT_TYPE>
__device__ size_t GpuReduceFunc(SubExecParam const &p, size_t const offset, size_t const N, int const unroll)
{
  switch (unroll)
  {
  case  1: return GpuReduceFuncImpl<FLOAT_TYPE,  1>(p, offset, N);
  case  2: return GpuReduceFuncImpl<FLOAT_TYPE,  2>(p, offset, N);
  case  3: return GpuReduceFuncImpl<FLOAT_TYPE,  3>(p, offset, N);
  case  4: return GpuReduceFuncImpl<FLOAT_TYPE,  4>(p, offset, N);
  case  5: return GpuReduceFuncImpl<FLOAT_TYPE,  5>(p, offset, N);
  case  6: return GpuReduceFuncImpl<FLOAT_TYPE,  6>(p, offset, N);
  case  7: return GpuReduceFuncImpl<FLOAT_TYPE,  7>(p, offset, N);
  case  8: return GpuReduceFuncImpl<FLOAT_TYPE,  8>(p, offset, N);
  case  9: return GpuReduceFuncImpl<FLOAT_TYPE,  9>(p, offset, N);
  case 10: return GpuReduceFuncImpl<FLOAT_TYPE, 10>(p, offset, N);
  case 11: return GpuReduceFuncImpl<FLOAT_TYPE, 11>(p, offset, N);
  case 12: return GpuReduceFuncImpl<FLOAT_TYPE, 12>(p, offset, N);
  case 13: return GpuReduceFuncImpl<FLOAT_TYPE, 13>(p, offset, N);
  case 14: return GpuReduceFuncImpl<FLOAT_TYPE, 14>(p, offset, N);
  case 15: return GpuReduceFuncImpl<FLOAT_TYPE, 15>(p, offset, N);
  case 16: return GpuReduceFuncImpl<FLOAT_TYPE, 16>(p, offset, N);
  default: return GpuReduceFuncImpl<FLOAT_TYPE,  1>(p, offset, N);
  }
}

// GPU copy kernel
__global__ void __launch_bounds__(MAX_BLOCKSIZE)
GpuReduceKernel2(SubExecParam* params)
{
  int64_t startCycle = wall_clock64();
  SubExecParam& p = params[blockIdx.y];

  size_t numFloatsLeft = GpuReduceFunc<float4>(p, 0, p.N, 8);
  if (numFloatsLeft)
    numFloatsLeft = GpuReduceFunc<float4>(p, p.N - numFloatsLeft, numFloatsLeft, 1);

  if (numFloatsLeft)
  GpuReduceFunc<float>(p, p.N - numFloatsLeft, numFloatsLeft, 1);

  __threadfence_system();
  if (threadIdx.x == 0)
  {
    p.startCycle = startCycle;
    p.stopCycle  = wall_clock64();
  }
}
template <int BLOCKSIZE, int UNROLL>
__global__ void __launch_bounds__(BLOCKSIZE)
  GpuReduceKernel3(SubExecParam* params, int waveOrder)
{
  int64_t startCycle;
  if (threadIdx.x == 0) startCycle = wall_clock64();

  SubExecParam& p = params[blockIdx.y];

  // (Experimental) Filter by XCC if desired
  int32_t xccId;
  GetXccId(xccId);
  if (p.preferredXccId != -1 && xccId != p.preferredXccId) return;

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
  int32_t const nFlt4PerWave = warpSize * 4;

  int32_t teamStride, waveStride, unrlStride;
  switch (waveOrder)
  {
  case 0: /* U,W,C */ unrlStride = 1; waveStride = UNROLL; teamStride = UNROLL * nWaves; break;
  case 1: /* U,C,W */ unrlStride = 1; teamStride = UNROLL; waveStride = UNROLL * nTeams; break;
  case 2: /* W,U,C */ waveStride = 1; unrlStride = nWaves; teamStride = nWaves * UNROLL; break;
  case 3: /* W,C,U */ waveStride = 1; teamStride = nWaves; unrlStride = nWaves * nTeams; break;
  case 4: /* C,U,W */ teamStride = 1; unrlStride = nTeams; waveStride = nTeams * UNROLL; break;
  case 5: /* C,W,U */ teamStride = 1; waveStride = nTeams; unrlStride = nTeams * nWaves; break;
  }

  // First loop: Each wavefront in the team works on UNROLL float4s per thread
  {
    size_t const loop1Stride = nTeams * nWaves * UNROLL * warpSize;
    size_t const loop1Limit  = numFloat4 / loop1Stride * loop1Stride;

    float4 val[UNROLL];
    if (numSrcs == 0)
    {
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

  // Wait for all threads to finish
  __syncthreads();
  if (threadIdx.x == 0)
  {
    __threadfence_system();
    p.stopCycle  = wall_clock64();
    p.startCycle = startCycle;
    p.xccId      = xccId;
    __trace_hwreg();
  }
}


typedef void (*GpuKernel1FuncPtr)(SubExecParam*);
GpuKernel1FuncPtr GpuKernel1Table[MAX_UNROLL] =
{
  GpuReduceKernel1<1>,
  GpuReduceKernel1<2>,
  GpuReduceKernel1<3>,
  GpuReduceKernel1<4>,
  GpuReduceKernel1<5>,
  GpuReduceKernel1<6>,
  GpuReduceKernel1<7>,
  GpuReduceKernel1<8>,
  GpuReduceKernel1<9>,
  GpuReduceKernel1<10>,
  GpuReduceKernel1<11>,
  GpuReduceKernel1<12>,
  GpuReduceKernel1<13>,
  GpuReduceKernel1<14>,
  GpuReduceKernel1<15>,
  GpuReduceKernel1<16>
};

typedef void (*GpuKernel3FuncPtr)(SubExecParam*, int);

#define GPU_KERNEL3_UNROLL_DECL(BLOCKSIZE) \
  {GpuReduceKernel3<BLOCKSIZE, 1>,  \
   GpuReduceKernel3<BLOCKSIZE, 2>,  \
   GpuReduceKernel3<BLOCKSIZE, 3>,  \
   GpuReduceKernel3<BLOCKSIZE, 4>,  \
   GpuReduceKernel3<BLOCKSIZE, 5>,  \
   GpuReduceKernel3<BLOCKSIZE, 6>,  \
   GpuReduceKernel3<BLOCKSIZE, 7>,  \
   GpuReduceKernel3<BLOCKSIZE, 8>,  \
   GpuReduceKernel3<BLOCKSIZE, 9>,  \
   GpuReduceKernel3<BLOCKSIZE, 10>, \
   GpuReduceKernel3<BLOCKSIZE, 11>, \
   GpuReduceKernel3<BLOCKSIZE, 12>, \
   GpuReduceKernel3<BLOCKSIZE, 13>, \
   GpuReduceKernel3<BLOCKSIZE, 14>, \
   GpuReduceKernel3<BLOCKSIZE, 15>, \
   GpuReduceKernel3<BLOCKSIZE, 16>}

GpuKernel3FuncPtr GpuKernel3Table[MAX_WAVEGROUPS][MAX_UNROLL] =
{
  GPU_KERNEL3_UNROLL_DECL(64),
  GPU_KERNEL3_UNROLL_DECL(128),
  GPU_KERNEL3_UNROLL_DECL(192),
  GPU_KERNEL3_UNROLL_DECL(256),
  GPU_KERNEL3_UNROLL_DECL(320),
  GPU_KERNEL3_UNROLL_DECL(384),
  GPU_KERNEL3_UNROLL_DECL(448),
  GPU_KERNEL3_UNROLL_DECL(512)
};

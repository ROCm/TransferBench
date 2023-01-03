/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#define WARP_SIZE       64
#define BLOCKSIZE       256
#define FLOATS_PER_PACK (sizeof(PackedFloat_t) / sizeof(float))
#define MEMSET_CHAR     75
#define MEMSET_VAL      13323083.0f
#define PACKED_VAL      make_float4(MEMSET_VAL, MEMSET_VAL, MEMSET_VAL, MEMSET_VAL)

// GPU copy kernel
__global__ void __launch_bounds__(BLOCKSIZE)
GpuReduceKernel(SubExecParam* params)
{
  int64_t startCycle = __builtin_amdgcn_s_memrealtime();

  // Operate on wavefront granularity
  SubExecParam& p    = params[blockIdx.x];
  int const numSrcs  = p.numSrcs;
  int const numDsts  = p.numDsts;
  int const numWaves = BLOCKSIZE   / WARP_SIZE; // Number of wavefronts per threadblock
  int const waveId   = threadIdx.x / WARP_SIZE; // Wavefront number
  int const threadId = threadIdx.x % WARP_SIZE; // Thread index within wavefront

  // 1st loop - each wavefront operates on LOOP1_UNROLL x FLOATS_PER_PACK per thread per iteration
  // Determine the number of packed floats processed by the first loop
  #define LOOP1_UNROLL 8
  size_t       Nrem        = p.N;
  size_t const loop1Npack  = (Nrem / (FLOATS_PER_PACK * LOOP1_UNROLL * WARP_SIZE)) * (LOOP1_UNROLL * WARP_SIZE);
  size_t const loop1Nelem  = loop1Npack * FLOATS_PER_PACK;
  size_t const loop1Inc    = BLOCKSIZE * LOOP1_UNROLL;
  size_t       loop1Offset = waveId * LOOP1_UNROLL * WARP_SIZE + threadId;

  while (loop1Offset < loop1Npack)
  {
    PackedFloat_t vals[LOOP1_UNROLL] = {};

    if (numSrcs == 0)
    {
      #pragma unroll
      for (int u = 0; u < LOOP1_UNROLL; ++u) vals[u] = PACKED_VAL;
    }
    else
    {
      for (int i = 0; i < numSrcs; ++i)
      {
        PackedFloat_t const* __restrict__ packedSrc = (PackedFloat_t const*)(p.src[i]) + loop1Offset;
        #pragma unroll
        for (int u = 0; u < LOOP1_UNROLL; ++u)
          vals[u] += *(packedSrc + u * WARP_SIZE);
      }
    }

    for (int i = 0; i < numDsts; ++i)
    {
      PackedFloat_t* __restrict__ packedDst = (PackedFloat_t*)(p.dst[i]) + loop1Offset;
      #pragma unroll
      for (int u = 0; u < LOOP1_UNROLL; ++u) *(packedDst + u * WARP_SIZE) = vals[u];
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
    int32_t const loop2Inc    = BLOCKSIZE;
    int32_t       loop2Offset = threadIdx.x;

    while (loop2Offset < loop2Npack)
    {
      PackedFloat_t val;
      if (numSrcs == 0)
      {
        val = PACKED_VAL;
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
          val += ((float const* __restrict__)p.src[i])[offset];
      }

      for (int i = 0; i < numDsts; ++i)
        ((float* __restrict__)p.dst[i])[offset] = val;
    }
  }

  __threadfence_system();
  if (threadIdx.x == 0)
  {
    p.startCycle = startCycle;
    p.stopCycle  = __builtin_amdgcn_s_memrealtime();
  }
}

void CpuReduceKernel(SubExecParam const& p)
{
  int const& numSrcs = p.numSrcs;
  int const& numDsts = p.numDsts;

  if (numSrcs == 0)
  {
    for (int i = 0; i < numDsts; ++i)
      memset((float* __restrict__)p.dst[i], MEMSET_CHAR, p.N * sizeof(float));
  }
  else if (numSrcs == 1)
  {
    float const* __restrict__ src = p.src[0];
    for (int i = 0; i < numDsts; ++i)
    {
      memcpy((float* __restrict__)p.dst[i], src, p.N * sizeof(float));
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

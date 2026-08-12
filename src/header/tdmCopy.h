/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
/// \file tdmCopy.h
/// \brief Helper functions to perform HBM to HBM copies via the Tensor Data Mover
///        which copies HBM to a LDS staging buffer, then from LDS out to HBM
///        all without touching cache
///
///        This is introduced via __device__ level memcpy-like API which can be
///        either blocking or asynchronous, and utilize all warps, or a team of
///        contiguous warps, to allow for other warps to do other tasks
///
/// \par Quick start
/// \code
///   #include "tdmCopy.h"
///   __shared__ uint8_t staging[N];                 // or dynamic extern __shared__
///   tdm::tdmCopy(dst, src, bytes, staging, N);     // block-collective, blocking
///   __syncthreads();                               // block-wide visibility
/// \endcode
///
/// \par Two axes of control
/// - Completion: tdmCopy() (blocking) vs. tdmCopyAsync() + tdmWait() (deferred).
/// - Participation: block-collective (all warps) vs. tdmCopyByTeam() (a
///   contiguous warp range, leaving the other warps free to compute).
///
/// \par Availability
/// TDM is a hardware feature present on some architectures only. On a target
/// without it, every entry point is `= delete`d: including the header is always
/// fine, but calling any tdm:: function is a hard compile-time error at the call
/// site. See the AVAILABILITY block below. For a runtime/host check (e.g. to pick
/// a kernel before launch), use IsTdmCopySupported(), which is always callable.
///
/// The design rationale, hardware model, and visibility rules live at the bottom
/// of this file under "IMPLEMENTATION NOTES"; the public API is right here.
#pragma once

// Two backends are supported:
// AMD's Tensor Data Mover (gfx1250) and NVIDIA's cp.async.bulk / TMA (Hopper, sm_90+). 
// The active backend is chosen from the compiler in use (see the AVAILABILITY block below).
#if defined(__CUDACC__) || defined(__NVCC__) || defined(__CUDA__)
#  include <cuda_runtime.h>
#  include <cuda/ptx>
#else
#  include <hip/hip_runtime.h>
#endif
#include <stdint.h>
#include <stddef.h>

// ============================================================================
//  AVAILABILITY
// ----------------------------------------------------------------------------
// TDM is supported on a subset of architectures. Detection is centralized in the
// single macro TDM_SUPPORTED, which is 1 only when we are compiling a device pass
// for a TDM-capable arch AND the compiler exposes the TDM builtin AND the arch's
// descriptor header is on the include path. Keying on __has_builtin / __has_include
// (not just the arch macro) means an older toolchain that predates the builtin, or
// a build without the descriptor header, degrades gracefully instead of failing;
// a new TDM-capable arch works as soon as its target macro is added below.
//
// Each public entry point is individually guarded. When TDM_SUPPORTED is 1,
// the real implementation is compiled. Otherwise every entry point is declared
// `= delete`, so including this header is always fine but CALLING any tdm::
// function on an unsupported target is a hard compile-time error at the call
// site ("call to deleted function"). The TDM_API / TDM_DELETED macro pair below
// applies that guard uniformly to every declaration.
// ============================================================================
#ifndef __has_builtin
#  define __has_builtin(x) 0
#endif
#ifndef __has_include
#  define __has_include(x) 0
#endif

// ---- backend selection -----------------------------------------------------
// TDM_PLATFORM_NV is a host-evaluable proxy for "this is the NVIDIA toolchain".
// Exactly one backend is enabled: TDM_BACKEND_AMD (gfx1250 TDM) or
// TDM_BACKEND_NV (Hopper+ cp.async.bulk). TDM_SUPPORTED is their OR.
#if defined(__CUDACC__) || defined(__NVCC__) || defined(__CUDA__)
#  define TDM_PLATFORM_NV 1
#else
#  define TDM_PLATFORM_NV 0
#endif

// Host-evaluable toolchain capability. TDM_SUPPORTED (below) keys on device arch
// macros (e.g. __gfx1250__) that are never defined during the host pass, so it is
// always 0 in host code and cannot gate the host-side IsTdmCopySupported() check.
// The descriptor header's presence is the reliable host-visible proxy for toolchain
// support (__has_builtin for the amdgcn intrinsic is unreliable in the host/x86
// pass), and it is also a prerequisite of TDM_SUPPORTED, so it is factored out here.
// Without this host gate, a build whose device pass fell back to the no-op TDM stub
// would still report support on gfx1250 hardware and silently dispatch a no-op copy.
#if !TDM_PLATFORM_NV
// AMD: TDM builds only when the arch, the builtin, AND the D# descriptor header
// are all present, so an older toolchain degrades gracefully (see rationale above).
#  if __has_include(<hip/amd_detail/amd_gfx1250_TDM.h>)
#    define TDM_TOOLCHAIN_AVAILABLE 1
#  else
#    define TDM_TOOLCHAIN_AVAILABLE 0
#  endif
#  if defined(__gfx1250__) && \
      __has_builtin(__builtin_amdgcn_tensor_load_to_lds) && \
      TDM_TOOLCHAIN_AVAILABLE
                                  /* extend: || (defined(__gfxNNNN__) && ...) */
#    define TDM_BACKEND_AMD 1
#  else
#    define TDM_BACKEND_AMD 0
#  endif
#  define TDM_BACKEND_NV 0
#else
// NVIDIA: cp.async.bulk (TMA) is present on Hopper and newer. __CUDA_ARCH__ is
// only defined in the device pass, so this is 0 in the host pass (as intended;
// the host-side IsTdmCopySupported() check below uses the runtime instead).
#  define TDM_TOOLCHAIN_AVAILABLE 0
#  define TDM_BACKEND_AMD 0
#  if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#    define TDM_BACKEND_NV 1
#  else
#    define TDM_BACKEND_NV 0
#  endif
#endif

#define TDM_SUPPORTED (TDM_BACKEND_AMD || TDM_BACKEND_NV)

#if TDM_BACKEND_AMD
#  include <hip/amd_detail/amd_gfx1250_TDM.h>   // D# descriptor types for the target's TDM
#endif

#if TDM_SUPPORTED
#  define TDM_API      inline     // normal inline declaration (defined below)
#  define TDM_DELETED             //   ... and not deleted
#else
#  define TDM_API                 // no linkage keyword on a deleted declaration
#  define TDM_DELETED  = delete   // unsupported target: any call is a compile error
#endif

namespace tdm {

// ============================================================================
//  PUBLIC API
// ============================================================================

/// \brief Report whether TDM copies are usable. Always available (never deleted),
///        so it is safe to call on any target as a guard before the copy fns.
///
/// The answer differs by compilation pass, because TDM availability is a property
/// of the specific GPU arch:
/// - DEVICE code: returns the compile-time constant TDM_SUPPORTED for the arch
///   this device pass was built for. It is a constant expression, so it folds away
///   and can drive `if constexpr` / dead-code elimination of the copy calls.
/// - HOST code: there is no single compile-time answer (a build may target many
///   archs), so it queries the given device's architecture via the HIP runtime and
///   reports whether it is TDM-capable. Returns false if the query fails.
///
/// \param deviceId HIP device to query (host only; ignored in device code).
/// \return true if tdm:: copies will run on the target/device in question.
///
/// \code
///   // Host: pick an implementation before launching.
///   if (tdm::IsTdmCopySupported(dev)) launchTdmKernel(...);
///   else                              launchFallbackKernel(...);
/// \endcode
__host__ __device__ inline bool IsTdmCopySupported(int deviceId = 0) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    (void)deviceId;
    return TDM_SUPPORTED;                 // compile-time constant for this arch pass
#elif TDM_PLATFORM_NV
    // Host (NVIDIA): cp.async.bulk (TMA) requires compute capability 9.0+ (Hopper).
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess) return false;
    return prop.major >= 9;
#else
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, deviceId) != hipSuccess) return false;
    // gcnArchName looks like "gfx1250:sramecc+:xnack-"; match the arch prefix.
    // Keep this list in sync with the TDM_SUPPORTED arch condition above.
    const char* arch = prop.gcnArchName;
    const char* p    = "gfx1250";
    while (*p && *arch == *p) { ++arch; ++p; }
    // Require BOTH a TDM-capable arch AND a toolchain that actually built TDM
    // (otherwise the device pass emitted a no-op stub and enabling the path here
    // would silently produce wrong results).
    return TDM_TOOLCHAIN_AVAILABLE && (*p == '\0');
#endif
}

/// \brief Blocking block-collective copy of [0, sizeBytes): dst <- src.
///
/// Call from ALL threads of the block with identical arguments (memcpy-style
/// order). Work is partitioned across every warp in the block. On return, the
/// calling warp's TDM ops are complete.
///
/// \param dst            Destination in global memory (HBM).
/// \param src            Source in global memory (HBM).
/// \param sizeBytes      Number of bytes to copy.
/// \param ldsBuffer      Per-block LDS staging area (shared memory).
/// \param ldsBufferBytes Size of \p ldsBuffer in bytes; subdivided among warps.
///
/// \note For BLOCK-wide visibility, follow with __syncthreads() (and see the
///       visibility notes at the bottom of the file).
/// \warning \p src and \p dst must be GLOBAL (HBM) pointers. Passing a shared /
///          LDS pointer compiles (it decays to void*) but is undefined at run
///          time -- the address is programmed into the descriptor's global field.
__device__ TDM_API void tdmCopy(void* dst, const void* src, size_t sizeBytes,
                                void* ldsBuffer, size_t ldsBufferBytes) TDM_DELETED;

/// \brief Non-blocking variant of tdmCopy(): issue and return.
///
/// Identical partitioning to tdmCopy(), but does NOT drain on return. The last
/// few TDM ops (bounded by the per-wave queue depth) stay in flight so they
/// overlap with whatever the calling warp does next. Pair with tdmWait().
///
/// \see tdmCopy for parameter meanings.
/// \see tdmWait to complete the copy.
__device__ TDM_API void tdmCopyAsync(void* dst, const void* src, size_t sizeBytes,
                                     void* ldsBuffer, size_t ldsBufferBytes) TDM_DELETED;

/// \brief Blocking WARP-SPECIALIZED copy performed by one contiguous warp team.
///
/// Only warps in the half-open range [\p startWarpId, \p stopWarpId) participate;
/// all other warps return immediately and are free to do compute. Work and LDS
/// are partitioned by RANK WITHIN THE TEAM (warpId - startWarpId), so several
/// disjoint teams can each run a different copy concurrently, each with its own
/// dst/src and its own \p ldsBuffer region.
///
/// \param dst            Destination in global memory (HBM).
/// \param src            Source in global memory (HBM).
/// \param sizeBytes      Number of bytes to copy.
/// \param ldsBuffer      THIS team's LDS region (distinct per team).
/// \param ldsBufferBytes Size of this team's LDS region, split among its warps.
/// \param startWarpId    First warp of the team (inclusive).
/// \param stopWarpId     One past the last warp (clamped to nWarps; ~0u = end).
///
/// \note This does NOT synchronize the block -- you choose the barrier. A single
///       __syncthreads() after the copy/compute branches is enough for
///       independent compute; use named/arrive-wait barriers for a pipelined
///       producer/consumer so the copy team can run ahead.
/// \warning Give each team (and the compute warps) a NON-OVERLAPPING LDS region;
///          the library trusts the pointer/size you pass.
///
/// \code
///   const uint32_t warpId = threadIdx.x / warpSize;   // (1D block)
///   if (warpId < COPY_WARPS)
///       tdm::tdmCopyByTeam(dst, src, n, staging, teamLdsBytes, 0u, COPY_WARPS);
///   else
///       /* compute -- use LDS past the team's window */;
///   __syncthreads();
/// \endcode
__device__ TDM_API void tdmCopyByTeam(void* dst, const void* src, size_t sizeBytes,
                                      void* ldsBuffer, size_t ldsBufferBytes,
                                      uint32_t startWarpId, uint32_t stopWarpId) TDM_DELETED;

/// \brief Non-blocking variant of tdmCopyByTeam(): issue and return.
/// \see tdmCopyByTeam for parameter meanings and team semantics.
/// \see tdmWait to complete the copy (called by each participating warp).
__device__ TDM_API void tdmCopyAsyncByTeam(void* dst, const void* src, size_t sizeBytes,
                                           void* ldsBuffer, size_t ldsBufferBytes,
                                           uint32_t startWarpId, uint32_t stopWarpId) TDM_DELETED;

/// \brief Drain the CALLING WARP's outstanding TDM ops (TENSORcnt -> 0).
///
/// TENSORcnt is a per-wave counter, so this waits only on the ops this warp
/// issued -- nothing else.
///
/// \note Multiple teams need no special handling: each participating warp calls
///       tdmWait() to drain its own ops, and one team's wait has no effect on
///       another's (there is no shared counter). Every ISSUING warp must call it
///       -- a warp cannot drain its teammates' ops. tdmCopy*()/*ByTeam() blocking
///       forms already do this internally.
/// \note For BLOCK-wide visibility follow with __syncthreads(); if copied data is
///       consumed within the block AND the vector head ran, also
///       __threadfence_block() so those ordinary global stores are observed.
__device__ TDM_API void tdmWait() TDM_DELETED;

} // namespace tdm

#undef TDM_API
#undef TDM_DELETED


// ############################################################################
// #                                                                          #
// #                         IMPLEMENTATION NOTES                             #
// #                                                                          #
// ############################################################################
//
// LAYOUT OF A COPY
//   [ head (VECTOR) ][ ---- aligned 256B rows (TDM) ---- ][ tail (TDM 1-D) ]
//   * Fixed choices for bandwidth: 4-byte data_size, 256B TDM row width.
//   * `head` brings the SOURCE up to a 128B boundary (the direct-copy
//     requirement); being the unaligned remainder, it stays a cooperative
//     vector copy done by the team's first warp.
//   * The aligned bulk is 2D TDM tiles (64 dwords x N rows of 256B).
//   * `tail` (< 256B sub-row remainder) is a separate 1-D TDM op at byte
//     granularity. TDM's out-of-bounds clamp is per-dimension (rectangular) and
//     cannot express "N full rows + a partial row", so the partial row must be
//     its own tile rather than riding the bulk descriptor's clamp.
//
// HARDWARE MODEL (why the per-tile waits are REQUIRED)
//   * TDM engines: 1 per SIMD-pair -> 2 per WGP; a warp runs on 1 SIMD32 and
//     shares its pair's engine. Bandwidth needs >=2 issuing warps/block (both
//     engines) and many blocks (many WGPs); a lone warp is latency-bound.
//   * Same-wave TDM ops ISSUE in order but their memory effects OVERLAP: up to 3
//     ops are outstanding per wave (that is what TENSORcnt counts). In-order issue
//     does NOT serialize completion, so it does NOT make LDS reuse hazard-free.
//   * This copy is single-buffered (one LDS window per warp), so each tile is a
//     load->store dependency chain on that window: the store reads what the load
//     wrote (RAW), and the next tile's load overwrites the window the store is
//     still draining (WAR). Both edges need an s_wait_tensorcnt, so issueRows()/
//     issueRow1d() wait after the load and after the store. Consequence: with a
//     single window the copy is effectively serialized; tdmCopyAsync() therefore
//     overlaps very little today. Regaining overlap needs DOUBLE-BUFFERING (>=2
//     LDS windows per warp) so a load into window B runs while window A stores.
//
// VISIBILITY
//   tdmWait() drains only the calling wave's TDM ops. Cross-warp / block-wide
//   visibility is the caller's __syncthreads(); the vector head's ordinary
//   global stores additionally want __threadfence_block() if consumed within the
//   block. Host-after-kernel and grid dependencies are handled by the stream.
//
// BUILTINS: verify names against your tree:
//   grep -iE 'tensor_(load|store)|tensorcnt' \
//        <llvm>/clang/include/clang/Basic/BuiltinsAMDGPU.def
// ============================================================================

namespace tdm {

#if TDM_SUPPORTED            // ===== real TDM implementation ================

#if TDM_BACKEND_AMD          // ----- AMD Tensor Data Mover (gfx1250) --------

namespace detail {

constexpr uint32_t WIDTH  = 256;          // bytes per TDM row (first tile dim)
constexpr uint32_t ELT    = 4;            // dword
constexpr uint32_t DS4    = 2;            // data_size code for 4-byte
constexpr uint32_t DS1    = 0;            // data_size code for 1-byte
constexpr uint32_t TD0    = WIDTH / ELT;  // 64 elements per row
constexpr uint32_t RWIDTH = 512;
constexpr uint32_t MAX_SRCS = 16;         // max sources reduced in one call
constexpr uint32_t MAX_DSTS = 16;         // max destinations broadcast to

constexpr uint8_t  MEMSET_CHAR = 75;
constexpr uint32_t MEMSET_WORD = 0x4B4B4B4Bu;
constexpr float    MEMSET_VAL  = 13323083.0f;

// Packed-float memset value (mirrors TransferBench.hpp's MemsetVal<T>()).
template <typename T> __device__ __forceinline__ T      MemsetVal();
template <>           __device__ __forceinline__ float  MemsetVal() { return MEMSET_VAL; }
template <>           __device__ __forceinline__ float2 MemsetVal() { return make_float2(MEMSET_VAL, MEMSET_VAL); }
template <>           __device__ __forceinline__ float4 MemsetVal() { return make_float4(MEMSET_VAL, MEMSET_VAL,
                                                                                         MEMSET_VAL, MEMSET_VAL); }

// ---- instruction emission (the only arch-specific piece) -------------------
// The tensor DMA is a single builtin taking the FULL descriptor: five register
// groups plus a constant cache policy. Per the clang reference
// (https://clang.llvm.org/docs/AMDGPUBuiltinReference.html):
//   void __builtin_amdgcn_tensor_load_to_lds  (v4u32 D0, v8i32 D1, v4i32 D2,
//                                              v4i32 D3, v8i32 D4, int cpol);
//   void __builtin_amdgcn_tensor_store_from_lds(<same signature>);
// D0=GROUP0 (addresses), D1=GROUP1 (2D shape). D2/D3/D4 carry the higher tensor
// dimensions; for a <=2D copy they are simply ZERO vectors ("unused"). This
// mirrors known-good example usage, which passes the group m_bitfields straight
// through (no signed cast) and zero raw vectors for the unused higher dims -- so
// we depend only on GROUP0/GROUP1 existing, not on GROUP2/3/4 by name.
using u32x4 = __attribute__((ext_vector_type(4))) uint32_t;   // D0, D2, D3
using u32x8 = __attribute__((ext_vector_type(8))) uint32_t;   // D1, D4

__device__ inline void load(const gfx1250_TDM_GROUP0& g0,
                            const gfx1250_TDM_GROUP1& g1) {
    __builtin_amdgcn_tensor_load_to_lds(
        g0.m_bitfield,               // D0  addresses
        g1.m_bitfield,               // D1  2D shape
        u32x4{}, u32x4{}, u32x8{},   // D2/D3/D4  higher dims: unused (zero)
        /*cpol=*/0);
}
__device__ inline void store(const gfx1250_TDM_GROUP0& g0,
                             const gfx1250_TDM_GROUP1& g1) {
    __builtin_amdgcn_tensor_store_from_lds(
        g0.m_bitfield,
        g1.m_bitfield,
        u32x4{}, u32x4{}, u32x8{},
        /*cpol=*/0);
}
__device__ inline void waitTensor0() { __builtin_amdgcn_s_wait_tensorcnt(0); }

// ---- cooperative vector copy of a small byte range, by one warp ------------
// All threads of the calling warp participate. Dword-wide where possible; the
// (<4 byte) ragged end is finished by the warp's first thread.
__device__ inline void warpVecCopy(const uint8_t* s, uint8_t* d, size_t n,
                                   uint32_t warpThread, uint32_t warpThreads) {
    size_t nd = n >> 2;
    const uint32_t* s32 = reinterpret_cast<const uint32_t*>(s);
    uint32_t*       d32 = reinterpret_cast<uint32_t*>(d);
    for (size_t i = warpThread; i < nd; i += warpThreads) d32[i] = s32[i];
    uint32_t rem = static_cast<uint32_t>(n & 3u);
    if (rem && warpThread == 0)
        for (uint32_t b = 0; b < rem; ++b) d[nd * 4 + b] = s[nd * 4 + b];
}

// ---- issue one chunk of whole 256B rows (2D tile) through ONE LDS window. ---
// This staging window is single-buffered, so the two TDM ops form a dependency
// chain that MUST be enforced with TENSORcnt waits -- up to 3 TDM ops are
// outstanding per wave (they overlap), so "same-wave in-order issue" does NOT
// serialize their memory effects:
//   * load -> wait: the store reads the LDS the load just wrote (RAW hazard).
//   * store -> wait: the caller reuses this same window next iteration; the next
//     load must not overwrite LDS the store is still draining (WAR hazard).
__device__ inline void issueRows(uint64_t src, uint32_t lds, uint64_t dst,
                                 uint32_t rows) {
    gfx1250_TDM_GROUP1 g1;
    g1.dataSize(DS4);
    g1.tileDim0(TD0);   g1.tileDim1(rows);
    g1.tensorDim0(TD0); g1.tensorDim1(rows);
    g1.tensorDim0Stride(TD0);                // rows back-to-back (contiguous)

    // Higher dims (D2/D3/D4) are unused for this 2D tile -> passed as zero inside
    // load()/store(), matching known-good example usage.
    gfx1250_TDM_GROUP0 g0l(lds, src);
    load(g0l, g1);   waitTensor0();          // RAW: fill LDS before store reads it
    gfx1250_TDM_GROUP0 g0s(lds, dst);
    store(g0s, g1);  waitTensor0();          // WAR: drain store before window reuse
}

// ---- issue a sub-row tail (<256B) as a 1-D tile at BYTE granularity. ---------
// Same single-buffered LDS window and the same required RAW/WAR waits as above.
__device__ inline void issueRow1d(uint64_t src, uint32_t lds, uint64_t dst,
                                  uint32_t nbytes) {
    gfx1250_TDM_GROUP1 g1;
    g1.dataSize(DS1);                        // 1-byte elements: exact length
    g1.tileDim0(nbytes);   g1.tileDim1(1);
    g1.tensorDim0(nbytes); g1.tensorDim1(1);
    g1.tensorDim0Stride(nbytes);

    gfx1250_TDM_GROUP0 g0l(lds, src);        // unused higher dims -> zero (see load())
    load(g0l, g1);   waitTensor0();          // RAW: fill LDS before store reads it
    gfx1250_TDM_GROUP0 g0s(lds, dst);
    store(g0s, g1);  waitTensor0();          // WAR: drain store before window reuse
}

// ---- core: partition + issue the whole copy for the team [start, stop). -----
// NO final wait. Work + LDS partition by rank within the team (warpId - start),
// so each team indexes its own `ldsBuffer` from zero. Safe to call collectively
// (warps outside the range return immediately) or only from the team's warps.
__device__ inline void issue(void* dst, const void* src, size_t sizeBytes,
                             void* ldsBuffer, size_t ldsBufferBytes,
                             uint32_t startWarpId, uint32_t stopWarpId) {
    const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
    uint8_t*       d = reinterpret_cast<uint8_t*>(dst);
    const uint32_t ldsBase  = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ldsBuffer));
    const uint32_t ldsBytes = static_cast<uint32_t>(ldsBufferBytes);  // LDS is small

    const uint32_t W          = warpSize;
    const uint32_t nThreads   = blockDim.x * blockDim.y * blockDim.z;
    const uint32_t tid        = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x
                                + threadIdx.x;
    const uint32_t warpThread = tid % W;               // thread index within its warp
    const uint32_t warpId     = tid / W;
    const uint32_t nWarps     = (nThreads + W - 1) / W;

    // --- team membership: this warp participates iff in [start, stop) --------
    const uint32_t teamStop = (stopWarpId > nWarps) ? nWarps : stopWarpId;
    if (startWarpId >= teamStop || warpId < startWarpId || warpId >= teamStop)
        return;                                        // not on this team
    const uint32_t rank      = warpId - startWarpId;    // rank within the team
    const uint32_t teamWarps = teamStop - startWarpId;  // >= 1

    // active threads in THIS warp (handles partial final warp); stride for vector.
    const uint32_t warpThreads = (nThreads - warpId * W < W) ? (nThreads - warpId * W) : W;

    // --- split the range: [head][ aligned 256B rows ][tail] ------------------
    uint64_t sAddr = reinterpret_cast<uint64_t>(s);
    uint32_t head  = static_cast<uint32_t>((128u - (sAddr & 127u)) & 127u);
    if (head > sizeBytes) head = static_cast<uint32_t>(sizeBytes);
    size_t   bulk     = sizeBytes - head;              // starts 128B-aligned
    size_t   rows     = bulk / WIDTH;                  // whole 256B rows
    size_t   tdmBytes = rows * WIDTH;
    size_t   tailOff  = head + tdmBytes;
    uint32_t tail     = static_cast<uint32_t>(sizeBytes - tailOff);   // < 256B

    // --- edges (team's FIRST warp = rank 0): vector head, TDM tail -----------
    if (rank == 0 && head) warpVecCopy(s, d, head, warpThread, warpThreads);
    if (rank == 0 && tail) {
        if (ldsBytes >= tail)                          // stage tail in rank 0's window
            issueRow1d(reinterpret_cast<uint64_t>(s + tailOff), ldsBase,
                       reinterpret_cast<uint64_t>(d + tailOff), tail);
        else
            warpVecCopy(s + tailOff, d + tailOff, tail, warpThread, warpThreads);
    }

    // --- aligned bulk via TDM ------------------------------------------------
    if (rows == 0) return;                             // no aligned bulk (edges done)
    uint32_t maxByLds = ldsBytes / WIDTH;              // #warps we can give a window
    if (maxByLds == 0) {                               // LDS < 256B: vector fallback
        if (rank == 0)
            warpVecCopy(s + head, d + head, tdmBytes, warpThread, warpThreads);
        return;
    }
    uint32_t issuers = teamWarps < maxByLds ? teamWarps : maxByLds;
    uint32_t window  = (ldsBytes / issuers) & ~(WIDTH - 1);   // per-warp 256B-multiple
    uint32_t rowsPerChunk = window / WIDTH;

    if (rank >= issuers) return;                       // this warp doesn't issue

    // distribute `rows` across issuers by team rank (contiguous row blocks)
    size_t base    = rows / issuers;
    size_t extra   = rows % issuers;
    size_t myRows  = base + (rank < extra ? 1u : 0u);
    size_t myStart = rank * base + (rank < extra ? rank : extra);
    if (myRows == 0) return;

    uint32_t myLds = ldsBase + rank * window;
    uint64_t sBase = reinterpret_cast<uint64_t>(s + head) + (uint64_t)myStart * WIDTH;
    uint64_t dBase = reinterpret_cast<uint64_t>(d + head) + (uint64_t)myStart * WIDTH;

    for (size_t r = 0; r < myRows; r += rowsPerChunk) {
        uint32_t chunkRows = (myRows - r < rowsPerChunk)
                             ? static_cast<uint32_t>(myRows - r) : rowsPerChunk;
        uint64_t off = (uint64_t)r * WIDTH;
        issueRows(sBase + off, myLds, dBase + off, chunkRows);
    }
}

// ############################################################################
// #        MULTI-SOURCE REDUCE / MULTI-DEST BROADCAST VARIANTS               #
// ############################################################################
// Reduce (element-wise sum) numSrcs sources and broadcast the result to numDsts
// destinations. numSrcs == 1 && numDsts == 1 degenerates to a plain copy. These
// are overloads of the single-copy helpers above; the reduce entry point routes
// here via detail::issue(dsts, srcs, numSrcs, numDsts, ...).

// ---- optimization guard for the multi-source reduce path --------------------
// The gfx1250 tensor load/store builtins are memory(inaccessiblemem): the backend
// does NOT model them as touching the LDS that the vector (ds) reduce reads/writes.
// At -O2/-O3 the scheduler/inliner exploits that missing dependency and miscompiles
// the reduce (drops the tensor store, or stores stale LDS) in ways that flip with
// unrelated codegen changes (inlining, register pressure). The generated code is
// correct at -O0/-O1, so the whole reduce path is pinned to no-optimization; the
// scalar overhead here is negligible (bandwidth comes from the TDM engine / ds
// pipe), and it keeps the result deterministically correct regardless of the TU's
// optimization level. Remove once the backend models the tensor<->LDS dependency.
#define TDM_REDUCE_NOOPT __attribute__((optnone, noinline))

// ---- vector fallback: reduce (sum) numSrcs sources, broadcast to numDsts. ----
// Element-wise sum of all srcs -> written to every dst. numSrcs == 1 is a plain
// copy; numDsts > 1 broadcasts the same reduced result to each destination.
// `srcs`/`dsts` hold base addresses; `offset` (in bytes) reaches the sub-range
// to copy (e.g. the tail start), so callers can share one base pointer array.
template <typename PACKED_FLOAT = float>
TDM_REDUCE_NOOPT __device__ void warpVecCopy(uint64_t* srcs, uint64_t* dsts,
                                   uint32_t numSrcs, uint32_t numDsts,
                                   size_t n, size_t offset,
                                   uint32_t warpThread, uint32_t warpThreads) {
    // Bulk: reduce whole PACKED_FLOAT elements.
    size_t np = n / sizeof(PACKED_FLOAT);
    for (size_t i = warpThread; i < np; i += warpThreads) {
        PACKED_FLOAT acc = numSrcs ? PACKED_FLOAT{} : MemsetVal<PACKED_FLOAT>();
        for (uint32_t s = 0; s < numSrcs; ++s)
            acc += reinterpret_cast<const PACKED_FLOAT*>(srcs[s] + offset)[i];
        for (uint32_t d = 0; d < numDsts; ++d)
            reinterpret_cast<PACKED_FLOAT*>(dsts[d] + offset)[i] = acc;
    }
    // Remaining whole floats (n not a PACKED_FLOAT multiple); empty when
    // PACKED_FLOAT == float.
    size_t nf     = n >> 2;
    size_t fStart = (np * sizeof(PACKED_FLOAT)) >> 2;
    for (size_t i = fStart + warpThread; i < nf; i += warpThreads) {
        float acc = numSrcs ? 0.f : MemsetVal<float>();
        for (uint32_t s = 0; s < numSrcs; ++s)
            acc += reinterpret_cast<const float*>(srcs[s] + offset)[i];
        for (uint32_t d = 0; d < numDsts; ++d)
            reinterpret_cast<float*>(dsts[d] + offset)[i] = acc;
    }
    // Ragged sub-float tail (< 4 bytes).
    uint32_t rem = static_cast<uint32_t>(n & 3u);
    if (rem && warpThread == 0) {
        for (uint32_t b = 0; b < rem; ++b) {
            uint8_t acc = numSrcs ? 0 : MEMSET_CHAR;
            for (uint32_t s = 0; s < numSrcs; ++s)
                acc += reinterpret_cast<const uint8_t*>(srcs[s] + offset)[nf * 4 + b];
            for (uint32_t d = 0; d < numDsts; ++d)
                reinterpret_cast<uint8_t*>(dsts[d] + offset)[nf * 4 + b] = acc;
        }
    }
}

// ---- LDS (address space 3) pointer helper --------------------------------
// `val`/`tmp` are 32-bit LDS byte offsets (exactly what the TDM load/store
// builtins consume). To touch that staging memory with ordinary scalar
// loads/stores we must build a pointer TAGGED as LDS (address space 3). A plain
// reinterpret_cast<T*> yields a generic/flat pointer whose numeric value lands
// in the GLOBAL aperture (an LDS offset like 0x19000 is a valid global VA), so
// dereferencing it faults. Casting to an address_space(3) pointer makes the
// compiler emit ds_* (LDS) accesses against the offset instead.
// ---- LDS pointer from the real base pointer + absolute byte offset --------
// `val`/`tmp` are ABSOLUTE LDS byte offsets (ldsBase + per-warp shift), exactly
// what the TDM load/store builtins consume. To touch that staging memory with
// ordinary scalar loads/stores we must use a pointer that carries the correct
// LDS aperture. Building one from a bare integer offset does NOT (it resolves to
// the GLOBAL aperture and reads/writes don't alias the tensor-written LDS);
// instead we offset from `ldsMem`, the real generic pointer to this block's LDS.
template <typename T>
__device__ inline T* ldsPtr(void* ldsMem, uint32_t absOff) {
    uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ldsMem));
    return reinterpret_cast<T*>(static_cast<char*>(ldsMem) + (absOff - base));
}

// ---- explicit LDS (address_space(3)) pointer from a segment byte offset -------
// The generic `ldsPtr` above resolves fine, but marking the result `volatile`
// (needed to stop the compiler DCE'ing the reduce RMW, since the tensor store
// intrinsic is not modeled as an LDS reader) blocks address-space inference and
// forces `flat_*` lowering. `flat_*` routes through the small FLAT LDS aperture,
// so large per-warp offsets fault (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION).
// An explicit address_space(3) pointer always lowers to `ds_*`, which uses a
// 32-bit LDS offset and reaches the full allocated LDS. `absOff` is already the
// 0-based LDS segment offset (low 32 bits of the block's flat LDS address).
template <typename T> using LdsPtrT = T __attribute__((address_space(3)))*;
template <typename T>
__device__ inline LdsPtrT<T> ldsPtr3(uint32_t absOff) {
    return reinterpret_cast<LdsPtrT<T>>(static_cast<size_t>(absOff));
}

// ---- LDS visibility fence between the tensor and vector memory pipes -------
// s_wait_tensorcnt only orders tensor-op vs tensor-op, so it makes the pure copy
// path (load->wait->store) correct but does NOT make the tensor engine's LDS
// writes visible to this wave's vector (ds) reads, nor the ds writes visible to a
// following tensor store. The reduce path interleaves ds reads/writes between the
// tensor load and store, so it needs an explicit workgroup-scope fence at those
// boundaries. (A __syncthreads() barrier is unusable here: warps with
// rank >= issuers or myRows == 0 return early and would deadlock it.)
__device__ __forceinline__ void ldsFence() {
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
}

// ---- issue one chunk of whole 256B rows (2D tile) through ONE LDS window. ---
// Same single-buffered LDS window / RAW+WAR waits as the single-copy issueRows().
// The first source lands in `val`; each subsequent source lands in `tmp` and is
// accumulated into `val`; the reduced `val` is stored to every destination.
template <typename PACKED_FLOAT>
TDM_REDUCE_NOOPT __device__ void issueRows(uint64_t* srcs, uint64_t* dsts,
                                 uint32_t numSrcs, uint32_t numDsts,
                                 uint32_t val, uint32_t tmp, void* ldsMem, uint32_t rows,
                                 uint32_t warpThread, uint32_t off = 0) {
    gfx1250_TDM_GROUP1 g1;
    g1.dataSize(DS4);
    g1.tileDim0(TD0);   g1.tileDim1(rows);
    g1.tensorDim0(TD0); g1.tensorDim1(rows);
    g1.tensorDim0Stride(TD0);                // rows back-to-back (contiguous)

    uint32_t nElems = (rows * WIDTH) / sizeof(PACKED_FLOAT);
    if (numSrcs) {
      // Seed the running-sum window `val` with src0.
      gfx1250_TDM_GROUP0 g0l(val, srcs[0] + off);
      load(g0l, g1);
      waitTensor0();   ldsFence();                       // RAW: src0 LDS write visible to vp reads
      LdsPtrT<volatile PACKED_FLOAT> vp = ldsPtr3<volatile PACKED_FLOAT>(val);
      for (size_t s = 1; s < numSrcs; s++) {
        gfx1250_TDM_GROUP0 g0lt(tmp, srcs[s] + off);      // stage next source in `tmp`
        load(g0lt, g1);
        waitTensor0();   ldsFence();                     // RAW: srcS LDS write visible to tp reads
        LdsPtrT<const volatile PACKED_FLOAT> tp = ldsPtr3<const volatile PACKED_FLOAT>(tmp);
        for (uint32_t u = warpThread; u < nElems; u += warpSize)
          vp[u] = vp[u] + tp[u];
        ldsFence();                                      // WAR: accumulate done before next reload of tmp
      }
    } else {
      // Empty source: fill this warp's reduce window with the MEMSET byte pattern
      // so the TDM store writes a memset() result to each dst (mirrors
      // GpuReduceKernel's numSrcs==0 path). No load is issued.
      LdsPtrT<volatile uint32_t> vp = ldsPtr3<volatile uint32_t>(val);
      uint32_t  nWords = (rows * WIDTH) / sizeof(uint32_t);
      for (uint32_t u = warpThread; u < nWords; u += warpSize) vp[u] = MEMSET_WORD;
    }

    ldsFence();  // ds writes (reduced/memset result) visible to the tensor store
    for (size_t d = 0; d < numDsts; d++) {
      gfx1250_TDM_GROUP0 g0s(val, dsts[d] + off);  // broadcast reduced result to each dst
      store(g0s, g1);  waitTensor0();              // WAR: drain store before window reuse / next dst
    }
}

// ---- core: partition + issue a reduce/broadcast for the team [start, stop). -
template <typename PACKED_FLOAT = float>
TDM_REDUCE_NOOPT __device__ void issue(void** dsts, const void** srcs, uint32_t numSrcs, uint32_t numDsts,
                             size_t sizeBytes, void* ldsBuffer, size_t ldsBufferBytes,
                             uint32_t startWarpId, uint32_t stopWarpId) {
    const uint32_t ldsBase  = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ldsBuffer));
    const uint32_t ldsBytes = static_cast<uint32_t>(ldsBufferBytes);  // LDS is small

    const uint32_t W          = warpSize;
    const uint32_t nThreads   = blockDim.x * blockDim.y * blockDim.z;
    const uint32_t tid        = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x
                                + threadIdx.x;
    const uint32_t warpThread = tid % W;               // thread index within its warp
    const uint32_t warpId     = tid / W;
    const uint32_t nWarps     = (nThreads + W - 1) / W;

    // --- team membership: this warp participates iff in [start, stop) --------
    const uint32_t teamStop = (stopWarpId > nWarps) ? nWarps : stopWarpId;
    if (startWarpId >= teamStop || warpId < startWarpId || warpId >= teamStop)
        return;                                        // not on this team
    const uint32_t rank      = warpId - startWarpId;    // rank within the team
    const uint32_t teamWarps = teamStop - startWarpId;  // >= 1

    // active threads in THIS warp (handles partial final warp); stride for vector.
    const uint32_t warpThreads = (nThreads - warpId * W < W) ? (nThreads - warpId * W) : W;

    // --- split the range: [256B rows ][tail] ------------------
    size_t   rows     = sizeBytes / WIDTH;                  // whole 256B rows
    size_t   tail     = sizeBytes % WIDTH;
    size_t   tdmBytes = rows * WIDTH;

    // --- base src/dst addresses for this team (byte offset applied per use) --
    uint64_t mySrcs[MAX_SRCS];
    uint64_t myDsts[MAX_DSTS];
    for (uint32_t i = 0; i < numSrcs; i++) mySrcs[i] = (uint64_t)srcs[i];
    for (uint32_t i = 0; i < numDsts; i++) myDsts[i] = (uint64_t)dsts[i];

    // --- LDS reduce buffers: val = running sum, tmp = staging for extra srcs --
    // Base offsets (rank 0); the TDM path shifts each warp by rank*window below.
    uint32_t val = ldsBase;
    uint32_t tmp = ldsBase + WIDTH;

    // --- edge (team's FIRST warp = rank 0): sub-256B tail via VECTOR reduce ----
    // The tail is < 256B (negligible for bandwidth), so it is reduced with plain
    // global vector loads/stores rather than a 1-D TDM tile. Routing the tail
    // through the tensor builtins needs a standalone (non-inlined) helper, and the
    // current gfx1250 backend miscompiles the ds-reduce -> inaccessiblemem tensor
    // store there (store dropped / stale LDS); inlining it instead bloats issue()
    // and corrupts the bulk path. A vector tail sidesteps both.
    if (rank == 0 && tail)
        warpVecCopy<PACKED_FLOAT>(mySrcs, myDsts, numSrcs, numDsts, tail, tdmBytes,
                    warpThread, warpThreads);

    // --- 256B rows copy via TDM ------------------------------------------------
    uint32_t maxByLds = ldsBytes / RWIDTH;             // #warps we can give a window, 2*WIDTH because of double buffering
    if (maxByLds == 0) {                               // LDS < 512B: vector fallback
        if (rank == 0)
            warpVecCopy<PACKED_FLOAT>(mySrcs, myDsts, numSrcs, numDsts, tdmBytes, 0,
                        warpThread, warpThreads);
        return;
    }
    uint32_t issuers = teamWarps < maxByLds ? teamWarps : maxByLds;
    uint32_t window  = (ldsBytes / issuers) & ~(RWIDTH - 1);   // per-warp 512B-multiple
    uint32_t rowsPerChunk = window / RWIDTH;

    if (rank >= issuers) return;                       // this warp doesn't issue

    // distribute `rows` across issuers by team rank (contiguous row blocks)
    size_t base    = rows / issuers;
    size_t extra   = rows % issuers;
    size_t myRows  = base + (rank < extra ? 1u : 0u);
    // TODO:if last warp, take the remaining edges
    size_t myStart = rank * base + (rank < extra ? rank : extra);
    if (myRows == 0) return;

    // Give this warp its own window and split it evenly into two halves:
    //   val = running sum (first half), tmp = staging for extra srcs (second half).
    // window is a multiple of RWIDTH (= 2*WIDTH), so window/2 is a WIDTH-multiple
    // large enough to hold rowsPerChunk (= window/RWIDTH) rows in each half.
    val = ldsBase + rank * window;
    tmp = val + window / 2;

    for (size_t r = 0; r < myRows; r += rowsPerChunk) {
        uint32_t chunkRows = (myRows - r < rowsPerChunk)
                             ? static_cast<uint32_t>(myRows - r) : rowsPerChunk;
        uint64_t off = (uint64_t)(myStart + r) * WIDTH;   // this warp's row block
        issueRows<PACKED_FLOAT>(mySrcs, myDsts, numSrcs, numDsts, val, tmp, ldsBuffer, chunkRows, warpThread, off);
    }
}

} // namespace detail

#elif TDM_BACKEND_NV         // ----- NVIDIA cp.async.bulk / TMA (sm_90+) -----

// The AMD backend expresses the aligned bulk as a 2D tensor tile (256B rows).
// cp.async.bulk is a FLAT 1-D byte copy, so the 2D descriptor machinery is gone:
// we copy contiguous 16B-aligned byte ranges through the same single-buffered LDS
// staging window. The load/store completion model also differs from AMD's single
// TENSORcnt: the global->shared load is tracked by an mbarrier (transaction bytes)
// and the shared->global store by a bulk async-group (commit + wait). The
// partition/team logic mirrors the AMD path so the public API is identical.
namespace detail {

namespace ptx = cuda::ptx;

constexpr uint32_t ALIGN = 16;           // cp.async.bulk addr/size granularity

// ---- cooperative vector copy of a small byte range, by one warp (== AMD) ----
__device__ inline void warpVecCopy(const uint8_t* s, uint8_t* d, size_t n,
                                   uint32_t warpThread, uint32_t warpThreads) {
    size_t nd = n >> 2;
    const uint32_t* s32 = reinterpret_cast<const uint32_t*>(s);
    uint32_t*       d32 = reinterpret_cast<uint32_t*>(d);
    for (size_t i = warpThread; i < nd; i += warpThreads) d32[i] = s32[i];
    uint32_t rem = static_cast<uint32_t>(n & 3u);
    if (rem && warpThread == 0)
        for (uint32_t b = 0; b < rem; ++b) d[nd * 4 + b] = s[nd * 4 + b];
}

// Drain the CALLING THREAD's outstanding bulk-store groups (the WAR/async edge).
// Named waitTensor0() so the shared public-API wrappers below are backend-agnostic.
__device__ inline void waitTensor0() {
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>{});
}

// ---- issue one contiguous chunk (nbytes, a 16B multiple) through ONE window --
// Single-buffered, so the same RAW/WAR edges as the AMD path apply, just with the
// cp.async.bulk completion mechanisms:
//   * G2S load  -> mbarrier wait      (RAW: store must see the filled LDS window)
//   * S2G store -> bulk-group wait     (WAR: next load must not clobber a window
//                                       whose store is still draining)
// Issued by the warp's leader thread only (bulk copies are single-thread ops).
__device__ inline void issueChunk(const uint8_t* src, void* lds, uint8_t* dst,
                                  uint32_t nbytes, uint64_t* bar, uint32_t& phase) {
    // G2S: arrive + expect nbytes, launch the async copy, then wait for the
    // mbarrier phase to flip (the copy deposits its tx-bytes on completion).
    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta,
                                   ptx::space_shared, bar, nbytes);
    ptx::cp_async_bulk(ptx::space_cluster, ptx::space_global, lds, src, nbytes, bar);
    while (!ptx::mbarrier_try_wait_parity(bar, phase)) {}
    phase ^= 1u;

    // Order the async-proxy LDS writes before the async-proxy store reads them.
    ptx::fence_proxy_async(ptx::space_shared);

    // S2G: launch the store, commit the bulk group, and drain before window reuse.
    ptx::cp_async_bulk(ptx::space_global, ptx::space_shared, dst, lds, nbytes);
    ptx::cp_async_bulk_commit_group();
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>{});
}

// ---- core: partition + issue the whole copy for the team [start, stop). ------
// Same partitioning contract as the AMD issue(): NO final wait beyond the
// per-chunk drains, work + LDS split by rank within the team.
__device__ inline void issue(void* dst, const void* src, size_t sizeBytes,
                             void* ldsBuffer, size_t ldsBufferBytes,
                             uint32_t startWarpId, uint32_t stopWarpId) {
    const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
    uint8_t*       d = reinterpret_cast<uint8_t*>(dst);
    const uint32_t ldsBytes = static_cast<uint32_t>(ldsBufferBytes);  // LDS is small

    const uint32_t W          = warpSize;
    const uint32_t nThreads   = blockDim.x * blockDim.y * blockDim.z;
    const uint32_t tid        = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x
                                + threadIdx.x;
    const uint32_t warpThread = tid % W;
    const uint32_t warpId     = tid / W;
    const uint32_t nWarps     = (nThreads + W - 1) / W;

    // --- team membership: this warp participates iff in [start, stop) --------
    const uint32_t teamStop = (stopWarpId > nWarps) ? nWarps : stopWarpId;
    if (startWarpId >= teamStop || warpId < startWarpId || warpId >= teamStop)
        return;
    const uint32_t rank      = warpId - startWarpId;
    const uint32_t teamWarps = teamStop - startWarpId;
    const uint32_t warpThreads = (nThreads - warpId * W < W) ? (nThreads - warpId * W) : W;
    const bool     leader    = (warpThread == 0);

    // Carve per-team mbarriers (one per warp in the team, indexed by rank) from the
    // FRONT of this team's ldsBuffer; the staging windows use the remainder. Keeping
    // the barriers in the passed-in dynamic LDS avoids any static __shared__, which
    // would otherwise push a full-size dynamic allocation past the per-block cap and
    // make the launch fail with "invalid argument".
    uint8_t*  ldsBase   = reinterpret_cast<uint8_t*>(ldsBuffer);
    uint32_t  barRegion = ((teamWarps * static_cast<uint32_t>(sizeof(uint64_t)))
                           + (ALIGN - 1)) & ~(ALIGN - 1);
    uint64_t* bars      = reinterpret_cast<uint64_t*>(ldsBase);
    uint8_t*  winBase   = ldsBase + barRegion;
    uint32_t  winBytes  = (ldsBytes > barRegion) ? (ldsBytes - barRegion) : 0u;

    // --- split the range: [head][ 16B-aligned bulk ][tail] -------------------
    // cp.async.bulk requires BOTH src and dst 16B-aligned. A single head can only
    // align both if they share the same 16B phase; if they don't, there is no
    // valid bulk split, so fall back to a pure cooperative vector copy.
    uint64_t sAddr = reinterpret_cast<uint64_t>(s);
    uint64_t dAddr = reinterpret_cast<uint64_t>(d);
    if (((sAddr ^ dAddr) & (ALIGN - 1)) != 0) {
        if (rank == 0) warpVecCopy(s, d, sizeBytes, warpThread, warpThreads);
        return;
    }
    uint32_t head = static_cast<uint32_t>((ALIGN - (sAddr & (ALIGN - 1))) & (ALIGN - 1));
    if (head > sizeBytes) head = static_cast<uint32_t>(sizeBytes);
    size_t   rest    = sizeBytes - head;
    size_t   bulk    = rest & ~static_cast<size_t>(ALIGN - 1);   // whole 16B units
    uint32_t tail    = static_cast<uint32_t>(rest - bulk);       // < 16B remainder
    size_t   tailOff = head + bulk;

    // --- edges (team's FIRST warp = rank 0): vector head and tail ------------
    if (rank == 0 && head) warpVecCopy(s, d, head, warpThread, warpThreads);
    if (rank == 0 && tail) warpVecCopy(s + tailOff, d + tailOff, tail,
                                       warpThread, warpThreads);

    // --- aligned bulk via cp.async.bulk --------------------------------------
    if (bulk == 0) return;
    uint32_t maxByLds = winBytes / ALIGN;              // #warps we can give a window
    if (maxByLds == 0) {                               // no window room: vector fallback
        if (rank == 0)
            warpVecCopy(s + head, d + head, bulk, warpThread, warpThreads);
        return;
    }
    uint32_t issuers = teamWarps < maxByLds ? teamWarps : maxByLds;
    uint32_t window  = (winBytes / issuers) & ~(ALIGN - 1);   // per-warp 16B-multiple
    if (rank >= issuers) return;                       // this warp doesn't issue

    // distribute the bulk across issuers by team rank (contiguous 16B units)
    size_t units   = bulk / ALIGN;
    size_t base    = units / issuers;
    size_t extra   = units % issuers;
    size_t myUnits = base + (rank < extra ? 1u : 0u);
    size_t myStart = rank * base + (rank < extra ? rank : extra);
    if (myUnits == 0) return;

    size_t         myBytes = myUnits * ALIGN;
    uint8_t*       myLds   = winBase + (size_t)rank * window;
    const uint8_t* sBase   = s + head + myStart * ALIGN;
    uint8_t*       dBase   = d + head + myStart * ALIGN;

    uint64_t* bar   = &bars[rank];
    uint32_t  phase = 0;
    if (leader) ptx::mbarrier_init(bar, 1);            // one arrival (the leader)
    __syncwarp();

    if (leader) {
        for (size_t off = 0; off < myBytes; off += window) {
            uint32_t chunk = (myBytes - off < window)
                             ? static_cast<uint32_t>(myBytes - off) : window;
            issueChunk(sBase + off, myLds, dBase + off, chunk, bar, phase);
        }
    }
}

} // namespace detail

#endif // backend selection

// ---- public API definitions (declared at the top of this file) -------------

__device__ inline void tdmWait() { detail::waitTensor0(); }

__device__ inline void tdmCopyAsync(void* dst, const void* src, size_t sizeBytes,
                                    void* ldsBuffer, size_t ldsBufferBytes) {
    detail::issue(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, /*start=*/0, /*stop=*/~0u);
}

__device__ inline void tdmCopy(void* dst, const void* src, size_t sizeBytes,
                               void* ldsBuffer, size_t ldsBufferBytes) {
    detail::issue(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, /*start=*/0, /*stop=*/~0u);
    tdmWait();
}

__device__ inline void tdmCopyAsyncByTeam(void* dst, const void* src, size_t sizeBytes,
                                          void* ldsBuffer, size_t ldsBufferBytes,
                                          uint32_t startWarpId, uint32_t stopWarpId) {
    detail::issue(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, startWarpId, stopWarpId);
}

__device__ inline void tdmCopyByTeam(void* dst, const void* src, size_t sizeBytes,
                                     void* ldsBuffer, size_t ldsBufferBytes,
                                     uint32_t startWarpId, uint32_t stopWarpId) {
    detail::issue(dst, src, sizeBytes, ldsBuffer, ldsBufferBytes, startWarpId, stopWarpId);
    tdmWait();   // no-op on any warp that issued nothing / is off-team
}

#if TDM_BACKEND_AMD          // multi-source reduce path only exists on the AMD (TDM) backend
template <typename PACKED_FLOAT = float>
__device__ inline void tdmReduce(void** dsts, const void** srcs, uint32_t numSrcs, uint32_t numDsts,
                                 size_t sizeBytes, void* ldsBuffer, size_t ldsBufferBytes) {
    detail::issue<PACKED_FLOAT>(dsts, srcs, numSrcs, numDsts, sizeBytes, ldsBuffer, ldsBufferBytes, /*start=*/0, /*stop=*/~0u);
    tdmWait();
}
#endif // TDM_BACKEND_AMD

#endif // TDM_SUPPORTED
// On an unsupported target the entry points were declared `= delete` at the top,
// so there is nothing to define here -- any call is a compile-time error.

} // namespace tdm
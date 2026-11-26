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
/// \file NicNearAid.hpp
/// \brief Layout, AID probe and work requests of NIC_NEAR_AID, which places the RDMA WRITEs of a
///        NIC Transfer only into the 16 KiB blocks of its GPU destination on one AID
///
///        In SPX/NPS1 GPU memory alternates between the two AIDs every 16 KiB. Drawn as an 8 x 8
///        grid of 16 KiB blocks, each 1 MiB region is a checkerboard of the two AIDs, regular or
///        inverted (N = AID next to the NIC, F = the other AID):
///
///            regular        inverted
///            N F N F ...    F N F N ...
///            F N F N ...    N F N F ...
///
///        A piece is 16 KiB of transfer data, a block is 16 KiB of the destination. Each region
///        holds 32 pieces on the blocks of one AID, so the destination is twice the transfer size.
#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <vector>

#include "IbvDynLoad.hpp"

#if defined(__x86_64__) && !defined(__NVCC__)
#include <x86intrin.h>
#endif

namespace NicNearAid
{
  size_t constexpr PIECE_BYTES       = 16 * 1024;
  size_t constexpr REGION_BYTES      = 1024 * 1024;
  size_t constexpr PIECES_PER_REGION = REGION_BYTES / 2 / PIECE_BYTES;
  size_t constexpr REGION_DATA_BYTES = PIECES_PER_REGION * PIECE_BYTES;
  size_t constexpr BLOCKS_PER_ROW    = 8;
  size_t constexpr CACHE_LINE_BYTES  = 64;
  // One RDMA WRITE per piece, posted as chains of one region with only the tail signaled. Long
  // unsignaled runs stall ionic: 8 chains in flight (248 unsignaled WRs per queue pair) hang now and
  // then, so at most 4 are in flight. ionic also stalls when those WRs fill more than half of the
  // send queue.
  size_t constexpr PIECES_PER_CHAIN  = PIECES_PER_REGION;
  size_t constexpr CHAINS_IN_FLIGHT  = 4;
  size_t constexpr MIN_SEND_WR       = 2 * CHAINS_IN_FLIGHT * PIECES_PER_CHAIN;

  // State of one NIC Transfer; inverted is empty when NIC_NEAR_AID is off
  struct State
  {
    std::vector<uint8_t>                   inverted;      ///< Per DST region, 1 if its checkerboard is inverted
    std::vector<std::vector<ibv_send_wr*>> chains;        ///< Per queue pair, heads of the WR chains
    std::vector<size_t>                    chainsPosted;  ///< Per queue pair, chains posted this iteration
    std::vector<size_t>                    chainsDone;    ///< Per queue pair, chains completed this iteration
  };

  inline size_t NumRegions(size_t numBytes)
  {
    return (numBytes + REGION_DATA_BYTES - 1) / REGION_DATA_BYTES;
  }

  inline size_t DstBytes(size_t numBytes)
  {
    return NumRegions(numBytes) * REGION_BYTES;
  }

  // Offset in the destination of logical destination byte `offset`. `inverted` has one entry per
  // region, 1 if its checkerboard is inverted; `far` selects the F blocks instead of the N blocks.
  inline size_t DstOffset(size_t offset, bool far, std::vector<uint8_t> const& inverted)
  {
    size_t const piece         = offset / PIECE_BYTES;
    size_t const region        = piece / PIECES_PER_REGION;
    size_t const pieceInRegion = piece % PIECES_PER_REGION;
    // Piece p takes block 2p or 2p+1. Which of the two is wanted alternates from row to row of the
    // checkerboard, and flips for an inverted region and for the F blocks.
    size_t const row           = pieceInRegion / (BLOCKS_PER_ROW / 2);
    int    const secondOfPair  = (row & 1) ^ inverted[region] ^ far;
    size_t const block         = 2 * pieceInRegion + secondOfPair;
    return region * REGION_BYTES + block * PIECE_BYTES + offset % PIECE_BYTES;
  }

  struct ProbeResult
  {
    size_t   numAmbiguous   = 0;   ///< Regions without a clear split, treated as regular
    uint64_t region0Nsec[2] = {};  ///< Read times of blocks 0 and 1 of region 0
  };

#if defined(__x86_64__) && !defined(__NVCC__)
  // Median time, in nanoseconds, of one CPU read at the start of `block`. The CPU caches the DMA-BUF
  // mapping, so each line is flushed first to make the read go to the GPU.
  inline uint64_t ReadNsec(char const* block)
  {
    int constexpr NUM_REPS = 16;
    uint64_t nsec[NUM_REPS];
    for (int rep = 0; rep < NUM_REPS; rep++) {
      volatile uint64_t const* word = (volatile uint64_t const*)(block + rep * CACHE_LINE_BYTES);
      _mm_clflush((void const*)word);
      _mm_mfence();
      auto const start = std::chrono::steady_clock::now();
      (void)*word;
      auto const elapsed = std::chrono::steady_clock::now() - start;
      nsec[rep] = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    }
    std::nth_element(nsec, nsec + NUM_REPS / 2, nsec + NUM_REPS);
    return nsec[NUM_REPS / 2];
  }

  // Finds which destination regions are inverted from CPU load latencies, read through an mmap of
  // the destination DMA-BUF, and returns 0 or an errno. The CPU reaches device memory through the
  // same PCIe entry as the NIC, so the faster block is taken as N.
  inline int ProbeInverted(int dmabufFd, size_t dmabufOffset, std::vector<uint8_t>& inverted, ProbeResult& result)
  {
    size_t const mapBytes = dmabufOffset + inverted.size() * REGION_BYTES;
    void* mapping = mmap(NULL, mapBytes, PROT_READ, MAP_SHARED, dmabufFd, 0);
    if (mapping == MAP_FAILED)
      return errno;

    char const* dst = (char const*)mapping + dmabufOffset;
    result = {};
    for (size_t region = 0; region < inverted.size(); region++) {
      char const* regionBase = dst + region * REGION_BYTES;
      uint64_t const t0 = ReadNsec(regionBase);
      uint64_t const t1 = ReadNsec(regionBase + 1 * PIECE_BYTES);
      uint64_t const t8 = ReadNsec(regionBase + 8 * PIECE_BYTES);
      uint64_t const t9 = ReadNsec(regionBase + 9 * PIECE_BYTES);
      if (region == 0) {
        result.region0Nsec[0] = t0;
        result.region0Nsec[1] = t1;
      }

      // Blocks 0 and 9 share an AID, as do blocks 1 and 8; each pair must differ by at least 10%.
      // Block 1 is N only in an inverted region.
      bool const block1Near = t1 < t0;
      bool const clearSplit = ((t8 < t9) == block1Near) &&
                              std::max(t0, t1) * 10 >= std::min(t0, t1) * 11 &&
                              std::max(t8, t9) * 10 >= std::min(t8, t9) * 11;
      if (!clearSplit) {
        result.numAmbiguous++;
        inverted[region] = 0;
        continue;
      }
      inverted[region] = block1Near;
    }
    munmap(mapping, mapBytes);
    return 0;
  }
#else
  inline int ProbeInverted(int dmabufFd, size_t dmabufOffset, std::vector<uint8_t>& inverted, ProbeResult& result)
  {
    return ENOTSUP;
  }
#endif

  // Builds the RDMA WRITEs of logical destination bytes [dstStart, dstStart + numBytes), one per
  // 16 KiB piece, in chains of PIECES_PER_CHAIN with only the tail signaled. `remoteBase` is the
  // remote address of logical byte 0; `chains` receives the chain heads.
  inline void BuildWorkRequests(uint64_t srcAddr, uint32_t lkey, uint64_t remoteBase, uint32_t rkey,
                                size_t dstStart, size_t numBytes, bool far,
                                std::vector<uint8_t> const& inverted, uint64_t wrId,
                                std::vector<ibv_sge>& sges, std::vector<ibv_send_wr>& wrs,
                                std::vector<ibv_send_wr*>& chains)
  {
    size_t const dstEnd     = dstStart + numBytes;
    size_t const firstPiece = dstStart / PIECE_BYTES;
    size_t const numPieces  = numBytes ? (dstEnd - 1) / PIECE_BYTES - firstPiece + 1 : 0;
    sges.assign(numPieces, {});
    wrs.assign(numPieces, {});
    chains.clear();

    size_t offset = dstStart;
    for (size_t pieceIdx = 0; pieceIdx < numPieces; pieceIdx++) {
      size_t const pieceEnd    = std::min((offset / PIECE_BYTES + 1) * PIECE_BYTES, dstEnd);
      bool   const isChainTail = (pieceIdx % PIECES_PER_CHAIN == PIECES_PER_CHAIN - 1) ||
                                 (pieceIdx == numPieces - 1);

      ibv_sge& sg = sges[pieceIdx];
      sg.addr   = srcAddr + (offset - dstStart);
      sg.length = pieceEnd - offset;
      sg.lkey   = lkey;

      ibv_send_wr& wr = wrs[pieceIdx];
      wr.wr_id               = wrId;
      wr.next                = isChainTail ? NULL : &wrs[pieceIdx + 1];
      wr.sg_list             = &sg;
      wr.num_sge             = 1;
      wr.send_flags          = isChainTail ? IBV_SEND_SIGNALED : 0;
      wr.opcode              = IBV_WR_RDMA_WRITE;
      wr.wr.rdma.remote_addr = remoteBase + DstOffset(offset, far, inverted);
      wr.wr.rdma.rkey        = rkey;

      if (pieceIdx % PIECES_PER_CHAIN == 0) chains.push_back(&wr);
      offset = pieceEnd;
    }
  }

  // Posts the next chain of queue pair `qpIndex` and returns the error code of ibv_post_send
  inline int PostNextChain(State& state, ibv_qp* qp, int qpIndex)
  {
    ibv_send_wr* badWorkReq;
    int const error = ibv_post_send(qp, state.chains[qpIndex][state.chainsPosted[qpIndex]], &badWorkReq);
    if (error)
      return error;
    state.chainsPosted[qpIndex]++;
    return 0;
  }

  // Starts an iteration with up to CHAINS_IN_FLIGHT chains per queue pair. On error, `failedQp` is the
  // queue pair whose post failed.
  inline int PostFirstChains(State& state, std::vector<ibv_qp*> const& qps, int& failedQp)
  {
    state.chainsPosted.assign(state.chains.size(), 0);
    state.chainsDone.assign(state.chains.size(), 0);
    for (int qpIndex = 0; qpIndex < (int)state.chains.size(); qpIndex++) {
      size_t const numFirst = std::min(CHAINS_IN_FLIGHT, state.chains[qpIndex].size());
      for (size_t chainIdx = 0; chainIdx < numFirst; chainIdx++) {
        int const error = PostNextChain(state, qps[qpIndex], qpIndex);
        if (error) {
          failedQp = qpIndex;
          return error;
        }
      }
    }
    return 0;
  }

  // Handles the completion of one chain of queue pair `qpIndex` by posting its next chain. Sets
  // `qpDone` once all chains of that queue pair have completed.
  inline int OnChainDone(State& state, ibv_qp* qp, int qpIndex, bool& qpDone)
  {
    size_t const numChains = state.chains[qpIndex].size();
    qpDone = (++state.chainsDone[qpIndex] == numChains);
    if (qpDone || state.chainsPosted[qpIndex] == numChains)
      return 0;
    return PostNextChain(state, qp, qpIndex);
  }

  // Copies the first `numBytes` logical bytes out of a host copy of the whole destination
  inline void GatherDst(uint8_t const* staged, size_t numBytes, bool far, std::vector<uint8_t> const& inverted,
                        uint8_t* gathered)
  {
    for (size_t offset = 0; offset < numBytes; offset += PIECE_BYTES) {
      size_t const pieceBytes = std::min(PIECE_BYTES, numBytes - offset);
      memcpy(gathered + offset, staged + DstOffset(offset, far, inverted), pieceBytes);
    }
  }
}

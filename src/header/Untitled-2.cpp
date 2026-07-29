// ---- vector fallback: reduce (sum) numSrcs sources, broadcast to numDsts. ----
// Element-wise sum of all srcs -> written to every dst. numSrcs == 1 is a plain
// copy; numDsts > 1 broadcasts the same reduced result to each destination.
// `srcs`/`dsts` hold base addresses; `offset` (in bytes) reaches the sub-range
// to copy (e.g. the tail start), so callers can share one base pointer array.
__device__ inline void warpVecCopy(uint64_t* srcs, uint64_t* dsts,
                                   uint32_t numSrcs, uint32_t numDsts,
                                   size_t n, size_t offset,
                                   uint32_t warpThread, uint32_t warpThreads) {
    size_t nd = n >> 2;
    for (size_t i = warpThread; i < nd; i += warpThreads) {
        uint32_t acc = 0;
        for (uint32_t s = 0; s < numSrcs; ++s)
            acc += reinterpret_cast<const uint32_t*>(srcs[s] + offset)[i];
        for (uint32_t d = 0; d < numDsts; ++d)
            reinterpret_cast<uint32_t*>(dsts[d] + offset)[i] = acc;
    }
    uint32_t rem = static_cast<uint32_t>(n & 3u);
    if (rem && warpThread == 0) {
        for (uint32_t b = 0; b < rem; ++b) {
            uint8_t acc = 0;
            for (uint32_t s = 0; s < numSrcs; ++s)
                acc += reinterpret_cast<const uint8_t*>(srcs[s] + offset)[nd * 4 + b];
            for (uint32_t d = 0; d < numDsts; ++d)
                reinterpret_cast<uint8_t*>(dsts[d] + offset)[nd * 4 + b] = acc;
        }
    }
}

// ---- issue one chunk of whole 256B rows (2D tile) through ONE LDS window. ---
// This staging window is single-buffered, so the two TDM ops form a dependency
// chain that MUST be enforced with TENSORcnt waits -- up to 3 TDM ops are
// outstanding per wave (they overlap), so "same-wave in-order issue" does NOT
// serialize their memory effects:
//   * load -> wait: the store reads the LDS the load just wrote (RAW hazard).
//   * store -> wait: the caller reuses this same window next iteration; the next
//     load must not overwrite LDS the store is still draining (WAR hazard).
__device__ inline void issueRows(uint64_t* srcs, uint64_t* dsts,
                                 uint32_t numSrcs, uint32_t numDsts,
                                 uint32_t val, uint32_t tmp, uint32_t rows, uint32_t off = 0) {
    gfx1250_TDM_GROUP1 g1;
    g1.dataSize(DS4);
    g1.tileDim0(TD0);   g1.tileDim1(rows);
    g1.tensorDim0(TD0); g1.tensorDim1(rows);
    g1.tensorDim0Stride(TD0);                // rows back-to-back (contiguous)

    if (numSrcs) {
      gfx1250_TDM_GROUP0 g0l(val, srcs[0] + off);
      load(g0l, g1);   waitTensor0();
      for (size_t s = 1; s < numSrcs; s++) {
        gfx1250_TDM_GROUP0 g0l(tmp, srcs[s] + off);
        load(g0l, g1);   waitTensor0();
        for (size_t u = 0; u < rows * WIDTH; u += Warp) {
          val[u] += tpm[u];
        }
      }
    }

    for (size_t d = 0; d < numDsts; d++) {
      gfx1250_TDM_GROUP0 g0s(tmp, dsts[d] + off);
      store(g0s, g1);  waitTensor0();
    }
}

// ---- issue a sub-row tail (<256B) as a 1-D tile at BYTE granularity. ---------
// Same single-buffered LDS window and the same required RAW/WAR waits as above.
__device__ inline void issueRow1d(uint64_t* srcs, uint64_t* dsts,
                                  uint32_t numSrcs, uint32_t numDsts,
                                  uint32_t val, uint32_t tmp, uint32_t nbytes,
                                  uint64_t off = 0) {
    gfx1250_TDM_GROUP1 g1;
    g1.dataSize(DS1);                        // 1-byte elements: exact length
    g1.tileDim0(nbytes);   g1.tileDim1(1);
    g1.tensorDim0(nbytes); g1.tensorDim1(1);
    g1.tensorDim0Stride(nbytes);

    if (numSrcs) {
      gfx1250_TDM_GROUP0 g0l(val, srcs[0] + off); // unused higher dims -> zero (see load())
      load(g0l, g1);   waitTensor0();        // RAW: fill LDS before store/reduce reads it
      for (size_t s = 1; s < numSrcs; s++) {
        gfx1250_TDM_GROUP0 g0l(tmp, srcs[s] + off);
        load(g0l, g1);   waitTensor0();
        for (size_t u = 0; u < nbytes; u += Warp) {
          val[u] += tmp[u];                  // reduce: accumulate into val
        }
      }
    }

    for (size_t d = 0; d < numDsts; d++) {
      gfx1250_TDM_GROUP0 g0s(val, dsts[d] + off);  // broadcast reduced result to each dst
      store(g0s, g1);  waitTensor0();        // WAR: drain store before window reuse
    }
}

__device__ inline void issue(void** dsts, const void** srcs, uint32_t numSrcs, uint32_t numDsts,
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

    // --- edges (team's FIRST warp = rank 0): vector head, TDM tail -----------
    if (rank == 0 && tail) {
        // LDS needed: val (+ tmp when reducing multiple srcs), each holding `tail` bytes
        uint32_t need = (numSrcs > 1 ? (tmp - ldsBase) : (val - ldsBase)) + tail;
        if (ldsBytes >= need) {                        // stage tail in rank 0's window
            issueRow1d(mySrcs, myDsts, numSrcs, numDsts, val, tmp, tail, tdmBytes);
        } else {
            warpVecCopy(mySrcs, myDsts, numSrcs, numDsts, tail, tdmBytes,
                        warpThread, warpThreads);
        }
    }

    // --- 256B rows copy via TDM ------------------------------------------------
    uint32_t maxByLds = ldsBytes / RWIDTH;             // #warps we can give a window, 2*WIDTH because of double buffering
    if (maxByLds == 0) {                               // LDS < 512B: vector fallback
        if (rank == 0)
            warpVecCopy(mySrcs, myDsts, numSrcs, numDsts, tdmBytes, 0,
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

    // shift this warp's reduce buffers into its own window
    val += rank * window;
    tmp += rank * window;

    for (size_t r = 0; r < myRows; r += rowsPerChunk) {
        uint32_t chunkRows = (myRows - r < rowsPerChunk)
                             ? static_cast<uint32_t>(myRows - r) : rowsPerChunk;
        uint64_t off = (uint64_t)(myStart + r) * WIDTH;   // this warp's row block
        issueRows(mySrcs, myDsts, numSrcs, numDsts, val, tmp, chunkRows, off);
    }
}
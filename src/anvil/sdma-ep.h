/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * SDMA Endpoint -- GPU-initiated DMA via AMD SDMA engines
 *
 * This header provides the complete public API for the SDMA
 * endpoint, including:
 *   - Host-side setup: init, connect, queue creation
 *   - Device-side operations: put, signal, wait, flush
 *   - CLI/validation helpers for the xio-tester
 *
 * The device handle (SdmaQueueHandle) and all device-side
 * operations are derived from the anvil library (AMD RAD).
 */

#ifndef SDMA_EP_H
#define SDMA_EP_H

#include <cassert>
#include <cstdint>
#include <string>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include "sdma_pkt_struct.h"

namespace xio {

struct XioEndpointConfig;

namespace sdma_ep {

/* ================================================================
 * Configuration
 * ================================================================ */

/**
 * @brief SDMA endpoint test configuration.
 *
 * Contains all user-facing options for the xio-tester
 * sdma-ep subcommand. Validated by validateConfig().
 */
struct SdmaEpConfig {
  std::string testType = "";  /**< Test subcommand name:
                                   "p2p", "ping-pong", or
                                   "buffer-reuse". */
  bool useHostDst = false;    /**< If true, destination is
                                   pinned host memory (single
                                   GPU, no P2P required). */
  bool verifyData = false;    /**< If true, validate the
                                   destination buffer after
                                   transfer. */
  bool useCounter = false;    /**< Use counter-based
                                   completion tracking. */
  bool useFlush = false;      /**< Use flush-based
                                   completion tracking. */
  int srcDeviceId = -1;       /**< Source HIP device ID.
                                   -1 = default (0). */
  int dstDeviceId = -1;       /**< Destination HIP device ID.
                                   -1 = default (1 for P2P,
                                   0 for --to-host). */
  size_t transferSize = 4096; /**< Per-iteration transfer
                                   size in bytes. Must be a
                                   multiple of 4. */
  unsigned iterations = 128;  /**< Number of SDMA transfers
                                    per run. */
};

/* ================================================================
 * Host-Side Setup Types
 * ================================================================ */

/**
 * @brief Information about an established SDMA connection.
 *
 * Returned by createConnection(). Contains the resolved
 * SDMA engine ID for the GPU pair, which is determined by
 * the XGMI/Infinity Fabric topology (MI300X OAM map).
 */
struct SdmaConnectionInfo {
  int srcDeviceId;   /**< Source HIP device ID. */
  int dstDeviceId;   /**< Destination HIP device ID. */
  uint32_t engineId; /**< XGMI-optimal SDMA engine ID
                          for this GPU pair. */
};

/**
 * @brief Information about a created SDMA queue.
 *
 * Returned by createQueue(). The deviceHandle pointer
 * is GPU-accessible and should be passed to GPU kernels
 * that use the device-side SDMA operations (put, signal,
 * waitSignal, flush, quiet).
 */
struct SdmaQueueInfo {
  void* deviceHandle; /**< GPU-accessible pointer to a
                           SdmaQueueHandle. Cast to
                           SdmaQueueHandle* in kernel
                           code. */
  int srcDeviceId;    /**< Source HIP device ID. */
  int dstDeviceId;    /**< Destination HIP device ID. */
  int channelIdx;     /**< Channel index within the
                           connection (0-based). */
};

/* ================================================================
 * Device-Side Constants
 * ================================================================ */

/** SDMA queue ring buffer size (matches ROCr). */
constexpr uint64_t SDMA_QUEUE_SIZE = 1024 * 1024;

/** Maximum spin-poll iterations before assert. */
constexpr int MAX_RETRIES = 1 << 30;

/** If true, assert on retry limit in device code. */
constexpr bool BREAK_ON_RETRIES = false;

/* ================================================================
 * Device-Side Packet Helpers (internal)
 * ================================================================ */

/**
 * @brief Build an SDMA linear copy packet.
 *
 * @param srcBuf Source address (GPU virtual).
 * @param dstBuf Destination address (GPU virtual).
 * @param packetSize Number of bytes to copy. Must be
 *        in the range [1, UINT32_MAX] (the HW count
 *        field is 30 bits, so max single-packet
 *        transfer is 1 GiB).
 * @return Populated SDMA_PKT_COPY_LINEAR.
 *
 * @note Device-only. The count field stores size-1 per
 *       the HW spec (0 means 1 byte).
 */
__device__ __forceinline__ SDMA_PKT_COPY_LINEAR
CreateCopyPacket(void* srcBuf, void* dstBuf, long long int packetSize) {
  assert(packetSize > 0 && "CreateCopyPacket: packetSize must be > 0");
  assert(packetSize <= 0xFFFFFFFFLL &&
         "CreateCopyPacket: packetSize exceeds 4 GiB");
  SDMA_PKT_COPY_LINEAR pkt = {};
  pkt.HEADER_UNION.op = SDMA_OP_COPY;
  pkt.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR;
  pkt.COUNT_UNION.count = (uint32_t)(packetSize - 1);
  pkt.SRC_ADDR_LO_UNION.src_addr_31_0 = (uint32_t)(uintptr_t)srcBuf;
  pkt.SRC_ADDR_HI_UNION.src_addr_63_32 = (uint32_t)((uintptr_t)srcBuf >> 32);
  pkt.DST_ADDR_LO_UNION.dst_addr_31_0 = (uint32_t)(uintptr_t)dstBuf;
  pkt.DST_ADDR_HI_UNION.dst_addr_63_32 = (uint32_t)((uintptr_t)dstBuf >> 32);
  return pkt;
}

/**
 * @brief Build an SDMA 2D sub-window copy packet.
 *
 * @param srcBuf   Source buffer base address.
 * @param dstBuf   Destination buffer base address.
 * @param tile_width  Tile width in bytes.
 * @param tile_height Tile height in rows.
 * @param src_buffer_pitch Source row stride in bytes.
 * @param dst_buffer_pitch Destination row stride in bytes.
 * @param src_x Source X offset in bytes.
 * @param src_y Source Y offset in rows.
 * @param dst_x Destination X offset in bytes.
 * @param dst_y Destination Y offset in rows.
 * @return Populated sub-window copy packet.
 */
__device__ __forceinline__ SDMA_PKT_LINEAR_LARGE_SUB_WINDOW_COPY
CreateLargeSubWindowCopyPacket(void* srcBuf, void* dstBuf, uint32_t tile_width,
                               uint32_t tile_height, uint32_t src_buffer_pitch,
                               uint32_t dst_buffer_pitch, uint32_t src_x,
                               uint32_t src_y, uint32_t dst_x, uint32_t dst_y) {
  SDMA_PKT_LINEAR_LARGE_SUB_WINDOW_COPY pkt = {};
  pkt.HEADER_UNION.op = SDMA_OP_COPY;
  pkt.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR_SUB_WINDOW;
  pkt.SRC_ADDR_LO_UNION.src_base_addr_31_0 = (uint32_t)(uintptr_t)srcBuf;
  pkt.SRC_ADDR_HI_UNION.src_base_addr_63_32 = (uint32_t)((uintptr_t)srcBuf >>
                                                         32);
  pkt.SRC_X_UNION.src_x = src_x;
  pkt.SRC_Y_UNION.src_y = src_y;
  pkt.SRC_Z_UNION.src_z = 0;
  pkt.SRC_PITCH_UNION.src_pitch = src_buffer_pitch - 1;
  uint64_t src_slice_pitch = 0;
  pkt.SRC_SLICE_PITCH_LO_UNION.src_slice_pitch_31_0 =
    (uint32_t)(src_slice_pitch & 0xFFFFFFFF);
  pkt.SRC_SLICE_PITCH_HI_UNION.src_slice_pitch_47_32 =
    (uint16_t)((src_slice_pitch >> 32) & 0xFFFF);
  pkt.DST_ADDR_LO_UNION.dst_data_31_0 = (uint32_t)(uintptr_t)dstBuf;
  pkt.DST_ADDR_HI_UNION.src_data_63_32 = (uint32_t)((uintptr_t)dstBuf >> 32);
  pkt.DST_X_UNION.dst_x = dst_x;
  pkt.DST_Y_UNION.dst_y = dst_y;
  pkt.DST_Z_UNION.dst_z = 0;
  pkt.DST_PITCH_UNION.dst_pitch = dst_buffer_pitch - 1;
  uint64_t dst_slice_pitch = 0;
  pkt.DST_SLICE_PITCH_LO_UNION.dst_slice_pitch_31_0 =
    (uint32_t)(dst_slice_pitch & 0xFFFFFFFF);
  pkt.DST_SLICE_PITCH_HI_UNION.dst_slice_pitch_47_32 =
    (uint16_t)((dst_slice_pitch >> 32) & 0xFFFF);
  pkt.RECT_X_UNION.rect_x = tile_width - 1;
  pkt.RECT_Y_UNION.rect_y = tile_height - 1;
  pkt.RECT_Z_UNION.rect_z = 0;
  return pkt;
}

/**
 * @brief Build an SDMA atomic increment packet.
 *
 * Atomically adds 1 to the 64-bit value at the given
 * address via the SDMA engine (not the shader ALU).
 *
 * @param addr Address of the 64-bit value to increment.
 * @return Populated SDMA_PKT_ATOMIC.
 */
__device__ __forceinline__ SDMA_PKT_ATOMIC
CreateAtomicIncPacket(uint64_t* addr) {
  SDMA_PKT_ATOMIC pkt = {};
  pkt.HEADER_UNION.op = SDMA_OP_ATOMIC;
  pkt.HEADER_UNION.operation = SDMA_ATOMIC_ADD64;
  pkt.ADDR_LO_UNION.addr_31_0 = (uint32_t)((uintptr_t)addr);
  pkt.ADDR_HI_UNION.addr_63_32 = (uint32_t)((uintptr_t)addr >> 32);
  pkt.SRC_DATA_LO_UNION.src_data_31_0 = 0x1;
  pkt.SRC_DATA_HI_UNION.src_data_63_32 = 0x0;
  return pkt;
}

/**
 * @brief Build an SDMA fence packet.
 *
 * Writes a data value to a memory address after all
 * preceding SDMA operations on this queue have completed.
 *
 * @param address Address to write the fence value to.
 * @param data    Value to write (default: 1).
 * @return Populated SDMA_PKT_FENCE.
 */
__device__ __forceinline__ SDMA_PKT_FENCE CreateFencePacket(uint64_t* address,
                                                            uint32_t data = 1) {
  SDMA_PKT_FENCE pkt = {};
  pkt.HEADER_UNION.op = SDMA_OP_FENCE;
  pkt.ADDR_LO_UNION.addr_31_0 = (uint32_t)((uintptr_t)address);
  pkt.ADDR_HI_UNION.addr_63_32 = (uint32_t)((uintptr_t)address >> 32);
  pkt.DATA_UNION.data = data;
  return pkt;
}

/* ================================================================
 * Device-Side Poll Helpers
 * ================================================================ */

/**
 * @brief Spin-poll until *addr >= expected.
 *
 * @tparam MAX_SPIN_COUNT Maximum iterations before assert
 *         (-1 = unlimited).
 * @param addr    Address to poll (device memory).
 * @param expected Minimum value to wait for.
 *
 * @note Device-only. Uses agent-scope relaxed atomics.
 */
template <int64_t MAX_SPIN_COUNT = -1>
__device__ __forceinline__ void poll_until_ge(uint64_t* addr,
                                              uint64_t expected) {
  [[maybe_unused]] int64_t spin_count = 0;
  while (__hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) <
         expected) {
    spin_count++;
    assert(MAX_SPIN_COUNT < 0 || spin_count != MAX_SPIN_COUNT);
  }
}

/* ================================================================
 * Device Handle
 * ================================================================ */

/**
 * @brief GPU-visible SDMA queue state.
 *
 * Contains ring buffer pointers, doorbell, and read/write
 * tracking state. Created on the host via createQueue()
 * and passed to GPU kernels for device-side SDMA
 * operations (put, signal, flush, quiet).
 *
 * Multi-producer safe: uses atomic CAS on cachedWptr for
 * queue space reservation. For single-producer kernels,
 * use SdmaQueueSingleProducerHandle for lower overhead.
 *
 * @note This struct is allocated in device memory and
 *       must not be copied by value across host/device
 *       boundary after initialization.
 */
struct SdmaQueueHandle {
  /** @brief Wrap a byte index into the ring buffer. */
  __device__ __forceinline__ uint64_t WrapIntoRing(uint64_t index) {
    return index % SDMA_QUEUE_SIZE;
  }

  /**
   * @brief Check if the queue has space up to uptoIndex.
   *
   * Uses a cached HW read index for fast-path; falls
   * back to reading the hardware register if the cached
   * value indicates the queue is full.
   */
  __device__ __forceinline__ bool CanWriteUpto(uint64_t uptoIndex) {
    if ((uptoIndex - cachedHwReadIndex) < SDMA_QUEUE_SIZE) {
      return true;
    }
    cachedHwReadIndex = __hip_atomic_load(rptr, __ATOMIC_RELAXED,
                                          __HIP_MEMORY_SCOPE_AGENT);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    return (uptoIndex - cachedHwReadIndex) < SDMA_QUEUE_SIZE;
  }

  /**
   * @brief Reserve space in the queue ring buffer.
   *
   * Atomically advances cachedWptr. Handles ring
   * wraparound by padding with NOPs. Spins until space
   * is available.
   *
   * @param size_in_bytes Bytes to reserve.
   * @param offset Output: NOP padding bytes inserted
   *        before the reserved region (for wraparound).
   * @return Base index (byte offset) of the reserved
   *         region.
   */
  __device__ __forceinline__ uint64_t
  ReserveQueueSpace(const size_t size_in_bytes, uint64_t& offset) {
    uint64_t cur_index;
    int retries = 0;
    while (true) {
      cur_index = __hip_atomic_load(cachedWptr, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_AGENT);
      offset = 0;
      if (WrapIntoRing(cur_index) + size_in_bytes > SDMA_QUEUE_SIZE) {
        offset = SDMA_QUEUE_SIZE - WrapIntoRing(cur_index);
      }
      uint64_t new_index = cur_index + size_in_bytes + offset;
      if (CanWriteUpto(new_index)) {
        if (__hip_atomic_compare_exchange_strong(cachedWptr, &cur_index,
                                                 new_index, __ATOMIC_RELAXED,
                                                 __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT)) {
          break;
        }
      }
      if constexpr (BREAK_ON_RETRIES) {
        if (retries++ == MAX_RETRIES) {
          assert(false && "Retry limit on reserve queue space");
          break;
        }
      }
    }
    return cur_index;
  }

  /**
   * @brief Place a packet into the queue ring buffer.
   *
   * Writes NOP padding (if offset > 0) followed by
   * the packet DWORDs using agent-scope relaxed stores.
   *
   * @tparam PacketType SDMA packet struct type.
   * @param packet      Reference to the packet data.
   * @param pendingWptr In/out write pointer tracking.
   * @param offset      NOP padding bytes before packet.
   */
  template <typename PacketType>
  __device__ __forceinline__ void placePacket(PacketType& packet,
                                              uint64_t& pendingWptr,
                                              uint64_t offset) {
    static_assert(sizeof(PacketType) / sizeof(uint32_t) <= 64);
    const uint32_t numOffsetDwords = offset / sizeof(uint32_t);
    const uint32_t numDwords = sizeof(PacketType) / sizeof(uint32_t);
    uint32_t* packetPtr = reinterpret_cast<uint32_t*>(&packet);
    uint64_t base_idx = WrapIntoRing(pendingWptr) / sizeof(uint32_t);
    for (uint32_t i = 0; i < numOffsetDwords; i++) {
      if (i == 0) {
        __hip_atomic_store(queueBuf + base_idx + i,
                           (((numOffsetDwords - 1) & 0xFFFF) << 16),
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      } else {
        __hip_atomic_store(queueBuf + base_idx + i, 0, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    pendingWptr += offset;
    base_idx = WrapIntoRing(pendingWptr) / sizeof(uint32_t);
    for (uint32_t i = 0; i < numDwords; i++) {
      __hip_atomic_store(queueBuf + base_idx + i, packetPtr[i],
                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    pendingWptr += sizeof(PacketType);
  }

  /**
   * @brief Submit a packet to the SDMA engine.
   *
   * Waits for committedWptr to match base (serializes
   * multi-producer submissions), then updates wptr,
   * rings the doorbell, and advances committedWptr.
   *
   * @param base        Base index from ReserveQueueSpace.
   * @param pendingWptr End index after placePacket.
   */
  __device__ __forceinline__ void submitPacket(uint64_t base,
                                               uint64_t pendingWptr) {
    int retries = 0;
    while (true) {
      uint64_t val = __hip_atomic_load(committedWptr, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
      __atomic_signal_fence(__ATOMIC_SEQ_CST);
      if (val == base)
        break;
      if constexpr (BREAK_ON_RETRIES) {
        if (retries++ == MAX_RETRIES) {
          assert(false && "submitPacket: Retry limit exceeded");
          break;
        }
      }
    }
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __hip_atomic_store(wptr, pendingWptr, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __hip_atomic_store(doorbell, pendingWptr, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __hip_atomic_store(committedWptr, pendingWptr, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
    maxWritePtr = pendingWptr;
  }

  /**
   * @brief Wait until HW has consumed up to upToIndex.
   *
   * Spin-polls the hardware read pointer until it
   * reaches or passes upToIndex.
   *
   * @param upToIndex Byte index to wait for.
   */
  __device__ __forceinline__ void flushTo(uint64_t upToIndex) {
    uint64_t hw_read_index;
    do {
      hw_read_index = __hip_atomic_load(rptr, __ATOMIC_RELAXED,
                                        __HIP_MEMORY_SCOPE_AGENT);
    } while (hw_read_index < upToIndex);
  }

  /**
   * @brief Wait for all submitted operations.
   *
   * Spin-polls the hardware read pointer until it
   * reaches maxWritePtr (the end of the last submitted
   * packet).
   */
  __device__ __forceinline__ void quietAll() {
    uint64_t hw_read_index;
    do {
      hw_read_index = __hip_atomic_load(rptr, __ATOMIC_RELAXED,
                                        __HIP_MEMORY_SCOPE_AGENT);
    } while (hw_read_index < maxWritePtr);
  }

  uint32_t* queueBuf;         /**< Ring buffer base. */
  uint64_t* rptr;             /**< HW read pointer. */
  uint64_t* wptr;             /**< HW write pointer. */
  uint64_t* doorbell;         /**< Doorbell register. */
  uint64_t* cachedWptr;       /**< Cached write pointer
                                   (shared, device mem). */
  uint64_t* committedWptr;    /**< Committed write pointer
                                   (shared, device mem). */
  uint64_t cachedHwReadIndex; /**< Cached HW read index
                                   (local). */
  uint64_t maxWritePtr;       /**< End of last submitted
                                   packet (local). */
};

/**
 * @brief Single-producer variant of SdmaQueueHandle.
 *
 * Avoids atomic CAS overhead in ReserveQueueSpace by
 * using direct pointer writes. Only safe when exactly
 * one thread submits to the queue.
 *
 * @note Same binary layout as SdmaQueueHandle (verified
 *       by static_assert).
 */
struct SdmaQueueSingleProducerHandle : SdmaQueueHandle {
  /** @brief Pad remaining ring space with NOPs and
   *         submit. */
  __device__ __forceinline__ void PadRingToEnd(uint64_t cur_index) {
    uint64_t new_index = cur_index +
                         (SDMA_QUEUE_SIZE - WrapIntoRing(cur_index));
    if (!CanWriteUpto(new_index))
      return;
    *cachedWptr = new_index;
    uint64_t idx = WrapIntoRing(cur_index) / sizeof(uint32_t);
    int nDwords = (new_index - cur_index) / sizeof(uint32_t);
    for (int i = 0; i < nDwords; i++) {
      queueBuf[idx + i] = (uint32_t)0;
    }
    submitPacket(cur_index, new_index);
  }

  /**
   * @brief Reserve space (single-producer, no CAS).
   * @param size_in_bytes Bytes to reserve.
   * @return Base index of the reserved region.
   */
  __device__ __forceinline__ uint64_t
  ReserveQueueSpace(const size_t size_in_bytes) {
    const uint32_t queue_size = SDMA_QUEUE_SIZE;
    uint64_t cur_index;
    while (true) {
      cur_index = *cachedWptr;
      uint64_t new_index = cur_index + size_in_bytes;
      if (WrapIntoRing(cur_index) + size_in_bytes > queue_size) {
        PadRingToEnd(cur_index);
        continue;
      }
      if (!CanWriteUpto(new_index))
        continue;
      *cachedWptr = new_index;
      break;
    }
    return cur_index;
  }

  /**
   * @brief Submit (single-producer, no committedWptr
   *        serialization).
   */
  __device__ __forceinline__ void submitPacket(uint64_t base,
                                               uint64_t pendingWptr) {
    *wptr = pendingWptr;
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    *doorbell = pendingWptr;
  }
};

static_assert(sizeof(SdmaQueueSingleProducerHandle) == sizeof(SdmaQueueHandle),
              "Single-producer handle must match base layout");

/* ================================================================
 * Device-Side Composite Operation (internal template)
 * ================================================================ */

/**
 * @brief Combined put/signal/counter operation.
 *
 * Reserves queue space for the enabled operations,
 * places the packets, and submits them as a single
 * batch. Template parameters select which operations
 * are included.
 *
 * @tparam PUT_EN     Include a linear copy packet.
 * @tparam SIGNAL_EN  Include a signal (atomic inc).
 * @tparam COUNTER_EN Include a counter (atomic inc).
 */
template <bool PUT_EN, bool SIGNAL_EN, bool COUNTER_EN>
__device__ __forceinline__ void put_signal_counter_impl(
  SdmaQueueHandle& handle, void* dst, void* src, size_t size, uint64_t* signal,
  uint64_t* counter, uint64_t* put_index = nullptr) {
  constexpr size_t space_required =
    ((PUT_EN) ? sizeof(SDMA_PKT_COPY_LINEAR) : 0) +
    ((SIGNAL_EN) ? sizeof(SDMA_PKT_ATOMIC) : 0) +
    ((COUNTER_EN) ? sizeof(SDMA_PKT_ATOMIC) : 0);
  uint64_t offset = 0;
  auto base = handle.ReserveQueueSpace(space_required, offset);
  uint64_t pendingWptr = base;
  if constexpr (PUT_EN) {
    auto pkt = CreateCopyPacket(src, dst, size);
    handle.placePacket(pkt, pendingWptr, offset);
    if (put_index != nullptr)
      *put_index = pendingWptr;
    offset = 0;
  }
  if constexpr (SIGNAL_EN) {
    auto pkt = CreateAtomicIncPacket(signal);
    handle.placePacket(pkt, pendingWptr, offset);
    offset = 0;
  }
  if constexpr (COUNTER_EN) {
    auto pkt = CreateAtomicIncPacket(counter);
    handle.placePacket(pkt, pendingWptr, offset);
    offset = 0;
  }
  handle.submitPacket(base, pendingWptr);
}

/* ================================================================
 * Device-Side Operations -- Data Transfer
 * ================================================================ */

/**
 * @brief DMA copy via SDMA engine.
 *
 * Submits an SDMA_PKT_COPY_LINEAR to transfer size
 * bytes from src to dst. The transfer is non-blocking:
 * it completes asynchronously after the SDMA engine
 * processes the packet.
 *
 * @param handle Queue handle (multi-producer safe).
 * @param dst    Destination address (GPU virtual).
 * @param src    Source address (GPU virtual).
 * @param size   Number of bytes to transfer.
 *
 * @note Device-only. Thread-safe for multi-producer
 *       queues (uses atomic CAS for reservation).
 * @note Use flush() or quiet() to wait for completion.
 */
__device__ __forceinline__ void put(SdmaQueueHandle& handle, void* dst,
                                    void* src, size_t size) {
  put_signal_counter_impl<true, false, false>(handle, dst, src, size, nullptr,
                                              nullptr);
}

/**
 * @brief 2D sub-window DMA copy via SDMA engine.
 *
 * Copies a rectangular tile from a source buffer to a
 * destination buffer using
 * SDMA_PKT_LINEAR_LARGE_SUB_WINDOW_COPY.
 *
 * @param handle    Queue handle.
 * @param dst       Destination buffer base address.
 * @param src       Source buffer base address.
 * @param tileWidth   Tile width in bytes.
 * @param tileHeight  Tile height in rows.
 * @param srcPitch  Source row stride in bytes.
 * @param dstPitch  Destination row stride in bytes.
 * @param srcX      Source X offset in bytes.
 * @param srcY      Source Y offset in rows.
 * @param dstX      Destination X offset in bytes.
 * @param dstY      Destination Y offset in rows.
 *
 * @note Device-only. Non-blocking.
 */
__device__ __forceinline__ void putTile(SdmaQueueHandle& handle, void* dst,
                                        void* src, uint32_t tileWidth,
                                        uint32_t tileHeight, uint32_t srcPitch,
                                        uint32_t dstPitch, uint32_t srcX,
                                        uint32_t srcY, uint32_t dstX,
                                        uint32_t dstY) {
  uint64_t offset = 0;
  auto base = handle.ReserveQueueSpace(sizeof(
                                         SDMA_PKT_LINEAR_LARGE_SUB_WINDOW_COPY),
                                       offset);
  auto pkt = CreateLargeSubWindowCopyPacket(src, dst, tileWidth, tileHeight,
                                            srcPitch, dstPitch, srcX, srcY,
                                            dstX, dstY);
  uint64_t pendingWptr = base;
  handle.placePacket(pkt, pendingWptr, offset);
  handle.submitPacket(base, pendingWptr);
}

/* ================================================================
 * Device-Side Operations -- Signaling
 * ================================================================ */

/**
 * @brief Atomically increment a signal via SDMA engine.
 *
 * Submits an SDMA_PKT_ATOMIC that adds 1 to the 64-bit
 * value at *signal. The increment is performed by the
 * SDMA engine, not the shader ALU.
 *
 * @param handle Queue handle.
 * @param signal Address of a 64-bit signal counter in
 *               device memory (uncached recommended).
 *
 * @note Device-only. Non-blocking.
 */
__device__ __forceinline__ void signal(SdmaQueueHandle& handle,
                                       uint64_t* signal) {
  put_signal_counter_impl<false, true, false>(handle, nullptr, nullptr, 0,
                                              signal, nullptr);
}

/**
 * @brief DMA copy with completion signal (batched).
 *
 * Combines a linear copy and an atomic signal increment
 * into a single queue submission. The signal is
 * incremented after the copy completes.
 *
 * @param handle Queue handle.
 * @param dst    Destination address (GPU virtual).
 * @param src    Source address (GPU virtual).
 * @param size   Number of bytes to transfer.
 * @param signal Address of a 64-bit signal counter.
 *
 * @note Device-only. Non-blocking.
 */
__device__ __forceinline__ void putSignal(SdmaQueueHandle& handle, void* dst,
                                          void* src, size_t size,
                                          uint64_t* signal) {
  put_signal_counter_impl<true, true, false>(handle, dst, src, size, signal,
                                             nullptr);
}

/**
 * @brief DMA copy with signal and counter (batched).
 *
 * Combines a linear copy, a signal increment, and a
 * counter increment into a single queue submission.
 *
 * @param handle  Queue handle.
 * @param dst     Destination address (GPU virtual).
 * @param src     Source address (GPU virtual).
 * @param size    Number of bytes to transfer.
 * @param signal  Address of a 64-bit signal counter.
 * @param counter Address of a 64-bit counter.
 *
 * @note Device-only. Non-blocking.
 */
__device__ __forceinline__ void putSignalCounter(SdmaQueueHandle& handle,
                                                 void* dst, void* src,
                                                 size_t size, uint64_t* signal,
                                                 uint64_t* counter) {
  put_signal_counter_impl<true, true, true>(handle, dst, src, size, signal,
                                            counter);
}

/**
 * @brief DMA copy with counter only (batched).
 *
 * @param handle  Queue handle.
 * @param dst     Destination address (GPU virtual).
 * @param src     Source address (GPU virtual).
 * @param size    Number of bytes to transfer.
 * @param counter Address of a 64-bit counter.
 *
 * @note Device-only. Non-blocking.
 */
__device__ __forceinline__ void putCounter(SdmaQueueHandle& handle, void* dst,
                                           void* src, size_t size,
                                           uint64_t* counter) {
  put_signal_counter_impl<true, false, true>(handle, dst, src, size, nullptr,
                                             counter);
}

/**
 * @brief Signal and counter increment (no copy).
 *
 * @param handle  Queue handle.
 * @param signal  Address of a 64-bit signal counter.
 * @param counter Address of a 64-bit counter.
 *
 * @note Device-only. Non-blocking.
 */
__device__ __forceinline__ void signalCounter(SdmaQueueHandle& handle,
                                              uint64_t* signal,
                                              uint64_t* counter) {
  put_signal_counter_impl<false, true, true>(handle, nullptr, nullptr, 0,
                                             signal, counter);
}

/* ================================================================
 * Device-Side Operations -- Completion Tracking
 * ================================================================ */

/**
 * @brief Wait for a signal to reach a value.
 *
 * Spin-polls the 64-bit value at *addr until it is >=
 * expected. Use after putSignal() or signal() to wait
 * for remote completion.
 *
 * @param addr     Address of a 64-bit signal in device
 *                 memory (uncached recommended).
 * @param expected Minimum value to wait for. Typically
 *                 the number of signals sent.
 *
 * @note Device-only. Blocking (spins until condition
 *       met). Uses agent-scope relaxed atomics.
 */
__device__ __forceinline__ void waitSignal(uint64_t* addr, uint64_t expected) {
  if constexpr (BREAK_ON_RETRIES) {
    poll_until_ge<MAX_RETRIES>(addr, expected);
  } else {
    poll_until_ge<-1>(addr, expected);
  }
}

/**
 * @brief Wait for a counter to reach a value.
 *
 * Identical semantics to waitSignal(); provided as a
 * separate function for clarity when waiting on a
 * counter rather than a signal.
 *
 * @param addr     Address of a 64-bit counter.
 * @param expected Minimum value to wait for.
 *
 * @note Device-only. Blocking.
 */
__device__ __forceinline__ void waitCounter(uint64_t* addr, uint64_t expected) {
  if constexpr (BREAK_ON_RETRIES) {
    poll_until_ge<MAX_RETRIES>(addr, expected);
  } else {
    poll_until_ge<-1>(addr, expected);
  }
}

/**
 * @brief Wait for a specific operation to complete.
 *
 * Spin-polls the hardware read pointer until it reaches
 * or passes upToIndex. Use with the put_index output of
 * put_signal_counter_impl to wait for a specific put
 * without waiting for subsequent signals.
 *
 * @param handle    Queue handle.
 * @param upToIndex Write pointer value to wait for
 *                  (from put_index tracking).
 *
 * @note Device-only. Blocking.
 */
__device__ __forceinline__ void flush(SdmaQueueHandle& handle,
                                      uint64_t upToIndex) {
  handle.flushTo(upToIndex);
}

/**
 * @brief Wait for ALL submitted operations to complete.
 *
 * Spin-polls the hardware read pointer until it reaches
 * maxWritePtr, meaning every packet submitted to this
 * queue has been consumed by the SDMA engine.
 *
 * @param handle Queue handle.
 *
 * @note Device-only. Blocking.
 */
__device__ __forceinline__ void quiet(SdmaQueueHandle& handle) {
  handle.quietAll();
}

/* ================================================================
 * Host-Side Setup Functions
 * ================================================================ */

/**
 * @brief Initialize the SDMA endpoint subsystem.
 *
 * Sets up the HSA runtime, enumerates GPU and CPU
 * agents, and opens the KFD (Kernel Fusion Driver)
 * interface. Must be called before createConnection()
 * or createQueue().
 *
 * Idempotent: safe to call multiple times; subsequent
 * calls are no-ops.
 *
 * @return 0 on success, negative error code on failure.
 */
__host__ int initEndpoint();

/**
 * @brief Mark the SDMA endpoint subsystem as inactive.
 *
 * Resets the internal initialization flag so that
 * subsequent createConnection() / createQueue() calls
 * will fail until initEndpoint() is called again.
 *
 * @note This does NOT destroy existing SDMA queues or
 *       shut down HSA/KFD. Queue and HSA resources are
 *       released when the AnvilLib singleton is
 *       destroyed at process exit. Call destroyQueue()
 *       on individual queues for explicit cleanup.
 * @note Because the underlying HSA init uses
 *       std::call_once, calling initEndpoint() after
 *       shutdownEndpoint() re-enables the flag but
 *       does not re-run HSA/KFD initialization.
 */
__host__ void shutdownEndpoint();

/**
 * @brief Create an SDMA connection between two GPUs.
 *
 * Enables P2P peer access from the source GPU to the
 * destination GPU and resolves the XGMI-topology-
 * optimal SDMA engine ID for this GPU pair (using the
 * MI300X OAM map). For bidirectional transfers, call
 * once for each direction.
 *
 * Must be called after initEndpoint() and before
 * createQueue() for the same GPU pair.
 *
 * @param srcDeviceId Source HIP device ID.
 * @param dstDeviceId Destination HIP device ID.
 * @param info        Output connection information.
 * @return 0 on success, negative error code on failure.
 */
__host__ int createConnection(int srcDeviceId, int dstDeviceId,
                              SdmaConnectionInfo* info);

/**
 * @brief Create an SDMA queue for a GPU pair.
 *
 * Allocates a 1 MiB ring buffer in device memory,
 * creates an SDMA queue via hsakmt, and populates a
 * GPU-accessible device handle (SdmaQueueHandle).
 *
 * Must be called after createConnection() for the same
 * GPU pair. The returned SdmaQueueInfo::deviceHandle is
 * a pointer in device memory that can be passed directly
 * to GPU kernels.
 *
 * @param srcDeviceId Source HIP device ID.
 * @param dstDeviceId Destination HIP device ID.
 * @param info        Output queue information.
 * @return 0 on success, negative error code on failure.
 */
__host__ int createQueue(int srcDeviceId, int dstDeviceId, SdmaQueueInfo* info);

/**
 * @brief Destroy an SDMA queue.
 *
 * Releases the ring buffer, device handle memory, and
 * hsakmt queue resources associated with the given
 * queue.
 *
 * @param info Queue information from createQueue().
 *             The deviceHandle becomes invalid after
 *             this call.
 */
__host__ void destroyQueue(SdmaQueueInfo* info);

/* ================================================================
 * CLI and Validation
 * ================================================================ */

/**
 * @brief Validate SDMA endpoint configuration.
 *
 * Checks that a test subcommand was selected, that
 * --use-counter and --use-flush are not both set, and
 * that transfer-size is > 0 and a multiple of 4.
 *
 * @param config Configuration to validate.
 * @return Empty string if valid, error message otherwise.
 */
__host__ std::string validateConfig(SdmaEpConfig* config);

/**
 * @brief Get the iteration count for this configuration.
 *
 * @param endpointConfig Opaque pointer to SdmaEpConfig.
 * @return Number of iterations to run.
 */
__host__ unsigned getIterations(void* endpointConfig);

/**
 * @brief Run the SDMA endpoint workload.
 */
__host__ hipError_t run(XioEndpointConfig* config);

} // namespace sdma_ep
} // namespace xio

namespace sdma_ep = xio::sdma_ep;

#endif // SDMA_EP_H

#pragma once

// Host/device SDMA packet builders for the host queue.
//
// Trimmed from rocm-xio sdma_packets.hpp @ 34bc5b5: only the types the host
// queue needs (CopyLinear, AtomicAdd, PollRegmem, Timestamp). Self-contained -
// they touch only the vendored packet structs and opcodes, so they build on
// host or device. See PROVENANCE.md.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__host__) && defined(__device__)
#define SDMA_HOST_DEVICE __host__ __device__
#else
#define SDMA_HOST_DEVICE
#endif

#if defined(__forceinline__)
#define SDMA_FORCEINLINE __forceinline__
#else
#define SDMA_FORCEINLINE inline
#endif

#include "sdma_opcodes.h"
#include "sdma_pkt_struct.h"

namespace anvil {
namespace packets {

struct CopyLinearPacket {
  SDMA_PKT_COPY_LINEAR value{};

  SDMA_HOST_DEVICE SDMA_FORCEINLINE explicit CopyLinearPacket(void* src,
                                                              void* dst,
                                                              size_t size) {
    assert(src != nullptr && dst != nullptr &&
           "CopyLinearPacket: nullptr address");
    // COUNT is a 30-bit field storing size-1, so one packet copies <= 1 GiB.
    // Larger host transfers are split by SdmaQueueHostHandle::put (chunking).
    assert(size > 0 && size <= 0x40000000ull &&
           "CopyLinearPacket: size exceeds 1 GiB (30-bit count)");

    value.HEADER_UNION.op = SDMA_OP_COPY;
    value.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR;
    value.COUNT_UNION.count = static_cast<uint32_t>(size - 1);

    uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
    value.SRC_ADDR_LO_UNION.src_addr_31_0 = static_cast<uint32_t>(src_addr);
    value.SRC_ADDR_HI_UNION.src_addr_63_32 =
      static_cast<uint32_t>(src_addr >> 32);

    uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
    value.DST_ADDR_LO_UNION.dst_addr_31_0 = static_cast<uint32_t>(dst_addr);
    value.DST_ADDR_HI_UNION.dst_addr_63_32 =
      static_cast<uint32_t>(dst_addr >> 32);
  }

  SDMA_HOST_DEVICE SDMA_FORCEINLINE const SDMA_PKT_COPY_LINEAR* data() const {
    return &value;
  }
  SDMA_HOST_DEVICE SDMA_FORCEINLINE SDMA_PKT_COPY_LINEAR* data() {
    return &value;
  }
  SDMA_HOST_DEVICE SDMA_FORCEINLINE static constexpr size_t size_bytes() {
    return sizeof(SDMA_PKT_COPY_LINEAR);
  }
};

template <typename T>
struct AtomicAddPacket {
  static_assert(std::is_integral_v<T>,
                "AtomicAddPacket requires integral type");
  static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                "AtomicAddPacket supports 32- or 64-bit values");

  SDMA_PKT_ATOMIC value{};

  SDMA_HOST_DEVICE SDMA_FORCEINLINE AtomicAddPacket(T* ptr, T delta) {
    assert(ptr != nullptr && "AtomicAddPacket: nullptr address");

    value.HEADER_UNION.op = SDMA_OP_ATOMIC;
    value.HEADER_UNION.operation = (sizeof(T) == 8) ? SDMA_ATOMIC_ADD64
                                                    : SDMA_ATOMIC_ADD32;

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    value.ADDR_LO_UNION.addr_31_0 = static_cast<uint32_t>(addr);
    value.ADDR_HI_UNION.addr_63_32 = static_cast<uint32_t>(addr >> 32);

    const uint64_t val64 = static_cast<uint64_t>(delta);
    value.SRC_DATA_LO_UNION.src_data_31_0 = static_cast<uint32_t>(val64);
    value.SRC_DATA_HI_UNION.src_data_63_32 = static_cast<uint32_t>(val64 >> 32);
  }

  SDMA_HOST_DEVICE SDMA_FORCEINLINE const SDMA_PKT_ATOMIC* data() const {
    return &value;
  }
  SDMA_HOST_DEVICE SDMA_FORCEINLINE SDMA_PKT_ATOMIC* data() {
    return &value;
  }
  SDMA_HOST_DEVICE SDMA_FORCEINLINE static constexpr size_t size_bytes() {
    return sizeof(SDMA_PKT_ATOMIC);
  }
};

// POLL_REGMEM: stall the SDMA engine until (*addr & mask) `func` value.
// Used to gate a following copy on a host/peer-written 32-bit flag. Poll is
// bounded by the engine (interval * retry_count); it is NOT an infinite wait.
struct PollRegmemPacket {
  SDMA_PKT_POLL_REGMEM value{};

  // interval: hardware poll interval (engine ticks between reads).
  // retry_count: max poll attempts (12-bit field, saturates at 0xFFF).
  SDMA_HOST_DEVICE SDMA_FORCEINLINE PollRegmemPacket(void* addr,
                                                     uint32_t expected,
                                                     uint32_t mask = 0xFFFFFFFFu,
                                                     uint32_t func =
                                                       SDMA_POLL_FUNC_GEQ,
                                                     uint32_t interval = 0x10,
                                                     uint32_t retry_count =
                                                       0xFFFu) {
    assert(addr != nullptr && "PollRegmemPacket: nullptr address");
    assert(retry_count <= 0xFFFu && "PollRegmemPacket: retry_count > 12 bits");

    value.HEADER_UNION.op = SDMA_OP_POLL_REGMEM;
    value.HEADER_UNION.func = func;
    value.HEADER_UNION.mem_poll = 1; // poll memory (not a register)

    uintptr_t p = reinterpret_cast<uintptr_t>(addr);
    value.ADDR_LO_UNION.addr_31_0 = static_cast<uint32_t>(p);
    value.ADDR_HI_UNION.addr_63_32 = static_cast<uint32_t>(p >> 32);

    value.VALUE_UNION.value = expected;
    value.MASK_UNION.mask = mask;
    value.DW5_UNION.interval = interval & 0xFFFFu;
    value.DW5_UNION.retry_count = retry_count & 0xFFFu;
  }

  SDMA_HOST_DEVICE SDMA_FORCEINLINE const SDMA_PKT_POLL_REGMEM* data() const {
    return &value;
  }
  SDMA_HOST_DEVICE SDMA_FORCEINLINE SDMA_PKT_POLL_REGMEM* data() {
    return &value;
  }
  SDMA_HOST_DEVICE SDMA_FORCEINLINE static constexpr size_t size_bytes() {
    return sizeof(SDMA_PKT_POLL_REGMEM);
  }
};

// TIMESTAMP (global): the engine writes its 64-bit GPU clock to *addr in
// packet order, giving an in-band submission/completion timestamp.
struct TimestampPacket {
  SDMA_PKT_TIMESTAMP value{};

  SDMA_HOST_DEVICE SDMA_FORCEINLINE explicit TimestampPacket(uint64_t* addr) {
    assert(addr != nullptr && "TimestampPacket: nullptr address");

    value.HEADER_UNION.op = SDMA_OP_TIMESTAMP;
    value.HEADER_UNION.sub_op = SDMA_SUBOP_TIMESTAMP_GLOBAL;

    uintptr_t p = reinterpret_cast<uintptr_t>(addr);
    value.ADDR_LO_UNION.addr_31_0 = static_cast<uint32_t>(p);
    value.ADDR_HI_UNION.addr_63_32 = static_cast<uint32_t>(p >> 32);
  }

  SDMA_HOST_DEVICE SDMA_FORCEINLINE const SDMA_PKT_TIMESTAMP* data() const {
    return &value;
  }
  SDMA_HOST_DEVICE SDMA_FORCEINLINE SDMA_PKT_TIMESTAMP* data() {
    return &value;
  }
  SDMA_HOST_DEVICE SDMA_FORCEINLINE static constexpr size_t size_bytes() {
    return sizeof(SDMA_PKT_TIMESTAMP);
  }
};

} // namespace packets
} // namespace anvil

#undef SDMA_FORCEINLINE
#undef SDMA_HOST_DEVICE

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "anvil_device.hpp"
#include "hsa/hsa_ext_amd.h"
#include "hsakmt/hsakmt.h"
#include "hsakmt/hsakmttypes.h"

namespace anvil {

// SDMA channel bucket key. Keying by {src,dst} (not dst alone) avoids
// misrouting when multiple sources target the same destination.
struct ChannelKey {
  int src;
  int dst;

  bool operator==(const ChannelKey& o) const {
    return src == o.src && dst == o.dst;
  }
};

struct ChannelKeyHash {
  size_t operator()(const ChannelKey& k) const {
    return (static_cast<size_t>(static_cast<uint32_t>(k.src)) << 32) ^
           static_cast<uint32_t>(k.dst);
  }
};

class SdmaQueue {
public:
  SdmaQueue(int localDeviceId, int remoteDeviceId, hsa_agent_t& localAgent,
            uint32_t engineId);
  ~SdmaQueue();

  SdmaQueueDeviceHandle* deviceHandle() const;

  // Source (local) / destination (remote) GPU ids this queue moves between.
  int localDeviceId() const { return localDeviceId_; }
  int remoteDeviceId() const { return remoteDeviceId_; }

  void dump(std::ofstream&);

private:
  // Host handle drives this ring from the CPU; needs the ring buffer, KFD
  // rptr/wptr/doorbell, and the host wptr bookkeeping below.
  friend class SdmaQueueHostHandle;

  int localDeviceId_;
  int remoteDeviceId_;
  uint64_t* cachedWptr_;
  uint64_t* committedWptr_;
  void* queueBuffer_;
  HsaQueueResource queue_;
  SdmaQueueDeviceHandle* deviceHandle_;

  // Host-side wptr bookkeeping, separate from the device cachedWptr_/
  // committedWptr_ in GPU memory. Only one driver (host or device) is active at
  // a time, so the two sets never race.
  std::atomic<uint64_t> hostCachedWptr_{0};
  std::atomic<uint64_t> hostCommittedWptr_{0};
};

// CPU-initiated SDMA over an SdmaQueue's ring: the CPU builds packets, copies
// them into the host-accessible ring, advances the wptr, and rings the doorbell.
class SdmaQueueHostHandle {
public:
  explicit SdmaQueueHostHandle(SdmaQueue* q) : queue_(q) {}

  // Linear copy src->dst (single COPY_LINEAR packet).
  void put(void* dst, void* src, size_t size);

  // Atomic-add `value` into *ptr (single ATOMIC packet). T is 32- or 64-bit.
  template <typename T>
  void signal(T* ptr, T value);

  // COPY_LINEAR then ATOMIC-add in one doorbell, so the flag is visible only
  // after the copy drains.
  template <typename T>
  void put_signal(void* dst, void* src, size_t size, T* flag_ptr, T flag_value);

  // POLL_REGMEM (wait until (*flag_ptr & mask) >= expected) then COPY_LINEAR in
  // one doorbell. T must be 32-bit; poll is bounded by interval * retry_count.
  template <typename T>
  void wait_flag_then_put(T* flag_ptr, T expected, void* dst, void* src,
                          size_t size, uint32_t mask = 0xFFFFFFFFu,
                          uint32_t interval = 0x10,
                          uint32_t retry_count = 0xFFFu);

  // TIMESTAMP packet: engine writes its 64-bit GPU clock to *ts_ptr in order.
  void timestamp(uint64_t* ts_ptr);

  // Block until the hardware has consumed everything submitted so far.
  void quiet();

private:
  uint64_t reserveQueueSpace(size_t size_in_bytes);
  void placePacket(const void* packet, size_t packet_size, uint64_t index);
  void submitPacket(uint64_t base, uint64_t pending_wptr);
  bool canWriteUpto(uint64_t uptoIndex);
  uint64_t wrapIntoRing(uint64_t index) const;
  void padRingToEnd(uint64_t cur_index);
  // Reserve, place, and submit a batch of packets under a single doorbell ring.
  void submitBatch(
    std::initializer_list<std::pair<const void*, size_t>> packets);

  SdmaQueue* queue_;
};

class AnvilLib {
private:
  // Make constructor private
  AnvilLib() = default;

public:
  ~AnvilLib();
  // access to singleton
  static AnvilLib& getInstance();

  AnvilLib(const AnvilLib&) = delete;
  AnvilLib& operator=(const AnvilLib&) = delete;

public:
  void init();
  bool connect(int srcDeviceId, int dstDeviceId, int numChannels = 1);
  SdmaQueue* getSdmaQueue(int srcDeviceId, int dstDeviceId, int channelIdx = 0);
  SdmaQueue* createSdmaQueue(int srcDeviceId, int dstDeviceId,
                             uint32_t engineId, int* channelIdx = nullptr);
  int getSdmaEngineId(int srcDeviceId, int dstDeviceId, int rotation = 0);

  // Host-initiated queue API. Host channels are independent CPU-driven SDMA
  // queues, separate from the device (GISDMA) channels. connectHost() is
  // idempotent; getHostHandle() returns nullptr for an unknown/out-of-range
  // {src,dst,channel}.
  bool connectHost(int srcDeviceId, int dstDeviceId, int numChannels = 1);
  SdmaQueueHostHandle* getHostHandle(int srcDeviceId, int dstDeviceId,
                                     int channelIdx = 0);

private:
  /*
   * MI300X OAM MAP (XGMI topology -> SDMA engine)
   * src\dst  0  1  2  3  4  5  6  7
   * 0        0  7  6  1  2  4  5  3
   * 1        7  0  1  5  4  2  3  6
   * 2        5  1  0  6  7  3  2  4
   * 3        1  6  5  0  3  7  4  2
   * 4        2  4  7  3  0  5  6  1
   * 5        4  2  3  7  6  0  1  5
   * 6        5  3  2  4  6  1  0  7
   * 7        3  6  4  2  1  5  7  0
   */
  std::array<std::array<int, 8>, 8> mi300xOamMap = {{{0, 7, 6, 1, 2, 4, 5, 3},
                                                     {7, 0, 1, 5, 4, 2, 3, 6},
                                                     {5, 1, 0, 6, 7, 3, 2, 4},
                                                     {1, 6, 5, 0, 3, 7, 4, 2},
                                                     {2, 4, 7, 3, 0, 5, 6, 1},
                                                     {4, 2, 3, 7, 6, 0, 1, 5},
                                                     {5, 3, 2, 4, 6, 1, 0, 7},
                                                     {3, 6, 4, 2, 1, 5, 7, 0}}};

  int getOamId(int deviceId);

  std::once_flag init_flag;
  std::unordered_map<ChannelKey, std::vector<std::unique_ptr<SdmaQueue>>,
                     ChannelKeyHash>
    sdma_channels_;

  // Host channels: each owns its SdmaQueue and the handle driving it, keyed by
  // {src,dst} like sdma_channels_.
  struct HostChannel {
    std::unique_ptr<SdmaQueue> queue;
    std::unique_ptr<SdmaQueueHostHandle> handle;
  };
  std::unordered_map<ChannelKey, std::vector<HostChannel>, ChannelKeyHash>
    host_sdma_channels_;
};

extern AnvilLib& anvil;

void EnablePeerAccess(int deviceId, int peerDeviceId);

} // namespace anvil

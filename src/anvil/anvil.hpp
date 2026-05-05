#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "anvil_device.hpp"
#include "hsa/hsa_ext_amd.h"
#include "hsakmt/hsakmt.h"
#include "hsakmt/hsakmttypes.h"

namespace anvil {

class SdmaQueue {
public:
  SdmaQueue(int localDeviceId, int remoteDeviceId, hsa_agent_t& localAgent,
            uint32_t engineId);
  ~SdmaQueue();

  SdmaQueueDeviceHandle* deviceHandle() const;

  void dump(std::ofstream&);

private:
  uint64_t* cachedWptr_;
  uint64_t* committedWptr_;
  void* queueBuffer_;
  HsaQueueResource queue_;
  SdmaQueueDeviceHandle* deviceHandle_;
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
  int getSdmaEngineId(int srcDeviceId, int dstDeviceId);

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
  std::unordered_map<int, std::vector<std::unique_ptr<SdmaQueue>>>
    sdma_channels_;
};

extern AnvilLib& anvil;

void EnablePeerAccess(int deviceId, int peerDeviceId);

} // namespace anvil

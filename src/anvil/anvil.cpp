#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

#include "anvil.hpp"
#include "sdma_packets.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>

namespace anvil {

static bool AnvilVerbose() {
  static bool v = (getenv("TB_VERBOSE") && atoi(getenv("TB_VERBOSE")) > 0);
  return v;
}

#define ANVIL_LOG(...) do { if (AnvilVerbose()) { printf("[ANVIL] " __VA_ARGS__); fflush(stdout); } } while(0)

[[maybe_unused]] auto checkHsaError = [](hsa_status_t s, const char* msg,
                                         const char* file, int line) {
  if (s != HSA_STATUS_SUCCESS) {
    const char* hsa_err_msg;
    hsa_status_string(s, &hsa_err_msg);
    throw(std::runtime_error{std::string("HSA error at ") + file +
                             std::string(":") + std::to_string(line) +
                             std::string(" - ") + hsa_err_msg});
  }
};

#define CHECK_HSA_ERROR(cmd) checkHsaError((cmd), #cmd, __FILE__, __LINE__)

#define CHECK_HSAKMT_SUCCESS(call, msg)                                        \
  do {                                                                         \
    if ((call) != HSAKMT_STATUS_SUCCESS) {                                     \
      std::cout << "ERROR code: " << std::dec << call << " " << msg            \
                << " (File: " << __FILE__ << ", Line: " << __LINE__ << ")"     \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

inline void checkHipError(hipError_t err, const char* msg, const char* file,
                          int line) {
  if (err != hipSuccess) {
    std::cerr << "HIP error at " << file << ":" << line << " — " << msg << "\n"
              << "  Code: " << err << " (" << hipGetErrorString(err) << ")"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_HIP_ERROR(cmd) checkHipError((cmd), #cmd, __FILE__, __LINE__)

// Allow access to peerDeviceId from deviceId
void EnablePeerAccess(int const deviceId, int const peerDeviceId) {
  int canAccess;
  CHECK_HIP_ERROR(hipDeviceCanAccessPeer(&canAccess, deviceId, peerDeviceId));
  if (!canAccess) {
    std::cerr << "Unable to enable peer access from GPU devices " << deviceId
              << " to " << peerDeviceId << "\n";
  }

  CHECK_HIP_ERROR(hipSetDevice(deviceId));
  hipError_t error = hipDeviceEnablePeerAccess(peerDeviceId, 0);
  if (error != hipSuccess && error != hipErrorPeerAccessAlreadyEnabled) {
    std::cerr << "Unable to enable peer to peer access from " << deviceId
              << "  to " << peerDeviceId << " (" << hipGetErrorString(error)
              << ")\n";
  }
}

// HSA agents
std::vector<hsa_agent_t> cpuAgents_;
std::vector<hsa_agent_t> gpuAgents_;

hsa_status_t rocm_hsa_agent_callback(hsa_agent_t agent,
                                     hsa_device_type_t target_device_type,
                                     [[maybe_unused]] void* vector) {
  std::vector<hsa_agent_t>* agents = static_cast<std::vector<hsa_agent_t>*>(
    vector);
  hsa_device_type_t device_type{};
  hsa_status_t status{
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type)};
  if (status != HSA_STATUS_SUCCESS) {
    printf("Failure to get device type: 0x%x", status);
    return status;
  }
  if (device_type == target_device_type) {
    agents->push_back(agent);
  }
  return status;
}

hsa_status_t rocm_hsa_gpu_agent_callback(hsa_agent_t agent,
                                         [[maybe_unused]] void* context) {
  return rocm_hsa_agent_callback(agent, HSA_DEVICE_TYPE_GPU, context);
}
hsa_status_t rocm_hsa_cpu_agent_callback(hsa_agent_t agent,
                                         [[maybe_unused]] void* context) {
  return rocm_hsa_agent_callback(agent, HSA_DEVICE_TYPE_CPU, context);
}

void SetUpKFD() {
  CHECK_HSAKMT_SUCCESS(hsaKmtOpenKFD(), "hsaKmtOpenKFD() failed!");
  HsaSystemProperties m_SystemProperties;
  std::memset(&m_SystemProperties, 0, sizeof(m_SystemProperties));
  CHECK_HSAKMT_SUCCESS(hsaKmtAcquireSystemProperties(&m_SystemProperties),
                       "Failed!");
}

// True only after init() has run (SetUpKFD). Avoids CloseKFD/hsa_shut_down at
// exit when user never ran sdma-ep (e.g. xio-tester -h).
static bool s_kfd_opened = false;

void CloseKFD() {
  CHECK_HSAKMT_SUCCESS(hsaKmtCloseKFD(), "hsaKmtCloseKFD() failed");
}

// Convert a logical deviceId index to the NVML device minor number
static const std::string getBusId(int deviceId) {
  // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
  // format. Still need to allocate proper space in case PCI domain goes
  // higher.
  char busIdChar[] = "00000000:00:00.0";
  CHECK_HIP_ERROR(hipDeviceGetPCIBusId(busIdChar, sizeof(busIdChar), deviceId));
  // we need the hex in lower case format
  for (size_t i = 0; i < sizeof(busIdChar); i++) {
    busIdChar[i] = std::tolower(busIdChar[i]);
  }
  return std::string(busIdChar);
}

std::ostream& operator<<(std::ostream& os, const SDMA_PKT_COPY_LINEAR& cmd) {
  os << "op: " << cmd.HEADER_UNION.op << " sub_op: " << cmd.HEADER_UNION.sub_op;
  os << " count: " << cmd.COUNT_UNION.count;
  os << " src[31:0] " << cmd.SRC_ADDR_LO_UNION.src_addr_31_0 << " src[63:32] "
     << cmd.SRC_ADDR_HI_UNION.src_addr_63_32;
  os << " dst[31:0] " << cmd.DST_ADDR_LO_UNION.dst_addr_31_0 << " dst[63:32] "
     << cmd.DST_ADDR_HI_UNION.dst_addr_63_32;
  return os;
}

std::ostream& operator<<(std::ostream& os, const SDMA_PKT_ATOMIC& cmd) {
  os << "op: " << cmd.HEADER_UNION.op << " sub_op: " << cmd.HEADER_UNION.sub_op
     << " operation: " << cmd.HEADER_UNION.operation;
  os << " add[31:0] " << cmd.ADDR_LO_UNION.addr_31_0 << " addr[63:32] "
     << cmd.ADDR_HI_UNION.addr_63_32;
  os << " data[31:0] " << cmd.SRC_DATA_LO_UNION.src_data_31_0 << " data[63:32] "
     << cmd.SRC_DATA_HI_UNION.src_data_63_32;
  return os;
}

#if XIO_SDMA_OSS7

std::ostream& operator<<(std::ostream& os,
                         const SDMA_PKT_COPY_LINEAR_PHY_MI4& cmd) {
  os << "MI4_PHY op: " << cmd.HEADER_UNION.op_code
     << " sub_op: " << cmd.HEADER_UNION.sub_op_code
     << " count: " << cmd.COUNT_UNION.count;
  os << " src[31:0] " << cmd.SRC_ADDR_LO_UNION.src_address_lo << " src[63:32] "
     << cmd.SRC_ADDR_HI_UNION.src_address_hi;
  os << " dst[31:0] " << cmd.DST_ADDR_LO_UNION.dst_address_lo << " dst[63:32] "
     << cmd.DST_ADDR_HI_UNION.dst_address_hi;
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4& cmd) {
  os << "MI4_WS op: " << cmd.HEADER_UNION.op
     << " sub_op: " << cmd.HEADER_UNION.subop
     << " wait: " << cmd.HEADER_UNION.wait
     << " signal: " << cmd.HEADER_UNION.signal
     << " count: " << cmd.COPY_COUNT_UNION.copy_count;
  os << " src[31:0] " << cmd.SRC_ADDR_LO_UNION.src_addr_31_0 << " src[63:32] "
     << cmd.SRC_ADDR_HI_UNION.src_addr_63_32;
  os << " dst[31:0] " << cmd.DST_ADDR_LO_UNION.dst_addr_31_0 << " dst[63:32] "
     << cmd.DST_ADDR_HI_UNION.dst_addr_63_32;
  if (cmd.HEADER_UNION.signal) {
    os << " sig_op: " << cmd.SIGNAL_CTRL_UNION.signal_operation;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const SDMA_PKT_FENCE_MI4& cmd) {
  os << "MI4_FENCE op: " << cmd.HEADER_UNION.op_code
     << " sub_op: " << cmd.HEADER_UNION.sub_op_code
     << " data: " << cmd.DATA_UNION.data;
  os << " addr[31:0] " << cmd.ADDR_LO_UNION.fence_addr_lo << " addr[63:32] "
     << cmd.ADDR_HI_UNION.fence_addr_hi;
  return os;
}

#endif /* XIO_SDMA_OSS7 */

SdmaQueue::SdmaQueue(int localDeviceId, int remoteDeviceId,
                     hsa_agent_t& localAgent, uint32_t engineId)
    : localDeviceId_(localDeviceId), remoteDeviceId_(remoteDeviceId) {
  int originalDeviceId;
  CHECK_HIP_ERROR(hipGetDevice(&originalDeviceId));
  (void)originalDeviceId;

  uint32_t localNodeId;
  hsa_status_t status = hsa_agent_get_info(localAgent, HSA_AGENT_INFO_NODE,
                                           &localNodeId);
  if (status != HSA_STATUS_SUCCESS) {
    printf("Failure to get device info: 0x%x", status);
    // return status;
  }

  ANVIL_LOG("SdmaQueue ctor: GPU %d -> GPU %d  engine %u  node %u\n",
            localDeviceId, remoteDeviceId, engineId, localNodeId);

  // Allocate SDMA queue buffer on device side, requires ExecuteAccess
  HsaMemFlags memFlags = {};
  memFlags.ui32.NonPaged = 1;
  memFlags.ui32.HostAccess = 1;
  memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
  memFlags.ui32.NoNUMABind = 1;
  memFlags.ui32.ExecuteAccess = 1;
  memFlags.ui32.Uncached = 1;
  CHECK_HSAKMT_SUCCESS(hsaKmtAllocMemory(localNodeId, SDMA_QUEUE_SIZE, memFlags,
                                         &queueBuffer_),
                       "Failed");
  ANVIL_LOG("  hsaKmtAllocMemory:   queueBuffer_=%p  size=%zu bytes\n",
            queueBuffer_, (size_t)SDMA_QUEUE_SIZE);
  CHECK_HSAKMT_SUCCESS(hsaKmtMapMemoryToGPU(queueBuffer_, SDMA_QUEUE_SIZE,
                                            NULL),
                       "Failed");
  ANVIL_LOG("  hsaKmtMapMemoryToGPU: queueBuffer_=%p mapped to GPU node %u\n",
            queueBuffer_, localNodeId);

  // Create SDMA Queue
  std::memset(&queue_, 0, sizeof(HsaQueueResource));

  CHECK_HSAKMT_SUCCESS(
    hsaKmtCreateQueueExt(localNodeId, HSA_QUEUE_SDMA_BY_ENG_ID,
                         DEFAULT_QUEUE_PERCENTAGE, DEFAULT_PRIORITY, engineId,
                         queueBuffer_, SDMA_QUEUE_SIZE, nullptr, &queue_),
    "Failed");
  ANVIL_LOG("  hsaKmtCreateQueueExt: QueueId=%lu  rptr=%p  wptr=%p  doorbell=%p\n",
            queue_.QueueId,
            queue_.Queue_read_ptr_aql,
            queue_.Queue_write_ptr_aql,
            queue_.Queue_DoorBell_aql);

  // Populate Device Handle (GPU-accessible pointers, allocated uncached)
  {
    if (hipMalloc((void**)&deviceHandle_, sizeof(SdmaQueueDeviceHandle)) != hipSuccess)
      throw std::runtime_error("hipMalloc(deviceHandle_) failed");
    ANVIL_LOG("  hipMalloc:           deviceHandle_=%p  size=%zu bytes\n",
              (void*)deviceHandle_, sizeof(SdmaQueueDeviceHandle));
    if (hipExtMallocWithFlags((void**)&cachedWptr_, sizeof(uint64_t),
                              hipDeviceMallocUncached) != hipSuccess)
      throw std::runtime_error("hipExtMallocWithFlags(cachedWptr_) failed");
    ANVIL_LOG("  hipExtMalloc(UC):    cachedWptr_=%p\n", (void*)cachedWptr_);
    if (hipExtMallocWithFlags((void**)&committedWptr_, sizeof(uint64_t),
                              hipDeviceMallocUncached) != hipSuccess)
      throw std::runtime_error("hipExtMallocWithFlags(committedWptr_) failed");
    ANVIL_LOG("  hipExtMalloc(UC):    committedWptr_=%p\n", (void*)committedWptr_);
  }

  uint64_t cachedWptr = *reinterpret_cast<uint64_t*>(
    queue_.Queue_write_ptr_aql);
  uint64_t committedWptr = *reinterpret_cast<uint64_t*>(
    queue_.Queue_write_ptr_aql);
  SdmaQueueDeviceHandle handle = {
    .queueBuf = static_cast<uint32_t*>(queueBuffer_),
    .rptr = queue_.Queue_read_ptr_aql,
    .wptr = queue_.Queue_write_ptr_aql,
    .doorbell = queue_.Queue_DoorBell_aql,
    .cachedWptr = cachedWptr_,
    .committedWptr = committedWptr_,
    .cachedHwReadIndex = *reinterpret_cast<uint64_t*>(
      queue_.Queue_read_ptr_aql),
    .maxWritePtr = *reinterpret_cast<uint64_t*>(queue_.Queue_read_ptr_aql),
  };

  CHECK_HIP_ERROR(hipMemcpy(deviceHandle_, &handle,
                            sizeof(SdmaQueueDeviceHandle),
                            hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(cachedWptr_, &cachedWptr, sizeof(uint64_t),
                            hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(committedWptr_, &committedWptr, sizeof(uint64_t),
                            hipMemcpyHostToDevice));

  // Seed host-side wptr bookkeeping so a SdmaQueueHostHandle can drive this ring.
  hostCachedWptr_.store(cachedWptr, std::memory_order_relaxed);
  hostCommittedWptr_.store(committedWptr, std::memory_order_relaxed);
}

SdmaQueue::~SdmaQueue() {
  ANVIL_LOG("SdmaQueue dtor: QueueId=%lu  queueBuffer_=%p\n",
            queue_.QueueId, queueBuffer_);
  CHECK_HSAKMT_SUCCESS(hsaKmtDestroyQueue(queue_.QueueId),
                       "Failed to destroy queue.");
  ANVIL_LOG("  hsaKmtDestroyQueue:  QueueId=%lu done\n", queue_.QueueId);
  (void)hipFree(deviceHandle_);
  ANVIL_LOG("  hipFree:             deviceHandle_=%p\n", (void*)deviceHandle_);
  (void)hipFree(cachedWptr_);
  ANVIL_LOG("  hipFree:             cachedWptr_=%p\n", (void*)cachedWptr_);
  (void)hipFree(committedWptr_);
  ANVIL_LOG("  hipFree:             committedWptr_=%p\n", (void*)committedWptr_);
  CHECK_HSAKMT_SUCCESS(hsaKmtUnmapMemoryToGPU(queueBuffer_), "Failed");
  ANVIL_LOG("  hsaKmtUnmapMemoryToGPU: queueBuffer_=%p done\n", queueBuffer_);
  CHECK_HSAKMT_SUCCESS(hsaKmtFreeMemory(queueBuffer_, SDMA_QUEUE_SIZE),
                       "Failed");
  ANVIL_LOG("  hsaKmtFreeMemory:    queueBuffer_=%p  size=%zu bytes done\n",
            queueBuffer_, (size_t)SDMA_QUEUE_SIZE);
}

SdmaQueueDeviceHandle* SdmaQueue::deviceHandle() const {
  return deviceHandle_;
}

void SdmaQueue::dump(std::ofstream& logFile) {
  logFile << "Queue "
          //  << " -> " << remoteDeviceId_ << ": "
          << "wptr: " << *deviceHandle_->wptr << ", "
          << "rptr: " << *deviceHandle_->rptr << ", "
          << "doorbell: " << *deviceHandle_->doorbell << ", "
          << "Queue cmd buffer address: " << deviceHandle_->queueBuf << ", "
          << "committedWptr: " << *deviceHandle_->committedWptr << ", "
          << "pendingWptr: " << *deviceHandle_->cachedWptr << ", " << std::endl;

  size_t dw_enqueued = std::min(*deviceHandle_->wptr,
                                (uint64_t)SDMA_QUEUE_SIZE) /
                       sizeof(uint32_t);
  size_t it = 0;
  uint32_t* dwPtr = deviceHandle_->queueBuf;
  uint64_t wrapped_rptr = *deviceHandle_->rptr % SDMA_QUEUE_SIZE;
  uint64_t wrapped_wptr = *deviceHandle_->wptr % SDMA_QUEUE_SIZE;

  logFile << "valid dw: " << dw_enqueued << "\nwrapped rptr : " << wrapped_rptr
          << " dw rptr: " << wrapped_rptr / sizeof(uint32_t)
          << "\nwrapper wptr " << wrapped_wptr
          << " dw wptr: " << wrapped_wptr / sizeof(uint32_t) << std::endl;
  while (it < dw_enqueued) {
    logFile << "[" << it << "] ";
    uint32_t opcode = *dwPtr & 0xFF;
    [[maybe_unused]] uint32_t subop = (*dwPtr >> 8) & 0xFF;
    if (opcode == SDMA_OP_COPY) {
#if XIO_SDMA_OSS7
      if (subop == SDMA_SUBOP_COPY_LINEAR_PHY_MI4) {
        auto* ptr = reinterpret_cast<SDMA_PKT_COPY_LINEAR_PHY_MI4*>(dwPtr);
        logFile << *ptr;
        constexpr size_t dw = sizeof(SDMA_PKT_COPY_LINEAR_PHY_MI4) /
                              sizeof(uint32_t);
        it += dw;
        dwPtr += dw;
      } else if (subop == SDMA_SUBOP_COPY_LINEAR_WAIT_SIGNAL_MI4) {
        auto* ptr = reinterpret_cast<SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4*>(
          dwPtr);
        logFile << *ptr;
        constexpr size_t dw = sizeof(SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4) /
                              sizeof(uint32_t);
        it += dw;
        dwPtr += dw;
      } else
#endif /* XIO_SDMA_OSS7 */
      {
        auto* ptr = reinterpret_cast<SDMA_PKT_COPY_LINEAR*>(dwPtr);
        logFile << *ptr;
        size_t dw = sizeof(SDMA_PKT_COPY_LINEAR) / sizeof(uint32_t);
        it += dw;
        dwPtr += dw;
      }
    } else if (opcode == SDMA_OP_ATOMIC) {
      auto* ptr = reinterpret_cast<SDMA_PKT_ATOMIC*>(dwPtr);
      logFile << *ptr;
      size_t dw = sizeof(SDMA_PKT_ATOMIC) / sizeof(uint32_t);
      it += dw;
      dwPtr += dw;
#if XIO_SDMA_OSS7
    } else if (opcode == SDMA_OP_FENCE && subop == SDMA_SUBOP_FENCE_MI4) {
      auto* ptr = reinterpret_cast<SDMA_PKT_FENCE_MI4*>(dwPtr);
      logFile << *ptr;
      constexpr size_t dw = sizeof(SDMA_PKT_FENCE_MI4) / sizeof(uint32_t);
      it += dw;
      dwPtr += dw;
#endif /* XIO_SDMA_OSS7 */
    } else {
      logFile << *dwPtr << std::endl;
      dwPtr++;
      it++;
    }
    logFile << std::endl;
  }
  logFile << "Queue "
          //  << " -> " << remoteDeviceId_ << ": "
          << "wptr: " << *deviceHandle_->wptr << ", "
          << "rptr: " << *deviceHandle_->rptr << ", "
          << "doorbell: " << *deviceHandle_->doorbell << ", "
          << "Queue cmd buffer address: " << deviceHandle_->queueBuf << ", "
          << "committedWptr: " << *deviceHandle_->committedWptr << ", "
          << "pendingWptr: " << *deviceHandle_->cachedWptr << ", " << std::endl;
}

AnvilLib::~AnvilLib() {
  for (auto& p : sdma_channels_) {
    p.second.clear();
  }
  for (auto& p : host_sdma_channels_) {
    p.second.clear();
  }
  if (s_kfd_opened) {
    CloseKFD();
    hsa_shut_down();
  }
}

void AnvilLib::init() {
  std::call_once(init_flag, []() {
    //   std::atexit(CloseKFD); // Register cleanup

    // HSA
    hsa_status_t status{hsa_init()};
    if (status != HSA_STATUS_SUCCESS) {
      printf("Failure to open HSA connection: 0x%x", status);
      // return 1;
    }
    status = hsa_iterate_agents(&rocm_hsa_gpu_agent_callback, &gpuAgents_);
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
      printf("Failure to iterate HSA agents: 0x%x", status);
      // return 1;
    }
    status = hsa_iterate_agents(&rocm_hsa_cpu_agent_callback, &cpuAgents_);
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
      printf("Failure to iterate HSA agents: 0x%x", status);
      // return 1;
    }

    SetUpKFD();
    s_kfd_opened = true;
  });
}

SdmaQueue* AnvilLib::createSdmaQueue(int srcDeviceId, int dstDeviceId,
                                     uint32_t engineId, int* channelIdx) {
  auto& vec = sdma_channels_[ChannelKey{srcDeviceId, dstDeviceId}];
  vec.emplace_back(std::make_unique<SdmaQueue>(srcDeviceId, dstDeviceId,
                                               gpuAgents_[srcDeviceId],
                                               engineId));
  if (channelIdx)
    *channelIdx = static_cast<int>(vec.size() - 1);
  return vec.back().get();
}

bool AnvilLib::connect(int srcDeviceId, int dstDeviceId, int numChannels) {
  if (numChannels <= 0) {
    throw std::invalid_argument("connect(): numChannels must be positive");
  }

  // Idempotent: only create the channels still missing for this {src,dst}.
  auto& queues = sdma_channels_[ChannelKey{srcDeviceId, dstDeviceId}];
  const int existingChannels = static_cast<int>(queues.size());
  if (existingChannels >= numChannels) {
    return true;
  }

  const uint32_t engineId = getSdmaEngineId(srcDeviceId, dstDeviceId);
  std::cout << "Connect from " << srcDeviceId << " to " << dstDeviceId
            << " with " << numChannels << " channels using engine " << engineId
            << std::endl;
  for (int c = existingChannels; c < numChannels; ++c) {
    createSdmaQueue(srcDeviceId, dstDeviceId, engineId);
  }
  return true;
}

SdmaQueue* AnvilLib::getSdmaQueue(int srcDeviceId, int dstDeviceId,
                                  int channel_idx) {
  auto it = sdma_channels_.find(ChannelKey{srcDeviceId, dstDeviceId});
  if (it == sdma_channels_.end()) {
    return nullptr;
  }

  const auto& channels = it->second;
  if (channel_idx < 0 || static_cast<size_t>(channel_idx) >= channels.size()) {
    return nullptr;
  }

  return channels[static_cast<size_t>(channel_idx)].get();
}

// ==================== SdmaQueueHostHandle (Phase 5) ====================
// CPU-driven SDMA submission: reserve / wrap-pad / place / doorbell over the
// same host-accessible, uncached KFD ring the device handle uses. A queue is
// driven by EITHER the host handle OR the device kernel, never both at once.

uint64_t SdmaQueueHostHandle::wrapIntoRing(uint64_t index) const {
  return index % SDMA_QUEUE_SIZE;
}

bool SdmaQueueHostHandle::canWriteUpto(uint64_t uptoIndex) {
  uint64_t hw_read_index = *queue_->queue_.Queue_read_ptr_aql;
  return (uptoIndex - hw_read_index) < SDMA_QUEUE_SIZE;
}

void SdmaQueueHostHandle::padRingToEnd(uint64_t cur_index) {
  uint64_t padding_size = SDMA_QUEUE_SIZE - wrapIntoRing(cur_index);
  uint64_t new_index = cur_index + padding_size;

  if (!canWriteUpto(new_index)) {
    return;
  }

  if (queue_->hostCachedWptr_.compare_exchange_weak(
        cur_index, new_index, std::memory_order_release,
        std::memory_order_relaxed)) {
    uint64_t num_nops = padding_size / sizeof(uint32_t);
    uint32_t* queueBuf = static_cast<uint32_t*>(queue_->queueBuffer_);
    uint64_t offset_dwords = wrapIntoRing(cur_index) / sizeof(uint32_t);
    for (uint64_t i = 0; i < num_nops; i++) {
      queueBuf[offset_dwords + i] = SDMA_OP_NOP;
    }
    submitPacket(cur_index, new_index);
  }
}

uint64_t SdmaQueueHostHandle::reserveQueueSpace(size_t size_in_bytes) {
  uint64_t cur_index;
  while (true) {
    cur_index = queue_->hostCachedWptr_.load(std::memory_order_relaxed);
    uint64_t new_index = cur_index + size_in_bytes;

    // Would this packet cross the ring boundary? Pad to end and retry.
    if (wrapIntoRing(cur_index) + size_in_bytes > SDMA_QUEUE_SIZE) {
      padRingToEnd(cur_index);
      continue;
    }
    // Wait for the hardware to free space if the ring is full.
    if (!canWriteUpto(new_index)) {
      std::this_thread::yield();
      continue;
    }
    if (queue_->hostCachedWptr_.compare_exchange_weak(
          cur_index, new_index, std::memory_order_release,
          std::memory_order_relaxed)) {
      break;
    }
    std::this_thread::yield();
  }
  return cur_index;
}

void SdmaQueueHostHandle::placePacket(const void* packet, size_t packet_size,
                                      uint64_t index) {
  uint64_t wrapped_index = wrapIntoRing(index);
  char* queueBuf = static_cast<char*>(queue_->queueBuffer_) + wrapped_index;
  std::memcpy(queueBuf, packet, packet_size);
}

void SdmaQueueHostHandle::submitPacket(uint64_t base, uint64_t pending_wptr) {
  // Advance the wptr only after the previous submission committed, so packets
  // reach the engine in ring order.
  while (queue_->hostCommittedWptr_.load(std::memory_order_acquire) != base) {
    std::this_thread::yield();
  }
  std::atomic_thread_fence(std::memory_order_release);
  *queue_->queue_.Queue_write_ptr_aql = pending_wptr;
  std::atomic_thread_fence(std::memory_order_release);
  *queue_->queue_.Queue_DoorBell_aql = pending_wptr;
  queue_->hostCommittedWptr_.store(pending_wptr, std::memory_order_release);
}

void SdmaQueueHostHandle::submitBatch(
  std::initializer_list<std::pair<const void*, size_t>> packets) {
  if (packets.size() == 0) {
    return;
  }
  size_t total_size = 0;
  for (const auto& p : packets) {
    total_size += p.second;
  }
  // Wraparound reserves one NOP, so a batch above SDMA_QUEUE_SIZE - sizeof(NOP)
  // would spin forever in reserveQueueSpace.
  constexpr size_t kMaxSubmitBytes = SDMA_QUEUE_SIZE - sizeof(uint32_t);
  if (total_size > kMaxSubmitBytes) {
    throw std::invalid_argument(
      "SdmaQueueHostHandle::submit: batch size " + std::to_string(total_size) +
      " bytes exceeds the maximum single submission of " +
      std::to_string(kMaxSubmitBytes) + " bytes");
  }

  uint64_t offset = reserveQueueSpace(total_size);
  uint64_t current = offset;
  for (const auto& p : packets) {
    placePacket(p.first, p.second, current);
    current += p.second;
  }
  submitPacket(offset, offset + total_size);
}

// Max bytes per COPY_LINEAR packet: 1 GiB HW limit, lowered via ANVIL_CHUNK_SIZE
// (raw bytes or K/M/G suffix). Read per call so tests can vary it at runtime.
static size_t anvilHostChunkBytes() {
  constexpr size_t kMax = 0x40000000ull; // 1 GiB
  char const* v = getenv("ANVIL_CHUNK_SIZE");
  if (!v || !*v)
    return kMax;
  char* end = nullptr;
  unsigned long long val = strtoull(v, &end, 10);
  if (end && *end) {
    switch (*end) {
      case 'g': case 'G': val <<= 30; break;
      case 'm': case 'M': val <<= 20; break;
      case 'k': case 'K': val <<= 10; break;
      default: break;
    }
  }
  if (val == 0 || val > kMax)
    return kMax;
  return static_cast<size_t>(val);
}

void SdmaQueueHostHandle::put(void* dst, void* src, size_t size) {
  if (!src || !dst || size == 0) {
    throw std::invalid_argument("SdmaQueueHostHandle::put: invalid args");
  }
  // Split into <=chunk COPY_LINEAR packets (one doorbell each).
  size_t const chunk = anvilHostChunkBytes();
  auto* d = static_cast<uint8_t*>(dst);
  auto* s = static_cast<uint8_t*>(src);
  size_t off = 0;
  while (size - off > chunk) {
    packets::CopyLinearPacket copy(s + off, d + off, chunk);
    submitBatch({{copy.data(), copy.size_bytes()}});
    off += chunk;
  }
  packets::CopyLinearPacket copy(s + off, d + off, size - off);
  submitBatch({{copy.data(), copy.size_bytes()}});
}

template <typename T>
void SdmaQueueHostHandle::signal(T* ptr, T value) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                "signal only supports 32-bit or 64-bit types");
  if (!ptr) {
    throw std::invalid_argument("SdmaQueueHostHandle::signal: nullptr");
  }
  packets::AtomicAddPacket<T> atom(ptr, value);
  submitBatch({{atom.data(), atom.size_bytes()}});
}

template <typename T>
void SdmaQueueHostHandle::put_signal(void* dst, void* src, size_t size,
                                     T* flag_ptr, T flag_value) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                "put_signal only supports 32-bit or 64-bit flag types");
  if (!src || !dst || size == 0 || !flag_ptr) {
    throw std::invalid_argument("SdmaQueueHostHandle::put_signal: invalid args");
  }
  // Only the final chunk carries the flag, so it fires once after all copies
  // drain (in-order queue).
  size_t const chunk = anvilHostChunkBytes();
  auto* d = static_cast<uint8_t*>(dst);
  auto* s = static_cast<uint8_t*>(src);
  size_t off = 0;
  while (size - off > chunk) {
    packets::CopyLinearPacket copy(s + off, d + off, chunk);
    submitBatch({{copy.data(), copy.size_bytes()}});
    off += chunk;
  }
  packets::CopyLinearPacket copy(s + off, d + off, size - off);
  packets::AtomicAddPacket<T> atom(flag_ptr, flag_value);
  submitBatch({{copy.data(), copy.size_bytes()},
               {atom.data(), atom.size_bytes()}});
}

template <typename T>
void SdmaQueueHostHandle::wait_flag_then_put(T* flag_ptr, T expected, void* dst,
                                             void* src, size_t size,
                                             uint32_t mask, uint32_t interval,
                                             uint32_t retry_count) {
  static_assert(sizeof(T) == 4,
                "wait_flag_then_put only supports 32-bit flag types "
                "(POLL_REGMEM compares a single DWORD)");
  if (!src || !dst || size == 0 || !flag_ptr) {
    throw std::invalid_argument(
      "SdmaQueueHostHandle::wait_flag_then_put: invalid args");
  }
  packets::PollRegmemPacket poll(flag_ptr, static_cast<uint32_t>(expected),
                                 mask, SDMA_POLL_FUNC_GEQ, interval,
                                 retry_count);
  // The poll gates the first copy chunk; remaining chunks follow once released.
  size_t const chunk = anvilHostChunkBytes();
  auto* d = static_cast<uint8_t*>(dst);
  auto* s = static_cast<uint8_t*>(src);
  size_t const first = (size > chunk) ? chunk : size;
  packets::CopyLinearPacket copy0(s, d, first);
  submitBatch({{poll.data(), poll.size_bytes()},
               {copy0.data(), copy0.size_bytes()}});
  for (size_t off = first; off < size;) {
    size_t const n = (size - off > chunk) ? chunk : size - off;
    packets::CopyLinearPacket copy(s + off, d + off, n);
    submitBatch({{copy.data(), copy.size_bytes()}});
    off += n;
  }
}

void SdmaQueueHostHandle::timestamp(uint64_t* ts_ptr) {
  if (!ts_ptr) {
    throw std::invalid_argument("SdmaQueueHostHandle::timestamp: nullptr");
  }
  packets::TimestampPacket ts(ts_ptr);
  submitBatch({{ts.data(), ts.size_bytes()}});
}

void SdmaQueueHostHandle::quiet() {
  uint64_t target_wptr = queue_->hostCommittedWptr_.load(
    std::memory_order_acquire);
  while (*queue_->queue_.Queue_read_ptr_aql != target_wptr) {
    std::atomic_thread_fence(std::memory_order_acquire);
    std::this_thread::yield();
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
}

// Explicit instantiations for the flag types TransferBench uses.
template void SdmaQueueHostHandle::signal<uint32_t>(uint32_t*, uint32_t);
template void SdmaQueueHostHandle::signal<uint64_t>(uint64_t*, uint64_t);
template void SdmaQueueHostHandle::put_signal<uint32_t>(void*, void*, size_t,
                                                        uint32_t*, uint32_t);
template void SdmaQueueHostHandle::put_signal<uint64_t>(void*, void*, size_t,
                                                        uint64_t*, uint64_t);
template void SdmaQueueHostHandle::wait_flag_then_put<uint32_t>(
  uint32_t*, uint32_t, void*, void*, size_t, uint32_t, uint32_t, uint32_t);

bool AnvilLib::connectHost(int srcDeviceId, int dstDeviceId, int numChannels) {
  if (numChannels <= 0) {
    throw std::invalid_argument("connectHost(): numChannels must be positive");
  }

  auto& channels = host_sdma_channels_[ChannelKey{srcDeviceId, dstDeviceId}];
  const int existingChannels = static_cast<int>(channels.size());
  if (existingChannels >= numChannels) {
    return true;
  }

  const uint32_t engineId = getSdmaEngineId(srcDeviceId, dstDeviceId);
  for (int c = existingChannels; c < numChannels; ++c) {
    HostChannel hc;
    hc.queue = std::make_unique<SdmaQueue>(srcDeviceId, dstDeviceId,
                                           gpuAgents_[srcDeviceId], engineId);
    hc.handle = std::make_unique<SdmaQueueHostHandle>(hc.queue.get());
    channels.push_back(std::move(hc));
  }
  return true;
}

SdmaQueueHostHandle* AnvilLib::getHostHandle(int srcDeviceId, int dstDeviceId,
                                             int channelIdx) {
  auto it = host_sdma_channels_.find(ChannelKey{srcDeviceId, dstDeviceId});
  if (it == host_sdma_channels_.end()) {
    return nullptr;
  }
  const auto& channels = it->second;
  if (channelIdx < 0 ||
      static_cast<size_t>(channelIdx) >= channels.size()) {
    return nullptr;
  }
  return channels[static_cast<size_t>(channelIdx)].handle.get();
}

AnvilLib& AnvilLib::getInstance() {
  static AnvilLib instance;
  return instance;
}

int AnvilLib::getOamId(int deviceId) {
  std::string busId = getBusId(deviceId);
  std::string file_str = "/sys/bus/pci/devices/" + busId + "/xgmi_physical_id";
  std::ifstream file(file_str);
  int xgmi_physical_id;
  if (file.is_open()) {
    if (!(file >> xgmi_physical_id)) {
      throw std::runtime_error("Failed to read xGMI physical id from file: " +
                               file_str);
    }
  } else {
    throw std::runtime_error("Failed to open file: " + file_str);
  }
  return xgmi_physical_id;
}

// Return the index of the n-th (0-based) set bit in mask, or -1 if fewer than
// n+1 bits are set. Used to spread transfers across the preferred engine set.
static int NthSetBit(uint32_t mask, int n) {
  for (int i = 0; i < 32; ++i) {
    if (mask & (1u << i)) {
      if (n == 0) return i;
      --n;
    }
  }
  return -1;
}

int AnvilLib::getSdmaEngineId(int srcDeviceId, int dstDeviceId, int rotation) {
  // ANVIL_USE_HSA_ENGINE=1 (default): query hsa_amd_memory_get_preferred_copy_engine
  // for the src->dst pair and return an engine index from the preferred mask.
  // ANVIL_USE_HSA_ENGINE=0: use the hardcoded MI300X OAM lookup table.
  static bool useHsaEngine = [] {
    char const* v = getenv("ANVIL_USE_HSA_ENGINE");
    return !v || atoi(v) != 0;
  }();

  // ANVIL_ENGINE_ROUND_ROBIN=1: distribute successive transfers across the set
  // bits of the preferred-engine mask (rotation selects the k-th engine).
  // Default (0): always use the lowest-set-bit engine (legacy deterministic).
  // Neutral on gfx1250 (bandwidth/latency-bound); opt-in for other topologies.
  static bool roundRobin = [] {
    char const* v = getenv("ANVIL_ENGINE_ROUND_ROBIN");
    return v && atoi(v) != 0;
  }();

  if (useHsaEngine) {
    uint32_t mask = 0;
    hsa_status_t status = hsa_amd_memory_get_preferred_copy_engine(
      gpuAgents_[dstDeviceId], gpuAgents_[srcDeviceId], &mask);
    if (status == HSA_STATUS_SUCCESS && mask != 0) {
      int const numEngines = __builtin_popcount(mask);
      int const slot = roundRobin ? (((rotation % numEngines) + numEngines) % numEngines) : 0;
      int const engine = NthSetBit(mask, slot);
      ANVIL_LOG("getSdmaEngineId: src=%d dst=%d mask=0x%x numEngines=%d rotation=%d -> engine=%d\n",
                srcDeviceId, dstDeviceId, mask, numEngines, rotation, engine);
      return engine;
    }
    ANVIL_LOG("getSdmaEngineId: HSA preferred engine query failed or returned 0 "
              "(src=%d dst=%d status=%u mask=%u), falling back to OAM map\n",
              srcDeviceId, dstDeviceId, (unsigned)status, mask);
  }

  // Hardcoded MI300X OAM map fallback (even engines only)
  int srcOamId = getOamId(srcDeviceId);
  int dstOamId = getOamId(dstDeviceId);
  return mi300xOamMap[srcOamId][dstOamId] * 2;
}

AnvilLib& anvil = anvil.getInstance();

} // namespace anvil

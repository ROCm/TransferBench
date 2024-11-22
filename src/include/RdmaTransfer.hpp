/*
Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef RdmaTransfer_HPP
#define RdmaTransfer_HPP
#ifndef LIB_IBVERBS_UNAVAILABLE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include "IbVerbsUtils.hpp"
/**
 * @class RdmaTransfer
 * @brief A class to manage RDMA operations using an RDMA capable NIC.
 *
 * This class provides functionalities to initialize RDMA devices, register memory,
 * post send requests, and tear down the RDMA setup.
 */
class RdmaTransfer 
{
public:
  /**
   * @brief Initializes the RDMA device and Queue Pairs (QPs) for communication.
   *
   * This function sets up the RDMA device and its associated resources, including
   * the device context, protection domain, completion queue, and queue pairs for
   * both sending and receiving. It also ensures that the device is active and 
   * ready for communication.
   *
   * @param source_device The index of the source RDMA device to be initialized.
   * @param destination_device The index of the destination RDMA device (currently unused).
   * @param gid_index The GID index to be used for RoCE (RDMA over Converged Ethernet).
   * @param roce_version RoCE version used for GID Indexing.
   * @param qpairs_count The number of QPs to use for each transfer.
   * @param ip_address_family The IP Address Family for source and destination adapters
   * @param port_num The port ID of the RDMA device to be used (default is 1).
   *
   * @note This function will exit the program if the selected RDMA device is down.
   */
  void InitDeviceAndQPs(int source_device, int destination_device, int gid_index, int roce_version, uint8_t qpairs_count, int ip_address_family, uint8_t port_num)
  {
    InitDeviceList();
    src_device_id = source_device;
    dst_device_id = destination_device;
    ib_device_port = port_num;
    qp_count = qpairs_count;
    int src_gid_index = gid_index;
    int dst_gid_index = gid_index;
    InitRDMAResources(src_device_id, port_num);
    InitRDMAResources(dst_device_id, port_num);
    auto && src_rdma = ib_attribute_mapper[src_device_id];
    auto && dst_rdma = ib_attribute_mapper[dst_device_id];
    bool isRoce = src_rdma->port_attr.link_layer == IBV_LINK_LAYER_ETHERNET;
    assert(src_rdma->port_attr.link_layer == dst_rdma->port_attr.link_layer);
    if(isRoce)
    {
      IBV_CALL(set_gid_index, src_rdma->device_context, port_num, src_rdma->port_attr.gid_tbl_len, roce_version, ip_address_family, &src_gid_index);
      IBV_CALL(set_gid_index, dst_rdma->device_context, port_num, dst_rdma->port_attr.gid_tbl_len, roce_version, ip_address_family, &dst_gid_index);
      IBV_CALL(set_ibv_gid, src_rdma->device_context,
                          port_num, src_gid_index, src_rdma->gid);
      IBV_CALL(set_ibv_gid, dst_rdma->device_context,
                          port_num, dst_gid_index, dst_rdma->gid);
    }
    assert(sender_qp == nullptr);
    assert(receiver_qp == nullptr);
    assert(qp_count >= 1);
    sender_qp = new ibv_qp* [qp_count];
    receiver_qp = new ibv_qp* [qp_count];
    for(int i = 0; i < qp_count; ++i) {
      IBV_PTR_CALL(sender_qp[i],
                qp_create, src_rdma->protection_domain,
                           src_rdma->completion_queue);

      IBV_PTR_CALL(receiver_qp[i],
                  qp_create, dst_rdma->protection_domain,
                             dst_rdma->completion_queue);

      IBV_CALL(qp_init, sender_qp[i], port_num,
                      rdma_flags);
      
      IBV_CALL(qp_init, receiver_qp[i], port_num,
                      rdma_flags);


      IBV_CALL(qp_transition_to_ready_to_receive, sender_qp[i],
                                                  dst_rdma->port_attr.lid,
                                                  receiver_qp[i]->qp_num,
                                                  dst_rdma->gid, dst_gid_index,
                                                  ib_device_port, isRoce,
                                                  src_rdma->port_attr.active_mtu
                                                );

      IBV_CALL(qp_transition_to_ready_to_send, sender_qp[i]);

      IBV_CALL(qp_transition_to_ready_to_receive, receiver_qp[i],
                                                src_rdma->port_attr.lid,
                                                sender_qp[i]->qp_num,
                                                src_rdma->gid, src_gid_index,
                                                ib_device_port, isRoce,
                                                dst_rdma->port_attr.active_mtu
                                                );

      IBV_CALL(qp_transition_to_ready_to_send, receiver_qp[i]);
    }
    
  }

  /**
   * @brief Registers memory for RDMA.
   * 
   * @param src Pointer to the source memory region.
   * @param dst Pointer to the destination memory region.
   * @param size Size of the memory region to register and send.
   * @return id to indentify the transfer and registered memory
   */
  size_t MemoryRegister(void *src, void *dst, size_t numBytes)
  {
    auto&& src_rdma_resource = ib_attribute_mapper[src_device_id];
    auto&& dst_rdma_resource = ib_attribute_mapper[dst_device_id];
    struct ibv_mr *src_mr;
    struct ibv_mr *dst_mr;
    IBV_PTR_CALL(src_mr, ibv_reg_mr, src_rdma_resource->protection_domain, src, numBytes, rdma_flags);
    IBV_PTR_CALL(dst_mr, ibv_reg_mr, dst_rdma_resource->protection_domain, dst, numBytes, rdma_flags);
    return AppendResources(src_mr, src, dst_mr, dst, numBytes);
  }

  /**
   * @brief Transfers data using RDMA.
   *
   * This function sets up and initiates an RDMA write operation to transfer data
   * from a source memory region to a destination memory region. It configures
   * the scatter-gather entry, work request, and posts the send request to the
   * sender queue pair. It also polls the completion queue to ensure the operation
   * completes successfully.
   *
   * @note This function assumes that the source and destination memory regions,
   *       as well as the sender queue pair and completion queue, have been
   *       properly initialized and configured.
   */
  void TransferData(int transferIdx) 
  {    
    assert((transferIdx % qp_count) == 0);
    uint64_t mem_id = transferIdx / qp_count;
    auto&& src_rdma_resource = ib_attribute_mapper[src_device_id];
    size_t chunk_size = messageSizes[mem_id] / qp_count;
    size_t remaining_size = messageSizes[mem_id] % qp_count;
    for (auto i = 0; i < qp_count; ++i) {
      struct ibv_sge sg = {};
      struct ibv_send_wr wr = {};
      size_t current_chunk_size = chunk_size + (i == qp_count - 1 ? remaining_size : 0);
      sg.addr = (uint64_t)source_mr[mem_id].second + i * chunk_size;
      sg.length = current_chunk_size;
      sg.lkey = source_mr[mem_id].first->lkey;
      struct ibv_send_wr *bad_wr;
      wr.wr_id = transferIdx + i;
      assert(wr.wr_id < receiveStatuses.size());
      wr.sg_list = &sg;
      wr.num_sge = 1;
      wr.opcode = IBV_WR_RDMA_WRITE;
      wr.send_flags = IBV_SEND_SIGNALED;
      wr.wr.rdma.remote_addr = (uint64_t)destination_mr[mem_id].second + i * chunk_size;
      wr.wr.rdma.rkey = destination_mr[mem_id].first->rkey;
      IBV_CALL(ibv_post_send, sender_qp[i], &wr, &bad_wr);
    }
    for(auto i = 0; i < qp_count; ++i) {
       IBV_CALL(poll_completion_queue, src_rdma_resource->completion_queue, transferIdx + i, receiveStatuses);
    }
  }


  /**
   * @brief Checks if RDMA functionality is supported.
   * 
   * @return true if the required features are supported, false otherwise.
   */
  static bool IsSupported()
  { 
    return true;
  }

  /**
   * @brief Tears down the RDMA setup by destroying all RDMA resources.
   */
  void TearDown() 
  {
    if (source_mr.size() > 0) 
    {
      for(auto mr : source_mr) 
      {
        IBV_CALL(ibv_dereg_mr, mr.first);
      }
      source_mr.clear();
    }
    if (destination_mr.size() > 0) 
    {
      for(auto mr : destination_mr) 
      {
        IBV_CALL(ibv_dereg_mr, mr.first);
      }
      destination_mr.clear();
    }
    receiveStatuses.clear();
    messageSizes.clear();
    if (sender_qp) 
    {
      for (int i = 0; i < qp_count; ++i) {
        IBV_CALL(ibv_destroy_qp, sender_qp[i]);
        sender_qp[i] = nullptr;
      }
      delete[] sender_qp;
      sender_qp = nullptr;
    }
    if (receiver_qp) 
    {
      for (int i = 0; i < qp_count; ++i) {
        IBV_CALL(ibv_destroy_qp, receiver_qp[i]);
        receiver_qp[i] = nullptr;
      }
      delete[] receiver_qp;
      receiver_qp = nullptr;
    }
    auto& src_rdma_resource = ib_attribute_mapper[src_device_id];
    auto& dst_rdma_resource = ib_attribute_mapper[dst_device_id];

    if (src_rdma_resource != nullptr) {
      if (src_rdma_resource->completion_queue) {
        IBV_CALL(ibv_destroy_cq, src_rdma_resource->completion_queue);
        src_rdma_resource->completion_queue = nullptr;
      }
      if (src_rdma_resource->protection_domain) {
        IBV_CALL(ibv_dealloc_pd, src_rdma_resource->protection_domain);
        src_rdma_resource->protection_domain = nullptr;
      }
      if (src_rdma_resource->device_context) {
        IBV_CALL(ibv_close_device, src_rdma_resource->device_context);
        src_rdma_resource->device_context = nullptr;
      }
      src_rdma_resource = nullptr;
    }

    if (dst_rdma_resource != nullptr) {
      if (dst_rdma_resource->completion_queue) {
        IBV_CALL(ibv_destroy_cq, dst_rdma_resource->completion_queue);
        dst_rdma_resource->completion_queue = nullptr;
      }
      if (dst_rdma_resource->protection_domain) {
        IBV_CALL(ibv_dealloc_pd, dst_rdma_resource->protection_domain);
        dst_rdma_resource->protection_domain = nullptr;
      }
      if (dst_rdma_resource->device_context) {
        IBV_CALL(ibv_close_device, dst_rdma_resource->device_context);
        dst_rdma_resource->device_context = nullptr;
      }
      dst_rdma_resource = nullptr;
    }
  }
  
  /**
   * @brief Initializes the device list if it is not already initialized.
   */
  static void InitDeviceList() 
  {
    if (device_list == NULL) 
    {
      IBV_PTR_CALL(device_list, ibv_get_device_list, &ib_device_count);
    }
  }

  /**
   * @brief Get RDMA device count.
   */
  static int GetNicCount()
  {
    if (device_list == NULL && ib_device_count < 0)
    {
      InitDeviceList();
    }
    return ib_device_count;
  }

private:
  void InitRDMAResources(int const& device_id, uint8_t const& port_num) {
    if (ib_attribute_mapper.size() <= device_id) {
      ib_attribute_mapper.resize(device_id + 1);
      ib_attribute_mapper[device_id] = nullptr;
    }
    if (!ib_attribute_mapper[device_id]) {
      ib_attribute_mapper[device_id] = new RDMA_Resources();
      auto& rdma = ib_attribute_mapper[device_id];
      IBV_PTR_CALL(rdma->device_context, ibv_open_device, device_list[device_id]);

      IBV_PTR_CALL(rdma->protection_domain, ibv_alloc_pd, rdma->device_context);

      IBV_PTR_CALL(rdma->completion_queue, ibv_create_cq, rdma->device_context, 100, NULL, NULL, 0);
      IBV_CALL(ibv_query_port, rdma->device_context, port_num, &rdma->port_attr);

      if (rdma->port_attr.state != IBV_PORT_ACTIVE) {
        std::cout << "[Error] selected RDMA device " << device_id << " is down. Select a different device" << std::endl;
        exit(1);
      }
    }
  }

  size_t AppendResources(ibv_mr *&src_mr, void *&src, ibv_mr *&dst_mr, void *&dst, size_t &numBytes)
  {
    source_mr.push_back(std::make_pair(src_mr, src));
    destination_mr.push_back(std::make_pair(dst_mr, dst));
    for(int i = 0; i < qp_count; ++i) {
      receiveStatuses.push_back(false);
    }
    messageSizes.push_back(numBytes);
    return receiveStatuses.size() - qp_count;
  }

  class RDMA_Resources {
    public:
      struct ibv_pd *protection_domain = nullptr; ///< Protection domain for RDMA operations.
      struct ibv_cq *completion_queue = nullptr; ///< Completion queue for RDMA operations.
      struct ibv_context *device_context = nullptr; ///< Device context for the RDMA capable NIC.  
      struct ibv_port_attr port_attr = {}; ///< Port attributes for the RDMA capable NIC.  
      union ibv_gid gid;                  ///< GID handler needed for RoCE support
  };  
  static int ib_device_count;          ///< Number of RDMA capable NICs.
  static struct ibv_device **device_list; ///< List of RDMA capable devices.
  std::vector<RDMA_Resources*> ib_attribute_mapper; ///< Store resoruce sensitive RDMA fields.
  std::vector<std::pair<struct ibv_mr *, void*>> source_mr; ///< Memory region for the source buffer.
  std::vector<std::pair<struct ibv_mr *, void*>> destination_mr; ///< Memory region for the destination buffer.
  std::vector<bool> receiveStatuses; ///< Keep track of send/recv statuses 
  std::vector<size_t> messageSizes; ///< Keep track of message sizes
  struct ibv_qp **sender_qp = nullptr; ///< Queue pair for sending RDMA requests.
  struct ibv_qp **receiver_qp = nullptr; ///< Queue pair for receiving RDMA requests.
  int src_device_id; ///< IB NIC device ID.
  int dst_device_id; ///< IB NIC device ID.
  int ib_device_port; ///< IB Port ID.  
  uint8_t qp_count; ///< Number of QPs to be used for transferring data
};
// Initialize the static member device_list
struct ibv_device **RdmaTransfer::device_list = NULL;
int RdmaTransfer::ib_device_count = -1;
//std::vector<RdmaTransfer::RDMA_Resources*> RdmaTransfer::ib_attribute_mapper;
#else
#warning "LIB Ibverbs is not installed. RDMA Executor is therefore disabled."
#define RDMA_NOT_SUPPORTED_ERROR()                           \
  do {                                                       \
    std::cout << "Error: RDMA Executor API not supported. "  \
              << "DISABLE_RDMA_EXECUTOR flag is set. "       \
              << "Executor API Call line " << __LINE__       \
              << " in file " << __FILE__ << "\n";            \
    exit(1);                                                 \
  } while(0)                                                 \

class RdmaTransfer
{
public:
  void InitDeviceAndQPs(int source_device, int destination_device, int gid_index, int roce_version, uint8_t qpairs_count, int ip_address_family, uint8_t port_num)
  {
    RDMA_NOT_SUPPORTED_ERROR();
  }
  size_t MemoryRegister(void *src, void *dst, size_t numBytes)
  {
    RDMA_NOT_SUPPORTED_ERROR();
  }
  void TransferData(int transferIdx)
  {
    RDMA_NOT_SUPPORTED_ERROR();
  }
  void TearDown()
  {
    RDMA_NOT_SUPPORTED_ERROR();
  }
  static bool IsSupported()
  {
    return false;
  }
  static void InitDeviceList()
  {
    RDMA_NOT_SUPPORTED_ERROR();
  }
  static int GetNicCount()
  {
    return 0;
  }
};
#endif
#endif
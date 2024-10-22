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
#include <infiniband/verbs.h>
#include <iostream>
#include <vector>

#define MAX_SEND_WR_PER_QP 12
#define MAX_RECV_WR_PER_QP 12

#define IB_PSN  0
const uint64_t WR_ID = 1789;

const unsigned int rdma_flags = IBV_ACCESS_LOCAL_WRITE  |
                                IBV_ACCESS_REMOTE_READ  |
                                IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_REMOTE_ATOMIC;


// Helper macro for catching RDMA errors
#define IBV_CALL(func)                                                                  \
    do {                                                                                \
        int error = (func);                                                             \
        if (error != 0)                                                                 \
        {                                                                               \
            std::cerr << "Encountered RDMA error " << error << " at line " << __LINE__  \
            << " in file " << __FILE__ << " and function " << __func__ << "\n";         \
            exit(-1);                                                                   \
        }                                                                               \
    } while (0)

// Helper macro for catching RDMA null return errors
#define IBV_PTR_CALL(ptr, func)                                                         \
    do {                                                                                \
        ptr = (func);                                                                   \
        if (ptr == NULL)                                                                \
        {                                                                               \
            std::cerr << "Encountered RDMA Null Pointer at line " << __LINE__           \
            << " in file " << __FILE__ << " and function " << __func__ << "\n";         \
            exit(-1);                                                                   \
        }                                                                               \
    } while (0)


static struct ibv_qp *qp_create(struct ibv_pd *pd, struct ibv_cq* cq);
static int qp_init(struct ibv_qp *qp, uint8_t port_num, unsigned flags);
static int qp_transition_to_ready_to_receive(struct ibv_qp *qp, uint16_t dlid, uint32_t dqpn, ibv_gid gid, uint8_t gid_index, uint8_t port, bool isRoCE);
static int qp_transition_to_ready_to_send(struct ibv_qp *qp);
static bool is_qp_ready_to_send(struct ibv_qp *qp);
static int poll_completion_queue(struct ibv_cq *cq, int transferIdx, std::vector<bool> &sendRecvStat);
static int set_ibv_gid(struct ibv_context *ctx, uint8_t port_num, int gid_index, ibv_gid& gid);
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
   * @param qpairs_count The number of QPs to use for each transfer.
   * @param port_num The port ID of the RDMA device to be used (default is 1).
   *
   * @note This function will exit the program if the selected RDMA device is down.
   */
  void InitDeviceAndQPs(int source_device, int destination_device, uint8_t gid_index, uint8_t qpairs_count, uint8_t port_num)
  {
    InitDeviceList();
    src_device_id = source_device;
    dst_device_id = destination_device;
    ib_device_port = port_num;
    qp_count = qpairs_count;
    InitRDMAResources(src_device_id, port_num);
    InitRDMAResources(dst_device_id, port_num);
    auto && src_rdma = ib_attribute_mapper[src_device_id];
    auto && dst_rdma = ib_attribute_mapper[dst_device_id];
    bool isRoce = src_rdma->port_attr.link_layer == IBV_LINK_LAYER_ETHERNET;
    assert(src_rdma->port_attr.link_layer == dst_rdma->port_attr.link_layer);
    if(isRoce)
    {
      IBV_CALL(set_ibv_gid(src_rdma->device_context,
                          port_num, gid_index, src_rdma->gid));
      IBV_CALL(set_ibv_gid(dst_rdma->device_context,
                          port_num, gid_index, dst_rdma->gid));
    }
    assert(sender_qp == nullptr);
    assert(receiver_qp == nullptr);
    assert(qp_count >= 1);
    sender_qp = new ibv_qp* [qp_count];
    receiver_qp = new ibv_qp* [qp_count];
    for(int i = 0; i < qp_count; ++i) {
      IBV_PTR_CALL(sender_qp[i],
                qp_create(src_rdma->protection_domain,
                          src_rdma->completion_queue));

      IBV_PTR_CALL(receiver_qp[i],
                  qp_create(dst_rdma->protection_domain,
                            dst_rdma->completion_queue));

      IBV_CALL(qp_init(sender_qp[i], port_num,
                      rdma_flags));
      
      IBV_CALL(qp_init(receiver_qp[i], port_num,
                      rdma_flags));


      IBV_CALL(qp_transition_to_ready_to_receive(sender_qp[i],
                                                  dst_rdma->port_attr.lid,
                                                  receiver_qp[i]->qp_num,
                                                  dst_rdma->gid, gid_index,
                                                  ib_device_port, isRoce));

      IBV_CALL(qp_transition_to_ready_to_send(sender_qp[i]));

      IBV_CALL(qp_transition_to_ready_to_receive(receiver_qp[i],
                                                src_rdma->port_attr.lid,
                                                sender_qp[i]->qp_num,
                                                src_rdma->gid, gid_index,
                                                ib_device_port, isRoce));

      IBV_CALL(qp_transition_to_ready_to_send(receiver_qp[i]));
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
    IBV_PTR_CALL(src_mr, ibv_reg_mr(src_rdma_resource->protection_domain, src, numBytes, rdma_flags));        
    IBV_PTR_CALL(dst_mr, ibv_reg_mr(dst_rdma_resource->protection_domain, dst, numBytes, rdma_flags));
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
   *
   * @param src_ptr Pointer to the source memory region.
   * @param size Size of the data to be transferred.
   * @param source_mr Pointer to the source memory region's memory registration structure.
   * @param dst_ptr Pointer to the destination memory region.
   * @param destination_mr Pointer to the destination memory region's memory registration structure.
   * @param sender_qp Pointer to the sender queue pair.
   * @param completion_queue Pointer to the completion queue.
   * @param WR_ID Work request ID.
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
      IBV_CALL(ibv_post_send(sender_qp[i], &wr, &bad_wr));
    }
    for(auto i = 0; i < qp_count; ++i) {
       IBV_CALL(poll_completion_queue(src_rdma_resource->completion_queue, transferIdx + i, receiveStatuses));
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
        IBV_CALL(ibv_dereg_mr(mr.first));        
      }
      source_mr.clear();
    }
    if (destination_mr.size() > 0) 
    {
      for(auto mr : destination_mr) 
      {
        IBV_CALL(ibv_dereg_mr(mr.first));
      }
      destination_mr.clear();
    }
    receiveStatuses.clear();
    messageSizes.clear();
    if (sender_qp) 
    {
      for (int i = 0; i < qp_count; ++i) {
        IBV_CALL(ibv_destroy_qp(sender_qp[i]));
        sender_qp[i] = nullptr;
      }
      delete[] sender_qp;
      sender_qp = nullptr;
    }
    if (receiver_qp) 
    {
      for (int i = 0; i < qp_count; ++i) {
        IBV_CALL(ibv_destroy_qp(receiver_qp[i]));
        receiver_qp[i] = nullptr;
      }
      delete[] receiver_qp;
      receiver_qp = nullptr;
    }
    auto& src_rdma_resource = ib_attribute_mapper[src_device_id];
    auto& dst_rdma_resource = ib_attribute_mapper[dst_device_id];

    if (src_rdma_resource != nullptr) {
      if (src_rdma_resource->completion_queue) {
        IBV_CALL(ibv_destroy_cq(src_rdma_resource->completion_queue));
        src_rdma_resource->completion_queue = nullptr;
      }
      if (src_rdma_resource->protection_domain) {
        IBV_CALL(ibv_dealloc_pd(src_rdma_resource->protection_domain));
        src_rdma_resource->protection_domain = nullptr;
      }
      if (src_rdma_resource->device_context) {
        IBV_CALL(ibv_close_device(src_rdma_resource->device_context));
        src_rdma_resource->device_context = nullptr;
      }
      src_rdma_resource = nullptr;
    }

    if (dst_rdma_resource != nullptr) {
      if (dst_rdma_resource->completion_queue) {
        IBV_CALL(ibv_destroy_cq(dst_rdma_resource->completion_queue));
        dst_rdma_resource->completion_queue = nullptr;
      }
      if (dst_rdma_resource->protection_domain) {
        IBV_CALL(ibv_dealloc_pd(dst_rdma_resource->protection_domain));
        dst_rdma_resource->protection_domain = nullptr;
      }
      if (dst_rdma_resource->device_context) {
        IBV_CALL(ibv_close_device(dst_rdma_resource->device_context));
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
      IBV_PTR_CALL(device_list, ibv_get_device_list(&ib_device_count));
    }
  }

  /**
   * @brief Get RDMA device count.
   */
  static int GetNicCount()
  {
    if (device_list == NULL && ib_device_count == 0)
    {
      InitDeviceList();
    }
    return ib_device_count;
  }

  /**
   * @brief Destructor that tears down the RDMA setup.
   */
  ~RdmaTransfer() 
  {
    //TearDown();
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
      IBV_PTR_CALL(rdma->device_context, ibv_open_device(device_list[device_id]));

      IBV_PTR_CALL(rdma->protection_domain, ibv_alloc_pd(rdma->device_context));

      IBV_PTR_CALL(rdma->completion_queue, ibv_create_cq(rdma->device_context, 100, NULL, NULL, 0));
      IBV_CALL(ibv_query_port(rdma->device_context, port_num, &rdma->port_attr));

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
int RdmaTransfer::ib_device_count = 0;
//std::vector<RdmaTransfer::RDMA_Resources*> RdmaTransfer::ib_attribute_mapper;

/**
 * @brief Creates an InfiniBand Queue Pair (QP).
 *
 * This function initializes and creates an InfiniBand Queue Pair (QP) with the specified
 * protection domain (PD) and completion queue (CQ). The QP is configured with the following
 * attributes:
 * - Both send and receive completion queues are set to the provided CQ.
 * - Maximum number of send work requests is set to 1.
 * - Maximum number of receive work requests is set to 1.
 * - Maximum number of scatter/gather elements in a send work request is set to 1.
 * - Maximum number of scatter/gather elements in a receive work request is set to 1.
 * - QP type is set to Reliable Connection (RC).
 *
 * @param pd Pointer to the protection domain (ibv_pd) to associate with the QP.
 * @param cq Pointer to the completion queue (ibv_cq) to use for both send and receive operations.
 * @return Pointer to the created Queue Pair (ibv_qp) on success, or nullptr on failure.
 */
static struct ibv_qp *qp_create(struct ibv_pd *pd, struct ibv_cq* cq)
{
  struct ibv_qp_init_attr attr = {};
  memset(&attr, 0, sizeof(struct ibv_qp_init_attr));
  attr.send_cq = cq;
  attr.recv_cq = cq;
  attr.cap.max_send_wr  = MAX_SEND_WR_PER_QP;
  attr.cap.max_recv_wr  = MAX_RECV_WR_PER_QP;
  attr.cap.max_send_sge = 1;
  attr.cap.max_recv_sge = 1;
  attr.qp_type = IBV_QPT_RC;
  return ibv_create_qp(pd, &attr);    
}

/**
 * @brief Sets the InfiniBand GID (Global Identifier) for a given port.
 *
 * This function queries and sets the GID for a specified port number and GID index
 * on the given InfiniBand context.
 *
 * @param ctx Pointer to the ibv_context structure representing the InfiniBand device context.
 * @param port_num Reference to the port number on the InfiniBand device.
 * @param gid_index Index of the GID to query.
 * @param gid Reference to the ibv_gid structure where the queried GID will be stored.
 * @return int Returns 0 on success, or the error code returned by ibv_query_gid on failure.
 */
static int set_ibv_gid(struct ibv_context *ctx, uint8_t port_num, int gid_index, ibv_gid& gid) 
{
  return ibv_query_gid(ctx, port_num, gid_index, &gid);
}

/**
 * @brief Initializes the given Queue Pair (QP) with the specified attributes.
 *
 * This function sets the QP state to INIT and configures the QP with the provided
 * access flags, port number, and pkey index. It then modifies the QP using the 
 * ibv_modify_qp function.
 *
 * @param qp Pointer to the ibv_qp structure representing the Queue Pair to be initialized.
 * @param flags Access flags to be set for the QP.
 * @return int Returns 0 on success, or the error code returned by ibv_modify_qp on failure.
 */
static int qp_init(struct ibv_qp *qp, uint8_t port_num, unsigned flags)
{
  struct ibv_qp_attr attr = {};        // Initialize the QP attributes structure to zero
  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state   = IBV_QPS_INIT;      // Set the QP state to INIT
  attr.pkey_index = 0;                 // Set the partition key index to 0
  attr.port_num   = port_num;           // Set the port number to the defined IB_PORT
  attr.qp_access_flags = flags;        // Set the QP access flags to the provided flags

  // Modify the QP with the specified attributes and return the result
  return ibv_modify_qp(qp, &attr,
              IBV_QP_STATE      |      // Modify the QP state
              IBV_QP_PKEY_INDEX |      // Modify the partition key index
              IBV_QP_PORT       |      // Modify the port number
              IBV_QP_ACCESS_FLAGS);    // Modify the access flags
}



/**
 * @brief Transition the Queue Pair (QP) to the Ready to Receive (RTR) state.
 *
 * This function modifies the attributes of a given Queue Pair (QP) to transition it to the 
 * Ready to Receive (RTR) state. It sets various attributes such as the QP state, path MTU, 
 * receive queue PSN, and others. It also handles both RoCE (RDMA over Converged Ethernet) 
 * and non-RoCE configurations.
 *
 * @param qp Pointer to the ibv_qp structure representing the Queue Pair.
 * @param dlid Destination Local Identifier (DLID) for non-RoCE configurations.
 * @param dqpn Destination Queue Pair Number (DQPN).
 * @param gid Global Identifier (GID) for RoCE configurations.
 * @param isRoCE Boolean flag indicating whether the configuration is for RoCE (true) or not (false).
 * @return int 0 on success, or the error code returned by ibv_modify_qp on failure.
 */
static int qp_transition_to_ready_to_receive(struct ibv_qp *qp, uint16_t dlid, uint32_t dqpn, ibv_gid gid, uint8_t gid_index, uint8_t port, bool isRoCE)
{
  struct ibv_qp_attr attr = {};
  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state       = IBV_QPS_RTR;
  attr.path_mtu       = IBV_MTU_4096;
  attr.rq_psn         = IB_PSN;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer  = 12;
  if(isRoCE) 
  {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid.global.subnet_prefix = gid.global.subnet_prefix;
    attr.ah_attr.grh.dgid.global.interface_id = gid.global.interface_id;
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.grh.hop_limit = 255;
  }
  else 
  {
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid   = dlid;
  }  
  attr.ah_attr.sl     = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num  = port;
  attr.dest_qp_num    = dqpn;

  return ibv_modify_qp(qp, &attr,
              IBV_QP_STATE              |
              IBV_QP_AV                 |
              IBV_QP_PATH_MTU           |
              IBV_QP_DEST_QPN           |
              IBV_QP_RQ_PSN             |
              IBV_QP_MAX_DEST_RD_ATOMIC |
              IBV_QP_MIN_RNR_TIMER);
}

/**
 * @brief Transition the Queue Pair (QP) to the Ready to Send (RTS) state.
 *
 * This function modifies the state of the given QP to IBV_QPS_RTS (Ready to Send).
 * It sets various attributes required for the transition, including the state,
 * send queue packet sequence number (PSN), timeout, retry count, RNR (Receiver Not Ready) retry count,
 * and maximum number of outstanding RDMA read/atomic operations.
 *
 * @param qp Pointer to the QP to be modified.
 * @return 0 on success, or the value returned by ibv_modify_qp on failure.
 */
static int qp_transition_to_ready_to_send(struct ibv_qp *qp)
{
  struct ibv_qp_attr attr = {};
  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state       = IBV_QPS_RTS;
  attr.sq_psn         = IB_PSN;
  attr.timeout        = 14;
  attr.retry_cnt      = 7;
  attr.rnr_retry      = 7;
  attr.max_rd_atomic  = 1;

  return ibv_modify_qp(qp, &attr,
              IBV_QP_STATE     |
              IBV_QP_TIMEOUT   |
              IBV_QP_RETRY_CNT |
              IBV_QP_RNR_RETRY |
              IBV_QP_SQ_PSN    |
              IBV_QP_MAX_QP_RD_ATOMIC);
}

/**
 * @brief Checks if the given Queue Pair (QP) is in the Ready to Send (RTS) state.
 *
 * This function queries the attributes of the provided QP and checks its current state.
 *
 * @param qp Pointer to the ibv_qp structure representing the Queue Pair to be checked.
 * @return int Returns 1 if the QP is in the IBV_QPS_RTS state, otherwise returns 0.
 */
static bool is_qp_ready_to_send(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = {};            // Initialize the QP attributes structure to zero
    struct ibv_qp_init_attr init_attr = {};  // Initialize the QP init attributes structure to zero
    int rc = ibv_query_qp(qp, &attr, IBV_QP_CUR_STATE, &init_attr); // Query the QP attributes

    // Return true if the current QP state is IBV_QPS_RTS, otherwise return 0
    return (attr.cur_qp_state == IBV_QPS_RTS);
}

/**
 * @brief Polls a completion queue (CQ) for work completions.
 *
 * This function continuously polls the given completion queue (CQ) until
 * at least one work completion (WC) is found. It asserts that the number
 * of completions polled is non-negative, the work request ID (wr_id) matches
 * the expected WR_ID, and the status of the work completion is successful.
 *
 * @param cq Pointer to the completion queue (CQ) to be polled.
 * @return Always returns 0.
 */
static int poll_completion_queue(struct ibv_cq *cq, int transferIdx, std::vector<bool> &sendRecvStat)
{
  int nc = 0;              // Number of completions polled
  struct ibv_wc wc;        // Work completion structure
  
  while (nc <= 0 && !sendRecvStat[transferIdx]) {   // Loop until at least one completion is found  
    nc = ibv_poll_cq(cq, 1, &wc);             // Poll the completion queue
     if(nc > 0) {
        assert(wc.status == IBV_WC_SUCCESS); // Ensure the status of the work completion is successful
        if(wc.wr_id == transferIdx) break;
        else {   
          sendRecvStat[wc.wr_id] = true;     // Lock is not needed.  ibv_poll_cq is thread-safe
          nc = 0;                            // reset to keep looping until my data is at least received
        }
      }
      assert(nc >= 0);                       // Ensure the number of completions polled is non-negative
  } 
  // No need to lock the shared vector. There are two cases
  // 1. If my receive was accomplished by another thread, my loop won't exit
  // unless unless the memory location has been sucessefully set by the receiving thread
  // 2. If my receive was accomplished by my thread, then it is guaranteed that I am the only
  // one trying to access this location
  // All of this will change if ibv_poll_cq was not thread-safe
  sendRecvStat[transferIdx] = false;
  return 0;               
}
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
  void InitDeviceAndQPs(int source_device, int destination_device, uint8_t gid_index, uint8_t qpairs_count, uint8_t port_num)
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
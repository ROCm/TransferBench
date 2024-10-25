#ifndef LIB_IBVERBS_UNAVAILABLE
#pragma once
#include <infiniband/verbs.h>
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
struct ibv_qp *qp_create(struct ibv_pd *pd, struct ibv_cq* cq)
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
int set_ibv_gid(struct ibv_context *ctx, uint8_t port_num, int gid_index, ibv_gid& gid) 
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
int qp_init(struct ibv_qp *qp, uint8_t port_num, unsigned flags)
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
int qp_transition_to_ready_to_receive(struct ibv_qp *qp, uint16_t dlid, uint32_t dqpn, ibv_gid gid, uint8_t gid_index, uint8_t port, bool isRoCE, enum ibv_mtu mtu)
{
  struct ibv_qp_attr attr = {};
  memset(&attr, 0, sizeof(struct ibv_qp_attr));
  attr.qp_state       = IBV_QPS_RTR;
  attr.path_mtu       = mtu;
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
int qp_transition_to_ready_to_send(struct ibv_qp *qp)
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
bool is_qp_ready_to_send(struct ibv_qp *qp) {
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
int poll_completion_queue(struct ibv_cq *cq, int transferIdx, std::vector<bool> &sendRecvStat)
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
#endif
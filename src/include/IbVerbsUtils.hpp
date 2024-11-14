#ifndef LIB_IBVERBS_UNAVAILABLE
#pragma once
#include <infiniband/verbs.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <arpa/inet.h>
#include <assert.h>
#include <fcntl.h>

#define MAX_SEND_WR_PER_QP 12
#define MAX_RECV_WR_PER_QP 12

#define IB_PSN  0
const uint64_t WR_ID = 1789;

const unsigned int rdma_flags = IBV_ACCESS_LOCAL_WRITE  |
                                IBV_ACCESS_REMOTE_READ  |
                                IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_REMOTE_ATOMIC;


// Helper macro for catching RDMA errors
#define IBV_CALL(__func__, ...)                                                     \
  do {                                                                              \
    int error = __func__(__VA_ARGS__);                                              \
    if (error != 0)                                                                 \
    {                                                                               \
      std::cerr << "Encountered RDMA error " << error << " at line " << __LINE__    \
            << " in file " << __FILE__ << " during " << #__func__ << "\n";          \
      exit(-1);                                                                     \
    }                                                                               \
  } while (0)

// Helper macro for catching RDMA null return errors
#define IBV_PTR_CALL(__ptr__, __func__, ...)                                        \
  do {                                                                              \
    __ptr__ = __func__(__VA_ARGS__);                                                \
    if (__ptr__ == NULL)                                                            \
    {                                                                               \
      std::cerr << "Encountered RDMA Null Pointer at line " << __LINE__             \
      << " in file " << __FILE__ << " during " << #__func__ << "\n";                \
      exit(-1);                                                                     \
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

static bool is_configured_gid(union ibv_gid* gid)
{
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
  if (((a->s6_addr32[0] | trailer) == 0UL) || ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL)))
  {
    return false;
  }
  return true;
}

static bool link_local_gid(union ibv_gid* gid)
{
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL)
  {
    return true;
  }
  return false;
}

static bool validGid(union ibv_gid* gid)
{
  return (is_configured_gid(gid) && !link_local_gid(gid));
}

static sa_family_t get_ib_address_family()
{
  sa_family_t family = AF_INET;
  const char* env = getenv("IB_ADDR_FAMILY");
  if (env == NULL || strlen(env) == 0) {
    return family;
  }

  printf("IB_ADDR_FAMILY set by environment to %s\n", env);

  if (strcmp(env, "AF_INET") == 0) {
    family = AF_INET;
  } else if (strcmp(env, "AF_INET6") == 0) {
    family = AF_INET6;
  }

  return family;
}

static sa_family_t get_gid_address_family(union ibv_gid* gid)
{
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  bool isIpV4Mapped = ((a->s6_addr32[0] | a->s6_addr32[1]) | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL;
  bool isIpV4MappedMulticast = (a->s6_addr32[0] == htonl(0xff0e0000) && ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL));
  return (isIpV4Mapped || isIpV4MappedMulticast) ? AF_INET : AF_INET6;
}

static bool match_gid_address_prefix(sa_family_t af, void* prefix, int prefixlen, union ibv_gid* gid)
{
  struct in_addr *base = NULL;
  struct in6_addr *base6 = NULL;
  struct in6_addr *addr6 = NULL;;
  if (af == AF_INET) {
    base = (struct in_addr *)prefix;
  } else {
    base6 = (struct in6_addr *)prefix;
  }
  addr6 = (struct in6_addr *)gid->raw;

#define NETMASK(bits) (htonl(0xffffffff ^ ((1 << (32 - bits)) - 1)))

  int i = 0;
  while (prefixlen > 0 && i < 4) {
    if (af == AF_INET) {
      int mask = NETMASK(prefixlen);
      if ((base->s_addr & mask) ^ (addr6->s6_addr32[3] & mask)) {
        break;
      }
      prefixlen = 0;
      break;
    } else {
      if (prefixlen >= 32) {
        if (base6->s6_addr32[i] ^ addr6->s6_addr32[i]) {
          break;
        }
        prefixlen -= 32;
        ++i;
      } else {
        int mask = NETMASK(prefixlen);
        if ((base6->s6_addr32[i] & mask) ^ (addr6->s6_addr32[i] & mask)) {
          break;
        }
        prefixlen = 0;
      }
    }
  }

  return (prefixlen == 0) ? true : false;
}

static int get_RoCE_version_number(const char* deviceName, int portNum, int gidIndex, int* version) {
  char gidRoceVerStr[16] = { 0 };
  char roceTypePath[PATH_MAX] = { 0 };
  sprintf(roceTypePath, "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d", deviceName, portNum, gidIndex);

  int fd = open(roceTypePath, O_RDONLY);
  if (fd == -1)
  {
    return 1;
  }

  int ret = read(fd, gidRoceVerStr, 15);
  close(fd);

  if (ret == -1)
  {
    return 1;
  }

  if (strlen(gidRoceVerStr))
  {
    if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0 || strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0)
    {
      *version = 1;
    }
    else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0)
    {
      *version = 2;
    }
  }

  return 0;
}

static int update_gid_index(struct ibv_context* context, uint8_t portNum, sa_family_t af, void* prefix, int prefixlen, int roceVer, int gidIndexCandidate, int* gidIndex)
{
  union ibv_gid gid, gidCandidate;
  IBV_CALL(ibv_query_gid, context, portNum, *gidIndex, &gid);
  IBV_CALL(ibv_query_gid, context, portNum, gidIndexCandidate, &gidCandidate);

  sa_family_t usrFam = af;
  sa_family_t gidFam = get_gid_address_family(&gid);
  sa_family_t gidCandidateFam = get_gid_address_family(&gidCandidate);
  bool gidCandidateMatchSubnet = match_gid_address_prefix(usrFam, prefix, prefixlen, &gidCandidate);

  if (gidCandidateFam != gidFam && gidCandidateFam == usrFam && gidCandidateMatchSubnet)
  {
    *gidIndex = gidIndexCandidate;
  }
  else
  {
    if (gidCandidateFam != usrFam || !validGid(&gidCandidate) || !gidCandidateMatchSubnet)
    {
      return 0;
    }
    int usrRoceVer = roceVer;
    int gidRoceVerNum, gidRoceVerNumCandidate;
    const char* deviceName = ibv_get_device_name(context->device);
    IBV_CALL(get_RoCE_version_number, deviceName, portNum, *gidIndex, &gidRoceVerNum);
    IBV_CALL(get_RoCE_version_number, deviceName, portNum, gidIndexCandidate, &gidRoceVerNumCandidate);
    if ((gidRoceVerNum != gidRoceVerNumCandidate || !validGid(&gid)) && gidRoceVerNumCandidate == usrRoceVer)
    {
      *gidIndex = gidIndexCandidate;
    }
  }
  return 0;
}

int set_gid_index(struct ibv_context *context, uint8_t portNum, int gidTblLen, int roce_version, int ip_address_family, int *gidIndex)
{
  if (*gidIndex >= 0)
  {
    return 0;
  }  
  sa_family_t userAddrFamily = (ip_address_family == 6)? AF_INET6 : AF_INET;

  int userRoceVersion = roce_version;

  // TODO: Get address range from user
  void *prefix = NULL;

  *gidIndex = 0;
  for (int gidIndexNext = 1; gidIndexNext < gidTblLen; ++gidIndexNext)
  {
    IBV_CALL(update_gid_index, context, portNum, userAddrFamily, prefix, 0, userRoceVersion, gidIndexNext, gidIndex);
  }
  return 0;
}

#endif
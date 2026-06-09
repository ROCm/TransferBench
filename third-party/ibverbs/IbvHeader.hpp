/*
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

#include <stdint.h>
#include <stddef.h>

extern "C" {

// ---------------------------------------------------------------------------
// Opaque handles (forward declarations only)
// ---------------------------------------------------------------------------
struct ibv_pd;
struct ibv_cq;
struct ibv_srq;
struct ibv_ah;
struct ibv_mw;
struct ibv_dm;
struct ibv_xrcd;
struct ibv_comp_channel;

// ---------------------------------------------------------------------------
// ibv_gid - 16-byte GID. verbs.h declares the inner fields as __be64 (big
// endian). TransferBench only memcpys/broadcasts/compares this opaque blob,
// so plain uint64_t preserves the layout without dragging in <linux/types.h>.
// ---------------------------------------------------------------------------
union ibv_gid {
  uint8_t raw[16];
  struct {
    uint64_t subnet_prefix;
    uint64_t interface_id;
  } global;
};

// ---------------------------------------------------------------------------
// Device enumeration types
// ---------------------------------------------------------------------------
enum ibv_node_type {
  IBV_NODE_UNKNOWN     = -1,
  IBV_NODE_CA          = 1,
  IBV_NODE_SWITCH,
  IBV_NODE_ROUTER,
  IBV_NODE_RNIC,
  IBV_NODE_USNIC,
  IBV_NODE_USNIC_UDP,
  IBV_NODE_UNSPECIFIED,
};

enum ibv_transport_type {
  IBV_TRANSPORT_UNKNOWN     = -1,
  IBV_TRANSPORT_IB          = 0,
  IBV_TRANSPORT_IWARP,
  IBV_TRANSPORT_USNIC,
  IBV_TRANSPORT_USNIC_UDP,
  IBV_TRANSPORT_UNSPECIFIED,
};

enum ibv_atomic_cap {
  IBV_ATOMIC_NONE,
  IBV_ATOMIC_HCA,
  IBV_ATOMIC_GLOB,
};

// ibv_device_ops: 2 opaque function pointers; preserved purely for layout.
struct _ibv_device_ops {
  struct ibv_context *(*_dummy1)(struct ibv_device *device, int cmd_fd);
  void                (*_dummy2)(struct ibv_context *context);
};

enum {
  IBV_SYSFS_NAME_MAX = 64,
  IBV_SYSFS_PATH_MAX = 256,
};

struct ibv_device {
  struct _ibv_device_ops  _ops;
  enum ibv_node_type      node_type;
  enum ibv_transport_type transport_type;
  char                    name[IBV_SYSFS_NAME_MAX];
  char                    dev_name[IBV_SYSFS_NAME_MAX];
  char                    dev_path[IBV_SYSFS_PATH_MAX];
  char                    ibdev_path[IBV_SYSFS_PATH_MAX];
};

// We only ever read ->device (offset 0). The remaining fields (ops table,
// fds, mutex, ...) are intentionally omitted - libibverbs allocates and
// frees these objects and we never take sizeof(ibv_context).
struct ibv_context {
  struct ibv_device *device;
};

// ---------------------------------------------------------------------------
// Device / port attributes (populated by ibv_query_device / ibv_query_port)
// ---------------------------------------------------------------------------
struct ibv_device_attr {
  char                 fw_ver[64];
  uint64_t             node_guid;
  uint64_t             sys_image_guid;
  uint64_t             max_mr_size;
  uint64_t             page_size_cap;
  uint32_t             vendor_id;
  uint32_t             vendor_part_id;
  uint32_t             hw_ver;
  int                  max_qp;
  int                  max_qp_wr;
  unsigned int         device_cap_flags;
  int                  max_sge;
  int                  max_sge_rd;
  int                  max_cq;
  int                  max_cqe;
  int                  max_mr;
  int                  max_pd;
  int                  max_qp_rd_atom;
  int                  max_ee_rd_atom;
  int                  max_res_rd_atom;
  int                  max_qp_init_rd_atom;
  int                  max_ee_init_rd_atom;
  enum ibv_atomic_cap  atomic_cap;
  int                  max_ee;
  int                  max_rdd;
  int                  max_mw;
  int                  max_raw_ipv6_qp;
  int                  max_raw_ethy_qp;
  int                  max_mcast_grp;
  int                  max_mcast_qp_attach;
  int                  max_total_mcast_qp_attach;
  int                  max_ah;
  int                  max_fmr;
  int                  max_map_per_fmr;
  int                  max_srq;
  int                  max_srq_wr;
  int                  max_srq_sge;
  uint16_t             max_pkeys;
  uint8_t              local_ca_ack_delay;
  uint8_t              phys_port_cnt;
};

enum ibv_mtu {
  IBV_MTU_256  = 1,
  IBV_MTU_512  = 2,
  IBV_MTU_1024 = 3,
  IBV_MTU_2048 = 4,
  IBV_MTU_4096 = 5,
};

enum ibv_port_state {
  IBV_PORT_NOP          = 0,
  IBV_PORT_DOWN         = 1,
  IBV_PORT_INIT         = 2,
  IBV_PORT_ARMED        = 3,
  IBV_PORT_ACTIVE       = 4,
  IBV_PORT_ACTIVE_DEFER = 5,
};

enum {
  IBV_LINK_LAYER_UNSPECIFIED,
  IBV_LINK_LAYER_INFINIBAND,
  IBV_LINK_LAYER_ETHERNET,
};

struct ibv_port_attr {
  enum ibv_port_state state;
  enum ibv_mtu        max_mtu;
  enum ibv_mtu        active_mtu;
  int                 gid_tbl_len;
  uint32_t            port_cap_flags;
  uint32_t            max_msg_sz;
  uint32_t            bad_pkey_cntr;
  uint32_t            qkey_viol_cntr;
  uint16_t            pkey_tbl_len;
  uint16_t            lid;
  uint16_t            sm_lid;
  uint8_t             lmc;
  uint8_t             max_vl_num;
  uint8_t             sm_sl;
  uint8_t             subnet_timeout;
  uint8_t             init_type_reply;
  uint8_t             active_width;
  uint8_t             active_speed;
  uint8_t             phys_state;
  uint8_t             link_layer;
  uint8_t             flags;
  uint16_t            port_cap_flags2;
  uint32_t            active_speed_ex;
};

// ---------------------------------------------------------------------------
// Memory region (populated by ibv_reg_mr / ibv_reg_dmabuf_mr)
// ---------------------------------------------------------------------------
struct ibv_mr {
  struct ibv_context *context;
  struct ibv_pd      *pd;
  void               *addr;
  size_t              length;
  uint32_t            handle;
  uint32_t            lkey;
  uint32_t            rkey;
};

// ---------------------------------------------------------------------------
// Address handle / global route (used inside ibv_qp_attr.ah_attr)
// ---------------------------------------------------------------------------
struct ibv_global_route {
  union ibv_gid dgid;
  uint32_t      flow_label;
  uint8_t       sgid_index;
  uint8_t       hop_limit;
  uint8_t       traffic_class;
};

struct ibv_ah_attr {
  struct ibv_global_route grh;
  uint16_t                dlid;
  uint8_t                 sl;
  uint8_t                 src_path_bits;
  uint8_t                 static_rate;
  uint8_t                 is_global;
  uint8_t                 port_num;
};

// ---------------------------------------------------------------------------
// Queue pair init / modify attributes
// ---------------------------------------------------------------------------
enum ibv_qp_type {
  IBV_QPT_RC          = 2,
  IBV_QPT_UC          = 3,
  IBV_QPT_UD          = 4,
  IBV_QPT_RAW_PACKET  = 8,
  IBV_QPT_XRC_SEND    = 9,
  IBV_QPT_XRC_RECV    = 10,
  IBV_QPT_DRIVER      = 0xff,
};

struct ibv_qp_cap {
  uint32_t max_send_wr;
  uint32_t max_recv_wr;
  uint32_t max_send_sge;
  uint32_t max_recv_sge;
  uint32_t max_inline_data;
};

struct ibv_qp_init_attr {
  void              *qp_context;
  struct ibv_cq     *send_cq;
  struct ibv_cq     *recv_cq;
  struct ibv_srq    *srq;
  struct ibv_qp_cap  cap;
  enum ibv_qp_type   qp_type;
  int                sq_sig_all;
};

enum ibv_qp_attr_mask {
  IBV_QP_STATE              = 1 <<  0,
  IBV_QP_CUR_STATE          = 1 <<  1,
  IBV_QP_EN_SQD_ASYNC_NOTIFY = 1 << 2,
  IBV_QP_ACCESS_FLAGS       = 1 <<  3,
  IBV_QP_PKEY_INDEX         = 1 <<  4,
  IBV_QP_PORT               = 1 <<  5,
  IBV_QP_QKEY               = 1 <<  6,
  IBV_QP_AV                 = 1 <<  7,
  IBV_QP_PATH_MTU           = 1 <<  8,
  IBV_QP_TIMEOUT            = 1 <<  9,
  IBV_QP_RETRY_CNT          = 1 << 10,
  IBV_QP_RNR_RETRY          = 1 << 11,
  IBV_QP_RQ_PSN             = 1 << 12,
  IBV_QP_MAX_QP_RD_ATOMIC   = 1 << 13,
  IBV_QP_ALT_PATH           = 1 << 14,
  IBV_QP_MIN_RNR_TIMER      = 1 << 15,
  IBV_QP_SQ_PSN             = 1 << 16,
  IBV_QP_MAX_DEST_RD_ATOMIC = 1 << 17,
  IBV_QP_PATH_MIG_STATE     = 1 << 18,
  IBV_QP_CAP                = 1 << 19,
  IBV_QP_DEST_QPN           = 1 << 20,
  IBV_QP_RATE_LIMIT         = 1 << 25,
};

enum ibv_qp_state {
  IBV_QPS_RESET,
  IBV_QPS_INIT,
  IBV_QPS_RTR,
  IBV_QPS_RTS,
  IBV_QPS_SQD,
  IBV_QPS_SQE,
  IBV_QPS_ERR,
  IBV_QPS_UNKNOWN,
};

// ibv_qp - layout matches libibverbs through qp_num. TransferBench only ever
// holds ibv_qp* returned by ibv_create_qp and reads qp_num, so the trailing
// libibverbs members (mutex/cond/events_completed) are intentionally omitted:
// we never allocate or sizeof an ibv_qp, and every accessed field sits at its
// real ABI offset. ibv_srq stays opaque (pointer only).
struct ibv_qp {
  struct ibv_context *context;
  void               *qp_context;
  struct ibv_pd      *pd;
  struct ibv_cq      *send_cq;
  struct ibv_cq      *recv_cq;
  struct ibv_srq     *srq;
  uint32_t            handle;
  uint32_t            qp_num;
  enum ibv_qp_state   state;
  enum ibv_qp_type    qp_type;
};

enum ibv_mig_state {
  IBV_MIG_MIGRATED,
  IBV_MIG_REARM,
  IBV_MIG_ARMED,
};

struct ibv_qp_attr {
  enum ibv_qp_state   qp_state;
  enum ibv_qp_state   cur_qp_state;
  enum ibv_mtu        path_mtu;
  enum ibv_mig_state  path_mig_state;
  uint32_t            qkey;
  uint32_t            rq_psn;
  uint32_t            sq_psn;
  uint32_t            dest_qp_num;
  unsigned int        qp_access_flags;
  struct ibv_qp_cap   cap;
  struct ibv_ah_attr  ah_attr;
  struct ibv_ah_attr  alt_ah_attr;
  uint16_t            pkey_index;
  uint16_t            alt_pkey_index;
  uint8_t             en_sqd_async_notify;
  uint8_t             sq_draining;
  uint8_t             max_rd_atomic;
  uint8_t             max_dest_rd_atomic;
  uint8_t             min_rnr_timer;
  uint8_t             port_num;
  uint8_t             timeout;
  uint8_t             retry_cnt;
  uint8_t             rnr_retry;
  uint8_t             alt_port_num;
  uint8_t             alt_timeout;
  uint32_t            rate_limit;
};

// ---------------------------------------------------------------------------
// Memory access / send flags
// ---------------------------------------------------------------------------
// IBV_ACCESS_RELAXED_ORDERING resolves to IB_UVERBS_ACCESS_OPTIONAL_FIRST,
// which the kernel uAPI defines as (1 << 20).
enum ibv_access_flags {
  IBV_ACCESS_LOCAL_WRITE      = 1,
  IBV_ACCESS_REMOTE_WRITE     = (1 << 1),
  IBV_ACCESS_REMOTE_READ      = (1 << 2),
  IBV_ACCESS_REMOTE_ATOMIC    = (1 << 3),
  IBV_ACCESS_MW_BIND          = (1 << 4),
  IBV_ACCESS_ZERO_BASED       = (1 << 5),
  IBV_ACCESS_ON_DEMAND        = (1 << 6),
  IBV_ACCESS_HUGETLB          = (1 << 7),
  IBV_ACCESS_FLUSH_GLOBAL     = (1 << 8),
  IBV_ACCESS_FLUSH_PERSISTENT = (1 << 9),
  IBV_ACCESS_RELAXED_ORDERING = (1 << 20),
};

enum ibv_wr_opcode {
  IBV_WR_RDMA_WRITE,
  IBV_WR_RDMA_WRITE_WITH_IMM,
  IBV_WR_SEND,
  IBV_WR_SEND_WITH_IMM,
  IBV_WR_RDMA_READ,
  IBV_WR_ATOMIC_CMP_AND_SWP,
  IBV_WR_ATOMIC_FETCH_AND_ADD,
  IBV_WR_LOCAL_INV,
  IBV_WR_BIND_MW,
  IBV_WR_SEND_WITH_INV,
  IBV_WR_TSO,
  IBV_WR_DRIVER1,
  IBV_WR_FLUSH        = 14,
  IBV_WR_ATOMIC_WRITE = 15,
};

enum ibv_send_flags {
  IBV_SEND_FENCE     = 1 << 0,
  IBV_SEND_SIGNALED  = 1 << 1,
  IBV_SEND_SOLICITED = 1 << 2,
  IBV_SEND_INLINE    = 1 << 3,
  IBV_SEND_IP_CSUM   = 1 << 4,
};

// ---------------------------------------------------------------------------
// Scatter/gather and work request (consumed by ibv_post_send)
// ---------------------------------------------------------------------------
struct ibv_sge {
  uint64_t addr;
  uint32_t length;
  uint32_t lkey;
};

// Forward decl needed by ibv_send_wr.bind_mw (kept for layout). Mirrors
// verbs.h's struct ibv_mw_bind_info exactly.
struct ibv_mw_bind_info {
  struct ibv_mr *mr;
  uint64_t       addr;
  uint64_t       length;
  unsigned int   mw_access_flags;
};

// Full ABI-exact ibv_send_wr. TransferBench only sets the rdma arm of `wr`,
// but the union's overall size must match the system layout because the
// driver may write through the entire struct.
struct ibv_send_wr {
  uint64_t            wr_id;
  struct ibv_send_wr *next;
  struct ibv_sge     *sg_list;
  int                 num_sge;
  enum ibv_wr_opcode  opcode;
  unsigned int        send_flags;
  union {
    uint32_t imm_data;
    uint32_t invalidate_rkey;
  };
  union {
    struct {
      uint64_t remote_addr;
      uint32_t rkey;
    } rdma;
    struct {
      uint64_t remote_addr;
      uint64_t compare_add;
      uint64_t swap;
      uint32_t rkey;
    } atomic;
    struct {
      struct ibv_ah *ah;
      uint32_t       remote_qpn;
      uint32_t       remote_qkey;
    } ud;
  } wr;
  union {
    struct {
      uint32_t remote_srqn;
    } xrc;
  } qp_type;
  union {
    struct {
      struct ibv_mw          *mw;
      uint32_t                rkey;
      struct ibv_mw_bind_info bind_info;
    } bind_mw;
    struct {
      void    *hdr;
      uint16_t hdr_sz;
      uint16_t mss;
    } tso;
  };
};

// ---------------------------------------------------------------------------
// Completion queue entry (populated by ibv_poll_cq)
// ---------------------------------------------------------------------------
enum ibv_wc_status {
  IBV_WC_SUCCESS,
  IBV_WC_LOC_LEN_ERR,
  IBV_WC_LOC_QP_OP_ERR,
  IBV_WC_LOC_EEC_OP_ERR,
  IBV_WC_LOC_PROT_ERR,
  IBV_WC_WR_FLUSH_ERR,
  IBV_WC_MW_BIND_ERR,
  IBV_WC_BAD_RESP_ERR,
  IBV_WC_LOC_ACCESS_ERR,
  IBV_WC_REM_INV_REQ_ERR,
  IBV_WC_REM_ACCESS_ERR,
  IBV_WC_REM_OP_ERR,
  IBV_WC_RETRY_EXC_ERR,
  IBV_WC_RNR_RETRY_EXC_ERR,
  IBV_WC_LOC_RDD_VIOL_ERR,
  IBV_WC_REM_INV_RD_REQ_ERR,
  IBV_WC_REM_ABORT_ERR,
  IBV_WC_INV_EECN_ERR,
  IBV_WC_INV_EEC_STATE_ERR,
  IBV_WC_FATAL_ERR,
  IBV_WC_RESP_TIMEOUT_ERR,
  IBV_WC_GENERAL_ERR,
  IBV_WC_TM_ERR,
  IBV_WC_TM_RNDV_INCOMPLETE,
};

enum ibv_wc_opcode {
  IBV_WC_SEND,
  IBV_WC_RDMA_WRITE,
  IBV_WC_RDMA_READ,
  IBV_WC_COMP_SWAP,
  IBV_WC_FETCH_ADD,
  IBV_WC_BIND_MW,
  IBV_WC_LOCAL_INV,
  IBV_WC_TSO,
  IBV_WC_FLUSH,
  IBV_WC_ATOMIC_WRITE       = 9,
  IBV_WC_RECV               = 1 << 7,
  IBV_WC_RECV_RDMA_WITH_IMM,
  IBV_WC_TM_ADD,
  IBV_WC_TM_DEL,
  IBV_WC_TM_SYNC,
  IBV_WC_TM_RECV,
  IBV_WC_TM_NO_TAG,
  IBV_WC_DRIVER1,
  IBV_WC_DRIVER2,
  IBV_WC_DRIVER3,
};

struct ibv_wc {
  uint64_t             wr_id;
  enum ibv_wc_status   status;
  enum ibv_wc_opcode   opcode;
  uint32_t             vendor_err;
  uint32_t             byte_len;
  union {
    uint32_t imm_data;
    uint32_t invalidated_rkey;
  };
  uint32_t             qp_num;
  uint32_t             src_qp;
  unsigned int         wc_flags;
  uint16_t             pkey_index;
  uint16_t             slid;
  uint8_t              sl;
  uint8_t              dlid_path_bits;
};
}  // extern "C"

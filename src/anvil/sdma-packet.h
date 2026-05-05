/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Auto-generated from OSS7.0 SDMA MAS by extract_sdma_headers.py
 * Source: OSS_70-sDMA_MAS.md
 */

// clang-format off
#ifndef __OSS70_SDMA_PKT_OPEN_H_
#define __OSS70_SDMA_PKT_OPEN_H_

/*
** SDMA Opcode definitions (OSS7.0)
*/
#define SDMA_OP_NOP                      0
#define SDMA_OP_COPY                     1
#define SDMA_OP_WRITE                    2
#define SDMA_OP_INDIRECT                 4
#define SDMA_OP_FENCE                    5
#define SDMA_OP_TRAP                     6
#define SDMA_OP_SEM                      7
#define SDMA_OP_POLL_REGMEM              8
#define SDMA_OP_COND_EXE                 9
#define SDMA_OP_ATOMIC                   10
#define SDMA_OP_CONST_FILL               11
#define SDMA_OP_PTEPDE                   12
#define SDMA_OP_TIMESTAMP                13
#define SDMA_OP_SRBM_WRITE               14
#define SDMA_OP_PRE_EXE                  15
#define SDMA_OP_GPUVM_INV                16
#define SDMA_OP_GCR_REQ                  17

/*
** SDMA Sub-opcode definitions (OSS7.0)
*/
#define SDMA_SUBOP_COPY_LINEAR_PHY_MI4                   0x8
#define SDMA_SUBOP_COPY_PAGE_TRANSFER_MI4                0xc
#define SDMA_SUBOP_COPY_LINEAR_MULTICAST_MI4             0xa
#define SDMA_SUBOP_COPY_LINEAR_WAIT_SIGNAL_MI4           0x0
#define SDMA_SUBOP_COPY_MULTICAST_WAIT_SIGNAL_MI4        0xa
#define SDMA_SUBOP_COPY_SWAP_WAIT_SIGNAL_MI4             0x9
#define SDMA_SUBOP_WRITE_LINEAR_MI4                      0x0
#define SDMA_SUBOP_FENCE_MI4                             0x0
#define SDMA_SUBOP_FENCE_COND_INT_NAVI4                  0x1
#define SDMA_SUBOP_FENCE_COND_INT_MI4                    0x1
#define SDMA_SUBOP_FENCE_64B_MI4                         0x2
#define SDMA_SUBOP_POLL_MEM_64B_MI4                      0x5
#define SDMA_SUBOP_POLL_MEM_64B_MES_SYNC_NAVI4           0x6
#define SDMA_SUBOP_EXCL_COND_EXEC                        0x1
#define SDMA_SUBOP_PTEPDE_GEN_MI4                        0x0
#define SDMA_SUBOP_PTEPDE_RMW_MI4                        0x2
#define SDMA_SUBOP_CONSTANT_FILL_MI4                     0x0
#define SDMA_SUBOP_CONSTANT_FILL_PAGE_MI4                0x4


/*
** Definitions for SDMA_PKT_COPY_LINEAR_PHY_MI4 packet
** Command: Physical Linear Copy (MI4xx)
** OP=0x1 SUB_OP=0x8
*/

/*define for HEADER word*/
/*define for op_code field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_op_code_offset 0
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_op_code_mask   0x000000FF
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_op_code_shift  0
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_OP_CODE(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_op_code_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_op_code_shift)

/*define for sub_op_code field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_sub_op_code_offset 0
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_sub_op_code_mask   0x000000FF
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_sub_op_code_shift  8
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_SUB_OP_CODE(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_sub_op_code_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_sub_op_code_shift)

/*define for tmz field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_tmz_offset 0
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_tmz_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_tmz_shift  18
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_TMZ(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_tmz_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_tmz_shift)

/*define for nsd field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_nsd_offset 0
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_nsd_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_nsd_shift  28
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_NSD(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_nsd_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_HEADER_nsd_shift)

/*define for DW_1 word*/
/*define for count field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_1_count_offset 1
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_1_count_mask   0x003FFFFF
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_1_count_shift  0
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_1_COUNT(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_1_count_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_1_count_shift)

/*define for addr_pair field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_1_addr_pair_offset 1
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_1_addr_pair_mask   0x000000FF
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_1_addr_pair_shift  24
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_1_ADDR_PAIR(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_1_addr_pair_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_1_addr_pair_shift)

/*define for DW_2 word*/
/*define for dst_mtype field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_mtype_offset 2
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_mtype_mask   0x00000003
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_mtype_shift  4
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_DST_MTYPE(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_mtype_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_mtype_shift)

/*define for dst_scope field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_scope_offset 2
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_scope_mask   0x00000003
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_scope_shift  6
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_DST_SCOPE(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_scope_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_scope_shift)

/*define for dst_temporal_hint field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_temporal_hint_offset 2
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_temporal_hint_shift  8
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_DST_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_temporal_hint_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_temporal_hint_shift)

/*define for src_mtype field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_mtype_offset 2
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_mtype_mask   0x00000003
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_mtype_shift  12
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_SRC_MTYPE(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_mtype_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_mtype_shift)

/*define for src_scope field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_scope_offset 2
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_scope_mask   0x00000003
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_scope_shift  14
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_SRC_SCOPE(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_scope_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_scope_shift)

/*define for src_temporal_hint field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_temporal_hint_offset 2
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_temporal_hint_shift  16
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_SRC_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_temporal_hint_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_temporal_hint_shift)

/*define for dst_gcc field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_gcc_offset 2
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_gcc_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_gcc_shift  19
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_DST_GCC(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_gcc_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_gcc_shift)

/*define for dst_sys field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_sys_offset 2
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_sys_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_sys_shift  20
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_DST_SYS(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_sys_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_sys_shift)

/*define for dst_snp field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_snp_offset 2
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_snp_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_snp_shift  22
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_DST_SNP(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_snp_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_snp_shift)

/*define for dst_gpa field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_gpa_offset 2
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_gpa_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_gpa_shift  23
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_DST_GPA(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_gpa_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_dst_gpa_shift)

/*define for src_gcc field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_gcc_offset 2
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_gcc_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_gcc_shift  19
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_SRC_GCC(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_gcc_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_gcc_shift)

/*define for src_sys field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_sys_offset 2
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_sys_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_sys_shift  28
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_SRC_SYS(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_sys_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_sys_shift)

/*define for src_snp field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_snp_offset 2
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_snp_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_snp_shift  30
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_SRC_SNP(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_snp_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_snp_shift)

/*define for src_gpa field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_gpa_offset 2
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_gpa_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_gpa_shift  31
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_SRC_GPA(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_gpa_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_2_src_gpa_shift)

/*define for DW_3 word*/
/*define for data_format field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_data_format_offset 3
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_data_format_mask   0x0000003F
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_data_format_shift  0
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_DATA_FORMAT(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_data_format_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_data_format_shift)

/*define for num_type field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_num_type_offset 3
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_num_type_mask   0x00000007
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_num_type_shift  9
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_NUM_TYPE(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_num_type_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_num_type_shift)

/*define for rd_cm field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_rd_cm_offset 3
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_rd_cm_mask   0x00000003
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_rd_cm_shift  16
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_RD_CM(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_rd_cm_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_rd_cm_shift)

/*define for wr_cm field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_wr_cm_offset 3
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_wr_cm_mask   0x00000003
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_wr_cm_shift  18
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_WR_CM(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_wr_cm_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_wr_cm_shift)

/*define for max_com field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_max_com_offset 3
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_max_com_mask   0x00000003
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_max_com_shift  24
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_MAX_COM(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_max_com_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_max_com_shift)

/*define for max_ucom field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_max_ucom_offset 3
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_max_ucom_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_max_ucom_shift  26
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_MAX_UCOM(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_max_ucom_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_max_ucom_shift)

/*define for dcc field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_dcc_offset 3
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_dcc_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_dcc_shift  31
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_DCC(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_dcc_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DW_3_dcc_shift)

/*define for SRC_ADDRESS_LO word*/
/*define for src_address_lo field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_SRC_ADDRESS_LO_src_address_lo_offset 4
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_SRC_ADDRESS_LO_src_address_lo_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_SRC_ADDRESS_LO_src_address_lo_shift  0
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_SRC_ADDRESS_LO_SRC_ADDRESS_LO(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_SRC_ADDRESS_LO_src_address_lo_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_SRC_ADDRESS_LO_src_address_lo_shift)

/*define for SRC_ADDRESS_HI word*/
/*define for src_address_hi field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_SRC_ADDRESS_HI_src_address_hi_offset 5
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_SRC_ADDRESS_HI_src_address_hi_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_SRC_ADDRESS_HI_src_address_hi_shift  0
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_SRC_ADDRESS_HI_SRC_ADDRESS_HI(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_SRC_ADDRESS_HI_src_address_hi_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_SRC_ADDRESS_HI_src_address_hi_shift)

/*define for DST_ADDRESS_LO word*/
/*define for dst_address_lo field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DST_ADDRESS_LO_dst_address_lo_offset 6
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DST_ADDRESS_LO_dst_address_lo_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DST_ADDRESS_LO_dst_address_lo_shift  0
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DST_ADDRESS_LO_DST_ADDRESS_LO(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DST_ADDRESS_LO_dst_address_lo_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DST_ADDRESS_LO_dst_address_lo_shift)

/*define for DST_ADDRESS_HI word*/
/*define for dst_address_hi field*/
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DST_ADDRESS_HI_dst_address_hi_offset 7
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DST_ADDRESS_HI_dst_address_hi_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DST_ADDRESS_HI_dst_address_hi_shift  0
#define SDMA_PKT_COPY_LINEAR_PHY_MI4_DST_ADDRESS_HI_DST_ADDRESS_HI(x) (((x) & SDMA_PKT_COPY_LINEAR_PHY_MI4_DST_ADDRESS_HI_dst_address_hi_mask) << SDMA_PKT_COPY_LINEAR_PHY_MI4_DST_ADDRESS_HI_dst_address_hi_shift)


/*
** Definitions for SDMA_PKT_COPY_PAGE_TRANSFER_MI4 packet
** Command: Page Transfer Copy (MI4xx)
** OP=0x1 SUB_OP=0xc
*/

/*define for HEADER word*/
/*define for op_code field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_op_code_offset 0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_op_code_mask   0x000000FF
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_op_code_shift  0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_OP_CODE(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_op_code_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_op_code_shift)

/*define for sub_op_code field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_sub_op_code_offset 0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_sub_op_code_mask   0x000000FF
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_sub_op_code_shift  8
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_SUB_OP_CODE(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_sub_op_code_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_sub_op_code_shift)

/*define for p_mtype field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_p_mtype_offset 0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_p_mtype_mask   0x00000003
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_p_mtype_shift  16
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_P_MTYPE(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_p_mtype_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_p_mtype_shift)

/*define for tmz field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_tmz_offset 0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_tmz_mask   0x00000001
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_tmz_shift  18
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_TMZ(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_tmz_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_tmz_shift)

/*define for l_scope field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_l_scope_offset 0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_l_scope_mask   0x00000003
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_l_scope_shift  19
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_L_SCOPE(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_l_scope_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_l_scope_shift)

/*define for s_scope field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_s_scope_offset 0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_s_scope_mask   0x00000003
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_s_scope_shift  21
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_S_SCOPE(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_s_scope_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_s_scope_shift)

/*define for page_size field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_page_size_offset 0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_page_size_mask   0x0000000F
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_page_size_shift  24
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_PAGE_SIZE(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_page_size_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_page_size_shift)

/*define for p_scope field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_p_scope_offset 0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_p_scope_mask   0x00000003
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_p_scope_shift  29
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_P_SCOPE(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_p_scope_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_p_scope_shift)

/*define for d field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_d_offset 0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_d_mask   0x00000001
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_d_shift  31
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_D(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_d_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_HEADER_d_shift)

/*define for DW_1 word*/
/*define for p_temporal_hint field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_p_temporal_hint_offset 1
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_p_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_p_temporal_hint_shift  0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_P_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_p_temporal_hint_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_p_temporal_hint_shift)

/*define for p_sys field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_p_sys_offset 1
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_p_sys_mask   0x00000001
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_p_sys_shift  4
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_P_SYS(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_p_sys_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_p_sys_shift)

/*define for p_snp field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_p_snp_offset 1
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_p_snp_mask   0x00000001
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_p_snp_shift  6
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_P_SNP(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_p_snp_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_p_snp_shift)

/*define for l_temporal_hint field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_l_temporal_hint_offset 1
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_l_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_l_temporal_hint_shift  8
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_L_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_l_temporal_hint_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_l_temporal_hint_shift)

/*define for l_mtype field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_l_mtype_offset 1
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_l_mtype_mask   0x00000003
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_l_mtype_shift  11
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_L_MTYPE(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_l_mtype_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_l_mtype_shift)

/*define for l_snp field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_l_snp_offset 1
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_l_snp_mask   0x00000001
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_l_snp_shift  14
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_L_SNP(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_l_snp_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_l_snp_shift)

/*define for s_temporal_hint field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_s_temporal_hint_offset 1
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_s_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_s_temporal_hint_shift  16
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_S_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_s_temporal_hint_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_s_temporal_hint_shift)

/*define for s_mtype field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_s_mtype_offset 1
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_s_mtype_mask   0x00000003
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_s_mtype_shift  19
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_S_MTYPE(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_s_mtype_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_s_mtype_shift)

/*define for s_snp field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_s_snp_offset 1
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_s_snp_mask   0x00000001
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_s_snp_shift  22
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_S_SNP(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_s_snp_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_s_snp_shift)

/*define for sysmem_addr_array_num field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_sysmem_addr_array_num_offset 1
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_sysmem_addr_array_num_mask   0x000000FF
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_sysmem_addr_array_num_shift  24
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_SYSMEM_ADDR_ARRAY_NUM(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_sysmem_addr_array_num_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_1_sysmem_addr_array_num_shift)

/*define for DW_2 word*/
/*define for data_format field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_data_format_offset 2
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_data_format_mask   0x0000003F
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_data_format_shift  0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_DATA_FORMAT(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_data_format_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_data_format_shift)

/*define for num_type field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_num_type_offset 2
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_num_type_mask   0x00000007
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_num_type_shift  9
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_NUM_TYPE(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_num_type_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_num_type_shift)

/*define for rd_cm field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_rd_cm_offset 2
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_rd_cm_mask   0x00000003
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_rd_cm_shift  16
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_RD_CM(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_rd_cm_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_rd_cm_shift)

/*define for wr_cm field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_wr_cm_offset 2
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_wr_cm_mask   0x00000003
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_wr_cm_shift  18
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_WR_CM(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_wr_cm_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_wr_cm_shift)

/*define for max_com field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_max_com_offset 2
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_max_com_mask   0x00000003
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_max_com_shift  24
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_MAX_COM(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_max_com_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_max_com_shift)

/*define for max_ucom field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_max_ucom_offset 2
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_max_ucom_mask   0x00000001
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_max_ucom_shift  26
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_MAX_UCOM(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_max_ucom_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_max_ucom_shift)

/*define for dcc field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_dcc_offset 2
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_dcc_mask   0x00000001
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_dcc_shift  31
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_DCC(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_dcc_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_2_dcc_shift)

/*define for PTE_MASK_LO word*/
/*define for pte_mask_lo field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_MASK_LO_pte_mask_lo_offset 3
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_MASK_LO_pte_mask_lo_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_MASK_LO_pte_mask_lo_shift  0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_MASK_LO_PTE_MASK_LO(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_MASK_LO_pte_mask_lo_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_MASK_LO_pte_mask_lo_shift)

/*define for PTE_MASK_HI word*/
/*define for pte_mask_hi field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_MASK_HI_pte_mask_hi_offset 4
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_MASK_HI_pte_mask_hi_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_MASK_HI_pte_mask_hi_shift  0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_MASK_HI_PTE_MASK_HI(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_MASK_HI_pte_mask_hi_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_MASK_HI_pte_mask_hi_shift)

/*define for PTE_ADDR_LO word*/
/*define for pte_addr_lo field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_ADDR_LO_pte_addr_lo_offset 5
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_ADDR_LO_pte_addr_lo_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_ADDR_LO_pte_addr_lo_shift  0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_ADDR_LO_PTE_ADDR_LO(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_ADDR_LO_pte_addr_lo_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_ADDR_LO_pte_addr_lo_shift)

/*define for PTE_ADDR_HI word*/
/*define for pte_addr_hi field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_ADDR_HI_pte_addr_hi_offset 6
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_ADDR_HI_pte_addr_hi_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_ADDR_HI_pte_addr_hi_shift  0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_ADDR_HI_PTE_ADDR_HI(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_ADDR_HI_pte_addr_hi_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_PTE_ADDR_HI_pte_addr_hi_shift)

/*define for LOCALMEM_ADDR_LO word*/
/*define for localmem_addr_lo field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_LOCALMEM_ADDR_LO_localmem_addr_lo_offset 7
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_LOCALMEM_ADDR_LO_localmem_addr_lo_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_LOCALMEM_ADDR_LO_localmem_addr_lo_shift  0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_LOCALMEM_ADDR_LO_LOCALMEM_ADDR_LO(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_LOCALMEM_ADDR_LO_localmem_addr_lo_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_LOCALMEM_ADDR_LO_localmem_addr_lo_shift)

/*define for LOCALMEM_ADDR_HI word*/
/*define for localmem_addr_hi field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_LOCALMEM_ADDR_HI_localmem_addr_hi_offset 8
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_LOCALMEM_ADDR_HI_localmem_addr_hi_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_LOCALMEM_ADDR_HI_localmem_addr_hi_shift  0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_LOCALMEM_ADDR_HI_LOCALMEM_ADDR_HI(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_LOCALMEM_ADDR_HI_localmem_addr_hi_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_LOCALMEM_ADDR_HI_localmem_addr_hi_shift)

/*define for SYSMEM_ADDR_0_LO word*/
/*define for sysmem_addr_0_lo field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_0_LO_sysmem_addr_0_lo_offset 9
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_0_LO_sysmem_addr_0_lo_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_0_LO_sysmem_addr_0_lo_shift  0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_0_LO_SYSMEM_ADDR_0_LO(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_0_LO_sysmem_addr_0_lo_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_0_LO_sysmem_addr_0_lo_shift)

/*define for SYSMEM_ADDR_0_HI word*/
/*define for sysmem_addr_0_hi field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_0_HI_sysmem_addr_0_hi_offset 10
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_0_HI_sysmem_addr_0_hi_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_0_HI_sysmem_addr_0_hi_shift  0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_0_HI_SYSMEM_ADDR_0_HI(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_0_HI_sysmem_addr_0_hi_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_0_HI_sysmem_addr_0_hi_shift)

/*define for DW_11 word*/
/*define for sysmem_addr_1_lo field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_11_sysmem_addr_1_lo_offset 11
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_11_sysmem_addr_1_lo_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_11_sysmem_addr_1_lo_shift  0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_11_SYSMEM_ADDR_1_LO(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_11_sysmem_addr_1_lo_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_11_sysmem_addr_1_lo_shift)

/*define for SYSMEM_ADDR_1_HI word*/
/*define for sysmem_addr_1_hi field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_1_HI_sysmem_addr_1_hi_offset 12
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_1_HI_sysmem_addr_1_hi_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_1_HI_sysmem_addr_1_hi_shift  0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_1_HI_SYSMEM_ADDR_1_HI(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_1_HI_sysmem_addr_1_hi_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_SYSMEM_ADDR_1_HI_sysmem_addr_1_hi_shift)

/*define for DW_11 word*/
/*define for sysmem_addr_n field*/
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_11_sysmem_addr_n_offset 11
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_11_sysmem_addr_n_mask   0xFFFFFFFFFFFFFFFF
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_11_sysmem_addr_n_shift  0
#define SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_11_SYSMEM_ADDR_N(x) (((x) & SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_11_sysmem_addr_n_mask) << SDMA_PKT_COPY_PAGE_TRANSFER_MI4_DW_11_sysmem_addr_n_shift)


/*
** Definitions for SDMA_PKT_COPY_LINEAR_MULTICAST_MI4 packet
** Command: Linear Multicast Copy (MI4xx)
** OP=0x1 SUB_OP=0xa
*/

/*define for HEADER word*/
/*define for op field*/
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_op_offset 0
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_op_mask   0x000000FF
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_op_shift  0
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_OP(x) (((x) & SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_op_mask) << SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_op_shift)

/*define for subop field*/
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_subop_offset 0
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_subop_mask   0x000000FF
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_subop_shift  8
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_SUBOP(x) (((x) & SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_subop_mask) << SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_subop_shift)

/*define for tmz field*/
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_tmz_offset 0
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_tmz_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_tmz_shift  18
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_TMZ(x) (((x) & SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_tmz_mask) << SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_tmz_shift)

/*define for npd field*/
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_npd_offset 0
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_npd_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_npd_shift  28
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_NPD(x) (((x) & SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_npd_mask) << SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_HEADER_npd_shift)

/*define for COUNT word*/
/*define for count field*/
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_COUNT_count_offset 1
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_COUNT_count_mask   0x3FFFFFFF
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_COUNT_count_shift  0
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_COUNT_COUNT(x) (((x) & SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_COUNT_count_mask) << SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_COUNT_count_shift)

/*define for DW_2 word*/
/*define for num_of_destination field*/
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_num_of_destination_offset 2
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_num_of_destination_mask   0x000003FF
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_num_of_destination_shift  0
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_NUM_OF_DESTINATION(x) (((x) & SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_num_of_destination_mask) << SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_num_of_destination_shift)

/*define for dst_scope field*/
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_dst_scope_offset 2
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_dst_scope_mask   0x00000003
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_dst_scope_shift  18
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_DST_SCOPE(x) (((x) & SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_dst_scope_mask) << SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_dst_scope_shift)

/*define for dst_temporal_hint field*/
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_dst_temporal_hint_offset 2
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_dst_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_dst_temporal_hint_shift  20
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_DST_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_dst_temporal_hint_mask) << SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_dst_temporal_hint_shift)

/*define for src_scope field*/
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_src_scope_offset 2
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_src_scope_mask   0x00000003
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_src_scope_shift  26
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_SRC_SCOPE(x) (((x) & SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_src_scope_mask) << SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_src_scope_shift)

/*define for src_temporal_hint field*/
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_src_temporal_hint_offset 2
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_src_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_src_temporal_hint_shift  28
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_SRC_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_src_temporal_hint_mask) << SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DW_2_src_temporal_hint_shift)

/*define for SRC_ADDR_31_0 word*/
/*define for src_addr_31_0 field*/
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_SRC_ADDR_31_0_src_addr_31_0_offset 3
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_SRC_ADDR_31_0_src_addr_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_SRC_ADDR_31_0_src_addr_31_0_shift  0
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_SRC_ADDR_31_0_SRC_ADDR_31_0(x) (((x) & SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_SRC_ADDR_31_0_src_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_SRC_ADDR_31_0_src_addr_31_0_shift)

/*define for SRC_ADDR_63_32 word*/
/*define for src_addr_63_32 field*/
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_SRC_ADDR_63_32_src_addr_63_32_offset 4
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_SRC_ADDR_63_32_src_addr_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_SRC_ADDR_63_32_src_addr_63_32_shift  0
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_SRC_ADDR_63_32_SRC_ADDR_63_32(x) (((x) & SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_SRC_ADDR_63_32_src_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_SRC_ADDR_63_32_src_addr_63_32_shift)

/*define for DST_ADDR_31_0 word*/
/*define for dst_addr_31_0 field*/
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DST_ADDR_31_0_dst_addr_31_0_offset 5
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DST_ADDR_31_0_dst_addr_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DST_ADDR_31_0_dst_addr_31_0_shift  0
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DST_ADDR_31_0_DST_ADDR_31_0(x) (((x) & SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DST_ADDR_31_0_dst_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DST_ADDR_31_0_dst_addr_31_0_shift)

/*define for DST_ADDR_63_32 word*/
/*define for dst_addr_63_32 field*/
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DST_ADDR_63_32_dst_addr_63_32_offset 6
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DST_ADDR_63_32_dst_addr_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DST_ADDR_63_32_dst_addr_63_32_shift  0
#define SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DST_ADDR_63_32_DST_ADDR_63_32(x) (((x) & SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DST_ADDR_63_32_dst_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_MULTICAST_MI4_DST_ADDR_63_32_dst_addr_63_32_shift)


/*
** Definitions for SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4 packet
** Command: Linear Copy with Wait/Signal (MI4xx)
** OP=0x1 SUB_OP=0x0
*/

/*define for HEADER word*/
/*define for op field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_op_offset 0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_op_mask   0x000000FF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_op_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_OP(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_op_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_op_shift)

/*define for subop field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_subop_offset 0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_subop_mask   0x000000FF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_subop_shift  8
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_SUBOP(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_subop_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_subop_shift)

/*define for tmz field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_tmz_offset 0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_tmz_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_tmz_shift  18
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_TMZ(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_tmz_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_tmz_shift)

/*define for npd field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_npd_offset 0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_npd_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_npd_shift  28
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_NPD(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_npd_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_npd_shift)

/*define for wait field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_wait_offset 0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_wait_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_wait_shift  30
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_WAIT(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_wait_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_wait_shift)

/*define for signal field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_signal_offset 0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_signal_mask   0x00000001
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_signal_shift  31
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_SIGNAL(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_signal_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_HEADER_signal_shift)

/*define for DW_1 word*/
/*define for wait_function field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_wait_function_offset 1
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_wait_function_mask   0x00000007
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_wait_function_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_WAIT_FUNCTION(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_wait_function_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_wait_function_shift)

/*define for wait_scope field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_wait_scope_offset 1
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_wait_scope_mask   0x00000003
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_wait_scope_shift  18
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_WAIT_SCOPE(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_wait_scope_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_wait_scope_shift)

/*define for wait_temporal_hint field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_wait_temporal_hint_offset 1
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_wait_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_wait_temporal_hint_shift  20
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_WAIT_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_wait_temporal_hint_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_1_wait_temporal_hint_shift)

/*define for WAIT_ADDR_31_3 word*/
/*define for wait_addr_31_3 field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_wait_addr_31_3_offset 2
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_wait_addr_31_3_mask   0x1FFFFFFF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_wait_addr_31_3_shift  3
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_WAIT_ADDR_31_3(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_wait_addr_31_3_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_wait_addr_31_3_shift)

/*define for WAIT_ADDR_63_32 word*/
/*define for wait_addr_63_32 field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_wait_addr_63_32_offset 3
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_wait_addr_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_wait_addr_63_32_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_WAIT_ADDR_63_32(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_wait_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_wait_addr_63_32_shift)

/*define for WAIT_REFERENCE_31_0 word*/
/*define for wait_reference_31_0 field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_wait_reference_31_0_offset 4
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_wait_reference_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_wait_reference_31_0_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_WAIT_REFERENCE_31_0(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_wait_reference_31_0_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_wait_reference_31_0_shift)

/*define for WAIT_REFERENCE_63_32 word*/
/*define for wait_reference_63_32 field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_wait_reference_63_32_offset 5
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_wait_reference_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_wait_reference_63_32_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_WAIT_REFERENCE_63_32(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_wait_reference_63_32_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_wait_reference_63_32_shift)

/*define for WAIT_MASK_31_0 word*/
/*define for wait_mask_31_0 field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_wait_mask_31_0_offset 6
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_wait_mask_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_wait_mask_31_0_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_WAIT_MASK_31_0(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_wait_mask_31_0_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_wait_mask_31_0_shift)

/*define for WAIT_MASK_63_32 word*/
/*define for wait_mask_63_32 field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_wait_mask_63_32_offset 7
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_wait_mask_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_wait_mask_63_32_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_WAIT_MASK_63_32(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_wait_mask_63_32_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_wait_mask_63_32_shift)

/*define for COPY_COUNT word*/
/*define for copy_count field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_COPY_COUNT_copy_count_offset 8
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_COPY_COUNT_copy_count_mask   0x3FFFFFFF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_COPY_COUNT_copy_count_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_COPY_COUNT_COPY_COUNT(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_COPY_COUNT_copy_count_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_COPY_COUNT_copy_count_shift)

/*define for DW_9 word*/
/*define for dst_scope field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_dst_scope_offset 9
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_dst_scope_mask   0x00000003
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_dst_scope_shift  18
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_DST_SCOPE(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_dst_scope_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_dst_scope_shift)

/*define for dst_temporal_hint field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_dst_temporal_hint_offset 9
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_dst_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_dst_temporal_hint_shift  20
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_DST_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_dst_temporal_hint_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_dst_temporal_hint_shift)

/*define for src_scope field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_src_scope_offset 9
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_src_scope_mask   0x00000003
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_src_scope_shift  26
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_SRC_SCOPE(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_src_scope_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_src_scope_shift)

/*define for src_temporal_hint field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_src_temporal_hint_offset 9
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_src_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_src_temporal_hint_shift  28
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_SRC_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_src_temporal_hint_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_9_src_temporal_hint_shift)

/*define for SRC_ADDR_31_0 word*/
/*define for src_addr_31_0 field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SRC_ADDR_31_0_src_addr_31_0_offset 10
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SRC_ADDR_31_0_src_addr_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SRC_ADDR_31_0_src_addr_31_0_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SRC_ADDR_31_0_SRC_ADDR_31_0(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SRC_ADDR_31_0_src_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SRC_ADDR_31_0_src_addr_31_0_shift)

/*define for SRC_ADDR_63_32 word*/
/*define for src_addr_63_32 field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SRC_ADDR_63_32_src_addr_63_32_offset 11
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SRC_ADDR_63_32_src_addr_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SRC_ADDR_63_32_src_addr_63_32_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SRC_ADDR_63_32_SRC_ADDR_63_32(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SRC_ADDR_63_32_src_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SRC_ADDR_63_32_src_addr_63_32_shift)

/*define for DST_ADDR_31_0 word*/
/*define for dst_addr_31_0 field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DST_ADDR_31_0_dst_addr_31_0_offset 12
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DST_ADDR_31_0_dst_addr_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DST_ADDR_31_0_dst_addr_31_0_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DST_ADDR_31_0_DST_ADDR_31_0(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DST_ADDR_31_0_dst_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DST_ADDR_31_0_dst_addr_31_0_shift)

/*define for DST_ADDR_63_32 word*/
/*define for dst_addr_63_32 field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DST_ADDR_63_32_dst_addr_63_32_offset 13
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DST_ADDR_63_32_dst_addr_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DST_ADDR_63_32_dst_addr_63_32_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DST_ADDR_63_32_DST_ADDR_63_32(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DST_ADDR_63_32_dst_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DST_ADDR_63_32_dst_addr_63_32_shift)

/*define for DW_14 word*/
/*define for signal_operation field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_signal_operation_offset 14
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_signal_operation_mask   0x0000007F
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_signal_operation_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_SIGNAL_OPERATION(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_signal_operation_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_signal_operation_shift)

/*define for signal_scope field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_signal_scope_offset 14
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_signal_scope_mask   0x00000003
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_signal_scope_shift  18
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_SIGNAL_SCOPE(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_signal_scope_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_signal_scope_shift)

/*define for signal_temporal_hint field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_signal_temporal_hint_offset 14
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_signal_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_signal_temporal_hint_shift  20
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_SIGNAL_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_signal_temporal_hint_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_DW_14_signal_temporal_hint_shift)

/*define for SIGNAL_ADDR_31_3 word*/
/*define for signal_addr_31_3 field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_signal_addr_31_3_offset 15
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_signal_addr_31_3_mask   0x1FFFFFFF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_signal_addr_31_3_shift  3
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_SIGNAL_ADDR_31_3(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_signal_addr_31_3_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_signal_addr_31_3_shift)

/*define for SIGNAL_ADDR_63_32 word*/
/*define for signal_addr_63_32 field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_signal_addr_63_32_offset 16
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_signal_addr_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_signal_addr_63_32_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_SIGNAL_ADDR_63_32(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_signal_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_signal_addr_63_32_shift)

/*define for SIGNAL_DATA_31_0 word*/
/*define for signal_data_31_0 field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_signal_data_31_0_offset 17
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_signal_data_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_signal_data_31_0_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_SIGNAL_DATA_31_0(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_signal_data_31_0_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_signal_data_31_0_shift)

/*define for SIGNAL_DATA_63_32 word*/
/*define for signal_data_63_32 field*/
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_signal_data_63_32_offset 18
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_signal_data_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_signal_data_63_32_shift  0
#define SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_SIGNAL_DATA_63_32(x) (((x) & SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_signal_data_63_32_mask) << SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_signal_data_63_32_shift)


/*
** Definitions for SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4 packet
** Command: Linear Multicast Copy with Wait/Signal (MI4xx)
** OP=0x1 SUB_OP=0xa
*/

/*define for HEADER word*/
/*define for op field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_op_offset 0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_op_mask   0x000000FF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_op_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_OP(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_op_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_op_shift)

/*define for subop field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_subop_offset 0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_subop_mask   0x000000FF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_subop_shift  8
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_SUBOP(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_subop_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_subop_shift)

/*define for tmz field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_tmz_offset 0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_tmz_mask   0x00000001
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_tmz_shift  18
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_TMZ(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_tmz_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_tmz_shift)

/*define for npd field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_npd_offset 0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_npd_mask   0x00000001
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_npd_shift  28
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_NPD(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_npd_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_npd_shift)

/*define for wait field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_wait_offset 0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_wait_mask   0x00000001
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_wait_shift  30
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_WAIT(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_wait_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_wait_shift)

/*define for signal field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_signal_offset 0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_signal_mask   0x00000001
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_signal_shift  31
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_SIGNAL(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_signal_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_HEADER_signal_shift)

/*define for DW_1 word*/
/*define for wait_function field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_wait_function_offset 1
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_wait_function_mask   0x00000007
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_wait_function_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_WAIT_FUNCTION(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_wait_function_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_wait_function_shift)

/*define for wait_scope field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_wait_scope_offset 1
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_wait_scope_mask   0x00000003
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_wait_scope_shift  18
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_WAIT_SCOPE(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_wait_scope_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_wait_scope_shift)

/*define for wait_temporal_hint field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_wait_temporal_hint_offset 1
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_wait_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_wait_temporal_hint_shift  20
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_WAIT_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_wait_temporal_hint_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_1_wait_temporal_hint_shift)

/*define for WAIT_ADDR_31_3 word*/
/*define for wait_addr_31_3 field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_wait_addr_31_3_offset 2
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_wait_addr_31_3_mask   0x1FFFFFFF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_wait_addr_31_3_shift  3
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_WAIT_ADDR_31_3(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_wait_addr_31_3_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_wait_addr_31_3_shift)

/*define for WAIT_ADDR_63_32 word*/
/*define for wait_addr_63_32 field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_wait_addr_63_32_offset 3
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_wait_addr_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_wait_addr_63_32_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_WAIT_ADDR_63_32(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_wait_addr_63_32_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_wait_addr_63_32_shift)

/*define for WAIT_REFERENCE_31_0 word*/
/*define for wait_reference_31_0 field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_wait_reference_31_0_offset 4
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_wait_reference_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_wait_reference_31_0_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_WAIT_REFERENCE_31_0(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_wait_reference_31_0_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_wait_reference_31_0_shift)

/*define for WAIT_REFERENCE_63_32 word*/
/*define for wait_reference_63_32 field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_wait_reference_63_32_offset 5
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_wait_reference_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_wait_reference_63_32_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_WAIT_REFERENCE_63_32(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_wait_reference_63_32_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_wait_reference_63_32_shift)

/*define for WAIT_MASK_31_0 word*/
/*define for wait_mask_31_0 field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_wait_mask_31_0_offset 6
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_wait_mask_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_wait_mask_31_0_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_WAIT_MASK_31_0(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_wait_mask_31_0_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_wait_mask_31_0_shift)

/*define for WAIT_MASK_63_32 word*/
/*define for wait_mask_63_32 field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_wait_mask_63_32_offset 7
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_wait_mask_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_wait_mask_63_32_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_WAIT_MASK_63_32(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_wait_mask_63_32_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_wait_mask_63_32_shift)

/*define for COUNT word*/
/*define for count field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_COUNT_count_offset 8
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_COUNT_count_mask   0x3FFFFFFF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_COUNT_count_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_COUNT_COUNT(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_COUNT_count_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_COUNT_count_shift)

/*define for DW_9 word*/
/*define for num_of_destination field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_num_of_destination_offset 9
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_num_of_destination_mask   0x000003FF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_num_of_destination_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_NUM_OF_DESTINATION(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_num_of_destination_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_num_of_destination_shift)

/*define for dst_scope field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_dst_scope_offset 9
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_dst_scope_mask   0x00000003
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_dst_scope_shift  18
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_DST_SCOPE(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_dst_scope_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_dst_scope_shift)

/*define for dst_temporal_hint field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_dst_temporal_hint_offset 9
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_dst_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_dst_temporal_hint_shift  20
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_DST_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_dst_temporal_hint_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_dst_temporal_hint_shift)

/*define for src_scope field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_src_scope_offset 9
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_src_scope_mask   0x00000003
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_src_scope_shift  26
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_SRC_SCOPE(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_src_scope_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_src_scope_shift)

/*define for src_temporal_hint field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_src_temporal_hint_offset 9
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_src_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_src_temporal_hint_shift  28
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_SRC_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_src_temporal_hint_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_9_src_temporal_hint_shift)

/*define for SRC_ADDR_31_0 word*/
/*define for src_addr_31_0 field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SRC_ADDR_31_0_src_addr_31_0_offset 10
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SRC_ADDR_31_0_src_addr_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SRC_ADDR_31_0_src_addr_31_0_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SRC_ADDR_31_0_SRC_ADDR_31_0(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SRC_ADDR_31_0_src_addr_31_0_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SRC_ADDR_31_0_src_addr_31_0_shift)

/*define for SRC_ADDR_63_32 word*/
/*define for src_addr_63_32 field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SRC_ADDR_63_32_src_addr_63_32_offset 11
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SRC_ADDR_63_32_src_addr_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SRC_ADDR_63_32_src_addr_63_32_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SRC_ADDR_63_32_SRC_ADDR_63_32(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SRC_ADDR_63_32_src_addr_63_32_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SRC_ADDR_63_32_src_addr_63_32_shift)

/*define for DST_ADDR_31_0 word*/
/*define for dst_addr_31_0 field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DST_ADDR_31_0_dst_addr_31_0_offset 12
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DST_ADDR_31_0_dst_addr_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DST_ADDR_31_0_dst_addr_31_0_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DST_ADDR_31_0_DST_ADDR_31_0(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DST_ADDR_31_0_dst_addr_31_0_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DST_ADDR_31_0_dst_addr_31_0_shift)

/*define for DST_ADDR_63_32 word*/
/*define for dst_addr_63_32 field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DST_ADDR_63_32_dst_addr_63_32_offset 13
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DST_ADDR_63_32_dst_addr_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DST_ADDR_63_32_dst_addr_63_32_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DST_ADDR_63_32_DST_ADDR_63_32(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DST_ADDR_63_32_dst_addr_63_32_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DST_ADDR_63_32_dst_addr_63_32_shift)

/*define for DW_14 word*/
/*define for signal_operation field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_signal_operation_offset 14
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_signal_operation_mask   0x0000007F
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_signal_operation_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_SIGNAL_OPERATION(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_signal_operation_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_signal_operation_shift)

/*define for signal_scope field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_signal_scope_offset 14
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_signal_scope_mask   0x00000003
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_signal_scope_shift  18
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_SIGNAL_SCOPE(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_signal_scope_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_signal_scope_shift)

/*define for signal_temporal_hint field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_signal_temporal_hint_offset 14
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_signal_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_signal_temporal_hint_shift  20
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_SIGNAL_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_signal_temporal_hint_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_DW_14_signal_temporal_hint_shift)

/*define for SIGNAL_ADDR_31_3 word*/
/*define for signal_addr_31_3 field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_signal_addr_31_3_offset 15
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_signal_addr_31_3_mask   0x1FFFFFFF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_signal_addr_31_3_shift  3
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_SIGNAL_ADDR_31_3(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_signal_addr_31_3_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_signal_addr_31_3_shift)

/*define for SIGNAL_ADDR_63_32 word*/
/*define for signal_addr_63_32 field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_signal_addr_63_32_offset 16
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_signal_addr_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_signal_addr_63_32_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_SIGNAL_ADDR_63_32(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_signal_addr_63_32_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_signal_addr_63_32_shift)

/*define for SIGNAL_DATA_31_0 word*/
/*define for signal_data_31_0 field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_signal_data_31_0_offset 17
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_signal_data_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_signal_data_31_0_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_SIGNAL_DATA_31_0(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_signal_data_31_0_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_signal_data_31_0_shift)

/*define for SIGNAL_DATA_63_32 word*/
/*define for signal_data_63_32 field*/
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_signal_data_63_32_offset 18
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_signal_data_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_signal_data_63_32_shift  0
#define SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_SIGNAL_DATA_63_32(x) (((x) & SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_signal_data_63_32_mask) << SDMA_PKT_COPY_MULTICAST_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_signal_data_63_32_shift)


/*
** Definitions for SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4 packet
** Command: Linear Swap Copy with Wait/Signal (MI4xx)
** OP=0x1 SUB_OP=0x9
*/

/*define for HEADER word*/
/*define for op field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_op_offset 0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_op_mask   0x000000FF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_op_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_OP(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_op_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_op_shift)

/*define for subop field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_subop_offset 0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_subop_mask   0x000000FF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_subop_shift  8
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_SUBOP(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_subop_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_subop_shift)

/*define for tmz field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_tmz_offset 0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_tmz_mask   0x00000001
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_tmz_shift  18
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_TMZ(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_tmz_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_tmz_shift)

/*define for wait field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_wait_offset 0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_wait_mask   0x00000001
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_wait_shift  30
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_WAIT(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_wait_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_wait_shift)

/*define for signal field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_signal_offset 0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_signal_mask   0x00000001
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_signal_shift  31
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_SIGNAL(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_signal_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_HEADER_signal_shift)

/*define for DW_1 word*/
/*define for wait_function field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_wait_function_offset 1
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_wait_function_mask   0x00000007
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_wait_function_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_WAIT_FUNCTION(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_wait_function_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_wait_function_shift)

/*define for wait_scope field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_wait_scope_offset 1
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_wait_scope_mask   0x00000003
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_wait_scope_shift  18
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_WAIT_SCOPE(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_wait_scope_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_wait_scope_shift)

/*define for wait_temporal_hint field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_wait_temporal_hint_offset 1
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_wait_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_wait_temporal_hint_shift  20
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_WAIT_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_wait_temporal_hint_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_1_wait_temporal_hint_shift)

/*define for WAIT_ADDR_31_3 word*/
/*define for wait_addr_31_3 field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_wait_addr_31_3_offset 2
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_wait_addr_31_3_mask   0x1FFFFFFF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_wait_addr_31_3_shift  3
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_WAIT_ADDR_31_3(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_wait_addr_31_3_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_ADDR_31_3_wait_addr_31_3_shift)

/*define for WAIT_ADDR_63_32 word*/
/*define for wait_addr_63_32 field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_wait_addr_63_32_offset 3
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_wait_addr_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_wait_addr_63_32_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_WAIT_ADDR_63_32(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_wait_addr_63_32_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_ADDR_63_32_wait_addr_63_32_shift)

/*define for WAIT_REFERENCE_31_0 word*/
/*define for wait_reference_31_0 field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_wait_reference_31_0_offset 4
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_wait_reference_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_wait_reference_31_0_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_WAIT_REFERENCE_31_0(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_wait_reference_31_0_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_REFERENCE_31_0_wait_reference_31_0_shift)

/*define for WAIT_REFERENCE_63_32 word*/
/*define for wait_reference_63_32 field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_wait_reference_63_32_offset 5
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_wait_reference_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_wait_reference_63_32_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_WAIT_REFERENCE_63_32(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_wait_reference_63_32_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_REFERENCE_63_32_wait_reference_63_32_shift)

/*define for WAIT_MASK_31_0 word*/
/*define for wait_mask_31_0 field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_wait_mask_31_0_offset 6
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_wait_mask_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_wait_mask_31_0_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_WAIT_MASK_31_0(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_wait_mask_31_0_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_MASK_31_0_wait_mask_31_0_shift)

/*define for WAIT_MASK_63_32 word*/
/*define for wait_mask_63_32 field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_wait_mask_63_32_offset 7
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_wait_mask_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_wait_mask_63_32_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_WAIT_MASK_63_32(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_wait_mask_63_32_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_WAIT_MASK_63_32_wait_mask_63_32_shift)

/*define for COUNT word*/
/*define for count field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_COUNT_count_offset 8
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_COUNT_count_mask   0x3FFFFFFF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_COUNT_count_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_COUNT_COUNT(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_COUNT_count_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_COUNT_count_shift)

/*define for DW_9 word*/
/*define for scope_b field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_scope_b_offset 9
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_scope_b_mask   0x00000003
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_scope_b_shift  18
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_SCOPE_B(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_scope_b_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_scope_b_shift)

/*define for temporal_hint_b field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_temporal_hint_b_offset 9
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_temporal_hint_b_mask   0x00000007
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_temporal_hint_b_shift  20
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_TEMPORAL_HINT_B(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_temporal_hint_b_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_temporal_hint_b_shift)

/*define for scope_a field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_scope_a_offset 9
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_scope_a_mask   0x00000003
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_scope_a_shift  26
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_SCOPE_A(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_scope_a_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_scope_a_shift)

/*define for temporal_hint_a field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_temporal_hint_a_offset 9
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_temporal_hint_a_mask   0x00000007
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_temporal_hint_a_shift  28
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_TEMPORAL_HINT_A(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_temporal_hint_a_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_9_temporal_hint_a_shift)

/*define for ADDR_A_31_0 word*/
/*define for addr_a_31_0 field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_A_31_0_addr_a_31_0_offset 10
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_A_31_0_addr_a_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_A_31_0_addr_a_31_0_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_A_31_0_ADDR_A_31_0(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_A_31_0_addr_a_31_0_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_A_31_0_addr_a_31_0_shift)

/*define for ADDR_A_63_32 word*/
/*define for addr_a_63_32 field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_A_63_32_addr_a_63_32_offset 11
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_A_63_32_addr_a_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_A_63_32_addr_a_63_32_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_A_63_32_ADDR_A_63_32(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_A_63_32_addr_a_63_32_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_A_63_32_addr_a_63_32_shift)

/*define for ADDR_B_31_0 word*/
/*define for addr_b_31_0 field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_B_31_0_addr_b_31_0_offset 12
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_B_31_0_addr_b_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_B_31_0_addr_b_31_0_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_B_31_0_ADDR_B_31_0(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_B_31_0_addr_b_31_0_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_B_31_0_addr_b_31_0_shift)

/*define for ADDR_B_63_32 word*/
/*define for addr_b_63_32 field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_B_63_32_addr_b_63_32_offset 13
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_B_63_32_addr_b_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_B_63_32_addr_b_63_32_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_B_63_32_ADDR_B_63_32(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_B_63_32_addr_b_63_32_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_ADDR_B_63_32_addr_b_63_32_shift)

/*define for DW_14 word*/
/*define for signal_operation field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_signal_operation_offset 14
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_signal_operation_mask   0x0000007F
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_signal_operation_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_SIGNAL_OPERATION(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_signal_operation_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_signal_operation_shift)

/*define for signal_scope field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_signal_scope_offset 14
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_signal_scope_mask   0x00000003
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_signal_scope_shift  18
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_SIGNAL_SCOPE(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_signal_scope_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_signal_scope_shift)

/*define for signal_temporal_hint field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_signal_temporal_hint_offset 14
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_signal_temporal_hint_mask   0x00000007
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_signal_temporal_hint_shift  20
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_SIGNAL_TEMPORAL_HINT(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_signal_temporal_hint_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_DW_14_signal_temporal_hint_shift)

/*define for SIGNAL_ADDR_31_3 word*/
/*define for signal_addr_31_3 field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_signal_addr_31_3_offset 15
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_signal_addr_31_3_mask   0x1FFFFFFF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_signal_addr_31_3_shift  3
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_SIGNAL_ADDR_31_3(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_signal_addr_31_3_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_ADDR_31_3_signal_addr_31_3_shift)

/*define for SIGNAL_ADDR_63_32 word*/
/*define for signal_addr_63_32 field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_signal_addr_63_32_offset 16
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_signal_addr_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_signal_addr_63_32_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_SIGNAL_ADDR_63_32(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_signal_addr_63_32_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_ADDR_63_32_signal_addr_63_32_shift)

/*define for SIGNAL_DATA_31_0 word*/
/*define for signal_data_31_0 field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_signal_data_31_0_offset 17
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_signal_data_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_signal_data_31_0_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_SIGNAL_DATA_31_0(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_signal_data_31_0_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_DATA_31_0_signal_data_31_0_shift)

/*define for SIGNAL_DATA_63_32 word*/
/*define for signal_data_63_32 field*/
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_signal_data_63_32_offset 18
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_signal_data_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_signal_data_63_32_shift  0
#define SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_SIGNAL_DATA_63_32(x) (((x) & SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_signal_data_63_32_mask) << SDMA_PKT_COPY_SWAP_WAIT_SIGNAL_MI4_SIGNAL_DATA_63_32_signal_data_63_32_shift)


/*
** Definitions for SDMA_PKT_WRITE_LINEAR_MI4 packet
** Command: Linear Write (MI4xx)
** OP=0x2 SUB_OP=0x0
*/

/*define for HEADER word*/
/*define for op_code field*/
#define SDMA_PKT_WRITE_LINEAR_MI4_HEADER_op_code_offset 0
#define SDMA_PKT_WRITE_LINEAR_MI4_HEADER_op_code_mask   0x000000FF
#define SDMA_PKT_WRITE_LINEAR_MI4_HEADER_op_code_shift  0
#define SDMA_PKT_WRITE_LINEAR_MI4_HEADER_OP_CODE(x) (((x) & SDMA_PKT_WRITE_LINEAR_MI4_HEADER_op_code_mask) << SDMA_PKT_WRITE_LINEAR_MI4_HEADER_op_code_shift)

/*define for sub_op_code field*/
#define SDMA_PKT_WRITE_LINEAR_MI4_HEADER_sub_op_code_offset 0
#define SDMA_PKT_WRITE_LINEAR_MI4_HEADER_sub_op_code_mask   0x000000FF
#define SDMA_PKT_WRITE_LINEAR_MI4_HEADER_sub_op_code_shift  8
#define SDMA_PKT_WRITE_LINEAR_MI4_HEADER_SUB_OP_CODE(x) (((x) & SDMA_PKT_WRITE_LINEAR_MI4_HEADER_sub_op_code_mask) << SDMA_PKT_WRITE_LINEAR_MI4_HEADER_sub_op_code_shift)

/*define for tmz field*/
#define SDMA_PKT_WRITE_LINEAR_MI4_HEADER_tmz_offset 0
#define SDMA_PKT_WRITE_LINEAR_MI4_HEADER_tmz_mask   0x00000001
#define SDMA_PKT_WRITE_LINEAR_MI4_HEADER_tmz_shift  18
#define SDMA_PKT_WRITE_LINEAR_MI4_HEADER_TMZ(x) (((x) & SDMA_PKT_WRITE_LINEAR_MI4_HEADER_tmz_mask) << SDMA_PKT_WRITE_LINEAR_MI4_HEADER_tmz_shift)

/*define for DST_ADDRESS_LO word*/
/*define for dst_address_lo field*/
#define SDMA_PKT_WRITE_LINEAR_MI4_DST_ADDRESS_LO_dst_address_lo_offset 1
#define SDMA_PKT_WRITE_LINEAR_MI4_DST_ADDRESS_LO_dst_address_lo_mask   0xFFFFFFFF
#define SDMA_PKT_WRITE_LINEAR_MI4_DST_ADDRESS_LO_dst_address_lo_shift  0
#define SDMA_PKT_WRITE_LINEAR_MI4_DST_ADDRESS_LO_DST_ADDRESS_LO(x) (((x) & SDMA_PKT_WRITE_LINEAR_MI4_DST_ADDRESS_LO_dst_address_lo_mask) << SDMA_PKT_WRITE_LINEAR_MI4_DST_ADDRESS_LO_dst_address_lo_shift)

/*define for DST_ADDRESS_HI word*/
/*define for dst_address_hi field*/
#define SDMA_PKT_WRITE_LINEAR_MI4_DST_ADDRESS_HI_dst_address_hi_offset 2
#define SDMA_PKT_WRITE_LINEAR_MI4_DST_ADDRESS_HI_dst_address_hi_mask   0xFFFFFFFF
#define SDMA_PKT_WRITE_LINEAR_MI4_DST_ADDRESS_HI_dst_address_hi_shift  0
#define SDMA_PKT_WRITE_LINEAR_MI4_DST_ADDRESS_HI_DST_ADDRESS_HI(x) (((x) & SDMA_PKT_WRITE_LINEAR_MI4_DST_ADDRESS_HI_dst_address_hi_mask) << SDMA_PKT_WRITE_LINEAR_MI4_DST_ADDRESS_HI_dst_address_hi_shift)

/*define for DW_3 word*/
/*define for count field*/
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_count_offset 3
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_count_mask   0x000FFFFF
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_count_shift  0
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_COUNT(x) (((x) & SDMA_PKT_WRITE_LINEAR_MI4_DW_3_count_mask) << SDMA_PKT_WRITE_LINEAR_MI4_DW_3_count_shift)

/*define for dst_sys field*/
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_sys_offset 3
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_sys_mask   0x00000001
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_sys_shift  20
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_DST_SYS(x) (((x) & SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_sys_mask) << SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_sys_shift)

/*define for dst_snp field*/
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_snp_offset 3
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_snp_mask   0x00000001
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_snp_shift  22
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_DST_SNP(x) (((x) & SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_snp_mask) << SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_snp_shift)

/*define for dst_gpa field*/
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_gpa_offset 3
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_gpa_mask   0x00000001
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_gpa_shift  23
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_DST_GPA(x) (((x) & SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_gpa_mask) << SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_gpa_shift)

/*define for dst_mtype field*/
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_mtype_offset 3
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_mtype_mask   0x00000003
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_mtype_shift  24
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_DST_MTYPE(x) (((x) & SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_mtype_mask) << SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_mtype_shift)

/*define for dst_scope field*/
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_scope_offset 3
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_scope_mask   0x00000003
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_scope_shift  26
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_DST_SCOPE(x) (((x) & SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_scope_mask) << SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_scope_shift)

/*define for dst_temporal_hint field*/
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_temporal_hint_offset 3
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_temporal_hint_mask   0x00000007
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_temporal_hint_shift  28
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_DST_TEMPORAL_HINT(x) (((x) & SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_temporal_hint_mask) << SDMA_PKT_WRITE_LINEAR_MI4_DW_3_dst_temporal_hint_shift)

/*define for data field*/
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_data_offset 3
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_data_mask   0xFFFFFFFF
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_data_shift  0
#define SDMA_PKT_WRITE_LINEAR_MI4_DW_3_DATA(x) (((x) & SDMA_PKT_WRITE_LINEAR_MI4_DW_3_data_mask) << SDMA_PKT_WRITE_LINEAR_MI4_DW_3_data_shift)


/*
** Definitions for SDMA_PKT_FENCE_MI4 packet
** Command: Fence (MI4xx)
** OP=0x5 SUB_OP=0x0
*/

/*define for HEADER word*/
/*define for op_code field*/
#define SDMA_PKT_FENCE_MI4_HEADER_op_code_offset 0
#define SDMA_PKT_FENCE_MI4_HEADER_op_code_mask   0x000000FF
#define SDMA_PKT_FENCE_MI4_HEADER_op_code_shift  0
#define SDMA_PKT_FENCE_MI4_HEADER_OP_CODE(x) (((x) & SDMA_PKT_FENCE_MI4_HEADER_op_code_mask) << SDMA_PKT_FENCE_MI4_HEADER_op_code_shift)

/*define for sub_op_code field*/
#define SDMA_PKT_FENCE_MI4_HEADER_sub_op_code_offset 0
#define SDMA_PKT_FENCE_MI4_HEADER_sub_op_code_mask   0x000000FF
#define SDMA_PKT_FENCE_MI4_HEADER_sub_op_code_shift  8
#define SDMA_PKT_FENCE_MI4_HEADER_SUB_OP_CODE(x) (((x) & SDMA_PKT_FENCE_MI4_HEADER_sub_op_code_mask) << SDMA_PKT_FENCE_MI4_HEADER_sub_op_code_shift)

/*define for mtype field*/
#define SDMA_PKT_FENCE_MI4_HEADER_mtype_offset 0
#define SDMA_PKT_FENCE_MI4_HEADER_mtype_mask   0x00000003
#define SDMA_PKT_FENCE_MI4_HEADER_mtype_shift  16
#define SDMA_PKT_FENCE_MI4_HEADER_MTYPE(x) (((x) & SDMA_PKT_FENCE_MI4_HEADER_mtype_mask) << SDMA_PKT_FENCE_MI4_HEADER_mtype_shift)

/*define for s field*/
#define SDMA_PKT_FENCE_MI4_HEADER_s_offset 0
#define SDMA_PKT_FENCE_MI4_HEADER_s_mask   0x00000001
#define SDMA_PKT_FENCE_MI4_HEADER_s_shift  20
#define SDMA_PKT_FENCE_MI4_HEADER_S(x) (((x) & SDMA_PKT_FENCE_MI4_HEADER_s_mask) << SDMA_PKT_FENCE_MI4_HEADER_s_shift)

/*define for snp field*/
#define SDMA_PKT_FENCE_MI4_HEADER_snp_offset 0
#define SDMA_PKT_FENCE_MI4_HEADER_snp_mask   0x00000001
#define SDMA_PKT_FENCE_MI4_HEADER_snp_shift  22
#define SDMA_PKT_FENCE_MI4_HEADER_SNP(x) (((x) & SDMA_PKT_FENCE_MI4_HEADER_snp_mask) << SDMA_PKT_FENCE_MI4_HEADER_snp_shift)

/*define for gpa field*/
#define SDMA_PKT_FENCE_MI4_HEADER_gpa_offset 0
#define SDMA_PKT_FENCE_MI4_HEADER_gpa_mask   0x00000001
#define SDMA_PKT_FENCE_MI4_HEADER_gpa_shift  23
#define SDMA_PKT_FENCE_MI4_HEADER_GPA(x) (((x) & SDMA_PKT_FENCE_MI4_HEADER_gpa_mask) << SDMA_PKT_FENCE_MI4_HEADER_gpa_shift)

/*define for scope field*/
#define SDMA_PKT_FENCE_MI4_HEADER_scope_offset 0
#define SDMA_PKT_FENCE_MI4_HEADER_scope_mask   0x00000003
#define SDMA_PKT_FENCE_MI4_HEADER_scope_shift  24
#define SDMA_PKT_FENCE_MI4_HEADER_SCOPE(x) (((x) & SDMA_PKT_FENCE_MI4_HEADER_scope_mask) << SDMA_PKT_FENCE_MI4_HEADER_scope_shift)

/*define for temporal_hint field*/
#define SDMA_PKT_FENCE_MI4_HEADER_temporal_hint_offset 0
#define SDMA_PKT_FENCE_MI4_HEADER_temporal_hint_mask   0x00000007
#define SDMA_PKT_FENCE_MI4_HEADER_temporal_hint_shift  26
#define SDMA_PKT_FENCE_MI4_HEADER_TEMPORAL_HINT(x) (((x) & SDMA_PKT_FENCE_MI4_HEADER_temporal_hint_mask) << SDMA_PKT_FENCE_MI4_HEADER_temporal_hint_shift)

/*define for FENCE_ADDR_LO word*/
/*define for fence_addr_lo field*/
#define SDMA_PKT_FENCE_MI4_FENCE_ADDR_LO_fence_addr_lo_offset 1
#define SDMA_PKT_FENCE_MI4_FENCE_ADDR_LO_fence_addr_lo_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_MI4_FENCE_ADDR_LO_fence_addr_lo_shift  0
#define SDMA_PKT_FENCE_MI4_FENCE_ADDR_LO_FENCE_ADDR_LO(x) (((x) & SDMA_PKT_FENCE_MI4_FENCE_ADDR_LO_fence_addr_lo_mask) << SDMA_PKT_FENCE_MI4_FENCE_ADDR_LO_fence_addr_lo_shift)

/*define for FENCE_ADDR_HI word*/
/*define for fence_addr_hi field*/
#define SDMA_PKT_FENCE_MI4_FENCE_ADDR_HI_fence_addr_hi_offset 2
#define SDMA_PKT_FENCE_MI4_FENCE_ADDR_HI_fence_addr_hi_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_MI4_FENCE_ADDR_HI_fence_addr_hi_shift  0
#define SDMA_PKT_FENCE_MI4_FENCE_ADDR_HI_FENCE_ADDR_HI(x) (((x) & SDMA_PKT_FENCE_MI4_FENCE_ADDR_HI_fence_addr_hi_mask) << SDMA_PKT_FENCE_MI4_FENCE_ADDR_HI_fence_addr_hi_shift)

/*define for DATA word*/
/*define for data field*/
#define SDMA_PKT_FENCE_MI4_DATA_data_offset 3
#define SDMA_PKT_FENCE_MI4_DATA_data_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_MI4_DATA_data_shift  0
#define SDMA_PKT_FENCE_MI4_DATA_DATA(x) (((x) & SDMA_PKT_FENCE_MI4_DATA_data_mask) << SDMA_PKT_FENCE_MI4_DATA_data_shift)


/*
** Definitions for SDMA_PKT_FENCE_COND_INT_NAVI4 packet
** Command: Fence Conditional Interrupt (Navi4x)
** OP=0x5 SUB_OP=0x1
*/

/*define for HEADER word*/
/*define for op_code field*/
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_op_code_offset 0
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_op_code_mask   0x000000FF
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_op_code_shift  0
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_OP_CODE(x) (((x) & SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_op_code_mask) << SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_op_code_shift)

/*define for sub_op_code field*/
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_sub_op_code_offset 0
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_sub_op_code_mask   0x000000FF
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_sub_op_code_shift  8
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_SUB_OP_CODE(x) (((x) & SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_sub_op_code_mask) << SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_sub_op_code_shift)

/*define for snp field*/
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_snp_offset 0
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_snp_mask   0x00000001
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_snp_shift  22
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_SNP(x) (((x) & SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_snp_mask) << SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_snp_shift)

/*define for gpa field*/
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_gpa_offset 0
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_gpa_mask   0x00000001
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_gpa_shift  23
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_GPA(x) (((x) & SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_gpa_mask) << SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_gpa_shift)

/*define for cache_policy field*/
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_cache_policy_offset 0
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_cache_policy_mask   0x00000007
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_cache_policy_shift  26
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_CACHE_POLICY(x) (((x) & SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_cache_policy_mask) << SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_cache_policy_shift)

/*define for qw field*/
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_qw_offset 0
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_qw_mask   0x00000001
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_qw_shift  31
#define SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_QW(x) (((x) & SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_qw_mask) << SDMA_PKT_FENCE_COND_INT_NAVI4_HEADER_qw_shift)

/*define for FENCE_ADDR_LO word*/
/*define for fence_addr_lo field*/
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_ADDR_LO_fence_addr_lo_offset 1
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_ADDR_LO_fence_addr_lo_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_ADDR_LO_fence_addr_lo_shift  0
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_ADDR_LO_FENCE_ADDR_LO(x) (((x) & SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_ADDR_LO_fence_addr_lo_mask) << SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_ADDR_LO_fence_addr_lo_shift)

/*define for FENCE_ADDR_HI word*/
/*define for fence_addr_hi field*/
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_ADDR_HI_fence_addr_hi_offset 2
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_ADDR_HI_fence_addr_hi_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_ADDR_HI_fence_addr_hi_shift  0
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_ADDR_HI_FENCE_ADDR_HI(x) (((x) & SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_ADDR_HI_fence_addr_hi_mask) << SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_ADDR_HI_fence_addr_hi_shift)

/*define for DATA_LO word*/
/*define for data_lo field*/
#define SDMA_PKT_FENCE_COND_INT_NAVI4_DATA_LO_data_lo_offset 3
#define SDMA_PKT_FENCE_COND_INT_NAVI4_DATA_LO_data_lo_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_NAVI4_DATA_LO_data_lo_shift  0
#define SDMA_PKT_FENCE_COND_INT_NAVI4_DATA_LO_DATA_LO(x) (((x) & SDMA_PKT_FENCE_COND_INT_NAVI4_DATA_LO_data_lo_mask) << SDMA_PKT_FENCE_COND_INT_NAVI4_DATA_LO_data_lo_shift)

/*define for DATA_HI word*/
/*define for data_hi field*/
#define SDMA_PKT_FENCE_COND_INT_NAVI4_DATA_HI_data_hi_offset 4
#define SDMA_PKT_FENCE_COND_INT_NAVI4_DATA_HI_data_hi_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_NAVI4_DATA_HI_data_hi_shift  0
#define SDMA_PKT_FENCE_COND_INT_NAVI4_DATA_HI_DATA_HI(x) (((x) & SDMA_PKT_FENCE_COND_INT_NAVI4_DATA_HI_data_hi_mask) << SDMA_PKT_FENCE_COND_INT_NAVI4_DATA_HI_data_hi_shift)

/*define for FENCE_REF_ADDR_LO word*/
/*define for fence_ref_addr_lo field*/
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_REF_ADDR_LO_fence_ref_addr_lo_offset 5
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_REF_ADDR_LO_fence_ref_addr_lo_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_REF_ADDR_LO_fence_ref_addr_lo_shift  0
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_REF_ADDR_LO_FENCE_REF_ADDR_LO(x) (((x) & SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_REF_ADDR_LO_fence_ref_addr_lo_mask) << SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_REF_ADDR_LO_fence_ref_addr_lo_shift)

/*define for FENCE_REF_ADDR_HI word*/
/*define for fence_ref_addr_hi field*/
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_REF_ADDR_HI_fence_ref_addr_hi_offset 6
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_REF_ADDR_HI_fence_ref_addr_hi_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_REF_ADDR_HI_fence_ref_addr_hi_shift  0
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_REF_ADDR_HI_FENCE_REF_ADDR_HI(x) (((x) & SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_REF_ADDR_HI_fence_ref_addr_hi_mask) << SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_REF_ADDR_HI_fence_ref_addr_hi_shift)

/*define for INTERRUPT_CONTEXT word*/
/*define for interrupt_context field*/
#define SDMA_PKT_FENCE_COND_INT_NAVI4_INTERRUPT_CONTEXT_interrupt_context_offset 7
#define SDMA_PKT_FENCE_COND_INT_NAVI4_INTERRUPT_CONTEXT_interrupt_context_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_NAVI4_INTERRUPT_CONTEXT_interrupt_context_shift  0
#define SDMA_PKT_FENCE_COND_INT_NAVI4_INTERRUPT_CONTEXT_INTERRUPT_CONTEXT(x) (((x) & SDMA_PKT_FENCE_COND_INT_NAVI4_INTERRUPT_CONTEXT_interrupt_context_mask) << SDMA_PKT_FENCE_COND_INT_NAVI4_INTERRUPT_CONTEXT_interrupt_context_shift)

/*define for FENCE_HANDLE word*/
/*define for fence_handle field*/
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_HANDLE_fence_handle_offset 8
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_HANDLE_fence_handle_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_HANDLE_fence_handle_shift  0
#define SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_HANDLE_FENCE_HANDLE(x) (((x) & SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_HANDLE_fence_handle_mask) << SDMA_PKT_FENCE_COND_INT_NAVI4_FENCE_HANDLE_fence_handle_shift)


/*
** Definitions for SDMA_PKT_FENCE_COND_INT_MI4 packet
** Command: Fence Conditional Interrupt (MI4xx)
** OP=0x5 SUB_OP=0x1
*/

/*define for HEADER word*/
/*define for op_code field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_op_code_offset 0
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_op_code_mask   0x000000FF
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_op_code_shift  0
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_OP_CODE(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_HEADER_op_code_mask) << SDMA_PKT_FENCE_COND_INT_MI4_HEADER_op_code_shift)

/*define for sub_op_code field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_sub_op_code_offset 0
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_sub_op_code_mask   0x000000FF
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_sub_op_code_shift  8
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_SUB_OP_CODE(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_HEADER_sub_op_code_mask) << SDMA_PKT_FENCE_COND_INT_MI4_HEADER_sub_op_code_shift)

/*define for mtype field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_mtype_offset 0
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_mtype_mask   0x00000003
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_mtype_shift  16
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_MTYPE(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_HEADER_mtype_mask) << SDMA_PKT_FENCE_COND_INT_MI4_HEADER_mtype_shift)

/*define for snp field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_snp_offset 0
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_snp_mask   0x00000001
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_snp_shift  22
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_SNP(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_HEADER_snp_mask) << SDMA_PKT_FENCE_COND_INT_MI4_HEADER_snp_shift)

/*define for gpa field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_gpa_offset 0
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_gpa_mask   0x00000001
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_gpa_shift  23
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_GPA(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_HEADER_gpa_mask) << SDMA_PKT_FENCE_COND_INT_MI4_HEADER_gpa_shift)

/*define for scope field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_scope_offset 0
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_scope_mask   0x00000003
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_scope_shift  24
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_SCOPE(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_HEADER_scope_mask) << SDMA_PKT_FENCE_COND_INT_MI4_HEADER_scope_shift)

/*define for temporal_hint field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_temporal_hint_offset 0
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_temporal_hint_mask   0x00000007
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_temporal_hint_shift  26
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_TEMPORAL_HINT(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_HEADER_temporal_hint_mask) << SDMA_PKT_FENCE_COND_INT_MI4_HEADER_temporal_hint_shift)

/*define for qw field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_qw_offset 0
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_qw_mask   0x00000001
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_qw_shift  31
#define SDMA_PKT_FENCE_COND_INT_MI4_HEADER_QW(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_HEADER_qw_mask) << SDMA_PKT_FENCE_COND_INT_MI4_HEADER_qw_shift)

/*define for FENCE_ADDR_LO word*/
/*define for fence_addr_lo field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ADDR_LO_fence_addr_lo_offset 1
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ADDR_LO_fence_addr_lo_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ADDR_LO_fence_addr_lo_shift  0
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ADDR_LO_FENCE_ADDR_LO(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ADDR_LO_fence_addr_lo_mask) << SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ADDR_LO_fence_addr_lo_shift)

/*define for FENCE_ADDR_HI word*/
/*define for fence_addr_hi field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ADDR_HI_fence_addr_hi_offset 2
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ADDR_HI_fence_addr_hi_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ADDR_HI_fence_addr_hi_shift  0
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ADDR_HI_FENCE_ADDR_HI(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ADDR_HI_fence_addr_hi_mask) << SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ADDR_HI_fence_addr_hi_shift)

/*define for DATA_LO word*/
/*define for data_lo field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_DATA_LO_data_lo_offset 3
#define SDMA_PKT_FENCE_COND_INT_MI4_DATA_LO_data_lo_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_MI4_DATA_LO_data_lo_shift  0
#define SDMA_PKT_FENCE_COND_INT_MI4_DATA_LO_DATA_LO(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_DATA_LO_data_lo_mask) << SDMA_PKT_FENCE_COND_INT_MI4_DATA_LO_data_lo_shift)

/*define for DATA_HI word*/
/*define for data_hi field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_DATA_HI_data_hi_offset 4
#define SDMA_PKT_FENCE_COND_INT_MI4_DATA_HI_data_hi_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_MI4_DATA_HI_data_hi_shift  0
#define SDMA_PKT_FENCE_COND_INT_MI4_DATA_HI_DATA_HI(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_DATA_HI_data_hi_mask) << SDMA_PKT_FENCE_COND_INT_MI4_DATA_HI_data_hi_shift)

/*define for FENCE_REF_ADDR_LO word*/
/*define for fence_ref_addr_lo field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_REF_ADDR_LO_fence_ref_addr_lo_offset 5
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_REF_ADDR_LO_fence_ref_addr_lo_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_REF_ADDR_LO_fence_ref_addr_lo_shift  0
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_REF_ADDR_LO_FENCE_REF_ADDR_LO(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_FENCE_REF_ADDR_LO_fence_ref_addr_lo_mask) << SDMA_PKT_FENCE_COND_INT_MI4_FENCE_REF_ADDR_LO_fence_ref_addr_lo_shift)

/*define for FENCE_REF_ADDR_HI word*/
/*define for fence_ref_addr_hi field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_REF_ADDR_HI_fence_ref_addr_hi_offset 6
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_REF_ADDR_HI_fence_ref_addr_hi_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_REF_ADDR_HI_fence_ref_addr_hi_shift  0
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_REF_ADDR_HI_FENCE_REF_ADDR_HI(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_FENCE_REF_ADDR_HI_fence_ref_addr_hi_mask) << SDMA_PKT_FENCE_COND_INT_MI4_FENCE_REF_ADDR_HI_fence_ref_addr_hi_shift)

/*define for INTERRUPT_CONTEXT word*/
/*define for interrupt_context field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_INTERRUPT_CONTEXT_interrupt_context_offset 7
#define SDMA_PKT_FENCE_COND_INT_MI4_INTERRUPT_CONTEXT_interrupt_context_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_MI4_INTERRUPT_CONTEXT_interrupt_context_shift  0
#define SDMA_PKT_FENCE_COND_INT_MI4_INTERRUPT_CONTEXT_INTERRUPT_CONTEXT(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_INTERRUPT_CONTEXT_interrupt_context_mask) << SDMA_PKT_FENCE_COND_INT_MI4_INTERRUPT_CONTEXT_interrupt_context_shift)

/*define for FENCE_ID word*/
/*define for fence_id field*/
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ID_fence_id_offset 8
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ID_fence_id_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ID_fence_id_shift  0
#define SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ID_FENCE_ID(x) (((x) & SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ID_fence_id_mask) << SDMA_PKT_FENCE_COND_INT_MI4_FENCE_ID_fence_id_shift)


/*
** Definitions for SDMA_PKT_FENCE_64B_MI4 packet
** Command: Fence 64b (MI4xx)
** OP=0x5 SUB_OP=0x2
*/

/*define for HEADER word*/
/*define for op field*/
#define SDMA_PKT_FENCE_64B_MI4_HEADER_op_offset 0
#define SDMA_PKT_FENCE_64B_MI4_HEADER_op_mask   0x000000FF
#define SDMA_PKT_FENCE_64B_MI4_HEADER_op_shift  0
#define SDMA_PKT_FENCE_64B_MI4_HEADER_OP(x) (((x) & SDMA_PKT_FENCE_64B_MI4_HEADER_op_mask) << SDMA_PKT_FENCE_64B_MI4_HEADER_op_shift)

/*define for subop field*/
#define SDMA_PKT_FENCE_64B_MI4_HEADER_subop_offset 0
#define SDMA_PKT_FENCE_64B_MI4_HEADER_subop_mask   0x000000FF
#define SDMA_PKT_FENCE_64B_MI4_HEADER_subop_shift  8
#define SDMA_PKT_FENCE_64B_MI4_HEADER_SUBOP(x) (((x) & SDMA_PKT_FENCE_64B_MI4_HEADER_subop_mask) << SDMA_PKT_FENCE_64B_MI4_HEADER_subop_shift)

/*define for mtype field*/
#define SDMA_PKT_FENCE_64B_MI4_HEADER_mtype_offset 0
#define SDMA_PKT_FENCE_64B_MI4_HEADER_mtype_mask   0x00000003
#define SDMA_PKT_FENCE_64B_MI4_HEADER_mtype_shift  16
#define SDMA_PKT_FENCE_64B_MI4_HEADER_MTYPE(x) (((x) & SDMA_PKT_FENCE_64B_MI4_HEADER_mtype_mask) << SDMA_PKT_FENCE_64B_MI4_HEADER_mtype_shift)

/*define for sys field*/
#define SDMA_PKT_FENCE_64B_MI4_HEADER_sys_offset 0
#define SDMA_PKT_FENCE_64B_MI4_HEADER_sys_mask   0x00000001
#define SDMA_PKT_FENCE_64B_MI4_HEADER_sys_shift  20
#define SDMA_PKT_FENCE_64B_MI4_HEADER_SYS(x) (((x) & SDMA_PKT_FENCE_64B_MI4_HEADER_sys_mask) << SDMA_PKT_FENCE_64B_MI4_HEADER_sys_shift)

/*define for snp field*/
#define SDMA_PKT_FENCE_64B_MI4_HEADER_snp_offset 0
#define SDMA_PKT_FENCE_64B_MI4_HEADER_snp_mask   0x00000001
#define SDMA_PKT_FENCE_64B_MI4_HEADER_snp_shift  22
#define SDMA_PKT_FENCE_64B_MI4_HEADER_SNP(x) (((x) & SDMA_PKT_FENCE_64B_MI4_HEADER_snp_mask) << SDMA_PKT_FENCE_64B_MI4_HEADER_snp_shift)

/*define for gpa field*/
#define SDMA_PKT_FENCE_64B_MI4_HEADER_gpa_offset 0
#define SDMA_PKT_FENCE_64B_MI4_HEADER_gpa_mask   0x00000001
#define SDMA_PKT_FENCE_64B_MI4_HEADER_gpa_shift  23
#define SDMA_PKT_FENCE_64B_MI4_HEADER_GPA(x) (((x) & SDMA_PKT_FENCE_64B_MI4_HEADER_gpa_mask) << SDMA_PKT_FENCE_64B_MI4_HEADER_gpa_shift)

/*define for scope field*/
#define SDMA_PKT_FENCE_64B_MI4_HEADER_scope_offset 0
#define SDMA_PKT_FENCE_64B_MI4_HEADER_scope_mask   0x00000003
#define SDMA_PKT_FENCE_64B_MI4_HEADER_scope_shift  24
#define SDMA_PKT_FENCE_64B_MI4_HEADER_SCOPE(x) (((x) & SDMA_PKT_FENCE_64B_MI4_HEADER_scope_mask) << SDMA_PKT_FENCE_64B_MI4_HEADER_scope_shift)

/*define for temporal_hint field*/
#define SDMA_PKT_FENCE_64B_MI4_HEADER_temporal_hint_offset 0
#define SDMA_PKT_FENCE_64B_MI4_HEADER_temporal_hint_mask   0x00000007
#define SDMA_PKT_FENCE_64B_MI4_HEADER_temporal_hint_shift  26
#define SDMA_PKT_FENCE_64B_MI4_HEADER_TEMPORAL_HINT(x) (((x) & SDMA_PKT_FENCE_64B_MI4_HEADER_temporal_hint_mask) << SDMA_PKT_FENCE_64B_MI4_HEADER_temporal_hint_shift)

/*define for ADDR_31_3 word*/
/*define for addr_31_3 field*/
#define SDMA_PKT_FENCE_64B_MI4_ADDR_31_3_addr_31_3_offset 1
#define SDMA_PKT_FENCE_64B_MI4_ADDR_31_3_addr_31_3_mask   0x1FFFFFFF
#define SDMA_PKT_FENCE_64B_MI4_ADDR_31_3_addr_31_3_shift  3
#define SDMA_PKT_FENCE_64B_MI4_ADDR_31_3_ADDR_31_3(x) (((x) & SDMA_PKT_FENCE_64B_MI4_ADDR_31_3_addr_31_3_mask) << SDMA_PKT_FENCE_64B_MI4_ADDR_31_3_addr_31_3_shift)

/*define for ADDR_63_32 word*/
/*define for addr_63_32 field*/
#define SDMA_PKT_FENCE_64B_MI4_ADDR_63_32_addr_63_32_offset 2
#define SDMA_PKT_FENCE_64B_MI4_ADDR_63_32_addr_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_64B_MI4_ADDR_63_32_addr_63_32_shift  0
#define SDMA_PKT_FENCE_64B_MI4_ADDR_63_32_ADDR_63_32(x) (((x) & SDMA_PKT_FENCE_64B_MI4_ADDR_63_32_addr_63_32_mask) << SDMA_PKT_FENCE_64B_MI4_ADDR_63_32_addr_63_32_shift)

/*define for DATA_31_0 word*/
/*define for data_31_0 field*/
#define SDMA_PKT_FENCE_64B_MI4_DATA_31_0_data_31_0_offset 3
#define SDMA_PKT_FENCE_64B_MI4_DATA_31_0_data_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_64B_MI4_DATA_31_0_data_31_0_shift  0
#define SDMA_PKT_FENCE_64B_MI4_DATA_31_0_DATA_31_0(x) (((x) & SDMA_PKT_FENCE_64B_MI4_DATA_31_0_data_31_0_mask) << SDMA_PKT_FENCE_64B_MI4_DATA_31_0_data_31_0_shift)

/*define for DATA_63_32 word*/
/*define for data_63_32 field*/
#define SDMA_PKT_FENCE_64B_MI4_DATA_63_32_data_63_32_offset 4
#define SDMA_PKT_FENCE_64B_MI4_DATA_63_32_data_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_FENCE_64B_MI4_DATA_63_32_data_63_32_shift  0
#define SDMA_PKT_FENCE_64B_MI4_DATA_63_32_DATA_63_32(x) (((x) & SDMA_PKT_FENCE_64B_MI4_DATA_63_32_data_63_32_mask) << SDMA_PKT_FENCE_64B_MI4_DATA_63_32_data_63_32_shift)


/*
** Definitions for SDMA_PKT_POLL_MEM_64B_MI4 packet
** Command: Poll Memory 64b (MI4xx)
** OP=0x8 SUB_OP=0x5
*/

/*define for HEADER word*/
/*define for op field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_op_offset 0
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_op_mask   0x000000FF
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_op_shift  0
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_OP(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_HEADER_op_mask) << SDMA_PKT_POLL_MEM_64B_MI4_HEADER_op_shift)

/*define for subop field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_subop_offset 0
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_subop_mask   0x000000FF
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_subop_shift  8
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_SUBOP(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_HEADER_subop_mask) << SDMA_PKT_POLL_MEM_64B_MI4_HEADER_subop_shift)

/*define for mtype field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_mtype_offset 0
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_mtype_mask   0x00000003
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_mtype_shift  16
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_MTYPE(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_HEADER_mtype_mask) << SDMA_PKT_POLL_MEM_64B_MI4_HEADER_mtype_shift)

/*define for sys field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_sys_offset 0
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_sys_mask   0x00000001
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_sys_shift  20
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_SYS(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_HEADER_sys_mask) << SDMA_PKT_POLL_MEM_64B_MI4_HEADER_sys_shift)

/*define for snp field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_snp_offset 0
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_snp_mask   0x00000001
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_snp_shift  22
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_SNP(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_HEADER_snp_mask) << SDMA_PKT_POLL_MEM_64B_MI4_HEADER_snp_shift)

/*define for gpa field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_gpa_offset 0
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_gpa_mask   0x00000001
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_gpa_shift  23
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_GPA(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_HEADER_gpa_mask) << SDMA_PKT_POLL_MEM_64B_MI4_HEADER_gpa_shift)

/*define for cache_policy field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_cache_policy_offset 0
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_cache_policy_mask   0x00000007
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_cache_policy_shift  24
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_CACHE_POLICY(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_HEADER_cache_policy_mask) << SDMA_PKT_POLL_MEM_64B_MI4_HEADER_cache_policy_shift)

/*define for func field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_func_offset 0
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_func_mask   0x00000007
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_func_shift  28
#define SDMA_PKT_POLL_MEM_64B_MI4_HEADER_FUNC(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_HEADER_func_mask) << SDMA_PKT_POLL_MEM_64B_MI4_HEADER_func_shift)

/*define for ADDR_31_3 word*/
/*define for addr_31_3 field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_ADDR_31_3_addr_31_3_offset 1
#define SDMA_PKT_POLL_MEM_64B_MI4_ADDR_31_3_addr_31_3_mask   0x1FFFFFFF
#define SDMA_PKT_POLL_MEM_64B_MI4_ADDR_31_3_addr_31_3_shift  3
#define SDMA_PKT_POLL_MEM_64B_MI4_ADDR_31_3_ADDR_31_3(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_ADDR_31_3_addr_31_3_mask) << SDMA_PKT_POLL_MEM_64B_MI4_ADDR_31_3_addr_31_3_shift)

/*define for ADDR_63_32 word*/
/*define for addr_63_32 field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_ADDR_63_32_addr_63_32_offset 2
#define SDMA_PKT_POLL_MEM_64B_MI4_ADDR_63_32_addr_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_POLL_MEM_64B_MI4_ADDR_63_32_addr_63_32_shift  0
#define SDMA_PKT_POLL_MEM_64B_MI4_ADDR_63_32_ADDR_63_32(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_ADDR_63_32_addr_63_32_mask) << SDMA_PKT_POLL_MEM_64B_MI4_ADDR_63_32_addr_63_32_shift)

/*define for REFERENCE_31_0 word*/
/*define for reference_31_0 field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_REFERENCE_31_0_reference_31_0_offset 3
#define SDMA_PKT_POLL_MEM_64B_MI4_REFERENCE_31_0_reference_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_POLL_MEM_64B_MI4_REFERENCE_31_0_reference_31_0_shift  0
#define SDMA_PKT_POLL_MEM_64B_MI4_REFERENCE_31_0_REFERENCE_31_0(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_REFERENCE_31_0_reference_31_0_mask) << SDMA_PKT_POLL_MEM_64B_MI4_REFERENCE_31_0_reference_31_0_shift)

/*define for REFERENCE_63_32 word*/
/*define for reference_63_32 field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_REFERENCE_63_32_reference_63_32_offset 4
#define SDMA_PKT_POLL_MEM_64B_MI4_REFERENCE_63_32_reference_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_POLL_MEM_64B_MI4_REFERENCE_63_32_reference_63_32_shift  0
#define SDMA_PKT_POLL_MEM_64B_MI4_REFERENCE_63_32_REFERENCE_63_32(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_REFERENCE_63_32_reference_63_32_mask) << SDMA_PKT_POLL_MEM_64B_MI4_REFERENCE_63_32_reference_63_32_shift)

/*define for MASK_31_0 word*/
/*define for mask_31_0 field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_MASK_31_0_mask_31_0_offset 5
#define SDMA_PKT_POLL_MEM_64B_MI4_MASK_31_0_mask_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_POLL_MEM_64B_MI4_MASK_31_0_mask_31_0_shift  0
#define SDMA_PKT_POLL_MEM_64B_MI4_MASK_31_0_MASK_31_0(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_MASK_31_0_mask_31_0_mask) << SDMA_PKT_POLL_MEM_64B_MI4_MASK_31_0_mask_31_0_shift)

/*define for MASK_63_32 word*/
/*define for mask_63_32 field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_MASK_63_32_mask_63_32_offset 6
#define SDMA_PKT_POLL_MEM_64B_MI4_MASK_63_32_mask_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_POLL_MEM_64B_MI4_MASK_63_32_mask_63_32_shift  0
#define SDMA_PKT_POLL_MEM_64B_MI4_MASK_63_32_MASK_63_32(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_MASK_63_32_mask_63_32_mask) << SDMA_PKT_POLL_MEM_64B_MI4_MASK_63_32_mask_63_32_shift)

/*define for DW_7 word*/
/*define for retry_count field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_DW_7_retry_count_offset 7
#define SDMA_PKT_POLL_MEM_64B_MI4_DW_7_retry_count_mask   0x000000FF
#define SDMA_PKT_POLL_MEM_64B_MI4_DW_7_retry_count_shift  16
#define SDMA_PKT_POLL_MEM_64B_MI4_DW_7_RETRY_COUNT(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_DW_7_retry_count_mask) << SDMA_PKT_POLL_MEM_64B_MI4_DW_7_retry_count_shift)

/*define for scope field*/
#define SDMA_PKT_POLL_MEM_64B_MI4_DW_7_scope_offset 7
#define SDMA_PKT_POLL_MEM_64B_MI4_DW_7_scope_mask   0x00000003
#define SDMA_PKT_POLL_MEM_64B_MI4_DW_7_scope_shift  28
#define SDMA_PKT_POLL_MEM_64B_MI4_DW_7_SCOPE(x) (((x) & SDMA_PKT_POLL_MEM_64B_MI4_DW_7_scope_mask) << SDMA_PKT_POLL_MEM_64B_MI4_DW_7_scope_shift)


/*
** Definitions for SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4 packet
** Command: Poll Memory 64b with MES Sync (Navi4x)
** OP=0x8 SUB_OP=0x6
*/

/*define for HEADER word*/
/*define for op_code field*/
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_op_code_offset 0
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_op_code_mask   0x000000FF
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_op_code_shift  0
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_OP_CODE(x) (((x) & SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_op_code_mask) << SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_op_code_shift)

/*define for sub_op_code field*/
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_sub_op_code_offset 0
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_sub_op_code_mask   0x000000FF
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_sub_op_code_shift  8
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_SUB_OP_CODE(x) (((x) & SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_sub_op_code_mask) << SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_sub_op_code_shift)

/*define for sys field*/
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_sys_offset 0
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_sys_mask   0x00000001
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_sys_shift  20
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_SYS(x) (((x) & SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_sys_mask) << SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_sys_shift)

/*define for snp field*/
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_snp_offset 0
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_snp_mask   0x00000001
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_snp_shift  22
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_SNP(x) (((x) & SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_snp_mask) << SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_snp_shift)

/*define for gpa field*/
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_gpa_offset 0
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_gpa_mask   0x00000001
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_gpa_shift  23
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_GPA(x) (((x) & SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_gpa_mask) << SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_gpa_shift)

/*define for cache_policy field*/
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_cache_policy_offset 0
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_cache_policy_mask   0x00000007
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_cache_policy_shift  24
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_CACHE_POLICY(x) (((x) & SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_cache_policy_mask) << SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_cache_policy_shift)

/*define for func field*/
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_func_offset 0
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_func_mask   0x00000007
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_func_shift  28
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_FUNC(x) (((x) & SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_func_mask) << SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_HEADER_func_shift)

/*define for ADDR_31_3 word*/
/*define for addr_31_3 field*/
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_ADDR_31_3_addr_31_3_offset 1
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_ADDR_31_3_addr_31_3_mask   0x3FFFFFFF
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_ADDR_31_3_addr_31_3_shift  2
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_ADDR_31_3_ADDR_31_3(x) (((x) & SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_ADDR_31_3_addr_31_3_mask) << SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_ADDR_31_3_addr_31_3_shift)

/*define for ADDR_63_32 word*/
/*define for addr_63_32 field*/
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_ADDR_63_32_addr_63_32_offset 2
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_ADDR_63_32_addr_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_ADDR_63_32_addr_63_32_shift  0
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_ADDR_63_32_ADDR_63_32(x) (((x) & SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_ADDR_63_32_addr_63_32_mask) << SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_ADDR_63_32_addr_63_32_shift)

/*define for REFERENCE_31_0 word*/
/*define for reference_31_0 field*/
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_REFERENCE_31_0_reference_31_0_offset 3
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_REFERENCE_31_0_reference_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_REFERENCE_31_0_reference_31_0_shift  0
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_REFERENCE_31_0_REFERENCE_31_0(x) (((x) & SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_REFERENCE_31_0_reference_31_0_mask) << SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_REFERENCE_31_0_reference_31_0_shift)

/*define for REFERENCE_63_32 word*/
/*define for reference_63_32 field*/
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_REFERENCE_63_32_reference_63_32_offset 4
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_REFERENCE_63_32_reference_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_REFERENCE_63_32_reference_63_32_shift  0
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_REFERENCE_63_32_REFERENCE_63_32(x) (((x) & SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_REFERENCE_63_32_reference_63_32_mask) << SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_REFERENCE_63_32_reference_63_32_shift)

/*define for MASK_31_0 word*/
/*define for mask_31_0 field*/
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_MASK_31_0_mask_31_0_offset 5
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_MASK_31_0_mask_31_0_mask   0xFFFFFFFF
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_MASK_31_0_mask_31_0_shift  0
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_MASK_31_0_MASK_31_0(x) (((x) & SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_MASK_31_0_mask_31_0_mask) << SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_MASK_31_0_mask_31_0_shift)

/*define for MASK_63_32 word*/
/*define for mask_63_32 field*/
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_MASK_63_32_mask_63_32_offset 6
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_MASK_63_32_mask_63_32_mask   0xFFFFFFFF
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_MASK_63_32_mask_63_32_shift  0
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_MASK_63_32_MASK_63_32(x) (((x) & SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_MASK_63_32_mask_63_32_mask) << SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_MASK_63_32_mask_63_32_shift)

/*define for RETRY_COUNT word*/
/*define for retry_count field*/
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_RETRY_COUNT_retry_count_offset 7
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_RETRY_COUNT_retry_count_mask   0x000000FF
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_RETRY_COUNT_retry_count_shift  16
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_RETRY_COUNT_RETRY_COUNT(x) (((x) & SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_RETRY_COUNT_retry_count_mask) << SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_RETRY_COUNT_retry_count_shift)

/*define for FENCE_HANDLE word*/
/*define for fence_handle field*/
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_FENCE_HANDLE_fence_handle_offset 8
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_FENCE_HANDLE_fence_handle_mask   0xFFFFFFFF
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_FENCE_HANDLE_fence_handle_shift  0
#define SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_FENCE_HANDLE_FENCE_HANDLE(x) (((x) & SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_FENCE_HANDLE_fence_handle_mask) << SDMA_PKT_POLL_MEM_64B_MES_SYNC_NAVI4_FENCE_HANDLE_fence_handle_shift)


/*
** Definitions for SDMA_PKT_EXCL_COND_EXEC packet
** Command: Exclusive Conditional Execution
** OP=0x9 SUB_OP=0x1
*/

/*define for HEADER word*/
/*define for op_code field*/
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_op_code_offset 0
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_op_code_mask   0x000000FF
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_op_code_shift  0
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_OP_CODE(x) (((x) & SDMA_PKT_EXCL_COND_EXEC_HEADER_op_code_mask) << SDMA_PKT_EXCL_COND_EXEC_HEADER_op_code_shift)

/*define for sub_op_code field*/
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_sub_op_code_offset 0
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_sub_op_code_mask   0x000000FF
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_sub_op_code_shift  8
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_SUB_OP_CODE(x) (((x) & SDMA_PKT_EXCL_COND_EXEC_HEADER_sub_op_code_mask) << SDMA_PKT_EXCL_COND_EXEC_HEADER_sub_op_code_shift)

/*define for p field*/
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_p_offset 0
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_p_mask   0x00000001
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_p_shift  30
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_P(x) (((x) & SDMA_PKT_EXCL_COND_EXEC_HEADER_p_mask) << SDMA_PKT_EXCL_COND_EXEC_HEADER_p_shift)

/*define for o field*/
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_o_offset 0
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_o_mask   0x00000001
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_o_shift  31
#define SDMA_PKT_EXCL_COND_EXEC_HEADER_O(x) (((x) & SDMA_PKT_EXCL_COND_EXEC_HEADER_o_mask) << SDMA_PKT_EXCL_COND_EXEC_HEADER_o_shift)

/*define for MUTEX_ADDR_LO word*/
/*define for mutex_addr_lo field*/
#define SDMA_PKT_EXCL_COND_EXEC_MUTEX_ADDR_LO_mutex_addr_lo_offset 1
#define SDMA_PKT_EXCL_COND_EXEC_MUTEX_ADDR_LO_mutex_addr_lo_mask   0xFFFFFFFF
#define SDMA_PKT_EXCL_COND_EXEC_MUTEX_ADDR_LO_mutex_addr_lo_shift  0
#define SDMA_PKT_EXCL_COND_EXEC_MUTEX_ADDR_LO_MUTEX_ADDR_LO(x) (((x) & SDMA_PKT_EXCL_COND_EXEC_MUTEX_ADDR_LO_mutex_addr_lo_mask) << SDMA_PKT_EXCL_COND_EXEC_MUTEX_ADDR_LO_mutex_addr_lo_shift)

/*define for MUTEX_ADDR_HI word*/
/*define for mutex_addr_hi field*/
#define SDMA_PKT_EXCL_COND_EXEC_MUTEX_ADDR_HI_mutex_addr_hi_offset 2
#define SDMA_PKT_EXCL_COND_EXEC_MUTEX_ADDR_HI_mutex_addr_hi_mask   0xFFFFFFFF
#define SDMA_PKT_EXCL_COND_EXEC_MUTEX_ADDR_HI_mutex_addr_hi_shift  0
#define SDMA_PKT_EXCL_COND_EXEC_MUTEX_ADDR_HI_MUTEX_ADDR_HI(x) (((x) & SDMA_PKT_EXCL_COND_EXEC_MUTEX_ADDR_HI_mutex_addr_hi_mask) << SDMA_PKT_EXCL_COND_EXEC_MUTEX_ADDR_HI_mutex_addr_hi_shift)

/*define for ADDR_LO word*/
/*define for addr_lo field*/
#define SDMA_PKT_EXCL_COND_EXEC_ADDR_LO_addr_lo_offset 3
#define SDMA_PKT_EXCL_COND_EXEC_ADDR_LO_addr_lo_mask   0xFFFFFFFF
#define SDMA_PKT_EXCL_COND_EXEC_ADDR_LO_addr_lo_shift  0
#define SDMA_PKT_EXCL_COND_EXEC_ADDR_LO_ADDR_LO(x) (((x) & SDMA_PKT_EXCL_COND_EXEC_ADDR_LO_addr_lo_mask) << SDMA_PKT_EXCL_COND_EXEC_ADDR_LO_addr_lo_shift)

/*define for ADDR_HI word*/
/*define for addr_hi field*/
#define SDMA_PKT_EXCL_COND_EXEC_ADDR_HI_addr_hi_offset 4
#define SDMA_PKT_EXCL_COND_EXEC_ADDR_HI_addr_hi_mask   0xFFFFFFFF
#define SDMA_PKT_EXCL_COND_EXEC_ADDR_HI_addr_hi_shift  0
#define SDMA_PKT_EXCL_COND_EXEC_ADDR_HI_ADDR_HI(x) (((x) & SDMA_PKT_EXCL_COND_EXEC_ADDR_HI_addr_hi_mask) << SDMA_PKT_EXCL_COND_EXEC_ADDR_HI_addr_hi_shift)

/*define for REF_LO word*/
/*define for ref_lo field*/
#define SDMA_PKT_EXCL_COND_EXEC_REF_LO_ref_lo_offset 5
#define SDMA_PKT_EXCL_COND_EXEC_REF_LO_ref_lo_mask   0xFFFFFFFF
#define SDMA_PKT_EXCL_COND_EXEC_REF_LO_ref_lo_shift  0
#define SDMA_PKT_EXCL_COND_EXEC_REF_LO_REF_LO(x) (((x) & SDMA_PKT_EXCL_COND_EXEC_REF_LO_ref_lo_mask) << SDMA_PKT_EXCL_COND_EXEC_REF_LO_ref_lo_shift)

/*define for REF_HI word*/
/*define for ref_hi field*/
#define SDMA_PKT_EXCL_COND_EXEC_REF_HI_ref_hi_offset 6
#define SDMA_PKT_EXCL_COND_EXEC_REF_HI_ref_hi_mask   0xFFFFFFFF
#define SDMA_PKT_EXCL_COND_EXEC_REF_HI_ref_hi_shift  0
#define SDMA_PKT_EXCL_COND_EXEC_REF_HI_REF_HI(x) (((x) & SDMA_PKT_EXCL_COND_EXEC_REF_HI_ref_hi_mask) << SDMA_PKT_EXCL_COND_EXEC_REF_HI_ref_hi_shift)

/*define for EXEC_COUNT word*/
/*define for exec_count field*/
#define SDMA_PKT_EXCL_COND_EXEC_EXEC_COUNT_exec_count_offset 7
#define SDMA_PKT_EXCL_COND_EXEC_EXEC_COUNT_exec_count_mask   0x00003FFF
#define SDMA_PKT_EXCL_COND_EXEC_EXEC_COUNT_exec_count_shift  0
#define SDMA_PKT_EXCL_COND_EXEC_EXEC_COUNT_EXEC_COUNT(x) (((x) & SDMA_PKT_EXCL_COND_EXEC_EXEC_COUNT_exec_count_mask) << SDMA_PKT_EXCL_COND_EXEC_EXEC_COUNT_exec_count_shift)


/*
** Definitions for SDMA_PKT_PTEPDE_GEN_MI4 packet
** Command: Generate PTE/PDE (MI4xx)
** OP=0xc SUB_OP=0x0
*/

/*define for HEADER word*/
/*define for op_code field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_op_code_offset 0
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_op_code_mask   0x000000FF
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_op_code_shift  0
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_OP_CODE(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_HEADER_op_code_mask) << SDMA_PKT_PTEPDE_GEN_MI4_HEADER_op_code_shift)

/*define for sub_op_code field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_sub_op_code_offset 0
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_sub_op_code_mask   0x000000FF
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_sub_op_code_shift  8
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_SUB_OP_CODE(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_HEADER_sub_op_code_mask) << SDMA_PKT_PTEPDE_GEN_MI4_HEADER_sub_op_code_shift)

/*define for mtype field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_mtype_offset 0
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_mtype_mask   0x00000003
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_mtype_shift  16
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_MTYPE(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_HEADER_mtype_mask) << SDMA_PKT_PTEPDE_GEN_MI4_HEADER_mtype_shift)

/*define for sys field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_sys_offset 0
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_sys_mask   0x00000001
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_sys_shift  20
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_SYS(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_HEADER_sys_mask) << SDMA_PKT_PTEPDE_GEN_MI4_HEADER_sys_shift)

/*define for snp field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_snp_offset 0
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_snp_mask   0x00000001
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_snp_shift  22
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_SNP(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_HEADER_snp_mask) << SDMA_PKT_PTEPDE_GEN_MI4_HEADER_snp_shift)

/*define for gpa field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_gpa_offset 0
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_gpa_mask   0x00000001
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_gpa_shift  23
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_GPA(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_HEADER_gpa_mask) << SDMA_PKT_PTEPDE_GEN_MI4_HEADER_gpa_shift)

/*define for scope field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_scope_offset 0
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_scope_mask   0x00000003
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_scope_shift  24
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_SCOPE(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_HEADER_scope_mask) << SDMA_PKT_PTEPDE_GEN_MI4_HEADER_scope_shift)

/*define for temporal_hint field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_temporal_hint_offset 0
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_temporal_hint_mask   0x00000007
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_temporal_hint_shift  26
#define SDMA_PKT_PTEPDE_GEN_MI4_HEADER_TEMPORAL_HINT(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_HEADER_temporal_hint_mask) << SDMA_PKT_PTEPDE_GEN_MI4_HEADER_temporal_hint_shift)

/*define for ADDR_LO word*/
/*define for addr_lo field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_ADDR_LO_addr_lo_offset 1
#define SDMA_PKT_PTEPDE_GEN_MI4_ADDR_LO_addr_lo_mask   0xFFFFFFFF
#define SDMA_PKT_PTEPDE_GEN_MI4_ADDR_LO_addr_lo_shift  0
#define SDMA_PKT_PTEPDE_GEN_MI4_ADDR_LO_ADDR_LO(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_ADDR_LO_addr_lo_mask) << SDMA_PKT_PTEPDE_GEN_MI4_ADDR_LO_addr_lo_shift)

/*define for ADDR_HI word*/
/*define for addr_hi field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_ADDR_HI_addr_hi_offset 2
#define SDMA_PKT_PTEPDE_GEN_MI4_ADDR_HI_addr_hi_mask   0xFFFFFFFF
#define SDMA_PKT_PTEPDE_GEN_MI4_ADDR_HI_addr_hi_shift  0
#define SDMA_PKT_PTEPDE_GEN_MI4_ADDR_HI_ADDR_HI(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_ADDR_HI_addr_hi_mask) << SDMA_PKT_PTEPDE_GEN_MI4_ADDR_HI_addr_hi_shift)

/*define for MASK_LO word*/
/*define for mask_lo field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_MASK_LO_mask_lo_offset 3
#define SDMA_PKT_PTEPDE_GEN_MI4_MASK_LO_mask_lo_mask   0xFFFFFFFF
#define SDMA_PKT_PTEPDE_GEN_MI4_MASK_LO_mask_lo_shift  0
#define SDMA_PKT_PTEPDE_GEN_MI4_MASK_LO_MASK_LO(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_MASK_LO_mask_lo_mask) << SDMA_PKT_PTEPDE_GEN_MI4_MASK_LO_mask_lo_shift)

/*define for MASK_HI word*/
/*define for mask_hi field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_MASK_HI_mask_hi_offset 4
#define SDMA_PKT_PTEPDE_GEN_MI4_MASK_HI_mask_hi_mask   0xFFFFFFFF
#define SDMA_PKT_PTEPDE_GEN_MI4_MASK_HI_mask_hi_shift  0
#define SDMA_PKT_PTEPDE_GEN_MI4_MASK_HI_MASK_HI(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_MASK_HI_mask_hi_mask) << SDMA_PKT_PTEPDE_GEN_MI4_MASK_HI_mask_hi_shift)

/*define for INIT_VALUE_LO word*/
/*define for init_value_lo field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_INIT_VALUE_LO_init_value_lo_offset 5
#define SDMA_PKT_PTEPDE_GEN_MI4_INIT_VALUE_LO_init_value_lo_mask   0xFFFFFFFF
#define SDMA_PKT_PTEPDE_GEN_MI4_INIT_VALUE_LO_init_value_lo_shift  0
#define SDMA_PKT_PTEPDE_GEN_MI4_INIT_VALUE_LO_INIT_VALUE_LO(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_INIT_VALUE_LO_init_value_lo_mask) << SDMA_PKT_PTEPDE_GEN_MI4_INIT_VALUE_LO_init_value_lo_shift)

/*define for INIT_VALUE_HI word*/
/*define for init_value_hi field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_INIT_VALUE_HI_init_value_hi_offset 6
#define SDMA_PKT_PTEPDE_GEN_MI4_INIT_VALUE_HI_init_value_hi_mask   0xFFFFFFFF
#define SDMA_PKT_PTEPDE_GEN_MI4_INIT_VALUE_HI_init_value_hi_shift  0
#define SDMA_PKT_PTEPDE_GEN_MI4_INIT_VALUE_HI_INIT_VALUE_HI(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_INIT_VALUE_HI_init_value_hi_mask) << SDMA_PKT_PTEPDE_GEN_MI4_INIT_VALUE_HI_init_value_hi_shift)

/*define for INCREMENT_LO word*/
/*define for increment_lo field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_INCREMENT_LO_increment_lo_offset 7
#define SDMA_PKT_PTEPDE_GEN_MI4_INCREMENT_LO_increment_lo_mask   0xFFFFFFFF
#define SDMA_PKT_PTEPDE_GEN_MI4_INCREMENT_LO_increment_lo_shift  0
#define SDMA_PKT_PTEPDE_GEN_MI4_INCREMENT_LO_INCREMENT_LO(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_INCREMENT_LO_increment_lo_mask) << SDMA_PKT_PTEPDE_GEN_MI4_INCREMENT_LO_increment_lo_shift)

/*define for INCREMENT_HI word*/
/*define for increment_hi field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_INCREMENT_HI_increment_hi_offset 8
#define SDMA_PKT_PTEPDE_GEN_MI4_INCREMENT_HI_increment_hi_mask   0xFFFFFFFF
#define SDMA_PKT_PTEPDE_GEN_MI4_INCREMENT_HI_increment_hi_shift  0
#define SDMA_PKT_PTEPDE_GEN_MI4_INCREMENT_HI_INCREMENT_HI(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_INCREMENT_HI_increment_hi_mask) << SDMA_PKT_PTEPDE_GEN_MI4_INCREMENT_HI_increment_hi_shift)

/*define for NUMPTE word*/
/*define for numpte field*/
#define SDMA_PKT_PTEPDE_GEN_MI4_NUMPTE_numpte_offset 9
#define SDMA_PKT_PTEPDE_GEN_MI4_NUMPTE_numpte_mask   0x0007FFFF
#define SDMA_PKT_PTEPDE_GEN_MI4_NUMPTE_numpte_shift  0
#define SDMA_PKT_PTEPDE_GEN_MI4_NUMPTE_NUMPTE(x) (((x) & SDMA_PKT_PTEPDE_GEN_MI4_NUMPTE_numpte_mask) << SDMA_PKT_PTEPDE_GEN_MI4_NUMPTE_numpte_shift)


/*
** Definitions for SDMA_PKT_PTEPDE_RMW_MI4 packet
** Command: RMW PTE/PDE (MI4xx)
** OP=0xc SUB_OP=0x2
*/

/*define for HEADER word*/
/*define for op_code field*/
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_op_code_offset 0
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_op_code_mask   0x000000FF
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_op_code_shift  0
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_OP_CODE(x) (((x) & SDMA_PKT_PTEPDE_RMW_MI4_HEADER_op_code_mask) << SDMA_PKT_PTEPDE_RMW_MI4_HEADER_op_code_shift)

/*define for sub_op_code field*/
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_sub_op_code_offset 0
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_sub_op_code_mask   0x000000FF
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_sub_op_code_shift  8
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_SUB_OP_CODE(x) (((x) & SDMA_PKT_PTEPDE_RMW_MI4_HEADER_sub_op_code_mask) << SDMA_PKT_PTEPDE_RMW_MI4_HEADER_sub_op_code_shift)

/*define for mtype field*/
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_mtype_offset 0
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_mtype_mask   0x00000003
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_mtype_shift  16
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_MTYPE(x) (((x) & SDMA_PKT_PTEPDE_RMW_MI4_HEADER_mtype_mask) << SDMA_PKT_PTEPDE_RMW_MI4_HEADER_mtype_shift)

/*define for sys field*/
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_sys_offset 0
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_sys_mask   0x00000001
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_sys_shift  20
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_SYS(x) (((x) & SDMA_PKT_PTEPDE_RMW_MI4_HEADER_sys_mask) << SDMA_PKT_PTEPDE_RMW_MI4_HEADER_sys_shift)

/*define for snp field*/
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_snp_offset 0
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_snp_mask   0x00000001
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_snp_shift  22
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_SNP(x) (((x) & SDMA_PKT_PTEPDE_RMW_MI4_HEADER_snp_mask) << SDMA_PKT_PTEPDE_RMW_MI4_HEADER_snp_shift)

/*define for gpa field*/
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_gpa_offset 0
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_gpa_mask   0x00000001
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_gpa_shift  23
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_GPA(x) (((x) & SDMA_PKT_PTEPDE_RMW_MI4_HEADER_gpa_mask) << SDMA_PKT_PTEPDE_RMW_MI4_HEADER_gpa_shift)

/*define for scope field*/
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_scope_offset 0
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_scope_mask   0x00000003
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_scope_shift  24
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_SCOPE(x) (((x) & SDMA_PKT_PTEPDE_RMW_MI4_HEADER_scope_mask) << SDMA_PKT_PTEPDE_RMW_MI4_HEADER_scope_shift)

/*define for temporal_hint field*/
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_temporal_hint_offset 0
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_temporal_hint_mask   0x00000003
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_temporal_hint_shift  26
#define SDMA_PKT_PTEPDE_RMW_MI4_HEADER_TEMPORAL_HINT(x) (((x) & SDMA_PKT_PTEPDE_RMW_MI4_HEADER_temporal_hint_mask) << SDMA_PKT_PTEPDE_RMW_MI4_HEADER_temporal_hint_shift)

/*define for ADDR_LO word*/
/*define for addr_lo field*/
#define SDMA_PKT_PTEPDE_RMW_MI4_ADDR_LO_addr_lo_offset 1
#define SDMA_PKT_PTEPDE_RMW_MI4_ADDR_LO_addr_lo_mask   0xFFFFFFFF
#define SDMA_PKT_PTEPDE_RMW_MI4_ADDR_LO_addr_lo_shift  0
#define SDMA_PKT_PTEPDE_RMW_MI4_ADDR_LO_ADDR_LO(x) (((x) & SDMA_PKT_PTEPDE_RMW_MI4_ADDR_LO_addr_lo_mask) << SDMA_PKT_PTEPDE_RMW_MI4_ADDR_LO_addr_lo_shift)

/*define for ADDR_HI word*/
/*define for addr_hi field*/
#define SDMA_PKT_PTEPDE_RMW_MI4_ADDR_HI_addr_hi_offset 2
#define SDMA_PKT_PTEPDE_RMW_MI4_ADDR_HI_addr_hi_mask   0xFFFFFFFF
#define SDMA_PKT_PTEPDE_RMW_MI4_ADDR_HI_addr_hi_shift  0
#define SDMA_PKT_PTEPDE_RMW_MI4_ADDR_HI_ADDR_HI(x) (((x) & SDMA_PKT_PTEPDE_RMW_MI4_ADDR_HI_addr_hi_mask) << SDMA_PKT_PTEPDE_RMW_MI4_ADDR_HI_addr_hi_shift)

/*define for MASK_LO word*/
/*define for mask_lo field*/
#define SDMA_PKT_PTEPDE_RMW_MI4_MASK_LO_mask_lo_offset 3
#define SDMA_PKT_PTEPDE_RMW_MI4_MASK_LO_mask_lo_mask   0xFFFFFFFF
#define SDMA_PKT_PTEPDE_RMW_MI4_MASK_LO_mask_lo_shift  0
#define SDMA_PKT_PTEPDE_RMW_MI4_MASK_LO_MASK_LO(x) (((x) & SDMA_PKT_PTEPDE_RMW_MI4_MASK_LO_mask_lo_mask) << SDMA_PKT_PTEPDE_RMW_MI4_MASK_LO_mask_lo_shift)

/*define for MASK_HI word*/
/*define for mask_hi field*/
#define SDMA_PKT_PTEPDE_RMW_MI4_MASK_HI_mask_hi_offset 4
#define SDMA_PKT_PTEPDE_RMW_MI4_MASK_HI_mask_hi_mask   0xFFFFFFFF
#define SDMA_PKT_PTEPDE_RMW_MI4_MASK_HI_mask_hi_shift  0
#define SDMA_PKT_PTEPDE_RMW_MI4_MASK_HI_MASK_HI(x) (((x) & SDMA_PKT_PTEPDE_RMW_MI4_MASK_HI_mask_hi_mask) << SDMA_PKT_PTEPDE_RMW_MI4_MASK_HI_mask_hi_shift)

/*define for VALUE_LO word*/
/*define for value_lo field*/
#define SDMA_PKT_PTEPDE_RMW_MI4_VALUE_LO_value_lo_offset 5
#define SDMA_PKT_PTEPDE_RMW_MI4_VALUE_LO_value_lo_mask   0xFFFFFFFF
#define SDMA_PKT_PTEPDE_RMW_MI4_VALUE_LO_value_lo_shift  0
#define SDMA_PKT_PTEPDE_RMW_MI4_VALUE_LO_VALUE_LO(x) (((x) & SDMA_PKT_PTEPDE_RMW_MI4_VALUE_LO_value_lo_mask) << SDMA_PKT_PTEPDE_RMW_MI4_VALUE_LO_value_lo_shift)

/*define for VALUE_HI word*/
/*define for value_hi field*/
#define SDMA_PKT_PTEPDE_RMW_MI4_VALUE_HI_value_hi_offset 6
#define SDMA_PKT_PTEPDE_RMW_MI4_VALUE_HI_value_hi_mask   0xFFFFFFFF
#define SDMA_PKT_PTEPDE_RMW_MI4_VALUE_HI_value_hi_shift  0
#define SDMA_PKT_PTEPDE_RMW_MI4_VALUE_HI_VALUE_HI(x) (((x) & SDMA_PKT_PTEPDE_RMW_MI4_VALUE_HI_value_hi_mask) << SDMA_PKT_PTEPDE_RMW_MI4_VALUE_HI_value_hi_shift)

/*define for NUMPTE word*/
/*define for numpte field*/
#define SDMA_PKT_PTEPDE_RMW_MI4_NUMPTE_numpte_offset 7
#define SDMA_PKT_PTEPDE_RMW_MI4_NUMPTE_numpte_mask   0x0007FFFF
#define SDMA_PKT_PTEPDE_RMW_MI4_NUMPTE_numpte_shift  0
#define SDMA_PKT_PTEPDE_RMW_MI4_NUMPTE_NUMPTE(x) (((x) & SDMA_PKT_PTEPDE_RMW_MI4_NUMPTE_numpte_mask) << SDMA_PKT_PTEPDE_RMW_MI4_NUMPTE_numpte_shift)


/*
** Definitions for SDMA_PKT_CONSTANT_FILL_MI4 packet
** Command: Constant Fill (MI4xx)
** OP=0xb SUB_OP=0x0
*/

/*define for HEADER word*/
/*define for op_code field*/
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_op_code_offset 0
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_op_code_mask   0x000000FF
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_op_code_shift  0
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_OP_CODE(x) (((x) & SDMA_PKT_CONSTANT_FILL_MI4_HEADER_op_code_mask) << SDMA_PKT_CONSTANT_FILL_MI4_HEADER_op_code_shift)

/*define for sub_op_code field*/
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_sub_op_code_offset 0
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_sub_op_code_mask   0x000000FF
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_sub_op_code_shift  8
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_SUB_OP_CODE(x) (((x) & SDMA_PKT_CONSTANT_FILL_MI4_HEADER_sub_op_code_mask) << SDMA_PKT_CONSTANT_FILL_MI4_HEADER_sub_op_code_shift)

/*define for mtype field*/
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_mtype_offset 0
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_mtype_mask   0x00000003
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_mtype_shift  16
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_MTYPE(x) (((x) & SDMA_PKT_CONSTANT_FILL_MI4_HEADER_mtype_mask) << SDMA_PKT_CONSTANT_FILL_MI4_HEADER_mtype_shift)

/*define for sys field*/
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_sys_offset 0
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_sys_mask   0x00000001
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_sys_shift  20
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_SYS(x) (((x) & SDMA_PKT_CONSTANT_FILL_MI4_HEADER_sys_mask) << SDMA_PKT_CONSTANT_FILL_MI4_HEADER_sys_shift)

/*define for snp field*/
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_snp_offset 0
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_snp_mask   0x00000001
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_snp_shift  22
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_SNP(x) (((x) & SDMA_PKT_CONSTANT_FILL_MI4_HEADER_snp_mask) << SDMA_PKT_CONSTANT_FILL_MI4_HEADER_snp_shift)

/*define for gpa field*/
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_gpa_offset 0
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_gpa_mask   0x00000001
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_gpa_shift  23
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_GPA(x) (((x) & SDMA_PKT_CONSTANT_FILL_MI4_HEADER_gpa_mask) << SDMA_PKT_CONSTANT_FILL_MI4_HEADER_gpa_shift)

/*define for scope field*/
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_scope_offset 0
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_scope_mask   0x00000003
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_scope_shift  24
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_SCOPE(x) (((x) & SDMA_PKT_CONSTANT_FILL_MI4_HEADER_scope_mask) << SDMA_PKT_CONSTANT_FILL_MI4_HEADER_scope_shift)

/*define for temporal_hint field*/
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_temporal_hint_offset 0
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_temporal_hint_mask   0x00000007
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_temporal_hint_shift  26
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_TEMPORAL_HINT(x) (((x) & SDMA_PKT_CONSTANT_FILL_MI4_HEADER_temporal_hint_mask) << SDMA_PKT_CONSTANT_FILL_MI4_HEADER_temporal_hint_shift)

/*define for npd field*/
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_npd_offset 0
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_npd_mask   0x00000001
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_npd_shift  29
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_NPD(x) (((x) & SDMA_PKT_CONSTANT_FILL_MI4_HEADER_npd_mask) << SDMA_PKT_CONSTANT_FILL_MI4_HEADER_npd_shift)

/*define for size field*/
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_size_offset 0
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_size_mask   0x00000003
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_size_shift  30
#define SDMA_PKT_CONSTANT_FILL_MI4_HEADER_SIZE(x) (((x) & SDMA_PKT_CONSTANT_FILL_MI4_HEADER_size_mask) << SDMA_PKT_CONSTANT_FILL_MI4_HEADER_size_shift)

/*define for ADDR_LO word*/
/*define for addr_lo field*/
#define SDMA_PKT_CONSTANT_FILL_MI4_ADDR_LO_addr_lo_offset 1
#define SDMA_PKT_CONSTANT_FILL_MI4_ADDR_LO_addr_lo_mask   0xFFFFFFFF
#define SDMA_PKT_CONSTANT_FILL_MI4_ADDR_LO_addr_lo_shift  0
#define SDMA_PKT_CONSTANT_FILL_MI4_ADDR_LO_ADDR_LO(x) (((x) & SDMA_PKT_CONSTANT_FILL_MI4_ADDR_LO_addr_lo_mask) << SDMA_PKT_CONSTANT_FILL_MI4_ADDR_LO_addr_lo_shift)

/*define for ADDR_HI word*/
/*define for addr_hi field*/
#define SDMA_PKT_CONSTANT_FILL_MI4_ADDR_HI_addr_hi_offset 2
#define SDMA_PKT_CONSTANT_FILL_MI4_ADDR_HI_addr_hi_mask   0xFFFFFFFF
#define SDMA_PKT_CONSTANT_FILL_MI4_ADDR_HI_addr_hi_shift  0
#define SDMA_PKT_CONSTANT_FILL_MI4_ADDR_HI_ADDR_HI(x) (((x) & SDMA_PKT_CONSTANT_FILL_MI4_ADDR_HI_addr_hi_mask) << SDMA_PKT_CONSTANT_FILL_MI4_ADDR_HI_addr_hi_shift)

/*define for SOURCE_DATA word*/
/*define for source_data field*/
#define SDMA_PKT_CONSTANT_FILL_MI4_SOURCE_DATA_source_data_offset 3
#define SDMA_PKT_CONSTANT_FILL_MI4_SOURCE_DATA_source_data_mask   0xFFFFFFFF
#define SDMA_PKT_CONSTANT_FILL_MI4_SOURCE_DATA_source_data_shift  0
#define SDMA_PKT_CONSTANT_FILL_MI4_SOURCE_DATA_SOURCE_DATA(x) (((x) & SDMA_PKT_CONSTANT_FILL_MI4_SOURCE_DATA_source_data_mask) << SDMA_PKT_CONSTANT_FILL_MI4_SOURCE_DATA_source_data_shift)

/*define for COUNT word*/
/*define for count field*/
#define SDMA_PKT_CONSTANT_FILL_MI4_COUNT_count_offset 4
#define SDMA_PKT_CONSTANT_FILL_MI4_COUNT_count_mask   0x3FFFFFFF
#define SDMA_PKT_CONSTANT_FILL_MI4_COUNT_count_shift  0
#define SDMA_PKT_CONSTANT_FILL_MI4_COUNT_COUNT(x) (((x) & SDMA_PKT_CONSTANT_FILL_MI4_COUNT_count_mask) << SDMA_PKT_CONSTANT_FILL_MI4_COUNT_count_shift)


/*
** Definitions for SDMA_PKT_CONSTANT_FILL_PAGE_MI4 packet
** Command: Constant Fill Page (MI4xx)
** OP=0xb SUB_OP=0x4
*/

/*define for HEADER word*/
/*define for op_code field*/
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_op_code_offset 0
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_op_code_mask   0x000000FF
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_op_code_shift  0
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_OP_CODE(x) (((x) & SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_op_code_mask) << SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_op_code_shift)

/*define for sub_op_code field*/
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_sub_op_code_offset 0
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_sub_op_code_mask   0x000000FF
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_sub_op_code_shift  8
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_SUB_OP_CODE(x) (((x) & SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_sub_op_code_mask) << SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_sub_op_code_shift)

/*define for mtype field*/
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_mtype_offset 0
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_mtype_mask   0x00000003
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_mtype_shift  16
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_MTYPE(x) (((x) & SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_mtype_mask) << SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_mtype_shift)

/*define for sys field*/
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_sys_offset 0
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_sys_mask   0x00000001
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_sys_shift  20
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_SYS(x) (((x) & SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_sys_mask) << SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_sys_shift)

/*define for snp field*/
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_snp_offset 0
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_snp_mask   0x00000001
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_snp_shift  22
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_SNP(x) (((x) & SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_snp_mask) << SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_snp_shift)

/*define for gpa field*/
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_gpa_offset 0
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_gpa_mask   0x00000001
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_gpa_shift  23
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_GPA(x) (((x) & SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_gpa_mask) << SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_gpa_shift)

/*define for scope field*/
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_scope_offset 0
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_scope_mask   0x00000003
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_scope_shift  24
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_SCOPE(x) (((x) & SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_scope_mask) << SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_scope_shift)

/*define for temporal_hint field*/
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_temporal_hint_offset 0
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_temporal_hint_mask   0x00000007
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_temporal_hint_shift  26
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_TEMPORAL_HINT(x) (((x) & SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_temporal_hint_mask) << SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_temporal_hint_shift)

/*define for size field*/
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_size_offset 0
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_size_mask   0x00000003
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_size_shift  30
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_SIZE(x) (((x) & SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_size_mask) << SDMA_PKT_CONSTANT_FILL_PAGE_MI4_HEADER_size_shift)

/*define for SOURCE_DATA word*/
/*define for source_data field*/
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_SOURCE_DATA_source_data_offset 1
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_SOURCE_DATA_source_data_mask   0x00000003
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_SOURCE_DATA_source_data_shift  30
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_SOURCE_DATA_SOURCE_DATA(x) (((x) & SDMA_PKT_CONSTANT_FILL_PAGE_MI4_SOURCE_DATA_source_data_mask) << SDMA_PKT_CONSTANT_FILL_PAGE_MI4_SOURCE_DATA_source_data_shift)

/*define for DW_2 word*/
/*define for page_unit field*/
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_page_unit_offset 2
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_page_unit_mask   0x0000000F
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_page_unit_shift  0
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_PAGE_UNIT(x) (((x) & SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_page_unit_mask) << SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_page_unit_shift)

/*define for page_number field*/
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_page_number_offset 2
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_page_number_mask   0x0000FFFF
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_page_number_shift  16
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_PAGE_NUMBER(x) (((x) & SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_page_number_mask) << SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_page_number_shift)

/*define for addr_i field*/
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_addr_i_offset 2
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_addr_i_mask   0xFFFFFFFF
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_addr_i_shift  0
#define SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_ADDR_I(x) (((x) & SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_addr_i_mask) << SDMA_PKT_CONSTANT_FILL_PAGE_MI4_DW_2_addr_i_shift)


/*
** Definitions for SDMA_PKT_AQL packet
** Command: AQL
*/

/*define for HEADER word*/
/*define for format field*/
#define SDMA_PKT_AQL_HEADER_format_offset 0
#define SDMA_PKT_AQL_HEADER_format_mask   0x000000FF
#define SDMA_PKT_AQL_HEADER_format_shift  0
#define SDMA_PKT_AQL_HEADER_FORMAT(x) (((x) & SDMA_PKT_AQL_HEADER_format_mask) << SDMA_PKT_AQL_HEADER_format_shift)

/*define for acquire_fence_scope field*/
#define SDMA_PKT_AQL_HEADER_acquire_fence_scope_offset 0
#define SDMA_PKT_AQL_HEADER_acquire_fence_scope_mask   0x00000003
#define SDMA_PKT_AQL_HEADER_acquire_fence_scope_shift  9
#define SDMA_PKT_AQL_HEADER_ACQUIRE_FENCE_SCOPE(x) (((x) & SDMA_PKT_AQL_HEADER_acquire_fence_scope_mask) << SDMA_PKT_AQL_HEADER_acquire_fence_scope_shift)

/*define for release_fence_scope field*/
#define SDMA_PKT_AQL_HEADER_release_fence_scope_offset 0
#define SDMA_PKT_AQL_HEADER_release_fence_scope_mask   0x00000003
#define SDMA_PKT_AQL_HEADER_release_fence_scope_shift  11
#define SDMA_PKT_AQL_HEADER_RELEASE_FENCE_SCOPE(x) (((x) & SDMA_PKT_AQL_HEADER_release_fence_scope_mask) << SDMA_PKT_AQL_HEADER_release_fence_scope_shift)

/*define for reserved field*/
#define SDMA_PKT_AQL_HEADER_reserved_offset 0
#define SDMA_PKT_AQL_HEADER_reserved_mask   0x00000007
#define SDMA_PKT_AQL_HEADER_reserved_shift  13
#define SDMA_PKT_AQL_HEADER_RESERVED(x) (((x) & SDMA_PKT_AQL_HEADER_reserved_mask) << SDMA_PKT_AQL_HEADER_reserved_shift)

// clang-format on
#endif /* __OSS70_SDMA_PKT_OPEN_H_ */

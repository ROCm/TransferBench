/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Pre-OSS7 (CDNA3 / MI300X) SDMA packet structures.
 */

#pragma once

#include "sdma_opcodes.h"

typedef struct SDMA_PKT_COPY_LINEAR_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 11;
      unsigned int broadcast : 1;
      unsigned int reserved_1 : 4;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int count : 30;
      unsigned int reserved_0 : 2;
    };
    unsigned int DW_1_DATA;
  } COUNT_UNION;

  union {
    struct {
      unsigned int reserved_0 : 16;
      unsigned int dst_sw : 2;
      unsigned int reserved_1 : 4;
      unsigned int dst_ha : 1;
      unsigned int reserved_2 : 1;
      unsigned int src_sw : 2;
      unsigned int reserved_3 : 4;
      unsigned int src_ha : 1;
      unsigned int reserved_4 : 1;
    };
    unsigned int DW_2_DATA;
  } PARAMETER_UNION;

  union {
    struct {
      unsigned int src_addr_31_0 : 32;
    };
    unsigned int DW_3_DATA;
  } SRC_ADDR_LO_UNION;

  union {
    struct {
      unsigned int src_addr_63_32 : 32;
    };
    unsigned int DW_4_DATA;
  } SRC_ADDR_HI_UNION;

  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_5_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_6_DATA;
  } DST_ADDR_HI_UNION;
} SDMA_PKT_COPY_LINEAR, *PSDMA_PKT_COPY_LINEAR;
static_assert(sizeof(SDMA_PKT_COPY_LINEAR) == 7 * sizeof(unsigned int),
              "SDMA PKT COPY_LINEAR must be 7 DWORDs");

typedef struct SDMA_PKT_WRITE_UNTILED_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 16;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } DST_ADDR_HI_UNION;

  union {
    struct {
      unsigned int count : 22;
      unsigned int reserved_0 : 2;
      unsigned int sw : 2;
      unsigned int reserved_1 : 6;
    };
    unsigned int DW_3_DATA;
  } DW_3_UNION;

  union {
    struct {
      unsigned int data0 : 32;
    };
    unsigned int DW_4_DATA;
  } DATA0_UNION;
} SDMA_PKT_WRITE_UNTILED, *PSDMA_PKT_WRITE_UNTILED;
static_assert(sizeof(SDMA_PKT_WRITE_UNTILED) == 5 * sizeof(unsigned int),
              "SDMA PKT WRITE_UNTILED must be 5 DWORDs");

typedef struct SDMA_PKT_FENCE_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 16;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } ADDR_LO_UNION;

  union {
    struct {
      unsigned int addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } ADDR_HI_UNION;

  union {
    struct {
      unsigned int data : 32;
    };
    unsigned int DW_3_DATA;
  } DATA_UNION;
} SDMA_PKT_FENCE, *PSDMA_PKT_FENCE;
static_assert(sizeof(SDMA_PKT_FENCE) == 4 * sizeof(unsigned int),
              "SDMA PKT FENCE must be 4 DWORDs");

typedef struct SDMA_PKT_CONSTANT_FILL_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int sw : 2;
      unsigned int reserved_0 : 12;
      unsigned int fillsize : 2;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int dst_addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int dst_addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } DST_ADDR_HI_UNION;

  union {
    struct {
      unsigned int src_data_31_0 : 32;
    };
    unsigned int DW_3_DATA;
  } DATA_UNION;

  union {
    struct {
      unsigned int count : 22;
      unsigned int reserved_0 : 10;
    };
    unsigned int DW_4_DATA;
  } COUNT_UNION;
} SDMA_PKT_CONSTANT_FILL, *PSDMA_PKT_CONSTANT_FILL;
static_assert(sizeof(SDMA_PKT_CONSTANT_FILL) == 5 * sizeof(unsigned int),
              "SDMA PKT CONSTANT_FILL must be 5 DWORDs");

typedef struct SDMA_PKT_TRAP_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 16;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int int_context : 28;
      unsigned int reserved_0 : 4;
    };
    unsigned int DW_1_DATA;
  } INT_CONTEXT_UNION;
} SDMA_PKT_TRAP, *PSDMA_PKT_TRAP;
static_assert(sizeof(SDMA_PKT_TRAP) == 2 * sizeof(unsigned int),
              "SDMA PKT TRAP must be 2 DWORDs");

typedef struct SDMA_PKT_POLL_REGMEM_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 10;
      unsigned int hdp_flush : 1;
      unsigned int reserved_1 : 1;
      unsigned int func : 3;
      unsigned int mem_poll : 1;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } ADDR_LO_UNION;

  union {
    struct {
      unsigned int addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } ADDR_HI_UNION;

  union {
    struct {
      unsigned int value : 32;
    };
    unsigned int DW_3_DATA;
  } VALUE_UNION;

  union {
    struct {
      unsigned int mask : 32;
    };
    unsigned int DW_4_DATA;
  } MASK_UNION;

  union {
    struct {
      unsigned int interval : 16;
      unsigned int retry_count : 12;
      unsigned int reserved_0 : 4;
    };
    unsigned int DW_5_DATA;
  } DW5_UNION;
} SDMA_PKT_POLL_REGMEM, *PSDMA_PKT_POLL_REGMEM;
static_assert(sizeof(SDMA_PKT_POLL_REGMEM) == 6 * sizeof(unsigned int),
              "SDMA PKT POLL_REGMEM must be 6 DWORDs");

typedef struct SDMA_PKT_TIMESTAMP_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int reserved_0 : 16;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } ADDR_LO_UNION;

  union {
    struct {
      unsigned int addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } ADDR_HI_UNION;
} SDMA_PKT_TIMESTAMP, *PSDMA_PKT_TIMESTAMP;
static_assert(sizeof(SDMA_PKT_TIMESTAMP) == 3 * sizeof(unsigned int),
              "SDMA PKT TIMESTAMP must be 3 DWORDs");

typedef struct SDMA_PKT_NOP_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int count : 14;
      unsigned int reserved_0 : 2;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int data0 : 32;
    };
    unsigned int DW_1_DATA;
  } DATA0_UNION;
} SDMA_PKT_NOP, *PSDMA_PKT_NOP;
static_assert(sizeof(SDMA_PKT_NOP) == 2 * sizeof(unsigned int),
              "SDMA PKT NOP must be 2 DWORDs");

typedef struct SDMA_PKT_ATOMIC_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int l : 1;
      unsigned int reserved_0 : 8;
      unsigned int operation : 7;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } ADDR_LO_UNION;

  union {
    struct {
      unsigned int addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } ADDR_HI_UNION;

  union {
    struct {
      unsigned int src_data_31_0 : 32;
    };
    unsigned int DW_3_DATA;
  } SRC_DATA_LO_UNION;

  union {
    struct {
      unsigned int src_data_63_32 : 32;
    };
    unsigned int DW_4_DATA;
  } SRC_DATA_HI_UNION;

  union {
    struct {
      unsigned int cmp_data_31_0 : 32;
    };
    unsigned int DW_5_DATA;
  } CMP_DATA_LO_UNION;

  union {
    struct {
      unsigned int cmp_data_63_32 : 32;
    };
    unsigned int DW_6_DATA;
  } CMP_DATA_HI_UNION;

  union {
    struct {
      unsigned int loop_interval : 13;
      unsigned int reserved_0 : 19;
    };
    unsigned int DW_7_DATA;
  } LOOP_UNION;
} SDMA_PKT_ATOMIC;
static_assert(sizeof(SDMA_PKT_ATOMIC) == 8 * sizeof(unsigned int),
              "SDMA PKT ATOMIC must be 8 DWORDs");

typedef struct SDMA_PKT_LINEAR_LARGE_SUB_WINDOW_COPY_TAG {
  union {
    struct {
      unsigned int op : 8;
      unsigned int sub_op : 8;
      unsigned int l : 1;
      unsigned int reserved_0 : 8;
      unsigned int operation : 7;
    };
    unsigned int DW_0_DATA;
  } HEADER_UNION;

  union {
    struct {
      unsigned int src_base_addr_31_0 : 32;
    };
    unsigned int DW_1_DATA;
  } SRC_ADDR_LO_UNION;

  union {
    struct {
      unsigned int src_base_addr_63_32 : 32;
    };
    unsigned int DW_2_DATA;
  } SRC_ADDR_HI_UNION;

  union {
    struct {
      unsigned int src_x : 32;
    };
    unsigned int DW_3_DATA;

  } SRC_X_UNION;

  union {
    struct {
      unsigned int src_y : 32;
    };
    unsigned int DW_4_DATA;
  } SRC_Y_UNION;

  union {
    struct {
      unsigned int src_z : 32;
    };
    unsigned int DW_5_DATA;

  } SRC_Z_UNION;

  union {
    struct {
      unsigned int src_pitch : 32;
    };
    unsigned int DW_6_DATA;
  } SRC_PITCH_UNION;

  union {
    struct {
      unsigned int src_slice_pitch_31_0 : 32;
    };
    unsigned int DW_7_DATA;
  } SRC_SLICE_PITCH_LO_UNION;

  union {
    struct {
      unsigned int src_slice_pitch_47_32 : 16;
      unsigned int reserved_0 : 16;
    };
    unsigned int DW_8_DATA;
  } SRC_SLICE_PITCH_HI_UNION;

  union {
    struct {
      unsigned int dst_data_31_0 : 32;
    };
    unsigned int DW_9_DATA;
  } DST_ADDR_LO_UNION;

  union {
    struct {
      unsigned int src_data_63_32 : 32;
    };
    unsigned int DW_10_DATA;
  } DST_ADDR_HI_UNION;

  union {
    struct {
      unsigned int dst_x : 32;
    };
    unsigned int DW_11_DATA;
  } DST_X_UNION;

  union {
    struct {
      unsigned int dst_y : 32;
    };
    unsigned int DW_12_DATA;
  } DST_Y_UNION;

  union {
    struct {
      unsigned int dst_z : 32;
    };
    unsigned int DW_13_DATA;
  } DST_Z_UNION;

  union {
    struct {
      unsigned int dst_pitch : 32;
    };
    unsigned int DW_14_DATA;
  } DST_PITCH_UNION;

  union {
    struct {
      unsigned int dst_slice_pitch_31_0 : 32;
    };
    unsigned int DW_15_DATA;
  } DST_SLICE_PITCH_LO_UNION;

  union {
    struct {
      unsigned int dst_slice_pitch_47_32 : 16;
      unsigned int reserved_0 : 16;
    };
    unsigned int DW_16_DATA;
  } DST_SLICE_PITCH_HI_UNION;

  union {
    struct {
      unsigned int rect_x : 32;
    };
    unsigned int DW_17_DATA;
  } RECT_X_UNION;

  union {
    struct {
      unsigned int rect_y : 32;
    };
    unsigned int DW_18_DATA;
  } RECT_Y_UNION;

  union {
    struct {
      unsigned int rect_z : 32;
    };
    unsigned int DW_19_DATA;
  } RECT_Z_UNION;
} SDMA_PKT_LINEAR_LARGE_SUB_WINDOW_COPY;
static_assert(sizeof(SDMA_PKT_LINEAR_LARGE_SUB_WINDOW_COPY) ==
                20 * sizeof(unsigned int),
              "SDMA PKT LINEAR_LARGE_SUB_WINDOW_COPY must be 20 DWORDS");

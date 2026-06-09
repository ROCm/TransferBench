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

#include <dlfcn.h>
#include <mutex>

#include "ibv_core.hpp"

/// @brief Outcome of the runtime libibverbs probe.
///
/// Stored on @ref IbvDynloadState::status and surfaced via
/// @ref TbIbvGetLoadStatus(). The integer values are stable and may be copied
/// directly into TransferBench's `System` fields.
///   - TB_IBV_OK         (0): all required symbols resolved (RDMA + DMA-BUF).
///   - TB_IBV_NO_DMABUF  (1): RDMA symbols resolved, but `ibv_reg_dmabuf_mr`
///                            is missing (older libibverbs / no DMA-BUF API).
///   - TB_IBV_NO_RDMA    (2): dlopen failed, or any non-DMA-BUF symbol is
///                            missing. Whole library is treated as unusable.
enum TbIbvLoadStatus {
  TB_IBV_OK        = 0,
  TB_IBV_NO_DMABUF = 1,
  TB_IBV_NO_RDMA   = 2,
};

#define IBV_FN(name, rettype, arglist) rettype(*name)arglist = nullptr;

namespace {

IBV_FN(ibv_alloc_pd, ibv_pd*, (ibv_context*))
IBV_FN(ibv_close_device, int, (ibv_context*))
IBV_FN(ibv_create_cq, ibv_cq*, (ibv_context*, int, void*, ibv_comp_channel*, int))
IBV_FN(ibv_create_qp, ibv_qp*, (ibv_pd*, ibv_qp_init_attr*))
IBV_FN(ibv_dealloc_pd, int, (ibv_pd*))
IBV_FN(ibv_dereg_mr, int, (ibv_mr*))
IBV_FN(ibv_destroy_cq, int, (ibv_cq*))
IBV_FN(ibv_destroy_qp, int, (ibv_qp*))
IBV_FN(ibv_free_device_list, void, (ibv_device**))
IBV_FN(ibv_get_device_list, ibv_device**, (int*))
IBV_FN(ibv_get_device_name, const char*, (ibv_device*))
IBV_FN(ibv_modify_qp, int, (ibv_qp*, ibv_qp_attr*, int))
IBV_FN(ibv_open_device, ibv_context*, (ibv_device*))
IBV_FN(ibv_poll_cq, int, (ibv_cq*, int, ibv_wc*))
IBV_FN(ibv_post_send, int, (ibv_qp*, ibv_send_wr*, ibv_send_wr**))
IBV_FN(ibv_query_device, int, (ibv_context*, ibv_device_attr*))
IBV_FN(ibv_query_gid, int, (ibv_context*, uint8_t, int, ibv_gid*))
// MEMO: Previously the IBV_DIRECT path bound to `___ibv_query_port` because
// older versions of <infiniband/verbs.h> only exposed `ibv_query_port` as an
// inline wrapper around that internal extern symbol. Now that we no longer
// include verbs.h (it has been vendored into ibv_core.hpp), we always link or
// dlsym the public `ibv_query_port` symbol from libibverbs.so.1 directly.
// Revisit only if support for pre-1.1 libibverbs ever becomes a requirement.
IBV_FN(ibv_query_port, int, (ibv_context*, uint8_t, ibv_port_attr*))
// `ibv_reg_dmabuf_mr` is always declared; whether the underlying symbol
// actually exists in the loaded libibverbs is decided at runtime by tryLoad().
IBV_FN(ibv_reg_dmabuf_mr, ibv_mr*, (ibv_pd*, uint64_t, size_t, uint64_t, int, int))
IBV_FN(ibv_reg_mr, ibv_mr*, (ibv_pd*, void*, size_t, int))

} // namespace

struct IbvDynloadState {
  std::once_flag   once{};
  void*            handle = nullptr;
  TbIbvLoadStatus  status = TB_IBV_NO_RDMA;

  /// @brief Run dlopen + dlsym once and classify the outcome.
  /// @return One of the @ref TbIbvLoadStatus values, also stored in @c status.
  TbIbvLoadStatus tryLoad()
  {
    status = TB_IBV_NO_RDMA;

    handle = dlopen("libibverbs.so.1", RTLD_NOW);
    if (handle == nullptr)
      return status;

    struct Symbol { void **ppfn; char const *name; };

    // Core RDMA symbols. Failure of any of these means RDMA is unusable, so we
    // tear the whole library back down and report TB_IBV_NO_RDMA.
    Symbol coreSymbols[] = {
        {(void**)&ibv_alloc_pd, "ibv_alloc_pd"},
        {(void**)&ibv_close_device, "ibv_close_device"},
        {(void**)&ibv_create_cq, "ibv_create_cq"},
        {(void**)&ibv_create_qp, "ibv_create_qp"},
        {(void**)&ibv_dealloc_pd, "ibv_dealloc_pd"},
        {(void**)&ibv_dereg_mr, "ibv_dereg_mr"},
        {(void**)&ibv_destroy_cq, "ibv_destroy_cq"},
        {(void**)&ibv_destroy_qp, "ibv_destroy_qp"},
        {(void**)&ibv_free_device_list, "ibv_free_device_list"},
        {(void**)&ibv_get_device_list, "ibv_get_device_list"},
        {(void**)&ibv_get_device_name, "ibv_get_device_name"},
        {(void**)&ibv_modify_qp, "ibv_modify_qp"},
        {(void**)&ibv_open_device, "ibv_open_device"},
        {(void**)&ibv_poll_cq, "ibv_poll_cq"},
        {(void**)&ibv_post_send, "ibv_post_send"},
        {(void**)&ibv_query_device, "ibv_query_device"},
        {(void**)&ibv_query_gid, "ibv_query_gid"},
        {(void**)&ibv_query_port, "ibv_query_port"},
        {(void**)&ibv_reg_mr, "ibv_reg_mr"},
    };

    for (Symbol const& s : coreSymbols) {
      void* sym = dlsym(handle, s.name);
      if (sym == nullptr) {
        // Roll back any pointer already wired so callers don't see a half-loaded library.
        for (Symbol const& r : coreSymbols) *r.ppfn = nullptr;
        dlclose(handle);
        handle = nullptr;
        return status; // TB_IBV_NO_RDMA
      }
      *s.ppfn = sym;
    }

    // DMA-BUF probe is independent: missing symbol downgrades to TB_IBV_NO_DMABUF
    // but RDMA stays usable.
    void* dmabufSym = dlsym(handle, "ibv_reg_dmabuf_mr");
    if (dmabufSym != nullptr) {
      *((void**)&ibv_reg_dmabuf_mr) = dmabufSym;
      status = TB_IBV_OK;
    } else {
      ibv_reg_dmabuf_mr = nullptr;
      status = TB_IBV_NO_DMABUF;
    }
    return status;
  }
};

inline IbvDynloadState& ibvDynloadState()
{
  static IbvDynloadState s;
  return s;
}

inline void TbIbvEnsureLoaded()
{
  IbvDynloadState& st = ibvDynloadState();
  std::call_once(st.once, [&]() { st.tryLoad(); });
}

inline TbIbvLoadStatus TbIbvGetLoadStatus()
{
  TbIbvEnsureLoaded();
  return ibvDynloadState().status;
}

inline bool TbIbvSymbolsReady()
{
  return TbIbvGetLoadStatus() != TB_IBV_NO_RDMA;
}

inline bool TbIbvDmabufPresent()
{
  return TbIbvGetLoadStatus() == TB_IBV_OK;
}

inline void* TbIbvDlHandle()
{
  TbIbvEnsureLoaded();
  return ibvDynloadState().handle;
}

inline void TbIbvUnload()
{
  IbvDynloadState& st = ibvDynloadState();
  if (st.handle != nullptr) {
    dlclose(st.handle);
    st.handle = nullptr;
    st.status = TB_IBV_NO_RDMA;
    ibv_alloc_pd = nullptr;
    ibv_close_device = nullptr;
    ibv_create_cq = nullptr;
    ibv_create_qp = nullptr;
    ibv_dealloc_pd = nullptr;
    ibv_dereg_mr = nullptr;
    ibv_destroy_cq = nullptr;
    ibv_destroy_qp = nullptr;
    ibv_free_device_list = nullptr;
    ibv_get_device_list = nullptr;
    ibv_get_device_name = nullptr;
    ibv_modify_qp = nullptr;
    ibv_open_device = nullptr;
    ibv_poll_cq = nullptr;
    ibv_post_send = nullptr;
    ibv_query_device = nullptr;
    ibv_query_gid = nullptr;
    ibv_query_port = nullptr;
    ibv_reg_dmabuf_mr = nullptr;
    ibv_reg_mr = nullptr;
  }
}

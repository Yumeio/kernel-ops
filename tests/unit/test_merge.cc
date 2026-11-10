// =============================================================================
// kernel-ops | tests/unit/test_merge.cc
// B1 attention merge kernels: merge_state (A14), merge_states (A14) and
// merge_split_kv (A04); B8 native attention: ragged / paged prefill (A01, A02,
// A06, A08) against kops::ref::attention_f64 over head dims, GQA, every
// AttentionModifiers rule, page layouts / tables, FP8 K/V, split-KV,
// determinism and batch invariance.
//
// Oracle: csrc/reference/reference.cc (merge_states_f64) for two- and N-state
// merges, plus a local fp64 log-sum-exp reference for split-KV with a sink
// (the reference header does not model sinks). Covers take-a, both/one -inf,
// N = 1 / 2 / 8, variable splits, zero splits, sink-once, total_rows_ptr,
// graph replay and the Status contract.
// =============================================================================
#include "tests/unit/harness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/quant.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/attention/attention_params.cuh"
#include "csrc/attention/prefill_kernels.cuh"
#include "csrc/attention/decode_kernels.cuh"
#include "csrc/attention/merge_kernels.cuh"
#include "csrc/attention/mla_kernels.cuh"
#include "csrc/attention/sparse_kernels.cuh"
#include "csrc/attention/ops.h"
#include "csrc/reference/ops.h"

// The public entry points come from csrc/attention/ops.h (this TU is compiled by
// nvcc, so the kernel parameter headers it needs are available). The internal
// seams below carry optional descriptors ops.h does not expose.
namespace kops::attn {
namespace internal {
Status merge_state_run(const Tensor& o, const Tensor& lse, const Tensor& o_a, const Tensor& lse_a,
                       const Tensor& o_b, const Tensor& lse_b, const Tensor& row_mask,
                       const LaunchContext& ctx);
Status merge_split_kv_run(const Tensor& o, const Tensor& lse, const SplitKVBuffers& split,
                          const Tensor& sink_logits, const Tensor& total_rows,
                          const LaunchContext& ctx);
}  // namespace internal
}  // namespace kops::attn

namespace {

using kops::DType;
using kops::LaunchContext;
using kops::Status;
using kops::Tensor;

constexpr float kNegInfF = -std::numeric_limits<float>::infinity();
constexpr double kNegInfD = -std::numeric_limits<double>::infinity();

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

void fill_uniform(std::vector<float>& v, unsigned seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (float& x : v) x = dist(rng);
}

template <typename T>
std::vector<double> to_double(const std::vector<T>& src) {
  std::vector<double> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) out[i] = static_cast<double>(kops_test::to_float(src[i]));
  return out;
}

template <typename T>
struct DevArray {
  T* p = nullptr;
  size_t n = 0;

  explicit DevArray(size_t count) : n(count) {
    p = kops_test::device_alloc<T>(count == 0 ? 1 : count);
  }
  ~DevArray() { kops_test::device_free(p); }
  DevArray(const DevArray&) = delete;
  DevArray& operator=(const DevArray&) = delete;
};

Tensor tensor1d(void* data, DType dtype, int64_t n) {
  Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 1;
  t.device = 0;
  t.shape[0] = n;
  t.stride[0] = 1;
  return t;
}

Tensor tensor4d(void* data, DType dtype, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
  Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 4;
  t.device = 0;
  t.shape[0] = d0;
  t.shape[1] = d1;
  t.shape[2] = d2;
  t.shape[3] = d3;
  t.stride[0] = d1 * d2 * d3;
  t.stride[1] = d2 * d3;
  t.stride[2] = d3;
  t.stride[3] = 1;
  return t;
}

Tensor tensor2d(void* data, DType dtype, int64_t d0, int64_t d1) {
  Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 2;
  t.device = 0;
  t.shape[0] = d0;
  t.shape[1] = d1;
  t.stride[0] = d1;
  t.stride[1] = 1;
  return t;
}

Tensor tensor3d(void* data, DType dtype, int64_t d0, int64_t d1, int64_t d2) {
  Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 3;
  t.device = 0;
  t.shape[0] = d0;
  t.shape[1] = d1;
  t.shape[2] = d2;
  t.stride[0] = d1 * d2;
  t.stride[1] = d2;
  t.stride[2] = 1;
  return t;
}

kops::kv::PagedKVDesc make_mla_desc(void* cache, int page, int pages, int rank, int pe) {
  kops::kv::PagedKVDesc d{};
  d.k_data = cache;
  d.v_data = nullptr;
  d.k_dtype = DType::kBFloat16;
  d.v_dtype = DType::kBFloat16;
  d.k_codec = kops::kv::KVCodec::kNative;
  d.v_codec = kops::kv::KVCodec::kNative;
  d.layout = kops::kv::KVLayout::kLatent;
  d.page_size = page;
  d.num_kv_heads = 1;
  d.head_dim_k = rank + pe;
  d.head_dim_v = rank;
  d.stride_unit = kops::StrideUnit::kElement;
  d.k_stride_page = static_cast<int64_t>(page) * (rank + pe);
  d.k_stride_token = rank + pe;
  d.k_stride_head = rank + pe;
  d.table_format = kops::kv::PageTableFormat::kSlotMapping;
  (void)pages;
  return d;
}

bool is_neg_inf(float x) { return std::isinf(x) && x < 0.0f; }
bool is_neg_inf(double x) { return std::isinf(x) && x < 0.0; }

template <typename T>
void check_zero(const std::vector<T>& v, const char* what) {
  for (size_t i = 0; i < v.size(); ++i) {
    if (kops_test::to_float(v[i]) != 0.0f) {
      kops_test::record_failure(what, "expected all zeros", __FILE__, __LINE__);
      return;
    }
  }
}

template <typename T>
void check_neg_inf(const std::vector<T>& v, const char* what) {
  for (size_t i = 0; i < v.size(); ++i) {
    if (!is_neg_inf(static_cast<double>(kops_test::to_float(v[i])))) {
      kops_test::record_failure(what, "expected all -inf", __FILE__, __LINE__);
      return;
    }
  }
}

bool sync_ok(const char* what) {
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    kops_test::record_failure(what, cudaGetErrorString(err), __FILE__, __LINE__);
    return false;
  }
  return true;
}

/// fp64 log-sum-exp merge of the states named by indptr (rows x states x heads).
/// A zero weight contributes nothing even when o is not finite; no mass at all
/// gives o = 0 and lse = -inf.
void split_ref(const std::vector<double>& o_partial, const std::vector<double>& lse_partial,
               const std::vector<int32_t>& indptr, int rows, int heads, int dim,
               const std::vector<double>& sink, std::vector<double>* out_o,
               std::vector<double>* out_lse) {
  out_o->assign(static_cast<size_t>(rows) * heads * dim, 0.0);
  out_lse->assign(static_cast<size_t>(rows) * heads, kNegInfD);
  for (int r = 0; r < rows; ++r) {
    const int32_t first = indptr[r];
    const int32_t last = indptr[r + 1];
    for (int h = 0; h < heads; ++h) {
      double m = sink.empty() ? kNegInfD : sink[h];
      for (int32_t s = first; s < last; ++s) {
        m = std::max(m, lse_partial[(static_cast<size_t>(s) * heads + h)]);
      }
      double denom = 0.0;
      if (m > kNegInfD) {
        if (!sink.empty()) denom += std::exp(sink[h] - m);
        for (int32_t s = first; s < last; ++s) {
          denom += std::exp(lse_partial[static_cast<size_t>(s) * heads + h] - m);
        }
      }
      if (denom > 0.0) {
        for (int d = 0; d < dim; ++d) {
          double acc = 0.0;
          for (int32_t s = first; s < last; ++s) {
            const double w = std::exp(lse_partial[static_cast<size_t>(s) * heads + h] - m);
            if (w == 0.0) continue;
            acc += w * o_partial[(static_cast<size_t>(s) * heads + h) * dim + d];
          }
          (*out_o)[(static_cast<size_t>(r) * heads + h) * dim + d] = acc / denom;
        }
        (*out_lse)[static_cast<size_t>(r) * heads + h] = m + std::log(denom);
      }
    }
  }
}

/// Uniform-state wrapper over split_ref: state s of row r is row (r, s).
void states_ref(const std::vector<double>& o_states, const std::vector<double>& lse_states,
                int rows, int n, int heads, int dim, std::vector<double>* out_o,
                std::vector<double>* out_lse) {
  std::vector<int32_t> indptr(static_cast<size_t>(rows) + 1);
  for (int r = 0; r <= rows; ++r) indptr[static_cast<size_t>(r)] = r * n;
  split_ref(o_states, lse_states, indptr, rows, heads, dim, {}, out_o, out_lse);
}

// -----------------------------------------------------------------------------
// merge_state
// -----------------------------------------------------------------------------

void test_merge_state_take_a_and_one_inf() {
  const int rows = 2, heads = 4, dim = 64;
  const size_t ne = static_cast<size_t>(rows) * heads * dim;
  const size_t nl = static_cast<size_t>(rows) * heads;

  std::vector<float> oa(ne), la(nl), lb(nl);
  fill_uniform(oa, 11u, -1.5f, 1.5f);
  fill_uniform(la, 12u, -3.0f, 0.0f);
  std::vector<float> ob(ne, std::numeric_limits<float>::quiet_NaN());
  std::fill(lb.begin(), lb.end(), kNegInfF);

  DevArray<float> d_oa(ne), d_ob(ne), d_la(nl), d_lb(nl), d_o(ne), d_l(nl);
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_oa.p, oa);
  kops_test::copy_in(d_ob.p, ob);
  kops_test::copy_in(d_la.p, la);
  kops_test::copy_in(d_lb.p, lb);

  Tensor t_o = kops_test::tensor3d(d_o.p, DType::kFloat32, rows, heads, dim);
  Tensor t_l = kops_test::tensor2d(d_l.p, DType::kFloat32, rows, heads);
  Tensor t_oa = kops_test::tensor3d(d_oa.p, DType::kFloat32, rows, heads, dim);
  Tensor t_ob = kops_test::tensor3d(d_ob.p, DType::kFloat32, rows, heads, dim);
  Tensor t_la = kops_test::tensor2d(d_la.p, DType::kFloat32, rows, heads);
  Tensor t_lb = kops_test::tensor2d(d_lb.p, DType::kFloat32, rows, heads);

  // b absent (lse_b = -inf): output must copy a, bit for bit, even though o_b
  // holds NaN. This is the public-API form of row_mask = 0.
  KOPS_EXPECT_OK(kops::attn::merge_state(t_o, t_l, t_oa, t_la, t_ob, t_lb, ctx),
                 "merge_state take-a");
  if (!sync_ok("merge_state take-a sync")) return;
  {
    const auto out = kops_test::copy_out(d_o.p, ne);
    const auto lse = kops_test::copy_out(d_l.p, nl);
    kops_test::compare(out, to_double(oa), 0.0, 0.0, "merge_state take-a output");
    kops_test::compare(lse, to_double(la), 0.0, 0.0, "merge_state take-a lse");
  }

  // a absent (lse_a = -inf): output must copy b.
  std::vector<float> la_inf(nl, kNegInfF);
  kops_test::copy_in(d_la.p, la_inf);
  std::vector<float> ob_finite(ne);
  fill_uniform(ob_finite, 13u, -1.5f, 1.5f);
  kops_test::copy_in(d_ob.p, ob_finite);
  kops_test::copy_in(d_lb.p, la);  // finite now
  KOPS_EXPECT_OK(kops::attn::merge_state(t_o, t_l, t_oa, t_la, t_ob, t_lb, ctx),
                 "merge_state take-b");
  if (!sync_ok("merge_state take-b sync")) return;
  {
    const auto out = kops_test::copy_out(d_o.p, ne);
    const auto lse = kops_test::copy_out(d_l.p, nl);
    kops_test::compare(out, to_double(ob_finite), 0.0, 0.0, "merge_state take-b output");
    kops_test::compare(lse, to_double(la), 0.0, 0.0, "merge_state take-b lse");
  }
}

void test_merge_state_row_mask() {
  const int rows = 3, heads = 4, dim = 128;
  const size_t ne = static_cast<size_t>(rows) * heads * dim;
  const size_t nl = static_cast<size_t>(rows) * heads;

  std::vector<float> oa(ne), ob(ne), la(nl), lb(nl);
  fill_uniform(oa, 21u, -1.0f, 1.0f);
  fill_uniform(ob, 22u, -1.0f, 1.0f);
  fill_uniform(la, 23u, -2.0f, 0.0f);
  fill_uniform(lb, 24u, -2.0f, 0.0f);
  // Row 0 would be dominated by b if it were read; make that impossible to miss.
  for (int h = 0; h < heads; ++h) {
    lb[static_cast<size_t>(h)] = 50.0f;
    for (int d = 0; d < dim; ++d) ob[static_cast<size_t>(h) * dim + d] = std::numeric_limits<float>::quiet_NaN();
  }

  std::vector<uint8_t> mask = {0u, 1u, 1u};
  DevArray<float> d_oa(ne), d_ob(ne), d_la(nl), d_lb(nl), d_o(ne), d_l(nl);
  DevArray<uint8_t> d_mask(mask.size());
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_oa.p, oa);
  kops_test::copy_in(d_ob.p, ob);
  kops_test::copy_in(d_la.p, la);
  kops_test::copy_in(d_lb.p, lb);
  kops_test::copy_in(d_mask.p, mask);

  Tensor t_o = kops_test::tensor3d(d_o.p, DType::kFloat32, rows, heads, dim);
  Tensor t_l = kops_test::tensor2d(d_l.p, DType::kFloat32, rows, heads);
  Tensor t_oa = kops_test::tensor3d(d_oa.p, DType::kFloat32, rows, heads, dim);
  Tensor t_ob = kops_test::tensor3d(d_ob.p, DType::kFloat32, rows, heads, dim);
  Tensor t_la = kops_test::tensor2d(d_la.p, DType::kFloat32, rows, heads);
  Tensor t_lb = kops_test::tensor2d(d_lb.p, DType::kFloat32, rows, heads);
  Tensor t_mask = tensor1d(d_mask.p, DType::kUInt8, mask.size());

  KOPS_EXPECT_OK(
      kops::attn::internal::merge_state_run(t_o, t_l, t_oa, t_la, t_ob, t_lb, t_mask, ctx),
      "merge_state row_mask");
  if (!sync_ok("merge_state row_mask sync")) return;

  // Expected rows 1-2 from the fp64 reference; row 0 is a verbatim copy of a.
  std::vector<double> states(static_cast<size_t>(rows - 1) * 2 * heads * dim);
  std::vector<double> lses(static_cast<size_t>(rows - 1) * 2 * heads);
  for (int r = 1; r < rows; ++r) {
    for (int s = 0; s < 2; ++s) {
      for (int h = 0; h < heads; ++h) {
        lses[(static_cast<size_t>(r - 1) * 2 + s) * heads + h] =
            static_cast<double>(s == 0 ? la[r * heads + h] : lb[r * heads + h]);
        for (int d = 0; d < dim; ++d) {
          const float v = s == 0 ? oa[(static_cast<size_t>(r) * heads + h) * dim + d]
                                 : ob[(static_cast<size_t>(r) * heads + h) * dim + d];
          states[((static_cast<size_t>(r - 1) * 2 + s) * heads + h) * dim + d] = v;
        }
      }
    }
  }
  std::vector<double> ref_o, ref_l;
  states_ref(states, lses, rows - 1, 2, heads, dim, &ref_o, &ref_l);

  const auto out = kops_test::copy_out(d_o.p, ne);
  const auto lse = kops_test::copy_out(d_l.p, nl);
  for (int h = 0; h < heads; ++h) {
    for (int d = 0; d < dim; ++d) {
      if (out[static_cast<size_t>(h) * dim + d] != oa[static_cast<size_t>(h) * dim + d]) {
        kops_test::record_failure("merge_state row_mask row 0 output", "not a verbatim copy",
                                  __FILE__, __LINE__);
        h = heads;
        break;
      }
    }
  }
  for (int h = 0; h < heads; ++h) {
    if (lse[static_cast<size_t>(h)] != la[static_cast<size_t>(h)]) {
      kops_test::record_failure("merge_state row_mask row 0 lse", "not copied", __FILE__, __LINE__);
      break;
    }
  }
  std::vector<float> got_o(out.begin() + static_cast<size_t>(heads) * dim, out.end());
  std::vector<float> got_l(lse.begin() + heads, lse.end());
  kops_test::compare(got_o, ref_o, 1e-5, 1e-5, "merge_state row_mask output");
  kops_test::compare(got_l, ref_l, 1e-5, 1e-5, "merge_state row_mask lse");

  // A row_mask of the wrong length is a contract error, not an out-of-bounds read.
  Tensor bad_mask = tensor1d(d_mask.p, DType::kUInt8, mask.size() - 1);
  KOPS_EXPECT_STATUS(
      kops::attn::internal::merge_state_run(t_o, t_l, t_oa, t_la, t_ob, t_lb, bad_mask, ctx),
      Status::kInvalidArgument, "merge_state row_mask length");
}

void test_merge_state_both_neg_inf() {
  const int rows = 2, heads = 4, dim = 64;
  const size_t ne = static_cast<size_t>(rows) * heads * dim;
  const size_t nl = static_cast<size_t>(rows) * heads;

  std::vector<float> nan(ne, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> inf(nl, kNegInfF);
  DevArray<float> d_oa(ne), d_ob(ne), d_la(nl), d_lb(nl), d_o(ne), d_l(nl);
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_oa.p, nan);
  kops_test::copy_in(d_ob.p, nan);
  kops_test::copy_in(d_la.p, inf);
  kops_test::copy_in(d_lb.p, inf);

  Tensor t_o = kops_test::tensor3d(d_o.p, DType::kFloat32, rows, heads, dim);
  Tensor t_l = kops_test::tensor2d(d_l.p, DType::kFloat32, rows, heads);
  Tensor t_oa = kops_test::tensor3d(d_oa.p, DType::kFloat32, rows, heads, dim);
  Tensor t_ob = kops_test::tensor3d(d_ob.p, DType::kFloat32, rows, heads, dim);
  Tensor t_la = kops_test::tensor2d(d_la.p, DType::kFloat32, rows, heads);
  Tensor t_lb = kops_test::tensor2d(d_lb.p, DType::kFloat32, rows, heads);

  KOPS_EXPECT_OK(kops::attn::merge_state(t_o, t_l, t_oa, t_la, t_ob, t_lb, ctx),
                 "merge_state both -inf");
  if (!sync_ok("merge_state both -inf sync")) return;
  const auto out = kops_test::copy_out(d_o.p, ne);
  const auto lse = kops_test::copy_out(d_l.p, nl);
  check_zero(out, "merge_state both -inf output");
  check_neg_inf(lse, "merge_state both -inf lse");
}

void test_merge_state_finite() {
  const int rows = 3, heads = 4, dim = 128;
  const size_t ne = static_cast<size_t>(rows) * heads * dim;
  const size_t nl = static_cast<size_t>(rows) * heads;

  std::vector<float> oa(ne), ob(ne), la(nl), lb(nl);
  fill_uniform(oa, 31u, -1.0f, 1.0f);
  fill_uniform(ob, 32u, -1.0f, 1.0f);
  fill_uniform(la, 33u, -4.0f, 0.0f);
  fill_uniform(lb, 34u, -4.0f, 0.0f);

  DevArray<float> d_oa(ne), d_ob(ne), d_la(nl), d_lb(nl), d_o(ne), d_l(nl);
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_oa.p, oa);
  kops_test::copy_in(d_ob.p, ob);
  kops_test::copy_in(d_la.p, la);
  kops_test::copy_in(d_lb.p, lb);

  Tensor t_o = kops_test::tensor3d(d_o.p, DType::kFloat32, rows, heads, dim);
  Tensor t_l = kops_test::tensor2d(d_l.p, DType::kFloat32, rows, heads);
  Tensor t_oa = kops_test::tensor3d(d_oa.p, DType::kFloat32, rows, heads, dim);
  Tensor t_ob = kops_test::tensor3d(d_ob.p, DType::kFloat32, rows, heads, dim);
  Tensor t_la = kops_test::tensor2d(d_la.p, DType::kFloat32, rows, heads);
  Tensor t_lb = kops_test::tensor2d(d_lb.p, DType::kFloat32, rows, heads);

  KOPS_EXPECT_OK(kops::attn::merge_state(t_o, t_l, t_oa, t_la, t_ob, t_lb, ctx),
                 "merge_state finite");
  if (!sync_ok("merge_state finite sync")) return;

  std::vector<double> states(static_cast<size_t>(rows) * 2 * heads * dim);
  std::vector<double> lses(static_cast<size_t>(rows) * 2 * heads);
  for (int r = 0; r < rows; ++r) {
    for (int s = 0; s < 2; ++s) {
      for (int h = 0; h < heads; ++h) {
        lses[(static_cast<size_t>(r) * 2 + s) * heads + h] =
            static_cast<double>(s == 0 ? la[r * heads + h] : lb[r * heads + h]);
        for (int d = 0; d < dim; ++d) {
          const float v = s == 0 ? oa[(static_cast<size_t>(r) * heads + h) * dim + d]
                                 : ob[(static_cast<size_t>(r) * heads + h) * dim + d];
          states[((static_cast<size_t>(r) * 2 + s) * heads + h) * dim + d] = v;
        }
      }
    }
  }
  std::vector<double> ref_o, ref_l;
  states_ref(states, lses, rows, 2, heads, dim, &ref_o, &ref_l);

  const auto out = kops_test::copy_out(d_o.p, ne);
  const auto lse = kops_test::copy_out(d_l.p, nl);
  kops_test::compare(out, ref_o, 1e-5, 1e-5, "merge_state finite output");
  kops_test::compare(lse, ref_l, 1e-5, 1e-5, "merge_state finite lse");
}

void test_merge_state_bf16() {
  const int rows = 2, heads = 4, dim = 64;
  const size_t ne = static_cast<size_t>(rows) * heads * dim;
  const size_t nl = static_cast<size_t>(rows) * heads;

  std::vector<float> oa(ne), ob(ne), la(nl), lb(nl);
  fill_uniform(oa, 35u, -1.0f, 1.0f);
  fill_uniform(ob, 36u, -1.0f, 1.0f);
  fill_uniform(la, 37u, -3.0f, 0.0f);
  fill_uniform(lb, 38u, -3.0f, 0.0f);

  DevArray<float> d_oa(ne), d_ob(ne), d_la(nl), d_lb(nl);
  DevArray<__nv_bfloat16> d_o(ne);
  DevArray<float> d_l(nl);
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_oa.p, oa);
  kops_test::copy_in(d_ob.p, ob);
  kops_test::copy_in(d_la.p, la);
  kops_test::copy_in(d_lb.p, lb);

  Tensor t_o = kops_test::tensor3d(d_o.p, DType::kBFloat16, rows, heads, dim);
  Tensor t_l = kops_test::tensor2d(d_l.p, DType::kFloat32, rows, heads);
  Tensor t_oa = kops_test::tensor3d(d_oa.p, DType::kFloat32, rows, heads, dim);
  Tensor t_ob = kops_test::tensor3d(d_ob.p, DType::kFloat32, rows, heads, dim);
  Tensor t_la = kops_test::tensor2d(d_la.p, DType::kFloat32, rows, heads);
  Tensor t_lb = kops_test::tensor2d(d_lb.p, DType::kFloat32, rows, heads);

  KOPS_EXPECT_OK(kops::attn::merge_state(t_o, t_l, t_oa, t_la, t_ob, t_lb, ctx),
                 "merge_state bf16");
  if (!sync_ok("merge_state bf16 sync")) return;

  std::vector<double> states(static_cast<size_t>(rows) * 2 * heads * dim);
  std::vector<double> lses(static_cast<size_t>(rows) * 2 * heads);
  for (int r = 0; r < rows; ++r) {
    for (int s = 0; s < 2; ++s) {
      for (int h = 0; h < heads; ++h) {
        lses[(static_cast<size_t>(r) * 2 + s) * heads + h] =
            static_cast<double>(s == 0 ? la[r * heads + h] : lb[r * heads + h]);
        for (int d = 0; d < dim; ++d) {
          const float v = s == 0 ? oa[(static_cast<size_t>(r) * heads + h) * dim + d]
                                 : ob[(static_cast<size_t>(r) * heads + h) * dim + d];
          states[((static_cast<size_t>(r) * 2 + s) * heads + h) * dim + d] = v;
        }
      }
    }
  }
  std::vector<double> ref_o, ref_l;
  states_ref(states, lses, rows, 2, heads, dim, &ref_o, &ref_l);

  const auto out = kops_test::copy_out(d_o.p, ne);
  const auto lse = kops_test::copy_out(d_l.p, nl);
  kops_test::compare(out, ref_o, 2e-2, 2e-2, "merge_state bf16 output");
  kops_test::compare(lse, ref_l, 1e-5, 1e-5, "merge_state bf16 lse");
}

void test_merge_state_inplace() {
  const int rows = 2, heads = 4, dim = 64;
  const size_t ne = static_cast<size_t>(rows) * heads * dim;
  const size_t nl = static_cast<size_t>(rows) * heads;

  std::vector<float> oa(ne), ob(ne), la(nl), lb(nl);
  fill_uniform(oa, 39u, -1.0f, 1.0f);
  fill_uniform(ob, 40u, -1.0f, 1.0f);
  fill_uniform(la, 41u, -3.0f, 0.0f);
  fill_uniform(lb, 42u, -3.0f, 0.0f);

  std::vector<double> states(static_cast<size_t>(rows) * 2 * heads * dim);
  std::vector<double> lses(static_cast<size_t>(rows) * 2 * heads);
  for (int r = 0; r < rows; ++r) {
    for (int s = 0; s < 2; ++s) {
      for (int h = 0; h < heads; ++h) {
        lses[(static_cast<size_t>(r) * 2 + s) * heads + h] =
            static_cast<double>(s == 0 ? la[r * heads + h] : lb[r * heads + h]);
        for (int d = 0; d < dim; ++d) {
          const float v = s == 0 ? oa[(static_cast<size_t>(r) * heads + h) * dim + d]
                                 : ob[(static_cast<size_t>(r) * heads + h) * dim + d];
          states[((static_cast<size_t>(r) * 2 + s) * heads + h) * dim + d] = v;
        }
      }
    }
  }
  std::vector<double> ref_o, ref_l;
  states_ref(states, lses, rows, 2, heads, dim, &ref_o, &ref_l);

  DevArray<float> d_oa(ne), d_ob(ne), d_la(nl), d_lb(nl);
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_oa.p, oa);
  kops_test::copy_in(d_ob.p, ob);
  kops_test::copy_in(d_la.p, la);
  kops_test::copy_in(d_lb.p, lb);

  Tensor t_oa = kops_test::tensor3d(d_oa.p, DType::kFloat32, rows, heads, dim);
  Tensor t_ob = kops_test::tensor3d(d_ob.p, DType::kFloat32, rows, heads, dim);
  Tensor t_la = kops_test::tensor2d(d_la.p, DType::kFloat32, rows, heads);
  Tensor t_lb = kops_test::tensor2d(d_lb.p, DType::kFloat32, rows, heads);

  // o may alias o_a and lse may alias lse_a (the header documents in-place use).
  KOPS_EXPECT_OK(kops::attn::merge_state(t_oa, t_la, t_oa, t_la, t_ob, t_lb, ctx),
                 "merge_state in-place");
  if (!sync_ok("merge_state in-place sync")) return;
  const auto out = kops_test::copy_out(d_oa.p, ne);
  const auto lse = kops_test::copy_out(d_la.p, nl);
  kops_test::compare(out, ref_o, 1e-5, 1e-5, "merge_state in-place output");
  kops_test::compare(lse, ref_l, 1e-5, 1e-5, "merge_state in-place lse");
}

// -----------------------------------------------------------------------------
// merge_states
// -----------------------------------------------------------------------------

void test_merge_states(int n, bool with_masked_state) {
  const int rows = 2, heads = 4, dim = 128;
  const size_t ne = static_cast<size_t>(rows) * heads * dim;
  const size_t ns = static_cast<size_t>(rows) * n * heads * dim;
  const size_t nl = static_cast<size_t>(rows) * n * heads;

  std::vector<float> states(ns), lses(nl);
  fill_uniform(states, 41u + static_cast<unsigned>(n), -1.0f, 1.0f);
  fill_uniform(lses, 51u + static_cast<unsigned>(n), -4.0f, 0.0f);
  if (with_masked_state) {
    if (n > 1) {
      // Row 0, state 1 is fully masked and holds NaN: it must not contaminate.
      for (int h = 0; h < heads; ++h) {
        lses[static_cast<size_t>(heads) + h] = kNegInfF;
        for (int d = 0; d < dim; ++d) {
          states[(static_cast<size_t>(heads) + h) * dim + d] =
              std::numeric_limits<float>::quiet_NaN();
        }
      }
    }
    // The last row is fully masked: zero output and -inf lse.
    for (int s = 0; s < n; ++s) {
      for (int h = 0; h < heads; ++h) {
        const size_t base = (static_cast<size_t>(rows - 1) * n + s) * heads + h;
        lses[base] = kNegInfF;
        for (int d = 0; d < dim; ++d) {
          states[base * dim + d] = std::numeric_limits<float>::quiet_NaN();
        }
      }
    }
  }

  DevArray<float> d_states(ns), d_lses(nl), d_o(ne), d_l(ne);
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_states.p, states);
  kops_test::copy_in(d_lses.p, lses);

  Tensor t_o = kops_test::tensor3d(d_o.p, DType::kFloat32, rows, heads, dim);
  Tensor t_l = kops_test::tensor2d(d_l.p, DType::kFloat32, rows, heads);
  Tensor t_states = tensor4d(d_states.p, DType::kFloat32, rows, n, heads, dim);
  Tensor t_lses = kops_test::tensor3d(d_lses.p, DType::kFloat32, rows, n, heads);

  std::vector<double> ref_o, ref_l;
  states_ref(to_double(states), to_double(lses), rows, n, heads, dim, &ref_o, &ref_l);

  KOPS_EXPECT_OK(kops::attn::merge_states(t_o, t_l, t_states, t_lses, ctx), "merge_states");
  if (!sync_ok("merge_states sync")) return;
  const auto out = kops_test::copy_out(d_o.p, ne);
  const auto lse = kops_test::copy_out(d_l.p, nl / n);
  kops_test::compare(out, ref_o, 1e-5, 1e-5, "merge_states output");
  kops_test::compare(lse, ref_l, 1e-5, 1e-5, "merge_states lse");
  if (with_masked_state) {
    const size_t row_off = static_cast<size_t>(rows - 1) * heads * dim;
    const size_t lse_off = static_cast<size_t>(rows - 1) * heads;
    check_zero(std::vector<float>(out.begin() + row_off, out.end()),
               "merge_states all-masked output");
    check_neg_inf(std::vector<float>(lse.begin() + lse_off, lse.end()),
                  "merge_states all-masked lse");
  }

  // Deterministic index order: an identical replay must be bit-identical.
  KOPS_EXPECT_OK(kops::attn::merge_states(t_o, t_l, t_states, t_lses, ctx), "merge_states repeat");
  if (!sync_ok("merge_states repeat sync")) return;
  const auto out2 = kops_test::copy_out(d_o.p, ne);
  if (std::memcmp(out.data(), out2.data(), ne * sizeof(float)) != 0) {
    kops_test::record_failure("merge_states deterministic", "second run differs", __FILE__,
                              __LINE__);
  }

  // lse is optional.
  KOPS_EXPECT_OK(kops::attn::merge_states(t_o, Tensor{}, t_states, t_lses, ctx),
                 "merge_states no lse");
  if (!sync_ok("merge_states no lse sync")) return;
  const auto out3 = kops_test::copy_out(d_o.p, ne);
  kops_test::compare(out3, ref_o, 1e-5, 1e-5, "merge_states no lse output");
}

// -----------------------------------------------------------------------------
// merge_split_kv
// -----------------------------------------------------------------------------

struct SplitCase {
  int rows;
  int heads;
  int dim;
  std::vector<int32_t> indptr;
};

/// Build partial buffers for a split case. Masked splits (lse = -inf) hold NaN
/// output to prove they are skipped.
void make_split_inputs(const SplitCase& c, std::vector<float>* o_partial,
                       std::vector<float>* lse_partial, unsigned seed) {
  const int partial_rows = c.indptr.back();
  const size_t np = static_cast<size_t>(partial_rows) * c.heads * c.dim;
  o_partial->resize(np);
  lse_partial->resize(static_cast<size_t>(partial_rows) * c.heads);
  fill_uniform(*o_partial, seed, -1.0f, 1.0f);
  fill_uniform(*lse_partial, seed + 1, -5.0f, 0.0f);
  if (partial_rows > 1) {
    // Split 0 of row 0 is fully masked; its o must never be read into the sum.
    for (int h = 0; h < c.heads; ++h) {
      (*lse_partial)[static_cast<size_t>(h)] = kNegInfF;
      for (int d = 0; d < c.dim; ++d) {
        (*o_partial)[static_cast<size_t>(h) * c.dim + d] = std::numeric_limits<float>::quiet_NaN();
      }
    }
  }
}

void run_split_case(const SplitCase& c, bool with_sink, bool with_lse) {
  std::vector<float> o_partial, lse_partial;
  make_split_inputs(c, &o_partial, &lse_partial, 61u + static_cast<unsigned>(c.heads));
  const size_t ne = static_cast<size_t>(c.rows) * c.heads * c.dim;
  const size_t nl = static_cast<size_t>(c.rows) * c.heads;

  std::vector<float> sink(c.heads);
  fill_uniform(sink, 71u, -2.0f, 0.0f);

  DevArray<float> d_partial(o_partial.size()), d_plse(lse_partial.size());
  DevArray<int32_t> d_indptr(c.indptr.size());
  DevArray<float> d_o(ne), d_l(nl), d_sink(c.heads);
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_partial.p, o_partial);
  kops_test::copy_in(d_plse.p, lse_partial);
  kops_test::copy_in(d_indptr.p, c.indptr);
  kops_test::copy_in(d_sink.p, sink);

  Tensor t_o = kops_test::tensor3d(d_o.p, DType::kFloat32, c.rows, c.heads, c.dim);
  Tensor t_l = kops_test::tensor2d(d_l.p, DType::kFloat32, c.rows, c.heads);
  Tensor t_sink = with_sink ? tensor1d(d_sink.p, DType::kFloat32, c.heads) : Tensor{};

  kops::attn::SplitKVBuffers split{};
  split.o_partial = d_partial.p;
  split.lse_partial = d_plse.p;
  split.merge_indptr = d_indptr.p;
  split.max_splits = 8;

  // Initialize outputs to a sentinel so zero-split rows are visibly written.
  std::vector<float> sentinel_o(ne, 7.0f), sentinel_l(nl, 7.0f);
  kops_test::copy_in(d_o.p, sentinel_o);
  kops_test::copy_in(d_l.p, sentinel_l);

  KOPS_EXPECT_OK(kops::attn::merge_split_kv(t_o, with_lse ? t_l : Tensor{}, split, t_sink, ctx),
                 "merge_split_kv");
  if (!sync_ok("merge_split_kv sync")) return;

  std::vector<double> ref_o, ref_l;
  split_ref(to_double(o_partial), to_double(lse_partial), c.indptr, c.rows, c.heads, c.dim,
            with_sink ? to_double(sink) : std::vector<double>{}, &ref_o, &ref_l);

  const auto out = kops_test::copy_out(d_o.p, ne);
  kops_test::compare(out, ref_o, 1e-5, 1e-5, "merge_split_kv output");
  if (with_lse) {
    const auto lse = kops_test::copy_out(d_l.p, nl);
    kops_test::compare(lse, ref_l, 1e-5, 1e-5, "merge_split_kv lse");
    // Zero-split rows: 0 output and -inf lse without a sink, sink with one.
    for (int r = 0; r < c.rows; ++r) {
      if (c.indptr[r + 1] != c.indptr[r]) continue;
      for (int h = 0; h < c.heads; ++h) {
        const size_t idx = static_cast<size_t>(r) * c.heads + h;
        for (int d = 0; d < c.dim; ++d) {
          if (out[static_cast<size_t>(r) * c.heads * c.dim + h * c.dim + d] != 0.0f) {
            kops_test::record_failure("merge_split_kv zero split output", "not zero", __FILE__,
                                      __LINE__);
            r = c.rows;
            break;
          }
        }
        if (with_sink ? (lse[idx] != sink[static_cast<size_t>(h)]) : !is_neg_inf(lse[idx])) {
          kops_test::record_failure("merge_split_kv zero split lse", "wrong lse", __FILE__,
                                    __LINE__);
          r = c.rows;
          break;
        }
      }
    }
  } else {
    // lse is optional; the kernel must not write it when absent.
    const auto lse = kops_test::copy_out(d_l.p, nl);
    for (size_t i = 0; i < nl; ++i) {
      if (lse[i] != 7.0f) {
        kops_test::record_failure("merge_split_kv lse absent", "lse was written", __FILE__,
                                  __LINE__);
        break;
      }
    }
  }
}

void test_merge_split_kv_variable_splits() {
  // Even head count -> HEADS_PER_CTA = 4; odd -> 1. Split counts {2, 0, 3}
  // include a zero-split row.
  run_split_case({3, 4, 64, {0, 2, 2, 5}}, false, true);
  run_split_case({3, 4, 64, {0, 2, 2, 5}}, true, true);
  run_split_case({3, 4, 64, {0, 2, 2, 5}}, true, false);
  run_split_case({2, 5, 128, {0, 1, 5}}, true, true);
}

void test_merge_split_kv_bf16() {
  const int rows = 2, heads = 4, dim = 64;
  const std::vector<int32_t> indptr = {0, 2, 3};
  const size_t ne = static_cast<size_t>(rows) * heads * dim;
  const size_t nl = static_cast<size_t>(rows) * heads;

  std::vector<float> o_partial, lse_partial;
  make_split_inputs({rows, heads, dim, indptr}, &o_partial, &lse_partial, 101u);
  std::vector<float> sink(heads);
  fill_uniform(sink, 102u, -2.0f, 0.0f);

  DevArray<float> d_partial(o_partial.size()), d_plse(lse_partial.size());
  DevArray<int32_t> d_indptr(indptr.size());
  DevArray<__nv_bfloat16> d_o(ne);
  DevArray<float> d_l(nl), d_sink(heads);
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_partial.p, o_partial);
  kops_test::copy_in(d_plse.p, lse_partial);
  kops_test::copy_in(d_indptr.p, indptr);
  kops_test::copy_in(d_sink.p, sink);

  Tensor t_o = kops_test::tensor3d(d_o.p, DType::kBFloat16, rows, heads, dim);
  Tensor t_l = kops_test::tensor2d(d_l.p, DType::kFloat32, rows, heads);
  Tensor t_sink = tensor1d(d_sink.p, DType::kFloat32, heads);

  kops::attn::SplitKVBuffers split{};
  split.o_partial = d_partial.p;
  split.lse_partial = d_plse.p;
  split.merge_indptr = d_indptr.p;
  split.max_splits = 4;

  KOPS_EXPECT_OK(kops::attn::merge_split_kv(t_o, t_l, split, t_sink, ctx),
                 "merge_split_kv bf16");
  if (!sync_ok("merge_split_kv bf16 sync")) return;

  std::vector<double> ref_o, ref_l;
  split_ref(to_double(o_partial), to_double(lse_partial), indptr, rows, heads, dim,
            to_double(sink), &ref_o, &ref_l);
  const auto out = kops_test::copy_out(d_o.p, ne);
  const auto lse = kops_test::copy_out(d_l.p, nl);
  kops_test::compare(out, ref_o, 2e-2, 2e-2, "merge_split_kv bf16 output");
  kops_test::compare(lse, ref_l, 1e-5, 1e-5, "merge_split_kv bf16 lse");
}

void test_merge_split_kv_total_rows() {
  const int rows = 4, heads = 4, dim = 64;
  const size_t ne = static_cast<size_t>(rows) * heads * dim;
  const size_t nl = static_cast<size_t>(rows) * heads;

  const std::vector<int32_t> indptr = {0, 1, 2, 3, 4};
  std::vector<float> o_partial, lse_partial;
  make_split_inputs({rows, heads, dim, indptr}, &o_partial, &lse_partial, 81u);

  DevArray<float> d_partial(o_partial.size()), d_plse(lse_partial.size());
  DevArray<int32_t> d_indptr(indptr.size());
  DevArray<float> d_o(ne), d_l(nl);
  DevArray<uint32_t> d_rows(1);
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_partial.p, o_partial);
  kops_test::copy_in(d_plse.p, lse_partial);
  kops_test::copy_in(d_indptr.p, indptr);

  Tensor t_o = kops_test::tensor3d(d_o.p, DType::kFloat32, rows, heads, dim);
  Tensor t_l = kops_test::tensor2d(d_l.p, DType::kFloat32, rows, heads);
  Tensor t_rows = tensor1d(d_rows.p, DType::kUInt32, 1);

  kops::attn::SplitKVBuffers split{};
  split.o_partial = d_partial.p;
  split.lse_partial = d_plse.p;
  split.merge_indptr = d_indptr.p;
  split.max_splits = 1;

  std::vector<double> ref_o, ref_l;
  split_ref(to_double(o_partial), to_double(lse_partial), indptr, rows, heads, dim, {}, &ref_o,
            &ref_l);

  const std::vector<float> sentinel_o(ne, 7.0f), sentinel_l(nl, 7.0f);
  const uint32_t two = 2;
  cudaMemcpy(d_rows.p, &two, sizeof(uint32_t), cudaMemcpyHostToDevice);
  kops_test::copy_in(d_o.p, sentinel_o);
  kops_test::copy_in(d_l.p, sentinel_l);

  KOPS_EXPECT_OK(kops::attn::internal::merge_split_kv_run(t_o, t_l, split, Tensor{}, t_rows, ctx),
                 "merge_split_kv total_rows");
  if (!sync_ok("merge_split_kv total_rows sync")) return;
  {
    const auto out = kops_test::copy_out(d_o.p, ne);
    const auto lse = kops_test::copy_out(d_l.p, nl);
    const size_t prefix = static_cast<size_t>(2) * heads * dim;
    const size_t prefix_l = static_cast<size_t>(2) * heads;
    std::vector<float> got_o(out.begin(), out.begin() + prefix);
    std::vector<float> got_l(lse.begin(), lse.begin() + prefix_l);
    std::vector<double> want_o(ref_o.begin(), ref_o.begin() + prefix);
    std::vector<double> want_l(ref_l.begin(), ref_l.begin() + prefix_l);
    kops_test::compare(got_o, want_o, 1e-5, 1e-5, "merge_split_kv total_rows=2 output");
    kops_test::compare(got_l, want_l, 1e-5, 1e-5, "merge_split_kv total_rows=2 lse");
    for (size_t i = prefix; i < ne; ++i) {
      if (out[i] != 7.0f) {
        kops_test::record_failure("merge_split_kv total_rows=2 tail", "row not gated", __FILE__,
                                  __LINE__);
        break;
      }
    }
    for (size_t i = prefix_l; i < nl; ++i) {
      if (lse[i] != 7.0f) {
        kops_test::record_failure("merge_split_kv total_rows=2 tail lse", "row not gated",
                                  __FILE__, __LINE__);
        break;
      }
    }
  }

  // Raise the device row count; the padded tail is now live.
  const uint32_t four = 4;
  cudaMemcpy(d_rows.p, &four, sizeof(uint32_t), cudaMemcpyHostToDevice);
  KOPS_EXPECT_OK(kops::attn::internal::merge_split_kv_run(t_o, t_l, split, Tensor{}, t_rows, ctx),
                 "merge_split_kv total_rows full");
  if (!sync_ok("merge_split_kv total_rows full sync")) return;
  {
    const auto out = kops_test::copy_out(d_o.p, ne);
    const auto lse = kops_test::copy_out(d_l.p, nl);
    kops_test::compare(out, ref_o, 1e-5, 1e-5, "merge_split_kv total_rows=4 output");
    kops_test::compare(lse, ref_l, 1e-5, 1e-5, "merge_split_kv total_rows=4 lse");
  }
}

void test_merge_split_kv_graph_replay() {
  const int rows = 3, heads = 4, dim = 64;
  const size_t ne = static_cast<size_t>(rows) * heads * dim;
  const size_t nl = static_cast<size_t>(rows) * heads;
  const std::vector<int32_t> indptr = {0, 1, 2, 3};

  std::vector<float> o_partial, lse_partial;
  make_split_inputs({rows, heads, dim, indptr}, &o_partial, &lse_partial, 91u);

  DevArray<float> d_partial(o_partial.size()), d_plse(lse_partial.size());
  DevArray<int32_t> d_indptr(indptr.size());
  DevArray<float> d_o(ne), d_l(nl);
  DevArray<uint32_t> d_rows(1);
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops_test::copy_in(d_partial.p, o_partial);
  kops_test::copy_in(d_plse.p, lse_partial);
  kops_test::copy_in(d_indptr.p, indptr);

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    kops_test::record_failure("cudaStreamCreate", cudaGetErrorString(cudaGetLastError()), __FILE__,
                              __LINE__);
    return;
  }
  ctx.stream = reinterpret_cast<kops::kops_stream_t>(stream);

  Tensor t_o = kops_test::tensor3d(d_o.p, DType::kFloat32, rows, heads, dim);
  Tensor t_l = kops_test::tensor2d(d_l.p, DType::kFloat32, rows, heads);
  Tensor t_rows = tensor1d(d_rows.p, DType::kUInt32, 1);
  kops::attn::SplitKVBuffers split{};
  split.o_partial = d_partial.p;
  split.lse_partial = d_plse.p;
  split.merge_indptr = d_indptr.p;
  split.max_splits = 1;

  std::vector<double> ref_o, ref_l;
  split_ref(to_double(o_partial), to_double(lse_partial), indptr, rows, heads, dim, {}, &ref_o,
            &ref_l);

  const std::vector<float> sentinel_o(ne, 7.0f), sentinel_l(nl, 7.0f);
  const uint32_t one = 1;
  cudaMemcpy(d_rows.p, &one, sizeof(uint32_t), cudaMemcpyHostToDevice);

  if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    kops_test::record_failure("cudaStreamBeginCapture", cudaGetErrorString(cudaGetLastError()),
                              __FILE__, __LINE__);
    cudaStreamDestroy(stream);
    return;
  }
  KOPS_EXPECT_OK(kops::attn::internal::merge_split_kv_run(t_o, t_l, split, Tensor{}, t_rows, ctx),
                 "merge_split_kv capture");
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  if (cudaStreamEndCapture(stream, &graph) != cudaSuccess ||
      cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
    kops_test::record_failure("graph instantiate", cudaGetErrorString(cudaGetLastError()), __FILE__,
                              __LINE__);
    cudaStreamDestroy(stream);
    return;
  }

  kops_test::copy_in(d_o.p, sentinel_o);
  kops_test::copy_in(d_l.p, sentinel_l);
  if (cudaGraphLaunch(exec, stream) != cudaSuccess) {
    kops_test::record_failure("graph launch", cudaGetErrorString(cudaGetLastError()), __FILE__,
                              __LINE__);
  }
  if (!sync_ok("graph launch sync")) {
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return;
  }
  {
    const auto out = kops_test::copy_out(d_o.p, ne);
    const auto lse = kops_test::copy_out(d_l.p, nl);
    const size_t prefix = static_cast<size_t>(heads) * dim;
    std::vector<float> got_o(out.begin(), out.begin() + prefix);
    std::vector<float> got_l(lse.begin(), lse.begin() + heads);
    std::vector<double> want_o(ref_o.begin(), ref_o.begin() + prefix);
    std::vector<double> want_l(ref_l.begin(), ref_l.begin() + heads);
    kops_test::compare(got_o, want_o, 1e-5, 1e-5, "graph replay total_rows=1 output");
    kops_test::compare(got_l, want_l, 1e-5, 1e-5, "graph replay total_rows=1 lse");
    for (size_t i = prefix; i < ne; ++i) {
      if (out[i] != 7.0f) {
        kops_test::record_failure("graph replay total_rows=1 tail", "row not gated", __FILE__,
                                  __LINE__);
        break;
      }
    }
    for (size_t i = static_cast<size_t>(heads); i < nl; ++i) {
      if (lse[i] != 7.0f) {
        kops_test::record_failure("graph replay total_rows=1 tail lse", "row not gated", __FILE__,
                                  __LINE__);
        break;
      }
    }
  }

  // Update only the device row count, then replay the same graph: the kernel
  // argument is a pointer, so no recapture is needed.
  const uint32_t three = 3;
  cudaMemcpy(d_rows.p, &three, sizeof(uint32_t), cudaMemcpyHostToDevice);
  if (cudaGraphLaunch(exec, stream) != cudaSuccess) {
    kops_test::record_failure("graph relaunch", cudaGetErrorString(cudaGetLastError()), __FILE__,
                              __LINE__);
  }
  if (sync_ok("graph relaunch sync")) {
    const auto out = kops_test::copy_out(d_o.p, ne);
    const auto lse = kops_test::copy_out(d_l.p, nl);
    kops_test::compare(out, ref_o, 1e-5, 1e-5, "graph replay total_rows=3 output");
    kops_test::compare(lse, ref_l, 1e-5, 1e-5, "graph replay total_rows=3 lse");
  }

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
}

// -----------------------------------------------------------------------------
// Status contract
// -----------------------------------------------------------------------------

void test_merge_hybrid() {
  const int rows = 2, heads = 2, dim = 64;
  const int64_t n = static_cast<int64_t>(rows) * heads * dim;
  const int64_t ln = static_cast<int64_t>(rows) * heads;
  float* o0 = kops_test::device_alloc<float>(static_cast<size_t>(n));
  float* o1 = kops_test::device_alloc<float>(static_cast<size_t>(n));
  float* l0 = kops_test::device_alloc<float>(static_cast<size_t>(ln));
  float* l1 = kops_test::device_alloc<float>(static_cast<size_t>(ln));
  float* out = kops_test::device_alloc<float>(static_cast<size_t>(n));
  float* lse = kops_test::device_alloc<float>(static_cast<size_t>(ln));
  float* sink = kops_test::device_alloc<float>(static_cast<size_t>(heads));
  std::vector<float> h0(static_cast<size_t>(n)), h1(static_cast<size_t>(n));
  std::vector<float> hl0(static_cast<size_t>(ln)), hl1(static_cast<size_t>(ln));
  std::vector<float> hs(static_cast<size_t>(heads));
  for (int64_t i = 0; i < n; ++i) {
    h0[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i % 17);
    h1[static_cast<size_t>(i)] = -0.05f * static_cast<float>(i % 13);
  }
  for (int64_t i = 0; i < ln; ++i) {
    hl0[static_cast<size_t>(i)] = -1.0f + 0.01f * static_cast<float>(i);
    hl1[static_cast<size_t>(i)] = -2.0f + 0.02f * static_cast<float>(i);
  }
  hs[0] = -0.5f;
  hs[1] = -0.75f;
  kops_test::copy_in(o0, h0);
  kops_test::copy_in(o1, h1);
  kops_test::copy_in(l0, hl0);
  kops_test::copy_in(l1, hl1);
  kops_test::copy_in(sink, hs);
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  // Ordinary host arrays: the launcher packs the pointer values into the kernel
  // parameter, so no pinned/mapped allocation is required.
  const float* o_ptrs[2] = {o0, o1};
  const float* l_ptrs[2] = {l0, l1};

  Tensor to = kops_test::tensor3d(out, DType::kFloat32, rows, heads, dim);
  Tensor tl = kops_test::tensor2d(lse, DType::kFloat32, rows, heads);
  Tensor ts{};
  ts.data = sink;
  ts.dtype = DType::kFloat32;
  ts.ndim = 1;
  ts.device = 0;
  ts.shape[0] = heads;
  ts.stride[0] = 1;
  kops::attn::HybridBranches branches{o_ptrs, l_ptrs, 2};
  KOPS_EXPECT_OK(kops::attn::merge_hybrid_branches(to, tl, branches, ts, ctx), "merge_hybrid");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const auto got = kops_test::copy_out(out, static_cast<size_t>(n));
  const auto got_l = kops_test::copy_out(lse, static_cast<size_t>(ln));
  std::vector<double> ref_o(static_cast<size_t>(n)), ref_l(static_cast<size_t>(ln));
  for (int r = 0; r < rows; ++r) {
    for (int h = 0; h < heads; ++h) {
      const int64_t idx = static_cast<int64_t>(r) * heads + h;
      const double m = std::max({static_cast<double>(hs[h]), static_cast<double>(hl0[idx]),
                                 static_cast<double>(hl1[idx])});
      const double den = std::exp(hs[h] - m) + std::exp(hl0[idx] - m) + std::exp(hl1[idx] - m);
      ref_l[idx] = m + std::log(den);
      for (int d = 0; d < dim; ++d) {
        const int64_t j = idx * dim + d;
        ref_o[j] = (std::exp(hs[h] - m) * 0.0 + std::exp(hl0[idx] - m) * h0[j] +
                    std::exp(hl1[idx] - m) * h1[j]) /
                   den;
      }
    }
  }
  kops_test::compare(got, ref_o, 1e-4, 1e-4, "merge_hybrid.output");
  kops_test::compare(got_l, ref_l, 1e-5, 1e-5, "merge_hybrid.lse");

  // Absent branch: null o pointer must be skipped.
  o_ptrs[1] = nullptr;
  branches.num_branches = 2;
  KOPS_EXPECT_OK(kops::attn::merge_hybrid_branches(to, tl, branches, ts, ctx), "merge_hybrid absent");
  KOPS_TEST_CHECK(cudaDeviceSynchronize() == cudaSuccess);

  // Graph capture with ordinary host arrays: only the pointer values are
  // captured, so this must not depend on pinned/mapped memory.
  o_ptrs[1] = o1;
  branches.num_branches = 2;
  cudaStream_t stream = nullptr;
  KOPS_TEST_CHECK(cudaStreamCreate(&stream) == cudaSuccess);
  LaunchContext gctx = ctx;
  gctx.stream = reinterpret_cast<kops::kops_stream_t>(stream);
  KOPS_TEST_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
  KOPS_EXPECT_OK(kops::attn::merge_hybrid_branches(to, tl, branches, ts, gctx),
                 "merge_hybrid graph capture");
  cudaGraph_t graph = nullptr;
  KOPS_TEST_CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
  cudaGraphExec_t exec = nullptr;
  KOPS_TEST_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess);
  KOPS_TEST_CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess);
  KOPS_TEST_CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);

  branches.num_branches = 0;
  KOPS_EXPECT_STATUS(kops::attn::merge_hybrid_branches(to, tl, branches, ts, ctx),
                     Status::kInvalidArgument, "merge_hybrid branch count");

  kops_test::device_free(o0); kops_test::device_free(o1);
  kops_test::device_free(l0); kops_test::device_free(l1);
  kops_test::device_free(out); kops_test::device_free(lse); kops_test::device_free(sink);
}

void test_concat_mla_q() {
  constexpr int64_t kT = 3, kH = 2, kNope = 16, kPe = 64;
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  DevArray<__nv_bfloat16> ql(static_cast<size_t>(kT * kH * kNope));
  DevArray<__nv_bfloat16> qp(static_cast<size_t>(kT * kH * kPe));
  DevArray<__nv_bfloat16> qo(static_cast<size_t>(kT * kH * (kNope + kPe)));
  if (ql.p == nullptr || qp.p == nullptr || qo.p == nullptr) return;
  std::vector<float> a(static_cast<size_t>(kT * kH * kNope));
  std::vector<float> b(static_cast<size_t>(kT * kH * kPe));
  fill_uniform(a, 301u, -2.0f, 2.0f);
  fill_uniform(b, 302u, -2.0f, 2.0f);
  std::vector<__nv_bfloat16> ah(a.size()), bh(b.size());
  for (size_t i = 0; i < a.size(); ++i) kops_test::from_float(ah[i], a[i]);
  for (size_t i = 0; i < b.size(); ++i) kops_test::from_float(bh[i], b[i]);
  kops_test::copy_in(ql.p, ah);
  kops_test::copy_in(qp.p, bh);
  const Tensor tql = tensor3d(ql.p, DType::kBFloat16, kT, kH, kNope);
  const Tensor tqp = tensor3d(qp.p, DType::kBFloat16, kT, kH, kPe);
  const Tensor tqo = tensor3d(qo.p, DType::kBFloat16, kT, kH, kNope + kPe);
  KOPS_EXPECT_OK(kops::attn::concat_mla_q(tqo, tql, tqp, ctx), "concat_mla_q");
  const auto got = kops_test::copy_out(qo.p, static_cast<size_t>(kT * kH * (kNope + kPe)));
  for (int64_t t = 0; t < kT; ++t) {
    for (int64_t h = 0; h < kH; ++h) {
      for (int64_t i = 0; i < kNope; ++i) {
        KOPS_TEST_CHECK(got[static_cast<size_t>((t * kH + h) * (kNope + kPe) + i)] ==
                        ah[static_cast<size_t>((t * kH + h) * kNope + i)]);
      }
      for (int64_t i = 0; i < kPe; ++i) {
        KOPS_TEST_CHECK(got[static_cast<size_t>((t * kH + h) * (kNope + kPe) + kNope + i)] ==
                        bh[static_cast<size_t>((t * kH + h) * kPe + i)]);
      }
    }
  }

  // Contract: q_pe must be 64 wide and the shapes must agree.
  const Tensor bad_pe = tensor3d(qp.p, DType::kBFloat16, kT, kH, 32);
  KOPS_EXPECT_STATUS(kops::attn::concat_mla_q(tqo, tql, bad_pe, ctx), Status::kInvalidArgument,
                     "concat_mla_q pe width");
  KOPS_EXPECT_STATUS(kops::attn::concat_mla_q(tqo, tqp, tqp, ctx), Status::kInvalidArgument,
                     "concat_mla_q shape mismatch");
}

void test_mla_context_gather() {
  constexpr int kPages = 4, kPage = 2, kRank = 8, kPe = 8, kTokens = 5, kBatch = 2;
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  DevArray<__nv_bfloat16> cache(static_cast<size_t>(kPages * kPage * (kRank + kPe)));
  DevArray<__nv_bfloat16> ckv(static_cast<size_t>(kTokens * kRank));
  DevArray<__nv_bfloat16> kpe(static_cast<size_t>(kTokens * kPe));
  DevArray<int32_t> table(static_cast<size_t>(kBatch * 3));
  DevArray<int32_t> indptr(static_cast<size_t>(kBatch + 1));
  DevArray<int32_t> starts(static_cast<size_t>(kBatch));
  if (cache.p == nullptr || ckv.p == nullptr || kpe.p == nullptr || table.p == nullptr ||
      indptr.p == nullptr || starts.p == nullptr) {
    return;
  }
  std::vector<float> cf(static_cast<size_t>(kPages * kPage * (kRank + kPe)));
  fill_uniform(cf, 311u, -2.0f, 2.0f);
  std::vector<__nv_bfloat16> ch(cf.size());
  for (size_t i = 0; i < cf.size(); ++i) kops_test::from_float(ch[i], cf[i]);
  kops_test::copy_in(cache.p, ch);
  kops_test::copy_in(table.p, std::vector<int32_t>{0, 1, 2, 2, 0, 1});
  kops_test::copy_in(indptr.p, std::vector<int32_t>{0, 2, 5});
  kops_test::copy_in(starts.p, std::vector<int32_t>{0, 3});
  const Tensor tc = tensor2d(ckv.p, DType::kBFloat16, kTokens, kRank);
  const Tensor tp = tensor2d(kpe.p, DType::kBFloat16, kTokens, kPe);
  const Tensor ti = tensor1d(indptr.p, DType::kInt32, kBatch + 1);
  const Tensor ts = tensor1d(starts.p, DType::kInt32, kBatch);
  kops::kv::PagedKVDesc d = make_mla_desc(cache.p, kPage, kPages, kRank, kPe);
  d.table_format = kops::kv::PageTableFormat::kBlockTable;
  d.block_table = table.p;
  d.block_table_stride = 3;
  KOPS_EXPECT_OK(kops::attn::mla_context_gather(tc, tp, d, ti, ts, ctx), "mla_context_gather");
  const auto gotc = kops_test::copy_out(ckv.p, static_cast<size_t>(kTokens * kRank));
  const auto gotp = kops_test::copy_out(kpe.p, static_cast<size_t>(kTokens * kPe));
  const int32_t tab[6] = {0, 1, 2, 2, 0, 1};
  const int32_t cus[3] = {0, 2, 5};
  const int32_t st[2] = {0, 3};
  for (int token = 0; token < kTokens; ++token) {
    int r = 0;
    while (cus[r + 1] <= token) ++r;
    const int64_t pos = st[r] + (token - cus[r]);
    const int64_t logical = pos / kPage;
    const int64_t page = tab[r * 3 + logical];
    const int64_t off = pos % kPage;
    const int64_t base = (page * kPage + off) * (kRank + kPe);
    for (int i = 0; i < kRank; ++i) {
      KOPS_TEST_CHECK(gotc[static_cast<size_t>(token * kRank + i)] == ch[static_cast<size_t>(base + i)]);
    }
    for (int i = 0; i < kPe; ++i) {
      KOPS_TEST_CHECK(gotp[static_cast<size_t>(token * kPe + i)] ==
                      ch[static_cast<size_t>(base + kRank + i)]);
    }
  }

  // Graph replay of the gather.
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) == cudaSuccess) {
    LaunchContext gc = ctx;
    gc.stream = reinterpret_cast<kops::kops_stream_t>(stream);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    KOPS_EXPECT_OK(kops::attn::mla_context_gather(tc, tp, d, ti, ts, gc), "graph mla gather");
    cudaStreamEndCapture(stream, &graph);
    if (graph != nullptr && cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess) {
      cudaGraphLaunch(exec, stream);
      cudaStreamSynchronize(stream);
      const auto again = kops_test::copy_out(ckv.p, static_cast<size_t>(kTokens * kRank));
      KOPS_TEST_CHECK(std::memcmp(again.data(), gotc.data(), again.size() * sizeof(__nv_bfloat16)) == 0);
      cudaGraphExecDestroy(exec);
    }
    if (graph != nullptr) cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
  }

  // Contract: a block table is mandatory.
  kops::kv::PagedKVDesc missing = d;
  missing.block_table = nullptr;
  KOPS_EXPECT_STATUS(kops::attn::mla_context_gather(tc, tp, missing, ti, ts, ctx),
                     Status::kInvalidArgument, "mla_context_gather block table");
  // Contract: descriptor entry width must match the output buffers.
  kops::kv::PagedKVDesc wrong = make_mla_desc(cache.p, kPage, kPages, kRank, 16);
  wrong.table_format = kops::kv::PageTableFormat::kBlockTable;
  wrong.block_table = table.p;
  wrong.block_table_stride = 3;
  KOPS_EXPECT_STATUS(kops::attn::mla_context_gather(tc, tp, wrong, ti, ts, ctx),
                     Status::kUnsupportedShape, "mla_context_gather entry width");
}

void test_contracts() {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  std::vector<float> data(8192, 1.0f);
  std::vector<int32_t> indptr = {0, 1};
  DevArray<float> d(data.size());
  DevArray<int32_t> d_indptr(indptr.size());
  kops_test::copy_in(d.p, data);
  kops_test::copy_in(d_indptr.p, indptr);

  Tensor o = kops_test::tensor3d(d.p, DType::kFloat32, 1, 4, 64);
  Tensor l = kops_test::tensor2d(d.p, DType::kFloat32, 1, 4);
  Tensor a16 = kops_test::tensor3d(d.p, DType::kFloat16, 1, 4, 64);
  Tensor a_short = kops_test::tensor3d(d.p, DType::kFloat32, 1, 4, 32);

  // merge_state
  KOPS_EXPECT_STATUS(kops::attn::merge_state(o, l, a16, l, a16, l, ctx),
                     Status::kUnsupportedDtype, "merge_state partial dtype");
  KOPS_EXPECT_STATUS(kops::attn::merge_state(o, l, a_short, l, a_short, l, ctx),
                     Status::kInvalidArgument, "merge_state shape");
  Tensor misaligned =
      kops_test::tensor3d(static_cast<char*>(static_cast<void*>(d.p)) + 4, DType::kFloat32, 1, 4, 64);
  KOPS_EXPECT_STATUS(kops::attn::merge_state(o, l, misaligned, l, misaligned, l, ctx),
                     Status::kMisaligned, "merge_state alignment");

  // merge_states
  Tensor states3d = kops_test::tensor3d(d.p, DType::kFloat32, 1, 4, 64);
  KOPS_EXPECT_STATUS(kops::attn::merge_states(o, l, states3d, l, ctx), Status::kInvalidArgument,
                     "merge_states states rank");
  Tensor states = tensor4d(d.p, DType::kFloat32, 1, 2, 4, 64);
  Tensor lses16 = kops_test::tensor3d(d.p, DType::kFloat16, 1, 2, 4);
  KOPS_EXPECT_STATUS(kops::attn::merge_states(o, l, states, lses16, ctx),
                     Status::kUnsupportedDtype, "merge_states lse dtype");
  Tensor lses_bad = kops_test::tensor3d(d.p, DType::kFloat32, 1, 3, 4);
  KOPS_EXPECT_STATUS(kops::attn::merge_states(o, l, states, lses_bad, ctx),
                     Status::kInvalidArgument, "merge_states state count");
  Tensor states_bad = tensor4d(d.p, DType::kFloat32, 1, 2, 3, 64);
  Tensor lses_ok = kops_test::tensor3d(d.p, DType::kFloat32, 1, 2, 4);
  KOPS_EXPECT_STATUS(kops::attn::merge_states(o, l, states_bad, lses_ok, ctx),
                     Status::kInvalidArgument, "merge_states head count");

  // merge_split_kv
  Tensor sink3 = tensor1d(d.p, DType::kFloat32, 3);
  Tensor sink4 = tensor1d(d.p, DType::kFloat32, 4);
  Tensor total_f32 = tensor1d(d.p, DType::kFloat32, 1);
  kops::attn::SplitKVBuffers split{};
  split.o_partial = d.p;
  split.lse_partial = d.p;
  split.merge_indptr = d_indptr.p;
  split.max_splits = 2;

  kops::attn::SplitKVBuffers no_indptr = split;
  no_indptr.merge_indptr = nullptr;
  KOPS_EXPECT_STATUS(kops::attn::merge_split_kv(o, l, no_indptr, Tensor{}, ctx),
                     Status::kInvalidArgument, "merge_split_kv indptr");

  kops::attn::SplitKVBuffers no_partial = split;
  no_partial.o_partial = nullptr;
  KOPS_EXPECT_STATUS(kops::attn::merge_split_kv(o, l, no_partial, Tensor{}, ctx),
                     Status::kInvalidArgument, "merge_split_kv partials");

  kops::attn::SplitKVBuffers zero_splits = split;
  zero_splits.max_splits = 0;
  KOPS_EXPECT_STATUS(kops::attn::merge_split_kv(o, l, zero_splits, Tensor{}, ctx),
                     Status::kInvalidArgument, "merge_split_kv max_splits");

  kops::attn::SplitKVBuffers bad_partial = split;
  bad_partial.o_partial = reinterpret_cast<float*>(reinterpret_cast<char*>(d.p) + 4);
  KOPS_EXPECT_STATUS(kops::attn::merge_split_kv(o, l, bad_partial, Tensor{}, ctx),
                     Status::kMisaligned, "merge_split_kv partial alignment");

  Tensor o72 = kops_test::tensor3d(d.p, DType::kFloat32, 1, 4, 72);
  KOPS_EXPECT_STATUS(kops::attn::merge_split_kv(o72, l, split, Tensor{}, ctx),
                     Status::kUnsupportedShape, "merge_split_kv head_dim");
  Tensor o96 = kops_test::tensor3d(d.p, DType::kFloat32, 1, 4, 96);
  Tensor a96 = kops_test::tensor3d(d.p, DType::kFloat32, 1, 4, 96);
  KOPS_EXPECT_STATUS(kops::attn::merge_state(o96, l, a96, l, a96, l, ctx),
                     Status::kUnsupportedShape, "merge_state head_dim");
  Tensor states96 = tensor4d(d.p, DType::kFloat32, 1, 2, 4, 96);
  KOPS_EXPECT_STATUS(kops::attn::merge_states(o96, l, states96, lses_ok, ctx),
                     Status::kUnsupportedShape, "merge_states head_dim");

  Tensor l_bad = kops_test::tensor2d(d.p, DType::kFloat32, 1, 5);
  KOPS_EXPECT_STATUS(kops::attn::merge_split_kv(o, l_bad, split, Tensor{}, ctx),
                     Status::kInvalidArgument, "merge_split_kv lse shape");

  KOPS_EXPECT_STATUS(kops::attn::merge_split_kv(o, l, split, sink3, ctx),
                     Status::kInvalidArgument, "merge_split_kv sink length");

  KOPS_EXPECT_STATUS(kops::attn::internal::merge_split_kv_run(o, l, split, sink4, total_f32, ctx),
                     Status::kUnsupportedDtype, "merge_split_kv total_rows dtype");

  Tensor strided = o;
  strided.stride[2] = 2;
  KOPS_EXPECT_STATUS(kops::attn::merge_split_kv(strided, l, split, Tensor{}, ctx),
                     Status::kUnsupportedLayout, "merge_split_kv last stride");
}

// -----------------------------------------------------------------------------
// B8 attention: prefill (A01/A02/A06/A08) against kops::ref::attention_f64
// -----------------------------------------------------------------------------

using kops::attn::MaskMode;

struct AttnCase {
  const char* name = "";
  std::vector<int> qo_lens, kv_lens;
  int Hq = 4, Hkv = 1, dqk = 128, dvo = 128;
  DType dt = DType::kBFloat16;
  MaskMode mask = MaskMode::kCausal;
  float softcap = 0.0f;
  int window_left = -1, window_right = -1, chunk = 0, num_sink_tokens = 0;
  bool alibi = false, sinks = false, q_pos_offset = false;
  // paged (page_size > 0)
  int page_size = 0;
  bool hnd = false, block_table = false;
  DType kv_dt = DType::kInvalid;  // kInvalid = same as dt; e4m3 / e5m2 = FP8 cache
  bool per_head_scale = false;
  float v_offset = 0.0f;  // added to every V value (makes an FP8 v_scale error visible)
  // decode entry (decode_plan / decode_run): every qo_len equals q_len, paged K/V
  bool decode = false, device_planner = false;
  int q_len = 1;
  // launch context
  int compute_units = 1;  // 1: never split; large: split whenever the q tiles cannot fill the GPU
  bool batch_invariant = false;
  double atol = 1e-2, rtol = 1e-2, lse_atol = 2e-4;
};

/// Device-side result of one prefill run (o as float, lse) plus the plan it used.
struct AttnRun {
  std::vector<float> o, lse;
  bool split = false;
  uint64_t kernel_id = 0;
  bool ok = false;
};

float round_to(DType dt, float x) {
  if (dt == DType::kBFloat16) return __bfloat162float(__float2bfloat16_rn(x));
  if (dt == DType::kFloat16) return __half2float(__float2half_rn(x));
  if (dt == DType::kFloat8E4M3) return static_cast<float>(__nv_fp8_e4m3(x));
  if (dt == DType::kFloat8E5M2) return static_cast<float>(__nv_fp8_e5m2(x));
  return x;
}

/// Host image of a 16-bit tensor from rounded floats.
std::vector<uint16_t> pack16(DType dt, const std::vector<float>& v) {
  std::vector<uint16_t> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    if (dt == DType::kBFloat16) {
      const __nv_bfloat16 b = __float2bfloat16_rn(v[i]);
      std::memcpy(&out[i], &b, 2);
    } else {
      const __half h = __float2half_rn(v[i]);
      std::memcpy(&out[i], &h, 2);
    }
  }
  return out;
}

float unpack16(DType dt, uint16_t x) {
  if (dt == DType::kBFloat16) {
    __nv_bfloat16 b;
    std::memcpy(&b, &x, 2);
    return __bfloat162float(b);
  }
  __half h;
  std::memcpy(&h, &x, 2);
  return __half2float(h);
}

uint8_t fp8_bits(DType dt, float x) {
  if (dt == DType::kFloat8E4M3) {
    const __nv_fp8_e4m3 f(x);
    return *reinterpret_cast<const uint8_t*>(&f);
  }
  const __nv_fp8_e5m2 f(x);
  return *reinterpret_cast<const uint8_t*>(&f);
}

/// Shared inputs of a case: rounded q / k / v (k, v already dequantized for FP8) + reference.
struct AttnData {
  std::vector<int32_t> qo_indptr, kv_indptr;
  std::vector<float> q, k, v;          // values the kernel sees (k, v: after dequant)
  std::vector<float> k_raw, v_raw;     // FP8: the stored code values (before scale)
  std::vector<float> k_scale, v_scale; // FP8: [1] or [Hkv]
  std::vector<float> alibi, sinks;
  std::vector<int32_t> q_pos;
  std::vector<uint8_t> mask_bits;
  std::vector<int64_t> mask_indptr;
  std::vector<double> o_ref, lse_ref;
  int total_q = 0, total_kv = 0;
};

AttnData make_attn_data(const AttnCase& c, unsigned seed) {
  AttnData d;
  const int B = static_cast<int>(c.qo_lens.size());
  d.qo_indptr.assign(B + 1, 0);
  d.kv_indptr.assign(B + 1, 0);
  for (int b = 0; b < B; ++b) {
    d.qo_indptr[b + 1] = d.qo_indptr[b] + c.qo_lens[b];
    d.kv_indptr[b + 1] = d.kv_indptr[b] + c.kv_lens[b];
  }
  d.total_q = d.qo_indptr[B];
  d.total_kv = d.kv_indptr[B];
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  d.q.resize(static_cast<size_t>(d.total_q) * c.Hq * c.dqk);
  for (float& x : d.q) x = round_to(c.dt, u(rng));
  const DType kvdt = c.kv_dt == DType::kInvalid ? c.dt : c.kv_dt;
  const bool fp8 = kvdt == DType::kFloat8E4M3 || kvdt == DType::kFloat8E5M2;
  const int nscale = c.per_head_scale ? c.Hkv : 1;
  if (fp8) {
    std::uniform_real_distribution<float> us(0.004f, 0.02f);
    d.k_scale.resize(nscale);
    d.v_scale.resize(nscale);
    for (int i = 0; i < nscale; ++i) {
      d.k_scale[i] = us(rng);
      d.v_scale[i] = us(rng);
    }
  }
  auto fill_kv = [&](std::vector<float>& out, std::vector<float>& raw, int dim,
                     const std::vector<float>& scale, float off) {
    out.resize(static_cast<size_t>(d.total_kv) * c.Hkv * dim);
    raw.resize(out.size());
    for (size_t i = 0; i < out.size(); ++i) {
      const float x = u(rng) * 2.0f + off;
      if (fp8) {
        const int h = static_cast<int>((i / dim) % c.Hkv);
        const float s = scale[c.per_head_scale ? h : 0];
        raw[i] = round_to(kvdt, x / s);
        out[i] = static_cast<float>(static_cast<double>(raw[i]) * s);
      } else {
        raw[i] = out[i] = round_to(c.dt, x * 0.5f);
      }
    }
  };
  fill_kv(d.k, d.k_raw, c.dqk, d.k_scale, 0.0f);
  fill_kv(d.v, d.v_raw, c.dvo, d.v_scale, c.v_offset);
  if (c.alibi) {
    d.alibi.resize(c.Hq);
    for (int h = 0; h < c.Hq; ++h) d.alibi[h] = std::pow(2.0f, -8.0f * (h + 1) / c.Hq);
  }
  if (c.sinks) {
    d.sinks.resize(c.Hq);
    for (float& x : d.sinks) x = u(rng) * 3.0f;
  }
  if (c.q_pos_offset) {
    d.q_pos.resize(B);
    for (int b = 0; b < B; ++b) d.q_pos[b] = std::max(0, c.kv_lens[b] - c.qo_lens[b] - 3 * b);
  }
  if (c.mask == MaskMode::kCustom) {
    d.mask_indptr.assign(B + 1, 0);
    for (int b = 0; b < B; ++b) {
      d.mask_indptr[b + 1] = d.mask_indptr[b] + static_cast<int64_t>(c.qo_lens[b]) * c.kv_lens[b];
    }
    d.mask_bits.assign(static_cast<size_t>((d.mask_indptr[B] + 7) / 8 + 1), 0);
    std::bernoulli_distribution keep(0.6);
    for (int64_t i = 0; i < d.mask_indptr[B]; ++i) {
      if (keep(rng)) d.mask_bits[static_cast<size_t>(i >> 3)] |= static_cast<uint8_t>(1u << (i & 7));
    }
  }
  // fp64 reference on exactly the values the kernel reads
  std::vector<double> q64(d.q.begin(), d.q.end()), k64(d.k.begin(), d.k.end()),
      v64(d.v.begin(), d.v.end()), al64(d.alibi.begin(), d.alibi.end()),
      sk64(d.sinks.begin(), d.sinks.end());
  d.o_ref.assign(static_cast<size_t>(d.total_q) * c.Hq * c.dvo, 0.0);
  d.lse_ref.assign(static_cast<size_t>(d.total_q) * c.Hq, 0.0);
  kops::ref::AttentionRefArgs ra;
  ra.q = q64.data();
  ra.k = k64.data();
  ra.v = v64.data();
  ra.qo_indptr = d.qo_indptr.data();
  ra.kv_indptr = d.kv_indptr.data();
  ra.q_pos_offset = c.q_pos_offset ? d.q_pos.data() : nullptr;
  ra.batch = B;
  ra.q_heads = c.Hq;
  ra.kv_heads = c.Hkv;
  ra.head_dim_qk = c.dqk;
  ra.head_dim_vo = c.dvo;
  ra.sm_scale = 1.0 / std::sqrt(static_cast<double>(c.dqk));
  ra.logits_softcap = c.softcap;
  ra.window_left = c.window_left;
  ra.window_right = c.window_right;
  ra.chunk_size = c.chunk;
  ra.num_sink_tokens = c.num_sink_tokens;
  ra.sink_logits = c.sinks ? sk64.data() : nullptr;
  ra.alibi_slopes = c.alibi ? al64.data() : nullptr;
  ra.causal = c.mask == MaskMode::kCausal;
  ra.custom_mask = c.mask == MaskMode::kCustom ? d.mask_bits.data() : nullptr;
  ra.mask_indptr = c.mask == MaskMode::kCustom ? d.mask_indptr.data() : nullptr;
  ra.o = d.o_ref.data();
  ra.lse = d.lse_ref.data();
  KOPS_EXPECT_OK(kops::ref::attention_f64(ra), "attention_f64");
  return d;
}

kops::attn::AttentionSpec spec_of(const AttnCase& c) {
  kops::attn::AttentionSpec s;
  s.num_q_heads = c.Hq;
  s.num_kv_heads = c.Hkv;
  s.head_dim_qk = c.dqk;
  s.head_dim_vo = c.dvo;
  s.q_dtype = s.o_dtype = c.dt;
  const DType kvdt = c.kv_dt == DType::kInvalid ? c.dt : c.kv_dt;
  s.kv_dtype = kvdt;
  if (kvdt != c.dt) {
    s.k_codec = s.v_codec = c.per_head_scale ? kops::kv::KVCodec::kFP8PerHead
                                             : kops::kv::KVCodec::kFP8PerTensor;
  }
  s.kv_layout = c.hnd ? kops::kv::KVLayout::kHND : kops::kv::KVLayout::kNHD;
  s.page_size = c.page_size;
  s.mask = c.mask;
  s.sliding_window = c.window_left >= 0 || c.window_right >= 0;
  s.softcap = c.softcap > 0.0f;
  s.alibi = c.alibi;
  s.sinks = c.sinks;
  s.chunked = c.chunk > 0;
  s.return_lse = true;
  return s;
}

/// Device buffers of one attention problem (ragged or paged).
struct AttnDev {
  DevArray<uint16_t> q;
  DevArray<uint16_t> o;
  DevArray<float> lse;
  DevArray<int32_t> qo_indptr, kv_indptr, q_pos;
  DevArray<uint8_t> k8, v8;          // FP8 caches
  DevArray<uint16_t> k16, v16;       // 16-bit K/V (ragged rows or pages)
  DevArray<int32_t> page_ids, page_indptr, last_len, block_table, seq_lens;
  DevArray<float> k_scale, v_scale, alibi, sinks;
  DevArray<uint8_t> mask_bits;
  DevArray<int64_t> mask_indptr;
  kops::kv::PagedKVDesc desc;
  kops::kv::RaggedKVDesc rdesc;
  AttnDev(const AttnCase& c, const AttnData& d, size_t pool_tokens)
      : q(d.q.size()), o(static_cast<size_t>(d.total_q) * c.Hq * c.dvo),
        lse(static_cast<size_t>(d.total_q) * c.Hq), qo_indptr(d.qo_indptr.size()),
        kv_indptr(d.kv_indptr.size()), q_pos(std::max<size_t>(d.q_pos.size(), 1)),
        k8(pool_tokens * c.Hkv * c.dqk), v8(pool_tokens * c.Hkv * c.dvo),
        k16(pool_tokens * c.Hkv * c.dqk), v16(pool_tokens * c.Hkv * c.dvo), page_ids(1),
        page_indptr(1), last_len(1), block_table(1), seq_lens(1),
        k_scale(std::max<size_t>(d.k_scale.size(), 1)), v_scale(std::max<size_t>(d.v_scale.size(), 1)),
        alibi(std::max<size_t>(d.alibi.size(), 1)), sinks(std::max<size_t>(d.sinks.size(), 1)),
        mask_bits(std::max<size_t>(d.mask_bits.size(), 1)),
        mask_indptr(std::max<size_t>(d.mask_indptr.size(), 1)) {}
};

void upload(void* dst, const void* src, size_t bytes) {
  if (bytes != 0) cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

/// Run one case end to end; `only` restricts the batch to one request (batch-invariance test).
AttnRun run_prefill(const AttnCase& c, const AttnData& d, unsigned page_seed) {
  AttnRun res;
  const int B = static_cast<int>(c.qo_lens.size());
  const bool paged = c.page_size > 0;
  const DType kvdt = c.kv_dt == DType::kInvalid ? c.dt : c.kv_dt;
  const bool fp8 = kvdt != c.dt;
  // Paged pool: requests get shuffled pages; two spare pages make ids non-contiguous.
  std::vector<std::vector<int32_t>> pages(B);
  int num_pages = 0;
  if (paged) {
    for (int b = 0; b < B; ++b) num_pages += (c.kv_lens[b] + c.page_size - 1) / c.page_size;
    num_pages += 2;
    std::vector<int32_t> perm(num_pages);
    for (int i = 0; i < num_pages; ++i) perm[i] = i;
    std::mt19937 rng(page_seed);
    std::shuffle(perm.begin(), perm.end(), rng);
    int at = 0;
    for (int b = 0; b < B; ++b) {
      const int np = (c.kv_lens[b] + c.page_size - 1) / c.page_size;
      for (int i = 0; i < np; ++i) pages[b].push_back(perm[at++]);
    }
  }
  const size_t pool_tokens = paged ? static_cast<size_t>(num_pages) * c.page_size : d.total_kv;
  AttnDev dev(c, d, std::max<size_t>(pool_tokens, 1));
  upload(dev.q.p, pack16(c.dt, d.q).data(), d.q.size() * 2);
  upload(dev.qo_indptr.p, d.qo_indptr.data(), d.qo_indptr.size() * 4);
  upload(dev.kv_indptr.p, d.kv_indptr.data(), d.kv_indptr.size() * 4);
  if (!d.q_pos.empty()) upload(dev.q_pos.p, d.q_pos.data(), d.q_pos.size() * 4);
  if (!d.alibi.empty()) upload(dev.alibi.p, d.alibi.data(), d.alibi.size() * 4);
  if (!d.sinks.empty()) upload(dev.sinks.p, d.sinks.data(), d.sinks.size() * 4);
  if (!d.mask_bits.empty()) upload(dev.mask_bits.p, d.mask_bits.data(), d.mask_bits.size());
  if (!d.mask_indptr.empty()) upload(dev.mask_indptr.p, d.mask_indptr.data(), d.mask_indptr.size() * 8);
  cudaMemset(dev.o.p, 0xFF, dev.o.n * 2);  // NaN-fill: every row must be written
  cudaMemset(dev.lse.p, 0xFF, dev.lse.n * 4);

  // K / V placement: element (token t of request b, head h, dim x).
  auto place = [&](const std::vector<float>& src, int dim, bool is_k) {
    const size_t n = pool_tokens * c.Hkv * dim;
    std::vector<float> img(n, 0.0f);
    for (int b = 0; b < B; ++b) {
      for (int t = 0; t < c.kv_lens[b]; ++t) {
        for (int h = 0; h < c.Hkv; ++h) {
          const size_t s0 = (static_cast<size_t>(d.kv_indptr[b] + t) * c.Hkv + h) * dim;
          size_t d0;
          if (!paged) {
            d0 = s0;
          } else {
            const size_t pg = pages[b][t / c.page_size], off = t % c.page_size;
            d0 = c.hnd ? ((pg * c.Hkv + h) * c.page_size + off) * dim
                       : ((pg * c.page_size + off) * c.Hkv + h) * dim;
          }
          for (int x = 0; x < dim; ++x) img[d0 + x] = src[s0 + x];
        }
      }
    }
    if (fp8) {
      std::vector<uint8_t> b8(n);
      for (size_t i = 0; i < n; ++i) b8[i] = fp8_bits(kvdt, img[i]);
      upload(is_k ? dev.k8.p : dev.v8.p, b8.data(), n);
    } else {
      upload(is_k ? dev.k16.p : dev.v16.p, pack16(c.dt, img).data(), n * 2);
    }
  };
  place(fp8 ? d.k_raw : d.k, c.dqk, true);
  place(fp8 ? d.v_raw : d.v, c.dvo, false);

  std::vector<int32_t> page_ids, page_indptr(B + 1, 0), last_len(B), bt, seq(c.kv_lens.begin(), c.kv_lens.end());
  int max_pages = 1;
  if (paged) {
    for (int b = 0; b < B; ++b) {
      page_indptr[b + 1] = page_indptr[b] + static_cast<int32_t>(pages[b].size());
      for (int32_t p : pages[b]) page_ids.push_back(p);
      last_len[b] = c.kv_lens[b] == 0 ? 0 : (c.kv_lens[b] - 1) % c.page_size + 1;
      max_pages = std::max<int>(max_pages, static_cast<int>(pages[b].size()));
    }
    bt.assign(static_cast<size_t>(B) * max_pages, 0);
    for (int b = 0; b < B; ++b) {
      for (size_t i = 0; i < pages[b].size(); ++i) bt[static_cast<size_t>(b) * max_pages + i] = pages[b][i];
    }
  }
  DevArray<int32_t> d_page_ids(std::max<size_t>(page_ids.size(), 1)), d_page_indptr(B + 1),
      d_last(std::max(B, 1)), d_bt(std::max<size_t>(bt.size(), 1)), d_seq(std::max(B, 1));
  upload(d_page_ids.p, page_ids.data(), page_ids.size() * 4);
  upload(d_page_indptr.p, page_indptr.data(), page_indptr.size() * 4);
  upload(d_last.p, last_len.data(), last_len.size() * 4);
  upload(d_bt.p, bt.data(), bt.size() * 4);
  upload(d_seq.p, seq.data(), seq.size() * 4);
  if (fp8) {
    upload(dev.k_scale.p, d.k_scale.data(), d.k_scale.size() * 4);
    upload(dev.v_scale.p, d.v_scale.data(), d.v_scale.size() * 4);
  }

  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return res;
  ctx.compute_units = c.compute_units;
  ctx.batch_invariant = c.batch_invariant;
  const kops::attn::AttentionSpec spec = spec_of(c);
  std::vector<int32_t> kvl(c.kv_lens.begin(), c.kv_lens.end());
  std::vector<int32_t> qoi(d.qo_indptr);
  Tensor qo_t = tensor1d(qoi.data(), DType::kInt32, B + 1);
  qo_t.device = -1;
  Tensor kv_t = tensor1d(kvl.data(), DType::kInt32, B);
  kv_t.device = -1;
  kops::attn::AttentionPlan plan;
  kops::Workspace ws{};
  const int max_kv = c.kv_lens.empty() ? 0 : *std::max_element(c.kv_lens.begin(), c.kv_lens.end());
  auto do_plan = [&]() {
    return c.decode ? kops::attn::decode_plan(spec, B, max_kv, c.q_len, c.device_planner, ctx, ws, &plan)
                    : kops::attn::prefill_plan(spec, qo_t, kv_t, B, ctx, ws, &plan);
  };
  Status s = do_plan();
  if (s != Status::kWorkspaceTooSmall) {
    KOPS_EXPECT_STATUS(s, Status::kWorkspaceTooSmall, c.name);
    return res;
  }
  DevArray<unsigned char> persistent(plan.workspace.device_persistent_bytes);
  DevArray<unsigned char> scratch(std::max<size_t>(plan.workspace.device_scratch_bytes, 1));
  ws.device_persistent = persistent.p;
  ws.device_persistent_bytes = plan.workspace.device_persistent_bytes;
  ws.device_scratch = scratch.p;
  ws.device_scratch_bytes = plan.workspace.device_scratch_bytes;
  s = do_plan();
  KOPS_EXPECT_OK(s, c.name);
  if (s != Status::kSuccess) return res;
  res.split = plan.split_kv;
  res.kernel_id = plan.kernel_id;

  kops::attn::AttentionModifiers mod;
  mod.sm_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(c.dqk)));
  mod.logits_softcap = c.softcap;
  mod.window_left = c.window_left;
  mod.window_right = c.window_right;
  mod.chunk_size = c.chunk;
  mod.num_sink_tokens = c.num_sink_tokens;
  mod.alibi_slopes = c.alibi ? dev.alibi.p : nullptr;
  mod.sink_logits = c.sinks ? dev.sinks.p : nullptr;
  auto common = [&](auto& p) {  // fields shared by the prefill and decode params
    p.q = dev.q.p;
    p.q_stride_n = static_cast<int64_t>(c.Hq) * c.dqk;
    p.q_stride_h = c.dqk;
    p.mod = mod;
    p.o = dev.o.p;
    p.o_stride_n = static_cast<int64_t>(c.Hq) * c.dvo;
    p.o_stride_h = c.dvo;
    p.lse = dev.lse.p;
    p.batch_size = B;
    p.num_qo_heads = c.Hq;
    p.num_kv_heads = c.Hkv;
  };
  auto prefill_common = [&](auto& p) {
    common(p);
    p.qo_indptr = dev.qo_indptr.p;
    p.q_pos_offset = c.q_pos_offset ? dev.q_pos.p : nullptr;
    p.custom_mask = c.mask == MaskMode::kCustom ? dev.mask_bits.p : nullptr;
    p.mask_indptr = c.mask == MaskMode::kCustom ? dev.mask_indptr.p : nullptr;
    p.total_q = d.total_q;
  };
  if (!paged) {
    kops::attn::RaggedPrefillParams p{};
    prefill_common(p);
    p.kv.k = dev.k16.p;
    p.kv.v = dev.v16.p;
    p.kv.k_stride_n = static_cast<int64_t>(c.Hkv) * c.dqk;
    p.kv.k_stride_h = c.dqk;
    p.kv.v_stride_n = static_cast<int64_t>(c.Hkv) * c.dvo;
    p.kv.v_stride_h = c.dvo;
    p.kv.kv_indptr = dev.kv_indptr.p;
    s = kops::attn::ragged_prefill_run(plan, p, ws, ctx);
  } else {
    kops::attn::PagedPrefillParams p{};
    prefill_common(p);
    kops::kv::PagedKVDesc& k = p.kv;
    k.k_data = fp8 ? static_cast<void*>(dev.k8.p) : static_cast<void*>(dev.k16.p);
    k.v_data = fp8 ? static_cast<void*>(dev.v8.p) : static_cast<void*>(dev.v16.p);
    k.k_dtype = k.v_dtype = kvdt;
    k.k_codec = k.v_codec = spec.k_codec;
    k.layout = spec.kv_layout;
    k.page_size = c.page_size;
    k.num_kv_heads = c.Hkv;
    k.head_dim_k = c.dqk;
    k.head_dim_v = c.dvo;
    const int64_t ps = c.page_size;
    k.k_stride_page = ps * c.Hkv * c.dqk;
    k.v_stride_page = ps * c.Hkv * c.dvo;
    k.k_stride_token = c.hnd ? c.dqk : static_cast<int64_t>(c.Hkv) * c.dqk;
    k.v_stride_token = c.hnd ? c.dvo : static_cast<int64_t>(c.Hkv) * c.dvo;
    k.k_stride_head = c.hnd ? ps * c.dqk : c.dqk;
    k.v_stride_head = c.hnd ? ps * c.dvo : c.dvo;
    if (c.block_table) {
      k.table_format = kops::kv::PageTableFormat::kBlockTable;
      k.block_table = d_bt.p;
      k.block_table_stride = max_pages;
      k.seq_lens = d_seq.p;
    } else {
      k.table_format = kops::kv::PageTableFormat::kCSR;
      k.kv_indptr = d_page_indptr.p;
      k.kv_page_indices = d_page_ids.p;
      k.kv_last_page_len = d_last.p;
    }
    if (fp8) {
      k.k_scale = dev.k_scale.p;
      k.v_scale = dev.v_scale.p;
      k.scale_head_stride = c.per_head_scale ? 1 : 0;
    }
    if (c.decode) {
      kops::attn::DecodeParams dp{};
      common(dp);
      dp.q_len = c.q_len;
      dp.kv = p.kv;
      s = kops::attn::decode_run(plan, dp, ws, ctx);
      if (c.device_planner && s == Status::kSuccess) {  // replay-safety of the task counter
        s = kops::attn::decode_run(plan, dp, ws, ctx);
      }
    } else {
      s = kops::attn::paged_prefill_run(plan, p, ws, ctx);
    }
  }
  KOPS_EXPECT_OK(s, c.name);
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    kops_test::record_failure(c.name, cudaGetErrorString(err), __FILE__, __LINE__);
    return res;
  }
  const std::vector<uint16_t> o16 = kops_test::copy_out(dev.o.p, dev.o.n);
  res.o.resize(o16.size());
  for (size_t i = 0; i < o16.size(); ++i) res.o[i] = unpack16(c.dt, o16[i]);
  res.lse = kops_test::copy_out(dev.lse.p, dev.lse.n);
  res.ok = s == Status::kSuccess;
  return res;
}

void check_attn(const AttnCase& c, const AttnData& d, const AttnRun& r, const char* tag) {
  if (!r.ok) return;
  char what[160];
  std::snprintf(what, sizeof(what), "%s%s o", c.name, tag);
  kops_test::compare(r.o, d.o_ref, c.atol, c.rtol, what);
  std::snprintf(what, sizeof(what), "%s%s lse", c.name, tag);
  kops_test::compare(r.lse, d.lse_ref, c.lse_atol, 1e-5, what);
  // compare() lets NaN through (NaN > tol is false): rows the kernel never wrote stay NaN-filled.
  size_t lost = 0;
  for (size_t i = 0; i < r.o.size(); ++i) lost += std::isfinite(d.o_ref[i]) && !std::isfinite(r.o[i]);
  for (size_t i = 0; i < r.lse.size(); ++i) lost += !std::isnan(d.lse_ref[i]) && std::isnan(r.lse[i]);
  std::snprintf(what, sizeof(what), "%s%s: %zu outputs not written (NaN)", c.name, tag, lost);
  if (lost != 0) kops_test::record_failure(what, nullptr, __FILE__, __LINE__);
}

void run_attn_case(const AttnCase& c, unsigned seed) {
  const AttnData d = make_attn_data(c, seed);
  const AttnRun r = run_prefill(c, d, seed + 17);
  check_attn(c, d, r, "");
}

AttnCase ragged_case(const char* name, int dqk, int dvo, int Hq, int Hkv) {
  AttnCase c;
  c.name = name;
  c.qo_lens = {37, 1, 130, 64};
  c.kv_lens = {37, 70, 200, 64};
  c.Hq = Hq;
  c.Hkv = Hkv;
  c.dqk = dqk;
  c.dvo = dvo;
  return c;
}

void test_prefill_head_dims() {
  const int dims[][2] = {{64, 64}, {80, 80}, {96, 96}, {128, 128}, {256, 256}, {192, 128}};
  unsigned seed = 100;
  for (const auto& dd : dims) {
    for (int g : {1, 4}) {
      char name[64];
      std::snprintf(name, sizeof(name), "prefill ragged d%d/%d G%d", dd[0], dd[1], g);
      AttnCase c = ragged_case(name, dd[0], dd[1], 2 * g, 2);
      run_attn_case(c, seed++);
    }
  }
  for (int d16 : {128, 80}) {  // fp16: tighter tolerance
    AttnCase f = ragged_case(d16 == 128 ? "prefill ragged fp16 d128 G4" : "prefill ragged fp16 d80 G2", d16, d16,
                             d16 == 128 ? 8 : 4, 2);
    f.dt = DType::kFloat16;
    f.atol = f.rtol = 2e-3;
    run_attn_case(f, seed++);
  }
  // An FP8 v_scale error only shows on outputs with a large mean: fp16, V offset 3, tight tolerance.
  AttnCase p8 = ragged_case("prefill paged e4m3 fp16 V+3", 128, 128, 8, 2);
  p8.dt = DType::kFloat16;
  p8.kv_dt = DType::kFloat8E4M3;
  p8.page_size = 16;
  p8.v_offset = 3.0f;
  p8.atol = p8.rtol = 2e-3;
  run_attn_case(p8, seed++);
}

/// One AttentionModifiers rule set (applied on top of a base case).
struct ModRule {
  const char* tag;
  MaskMode mask;
  int window_left, window_right, num_sink_tokens, chunk;
  float softcap;
  bool alibi, sinks;
};
const ModRule kModRules[] = {
    {"window", MaskMode::kCausal, 37, -1, 0, 0, 0.0f, false, false},
    {"window+sink tokens", MaskMode::kCausal, 50, -1, 4, 0, 0.0f, false, false},
    {"softcap", MaskMode::kCausal, -1, -1, 0, 0, 3.0f, false, false},
    {"alibi", MaskMode::kCausal, -1, -1, 0, 0, 0.0f, true, false},
    {"sinks + SWA", MaskMode::kCausal, 127, -1, 0, 0, 0.0f, false, true},
    {"chunked", MaskMode::kCausal, -1, -1, 0, 48, 0.0f, false, false},
    {"no mask + bidir window", MaskMode::kNone, 20, 11, 0, 0, 0.0f, false, false},
    {"custom mask", MaskMode::kCustom, -1, -1, 0, 0, 0.0f, false, false},
};

AttnCase with_rule(AttnCase c, const ModRule& m) {
  c.mask = m.mask;
  c.window_left = m.window_left;
  c.window_right = m.window_right;
  c.num_sink_tokens = m.num_sink_tokens;
  c.chunk = m.chunk;
  c.softcap = m.softcap;
  c.alibi = m.alibi;
  c.sinks = m.sinks;
  return c;
}

void test_prefill_modifiers() {  // every rule x every compiled head-dim pair
  unsigned seed = 200;
  const int dims[][2] = {{64, 64}, {80, 80}, {96, 96}, {128, 128}, {256, 256}, {192, 128}};
  for (const auto& dd : dims) {
    for (const ModRule& m : kModRules) {
      char name[96];
      std::snprintf(name, sizeof(name), "prefill %s d%d/%d", m.tag, dd[0], dd[1]);
      AttnCase c = with_rule(ragged_case(name, dd[0], dd[1], 8, 2), m);
      c.qo_lens = {150, 9, 77};
      c.kv_lens = {300, 9, 140};
      run_attn_case(c, seed++);
    }
  }
  AttnCase c = ragged_case("prefill q_pos_offset + all modifiers", 128, 128, 8, 2);
  c.qo_lens = {40, 70};
  c.kv_lens = {200, 90};
  c.q_pos_offset = true;
  c.window_left = 60;
  c.num_sink_tokens = 3;
  c.softcap = 5.0f;
  c.alibi = true;
  c.sinks = true;
  c.chunk = 64;
  run_attn_case(c, seed++);
  // Fully masked rows: qo_len > kv_len puts the first rows at negative positions.
  AttnCase m = ragged_case("prefill masked rows (qo > kv)", 64, 64, 2, 1);
  m.qo_lens = {20};
  m.kv_lens = {5};
  run_attn_case(m, seed++);
  m.name = "prefill masked rows + sinks";
  m.sinks = true;
  run_attn_case(m, seed++);
}

void test_prefill_paged() {
  unsigned seed = 300;
  struct P {
    int ps;
    bool hnd, bt;
    DType kv;
    bool per_head;
    int dim;
  };
  const P cases[] = {
      {16, false, false, DType::kInvalid, false, 128}, {1, false, true, DType::kInvalid, false, 64},
      {64, true, false, DType::kInvalid, false, 128},  {32, true, true, DType::kInvalid, false, 96},
      {16, false, false, DType::kFloat8E4M3, false, 128}, {16, true, true, DType::kFloat8E4M3, true, 128},
      {32, false, true, DType::kFloat8E5M2, false, 64}, {16, false, false, DType::kFloat8E4M3, true, 256},
  };
  for (const P& pc : cases) {
    char name[96];
    std::snprintf(name, sizeof(name), "prefill paged ps%d %s %s %s%s d%d", pc.ps, pc.hnd ? "HND" : "NHD",
                  pc.bt ? "block-table" : "CSR",
                  pc.kv == DType::kInvalid ? "bf16" : (pc.kv == DType::kFloat8E4M3 ? "e4m3" : "e5m2"),
                  pc.per_head ? " per-head" : "", pc.dim);
    AttnCase c = ragged_case(name, pc.dim, pc.dim, 8, 2);
    c.qo_lens = {17, 1, 90};
    c.kv_lens = {120, 33, 90};
    c.page_size = pc.ps;
    c.hnd = pc.hnd;
    c.block_table = pc.bt;
    c.kv_dt = pc.kv;
    c.per_head_scale = pc.per_head;
    run_attn_case(c, seed++);
  }
  // Verification (A06): 5 draft tokens + tree mask over a cached prefix.
  AttnCase v = ragged_case("prefill paged verify tree mask", 128, 128, 8, 1);
  v.qo_lens = {6, 6};
  v.kv_lens = {100, 260};
  v.page_size = 16;
  v.mask = MaskMode::kCustom;
  run_attn_case(v, seed++);
}

void test_prefill_split_and_invariance() {
  // Split-KV vs one pass: same request, tolerance only.
  AttnCase c = ragged_case("prefill split-kv", 128, 128, 4, 1);
  c.qo_lens = {16};
  c.kv_lens = {3000};
  c.sinks = true;
  c.window_left = 2000;
  c.compute_units = 64;
  const AttnData d = make_attn_data(c, 401);
  const AttnRun r1 = run_prefill(c, d, 9);
  KOPS_TEST_CHECK(r1.split);
  check_attn(c, d, r1, " (split)");
  const AttnRun r2 = run_prefill(c, d, 9);
  KOPS_TEST_CHECK(r1.o == r2.o && r1.lse == r2.lse);  // deterministic: bit-identical rerun
  AttnCase nc = c;
  nc.compute_units = 1;
  const AttnRun r3 = run_prefill(nc, d, 9);
  KOPS_TEST_CHECK(!r3.split);
  check_attn(nc, d, r3, " (one pass)");

  // batch_invariant: a request alone and inside a batch gives the same bits.
  AttnCase big = ragged_case("prefill batch-invariant", 128, 128, 8, 2);
  big.qo_lens = {33, 70, 5};
  big.kv_lens = {5000, 70, 9000};
  big.batch_invariant = true;
  big.compute_units = 64;
  const AttnData db = make_attn_data(big, 402);
  const AttnRun rb = run_prefill(big, db, 3);
  check_attn(big, db, rb, " (batch)");
  AttnCase one = big;
  one.qo_lens = {big.qo_lens[2]};
  one.kv_lens = {big.kv_lens[2]};
  AttnData d1 = make_attn_data(one, 403);
  // Same inputs as request 2 of the batch.
  const size_t qrow = static_cast<size_t>(big.Hq) * big.dqk;
  d1.q.assign(db.q.begin() + db.qo_indptr[2] * qrow, db.q.begin() + db.qo_indptr[3] * qrow);
  const size_t krow = static_cast<size_t>(big.Hkv) * big.dqk, vrow = static_cast<size_t>(big.Hkv) * big.dvo;
  d1.k.assign(db.k.begin() + db.kv_indptr[2] * krow, db.k.begin() + db.kv_indptr[3] * krow);
  d1.v.assign(db.v.begin() + db.kv_indptr[2] * vrow, db.v.begin() + db.kv_indptr[3] * vrow);
  d1.k_raw = d1.k;
  d1.v_raw = d1.v;
  const AttnRun r_one = run_prefill(one, d1, 4);
  const size_t orow = static_cast<size_t>(big.Hq) * big.dvo;
  bool same = r_one.ok && rb.ok;
  for (size_t i = 0; same && i < r_one.o.size(); ++i) {
    const float a = r_one.o[i], b = rb.o[db.qo_indptr[2] * orow + i];
    same = std::memcmp(&a, &b, 4) == 0;
  }
  KOPS_TEST_CHECK(same);
}

void test_prefill_models() {
  unsigned seed = 500;
  AttnCase g = ragged_case("model Gemma2 softcap 50 d256", 256, 256, 8, 4);
  g.softcap = 50.0f;
  g.window_left = 4096;
  run_attn_case(g, seed++);
  AttnCase o = ragged_case("model GPT-OSS sinks + SWA 128 d64", 64, 64, 8, 1);
  o.qo_lens = {300, 20};
  o.kv_lens = {300, 400};
  o.sinks = true;
  o.window_left = 127;  // HF sliding_window 128 covers the query and 127 keys before it
  run_attn_case(o, seed++);
  AttnCase l = ragged_case("model Llama4 chunk 8192", 128, 128, 4, 1);
  l.qo_lens = {96};
  l.kv_lens = {8192 + 40};
  l.chunk = 8192;
  l.page_size = 16;
  run_attn_case(l, seed++);
  AttnCase m = ragged_case("model Mistral SWA d128", 128, 128, 8, 2);
  m.qo_lens = {200};
  m.kv_lens = {600};
  m.window_left = 255;
  run_attn_case(m, seed++);
  AttnCase p = ragged_case("model Phi-3 SWA d96", 96, 96, 4, 4);
  p.qo_lens = {100, 3};
  p.kv_lens = {300, 50};
  p.window_left = 127;
  run_attn_case(p, seed++);
  AttnCase b = ragged_case("model BLOOM ALiBi d64", 64, 64, 8, 8);
  b.alibi = true;
  run_attn_case(b, seed++);
}

/// Graph strategy: plan at capacity, capture ragged_prefill_run once (device planner + kernel +
/// merge), replay with different device lengths and data in the same buffers.
void test_prefill_graph() {
  for (int inv = 0; inv < 2; ++inv) {
    AttnCase cap = ragged_case(inv ? "prefill graph batch-invariant" : "prefill graph split", 128, 128, 8, 2);
    cap.qo_lens = {40, 10, 70};
    cap.kv_lens = {300, 10, 5000};
    cap.batch_invariant = inv != 0;
    cap.compute_units = 64;
    cap.sinks = true;
    const int B = 3;
    const AttnData dcap = make_attn_data(cap, 600 + inv);
    DevArray<uint16_t> q(dcap.q.size()), k(dcap.k.size()), v(dcap.v.size()),
        o(static_cast<size_t>(dcap.total_q) * cap.Hq * cap.dvo);
    DevArray<float> lse(static_cast<size_t>(dcap.total_q) * cap.Hq), sinks(cap.Hq);
    DevArray<int32_t> qo(B + 1), kvi(B + 1);
    LaunchContext ctx{};
    if (!kops_test::make_context(&ctx)) return;
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    ctx.stream = reinterpret_cast<kops::kops_stream_t>(stream);
    ctx.compute_units = cap.compute_units;
    ctx.batch_invariant = cap.batch_invariant;
    ctx.strategy = kops::ExecutionStrategy::kGraph;
    ctx.capture_generation = 7;
    std::vector<int32_t> qoi(dcap.qo_indptr), kvl(cap.kv_lens.begin(), cap.kv_lens.end());
    Tensor qo_t = tensor1d(qoi.data(), DType::kInt32, B + 1);
    qo_t.device = -1;
    Tensor kv_t = tensor1d(kvl.data(), DType::kInt32, B);
    kv_t.device = -1;
    const kops::attn::AttentionSpec spec = spec_of(cap);
    kops::attn::AttentionPlan plan;
    kops::Workspace ws{};
    KOPS_EXPECT_STATUS(kops::attn::prefill_plan(spec, qo_t, kv_t, B, ctx, ws, &plan),
                       Status::kWorkspaceTooSmall, cap.name);
    DevArray<unsigned char> pers(plan.workspace.device_persistent_bytes),
        scr(std::max<size_t>(plan.workspace.device_scratch_bytes, 1));
    ws.device_persistent = pers.p;
    ws.device_persistent_bytes = plan.workspace.device_persistent_bytes;
    ws.device_scratch = scr.p;
    ws.device_scratch_bytes = plan.workspace.device_scratch_bytes;
    KOPS_EXPECT_OK(kops::attn::prefill_plan(spec, qo_t, kv_t, B, ctx, ws, &plan), cap.name);
    KOPS_TEST_CHECK(plan.split_kv && plan.num_launches == 3);

    kops::attn::RaggedPrefillParams p{};
    p.q = q.p;
    p.q_stride_n = static_cast<int64_t>(cap.Hq) * cap.dqk;
    p.q_stride_h = cap.dqk;
    p.qo_indptr = qo.p;
    p.kv.k = k.p;
    p.kv.v = v.p;
    p.kv.k_stride_n = static_cast<int64_t>(cap.Hkv) * cap.dqk;
    p.kv.k_stride_h = cap.dqk;
    p.kv.v_stride_n = static_cast<int64_t>(cap.Hkv) * cap.dvo;
    p.kv.v_stride_h = cap.dvo;
    p.kv.kv_indptr = kvi.p;
    p.mod.sm_scale = static_cast<float>(1.0 / std::sqrt(128.0));
    p.mod.sink_logits = sinks.p;
    p.o = o.p;
    p.o_stride_n = static_cast<int64_t>(cap.Hq) * cap.dvo;
    p.o_stride_h = cap.dvo;
    p.lse = lse.p;
    p.batch_size = B;
    p.num_qo_heads = cap.Hq;
    p.num_kv_heads = cap.Hkv;
    p.total_q = dcap.total_q;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    const Status rs = kops::attn::ragged_prefill_run(plan, p, ws, ctx);
    cudaStreamEndCapture(stream, &graph);
    KOPS_EXPECT_OK(rs, cap.name);
    if (rs != Status::kSuccess || graph == nullptr ||
        cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
      kops_test::record_failure(cap.name, "graph capture failed", __FILE__, __LINE__);
      cudaStreamDestroy(stream);
      return;
    }
    const std::vector<std::vector<int>> qlens = {cap.qo_lens, {5, 33, 1}, {0, 64, 12}};
    const std::vector<std::vector<int>> klens = {cap.kv_lens, {4999, 40, 7}, {9, 64, 4096}};
    for (size_t r = 0; r < qlens.size(); ++r) {
      AttnCase sc = cap;
      sc.qo_lens = qlens[r];
      sc.kv_lens = klens[r];
      const AttnData d = make_attn_data(sc, 610 + static_cast<unsigned>(r) + 10 * inv);
      upload(q.p, pack16(sc.dt, d.q).data(), d.q.size() * 2);
      upload(k.p, pack16(sc.dt, d.k).data(), d.k.size() * 2);
      upload(v.p, pack16(sc.dt, d.v).data(), d.v.size() * 2);
      upload(qo.p, d.qo_indptr.data(), (B + 1) * 4);
      upload(kvi.p, d.kv_indptr.data(), (B + 1) * 4);
      upload(sinks.p, d.sinks.data(), d.sinks.size() * 4);
      cudaMemset(o.p, 0xFF, o.n * 2);
      cudaGraphLaunch(exec, stream);
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        kops_test::record_failure(cap.name, "graph replay failed", __FILE__, __LINE__);
        break;
      }
      const std::vector<uint16_t> o16 = kops_test::copy_out(o.p, static_cast<size_t>(d.total_q) * sc.Hq * sc.dvo);
      AttnRun res;
      res.o.resize(o16.size());
      for (size_t i = 0; i < o16.size(); ++i) res.o[i] = unpack16(sc.dt, o16[i]);
      res.lse = kops_test::copy_out(lse.p, static_cast<size_t>(d.total_q) * sc.Hq);
      res.ok = true;
      char tag[32];
      std::snprintf(tag, sizeof(tag), " replay %zu", r);
      check_attn(sc, d, res, tag);
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
  }
}

// ---- decode (A03 / A05 / A06 / A08) -------------------------------------------------------------

/// decode_plan path recorded in the plan's kernel_id: 0 CUDA core, 1 tensor core, 2 dynamic.
int decode_path(const AttnRun& r) { return static_cast<int>((r.kernel_id >> 52) & 7); }

AttnCase decode_case(const char* name, int dim, int Hq, int Hkv, int q_len, std::vector<int> kv_lens) {
  AttnCase c;
  c.name = name;
  c.decode = true;
  c.q_len = q_len;
  c.kv_lens = std::move(kv_lens);
  c.qo_lens.assign(c.kv_lens.size(), q_len);
  c.Hq = Hq;
  c.Hkv = Hkv;
  c.dqk = c.dvo = dim;
  c.page_size = 16;
  return c;
}

void run_decode_case(const AttnCase& c, unsigned seed, int want_path) {
  const AttnData d = make_attn_data(c, seed);
  const AttnRun r = run_prefill(c, d, seed + 5);
  if (r.ok && want_path >= 0 && decode_path(r) != want_path) {
    char msg[96];
    std::snprintf(msg, sizeof(msg), "path %d, expected %d", decode_path(r), want_path);
    kops_test::record_failure(c.name, msg, __FILE__, __LINE__);
  }
  check_attn(c, d, r, "");
}

void test_decode_dims_and_paths() {
  unsigned seed = 700;
  for (int dim : {64, 80, 96, 128, 192, 256}) {
    const bool tc = dim == 64 || dim == 128 || dim == 256;
    char n1[64], n2[64], n3[64];
    std::snprintf(n1, sizeof(n1), "decode simt d%d MHA", dim);
    run_decode_case(decode_case(n1, dim, 4, 4, 1, {1, 37, 300, 129}), seed++, 0);
    std::snprintf(n2, sizeof(n2), "decode d%d GQA8 q_len 1", dim);
    run_decode_case(decode_case(n2, dim, 16, 2, 1, {5, 260, 17}), seed++, tc ? 1 : 0);
    std::snprintf(n3, sizeof(n3), "decode d%d GQA4 verify q_len 3", dim);
    run_decode_case(decode_case(n3, dim, 8, 2, 3, {3, 90, 400}), seed++, tc ? 1 : 0);
  }
  AttnCase m = decode_case("decode tc G16 x q_len 4 (4 row blocks)", 128, 32, 2, 4, {50, 777});
  run_decode_case(m, seed++, 1);
  AttnCase h = decode_case("decode fp16 tc d128", 128, 8, 1, 1, {1000, 3});
  h.dt = DType::kFloat16;
  h.atol = h.rtol = 2e-3;
  run_decode_case(h, seed++, 1);
  AttnCase hs = decode_case("decode fp16 simt d96", 96, 2, 2, 1, {1000, 3});
  hs.dt = DType::kFloat16;
  hs.atol = hs.rtol = 2e-3;
  run_decode_case(hs, seed++, 0);
  for (int g : {1, 8}) {  // FP8 v_scale check (fp16, V offset 3) on the CUDA-core and tensor-core paths
    AttnCase f8 = decode_case(g == 1 ? "decode simt e4m3 fp16 V+3" : "decode tc e4m3 fp16 V+3", 128, 2 * g, 2, 1,
                              {700, 5});
    f8.dt = DType::kFloat16;
    f8.kv_dt = DType::kFloat8E4M3;
    f8.v_offset = 3.0f;
    f8.atol = f8.rtol = 2e-3;
    run_decode_case(f8, seed++, g == 1 ? 0 : 1);
  }
  AttnCase z = decode_case("decode empty request", 64, 4, 4, 1, {0, 9});
  run_decode_case(z, seed++, 0);
}

void test_decode_modifiers_and_caches() {
  unsigned seed = 800;
  struct Cache {
    const char* tag;
    DType kv;
    bool hnd, bt, per_head;
    int ps;
  };
  const Cache caches[] = {{"fp8 e4m3 HND block-table", DType::kFloat8E4M3, true, true, false, 16},
                          {"fp8 e5m2 per-head ps1", DType::kFloat8E5M2, false, false, true, 1},
                          {"bf16 ps64", DType::kInvalid, false, false, false, 64}};
  for (int g = 0; g < 2; ++g) {  // g = 0: CUDA core (MHA), 1: tensor core (GQA 8)
    const int Hq = g ? 16 : 4, Hkv = g ? 2 : 4;
    for (const ModRule& m : kModRules) {
      if (m.mask != MaskMode::kCausal) continue;  // decode is causal among its q_len rows
      char name[96];
      std::snprintf(name, sizeof(name), "decode %s %s", g ? "tc" : "simt", m.tag);
      AttnCase c = with_rule(decode_case(name, 128, Hq, Hkv, 2, {300, 41, 1}), m);
      run_decode_case(c, seed++, g);
    }
    for (const Cache& k : caches) {
      char name[96];
      std::snprintf(name, sizeof(name), "decode %s %s + sinks", g ? "tc" : "simt", k.tag);
      AttnCase c = decode_case(name, 128, Hq, Hkv, 2, {300, 41, 1});
      c.kv_dt = k.kv;
      c.hnd = k.hnd;
      c.block_table = k.bt;
      c.per_head_scale = k.per_head;
      c.page_size = k.ps;
      c.sinks = true;
      run_decode_case(c, seed++, g);
    }
  }
  AttnCase f = decode_case("decode simt fp8 d80", 80, 4, 4, 1, {200, 33});
  f.kv_dt = DType::kFloat8E4M3;
  run_decode_case(f, seed++, 0);
  // A 16-key tile that straddles a page boundary: the sliding window moves the first visible key off
  // the tile grid, so the tensor-core V/K addressing has to fall back from one page base per tile to
  // a per-row page lookup in the middle of the tile (page 4 offset 49 of the 64-token page, tile
  // 305..320 crosses into page 5).
  for (DType pad : {DType::kInvalid, DType::kFloat8E4M3}) {
    AttnCase s = decode_case(pad == DType::kInvalid ? "decode tc page-straddling tile"
                                                    : "decode tc fp8 page-straddling tile",
                             128, 16, 2, 1, {380, 70});
    s.page_size = 64;
    s.window_left = 74;
    s.kv_dt = pad;
    run_decode_case(s, seed++, 1);
  }
}

void test_decode_split_dynamic() {
  unsigned seed = 900;
  for (int g = 0; g < 2; ++g) {
    AttnCase c = decode_case(g ? "decode tc split" : "decode simt split", 128, g ? 8 : 2, g ? 1 : 2, 1, {4000, 9, 2500});
    c.compute_units = 64;
    c.sinks = true;
    const AttnData d = make_attn_data(c, seed++);
    const AttnRun r1 = run_prefill(c, d, 3);
    KOPS_TEST_CHECK(r1.split && decode_path(r1) == g);
    check_attn(c, d, r1, " (split)");
    const AttnRun r2 = run_prefill(c, d, 3);
    KOPS_TEST_CHECK(r1.o == r2.o && r1.lse == r2.lse);
    AttnCase nc = c;
    nc.compute_units = 1;
    const AttnRun r3 = run_prefill(nc, d, 3);
    KOPS_TEST_CHECK(!r3.split);
    check_attn(nc, d, r3, " (one pass)");
  }
  // batch_invariant: the last request alone vs in the batch.
  AttnCase big = decode_case("decode batch-invariant", 128, 8, 2, 1, {5000, 30, 4500});
  big.batch_invariant = true;
  big.compute_units = 64;
  const AttnData db = make_attn_data(big, seed++);
  const AttnRun rb = run_prefill(big, db, 7);
  check_attn(big, db, rb, " (batch)");
  AttnCase one = big;
  one.kv_lens = {big.kv_lens[2]};
  one.qo_lens = {1};
  AttnData d1 = make_attn_data(one, seed++);
  const size_t qrow = static_cast<size_t>(big.Hq) * big.dqk, krow = static_cast<size_t>(big.Hkv) * big.dqk;
  d1.q.assign(db.q.begin() + db.qo_indptr[2] * qrow, db.q.begin() + db.qo_indptr[3] * qrow);
  d1.k.assign(db.k.begin() + db.kv_indptr[2] * krow, db.k.begin() + db.kv_indptr[3] * krow);
  d1.v.assign(db.v.begin() + db.kv_indptr[2] * krow, db.v.begin() + db.kv_indptr[3] * krow);
  d1.k_raw = d1.k;
  d1.v_raw = d1.v;
  const AttnRun r_one = run_prefill(one, d1, 8);
  const size_t orow = static_cast<size_t>(big.Hq) * big.dvo;
  bool same = r_one.ok && rb.ok;
  for (size_t i = 0; same && i < r_one.o.size(); ++i) {
    const float a = r_one.o[i], b = rb.o[2 * orow + i];
    same = std::memcmp(&a, &b, 4) == 0;
  }
  KOPS_TEST_CHECK(same);

  // A05 dynamic: one long request among short ones, q_len 1 and 2, every cache flavour.
  AttnCase dy = decode_case("decode dynamic skewed", 128, 16, 2, 1, {20000, 300, 300, 1, 300, 4096, 300, 77});
  dy.device_planner = true;
  dy.compute_units = 16;
  dy.sinks = true;
  run_decode_case(dy, seed++, 2);
  AttnCase dy2 = decode_case("decode dynamic verify q_len 2 fp8 block-table", 64, 8, 8, 2, {700, 3, 1500});
  dy2.device_planner = true;
  dy2.kv_dt = DType::kFloat8E4M3;
  dy2.block_table = true;
  dy2.window_left = 255;
  run_decode_case(dy2, seed++, 2);
  AttnCase dy3 = decode_case("decode dynamic d256 fp16", 256, 4, 1, 1, {900, 2});
  dy3.device_planner = true;
  dy3.dt = DType::kFloat16;
  dy3.atol = dy3.rtol = 2e-3;
  run_decode_case(dy3, seed++, 2);
}

void test_prefill_contracts() {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  kops::attn::AttentionSpec s = spec_of(ragged_case("x", 128, 128, 8, 2));
  KOPS_EXPECT_OK(kops::attn::attention_supports(s, ctx, nullptr, 0, nullptr), "supports d128");
  kops::attn::AttentionSpec bad = s;
  bad.head_dim_qk = bad.head_dim_vo = 72;
  KOPS_EXPECT_STATUS(kops::attn::attention_supports(bad, ctx, nullptr, 0, nullptr),
                     Status::kUnsupportedShape, "supports d72");
  bad = s;
  bad.num_q_heads = 7;
  KOPS_EXPECT_STATUS(kops::attn::attention_supports(bad, ctx, nullptr, 0, nullptr),
                     Status::kUnsupportedShape, "supports Hq % Hkv");
  bad = s;
  bad.kv_dtype = DType::kFloat8E4M3;
  bad.k_codec = bad.v_codec = kops::kv::KVCodec::kFP8PerTensor;
  KOPS_EXPECT_STATUS(kops::attn::attention_supports(bad, ctx, nullptr, 0, nullptr),
                     Status::kUnsupportedDtype, "supports ragged fp8");
  bad.page_size = 16;
  KOPS_EXPECT_OK(kops::attn::attention_supports(bad, ctx, nullptr, 0, nullptr), "supports paged fp8");
  bad.head_dim_qk = bad.head_dim_vo = 96;
  KOPS_EXPECT_STATUS(kops::attn::attention_supports(bad, ctx, nullptr, 0, nullptr),
                     Status::kUnsupportedShape, "supports paged fp8 d96");
  kops::attn::RejectReason why[4];
  int32_t n = 0;
  bad = s;
  bad.q_dtype = DType::kFloat32;
  bad.head_dim_qk = 72;
  (void)kops::attn::attention_supports(bad, ctx, why, 4, &n);
  KOPS_TEST_CHECK(n == 3);  // head dims, q dtype, K/V dtype
  // Plan: host tensors only, two-call workspace protocol.
  std::vector<int32_t> qo = {0, 10}, kvl = {10};
  Tensor qo_t = tensor1d(qo.data(), DType::kInt32, 2);
  qo_t.device = -1;
  Tensor kv_t = tensor1d(kvl.data(), DType::kInt32, 1);
  kv_t.device = -1;
  kops::attn::AttentionPlan plan;
  KOPS_EXPECT_STATUS(kops::attn::prefill_plan(s, qo_t, kv_t, 1, ctx, kops::Workspace{}, &plan),
                     Status::kWorkspaceTooSmall, "prefill_plan empty workspace");
  KOPS_TEST_CHECK(plan.workspace.device_persistent_bytes > 0);
  KOPS_EXPECT_STATUS(kops::attn::prefill_plan(s, qo_t, kv_t, 0 + 1, ctx, kops::Workspace{}, nullptr),
                     Status::kInvalidArgument, "prefill_plan null plan");
  Tensor dev_qo = qo_t;
  dev_qo.device = 0;
  KOPS_EXPECT_STATUS(kops::attn::prefill_plan(s, dev_qo, kv_t, 1, ctx, kops::Workspace{}, &plan),
                     Status::kInvalidArgument, "prefill_plan device qo_indptr");
  // A ragged plan cannot run the paged entry point.
  kops::attn::PagedPrefillParams pp{};
  KOPS_EXPECT_STATUS(kops::attn::paged_prefill_run(plan, pp, kops::Workspace{}, ctx),
                     Status::kInvalidArgument, "paged run without kv");
  KOPS_EXPECT_STATUS(kops::attn::block_sparse_prefill_run(plan, pp, kops::attn::BlockSparseMask{},
                                                          kops::Workspace{}, ctx),
                     Status::kNotImplemented, "block sparse is A10");
}

}  // namespace

int main() {
  LaunchContext probe{};
  if (!kops_test::make_context(&probe)) return 77;

  test_merge_state_take_a_and_one_inf();
  test_merge_state_row_mask();
  test_merge_state_both_neg_inf();
  test_merge_state_finite();
  test_merge_state_bf16();
  test_merge_state_inplace();

  test_merge_states(1, false);
  test_merge_states(2, true);
  test_merge_states(8, true);

  test_merge_split_kv_variable_splits();
  test_merge_split_kv_bf16();
  test_merge_split_kv_total_rows();
  test_merge_split_kv_graph_replay();
  test_merge_hybrid();
  test_concat_mla_q();
  test_mla_context_gather();

  test_contracts();

  test_prefill_head_dims();
  test_prefill_modifiers();
  test_prefill_paged();
  test_prefill_split_and_invariance();
  test_prefill_models();
  test_prefill_graph();
  test_decode_dims_and_paths();
  test_decode_modifiers_and_caches();
  test_decode_split_dynamic();
  test_prefill_contracts();
  return kops_test::finish("test_merge");
}

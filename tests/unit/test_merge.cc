// =============================================================================
// kernel-ops | tests/unit/test_merge.cc
// B1 attention merge kernels: merge_state (A14), merge_states (A14) and
// merge_split_kv (A04).
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
#include "csrc/reference/ops.h"

// The public entry points are declared in csrc/attention/ops.h, which pulls in
// kernel parameter types that only exist in a device pass. This test only calls
// host launchers, so it repeats the three declarations it uses, plus the two
// internal seams for the optional descriptors ops.h does not carry yet.
namespace kops::attn {
Status merge_state(const Tensor& o, const Tensor& lse, const Tensor& o_a, const Tensor& lse_a,
                   const Tensor& o_b, const Tensor& lse_b, const LaunchContext& ctx);
Status merge_states(const Tensor& o, const Tensor& lse, const Tensor& o_states,
                    const Tensor& lse_states, const LaunchContext& ctx);
Status merge_split_kv(const Tensor& o, const Tensor& lse, const SplitKVBuffers& split,
                      const Tensor& sink_logits, const LaunchContext& ctx);
struct HybridBranches {
  const float* const* o = nullptr;
  const float* const* lse = nullptr;
  int32_t num_branches = 0;
};
Status merge_hybrid_branches(const Tensor& o, const Tensor& lse, const HybridBranches& branches,
                             const Tensor& sink_logits, const LaunchContext& ctx);
Status concat_mla_q(const Tensor& q_out, const Tensor& ql_nope, const Tensor& q_pe,
                    const LaunchContext& ctx);
Status mla_context_gather(const Tensor& c_kv_out, const Tensor& k_pe_out,
                          const kv::PagedKVDesc& cache, const Tensor& ctx_token_indptr,
                          const Tensor& ctx_start, const LaunchContext& ctx);
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

  Tensor o96 = kops_test::tensor3d(d.p, DType::kFloat32, 1, 4, 96);
  KOPS_EXPECT_STATUS(kops::attn::merge_split_kv(o96, l, split, Tensor{}, ctx),
                     Status::kUnsupportedShape, "merge_split_kv head_dim");
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
  return kops_test::finish("test_merge");
}

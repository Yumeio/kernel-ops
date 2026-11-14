// =============================================================================
// kernel-ops | tests/unit/test_sparse.cc
// B14 A11: sparse_topk — exact top-K over fp32 score rows.
//
// Oracle: a local host reference (std::sort by (value desc, index asc), NaN
// ranked with -inf, -0.0 equal to +0.0) over the visible [row_starts,
// row_ends) range. Covers rows shorter than K, duplicate / boundary ties, the
// all-equal fallback tie path, absolute vs relative indices, page-slot
// conversion (including an unmapped page and token_to_request), rows past
// num_rows, topk 512 / 1024 / 2048, graph capture + replay and the Status
// contract.
// =============================================================================
#include "tests/unit/harness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/tensor.h"
#include "csrc/common/op_types.cuh"
#include "csrc/kvcache/kv_layout.cuh"
#include "csrc/attention/attention_params.cuh"
#include "csrc/attention/prefill_kernels.cuh"
#include "csrc/attention/decode_kernels.cuh"
#include "csrc/attention/mla_kernels.cuh"
#include "csrc/attention/sparse_kernels.cuh"
#include "csrc/attention/ops.h"

namespace {

using kops::DType;
using kops::LaunchContext;
using kops::Status;
using kops::Tensor;

constexpr int32_t kSentinelIndex = 0x7f7f7f7f;

float neg_inf() { return -std::numeric_limits<float>::infinity(); }

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

Tensor tensor2d(void* data, DType dtype, int64_t rows, int64_t cols) {
  Tensor t{};
  t.data = data;
  t.dtype = dtype;
  t.ndim = 2;
  t.device = 0;
  t.shape[0] = rows;
  t.shape[1] = cols;
  t.stride[0] = cols;
  t.stride[1] = 1;
  return t;
}

bool sync_ok(const char* what) {
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    kops_test::record_failure(what, cudaGetErrorString(err), __FILE__, __LINE__);
    return false;
  }
  return true;
}

bool same_value(float a, float b) {
  if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
  return a == b;
}

// -----------------------------------------------------------------------------
// Host reference
// -----------------------------------------------------------------------------

/// Reference key order: NaN ranks with -inf, -0.0 ties with +0.0, then
/// (value desc, index asc). This is exactly the order the kernel's monotone
/// unsigned key implements.
float ref_rank(float v) {
  if (std::isnan(v)) return neg_inf();
  return v == 0.0f ? 0.0f : v;
}

int32_t ref_map_column(int row, int64_t col, const std::vector<int32_t>& table, int table_stride,
                       const std::vector<int32_t>& request, int page_size, bool absolute,
                       bool slots, int64_t row_start) {
  if (!slots) return static_cast<int32_t>(absolute ? col : col - row_start);
  const int r = request.empty() ? row : request[static_cast<size_t>(row)];
  const int64_t page_index = col / page_size;
  if (r < 0 || page_index >= table_stride) return -1;
  const int32_t page = table[static_cast<size_t>(r) * table_stride + page_index];
  if (page < 0) return -1;
  return static_cast<int32_t>(static_cast<int64_t>(page) * page_size + col % page_size);
}

struct TopkCase {
  int rows = 1;
  int max_kv = 64;
  int topk = 512;
  std::vector<float> scores;      // rows * max_kv
  std::vector<int32_t> starts;    // empty -> row_starts absent
  std::vector<int32_t> ends;      // rows
  std::vector<int32_t> table;     // empty -> block_table absent
  int table_stride = 0;
  std::vector<int32_t> request;   // empty -> token_to_request absent
  int page_size = 0;
  bool absolute = true;
  bool slots = false;
  bool with_values = true;
  int buffer_rows = 0;            // > rows: sentinel index rows past num_rows
  /// 0: workspace-free entry (sparse_topk); 1: sparse_topk_ws with the single-row
  /// lease (max_kv >= 256 Ki then takes the multi-CTA fast path); 2: sparse_topk_ws
  /// with a 1 KiB lease, which must fall back and stay exact; 3: sparse_topk_ws
  /// with sparse_topk_rows_workspace_bytes(rows), one scratch slice per wide row.
  int workspace = 0;
};

void reference_topk(const TopkCase& c, std::vector<int32_t>* out_idx,
                    std::vector<float>* out_val) {
  const int topk = c.topk;
  out_idx->assign(static_cast<size_t>(c.rows) * topk, -1);
  out_val->assign(static_cast<size_t>(c.rows) * topk, neg_inf());
  const int table_rows =
      c.table_stride > 0 ? static_cast<int>(c.table.size() / c.table_stride) : 0;
  (void)table_rows;
  std::vector<std::pair<float, int32_t>> cand;
  for (int row = 0; row < c.rows; ++row) {
    const int32_t start = c.starts.empty() ? 0 : c.starts[static_cast<size_t>(row)];
    const int32_t end = c.ends[static_cast<size_t>(row)];
    cand.clear();
    for (int32_t col = start; col < end; ++col) {
      cand.emplace_back(c.scores[static_cast<size_t>(row) * c.max_kv + col], col);
    }
    std::sort(cand.begin(), cand.end(), [](const std::pair<float, int32_t>& a,
                                           const std::pair<float, int32_t>& b) {
      const float av = ref_rank(a.first);
      const float bv = ref_rank(b.first);
      if (av != bv) return av > bv;
      return a.second < b.second;
    });
    const size_t m = std::min<size_t>(cand.size(), static_cast<size_t>(topk));
    for (size_t i = 0; i < m; ++i) {
      const size_t dst = static_cast<size_t>(row) * topk + i;
      (*out_idx)[dst] = ref_map_column(row, cand[i].second, c.table, c.table_stride, c.request,
                                       c.page_size, c.absolute, c.slots, start);
      (*out_val)[dst] = cand[i].first;
    }
  }
}

struct Entry {
  int32_t row;
  int32_t idx;
  float val;
  uint32_t val_key;
};

/// Total order over an fp32 value, used only to sort unordered output so equal
/// (row, index, value) triples line up on both sides.
uint32_t value_sort_key(float v) {
  uint32_t u = 0;
  std::memcpy(&u, &v, sizeof(u));
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0xffffffffu;  // NaN last
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/// Output order is unspecified: compare each row's index multiset and the value
/// of every selected index. Padding is index -1 with value -inf; an unmapped
/// page slot is also index -1 but keeps the selected score as its value.
void check_topk(const std::vector<int32_t>& got_idx, const std::vector<float>& got_val,
                const std::vector<int32_t>& want_idx, const std::vector<float>& want_val,
                int topk, bool check_values, const char* what) {
  if (got_idx.size() != want_idx.size() || got_val.size() != want_val.size()) {
    kops_test::record_failure(what, "result size mismatch", __FILE__, __LINE__);
    return;
  }
  const int rows = topk > 0 ? static_cast<int>(got_idx.size() / topk) : 0;
  std::vector<Entry> g;
  std::vector<Entry> w;
  g.reserve(got_idx.size());
  w.reserve(want_idx.size());
  for (int row = 0; row < rows; ++row) {
    for (int i = 0; i < topk; ++i) {
      const size_t k = static_cast<size_t>(row) * topk + i;
      g.push_back({row, got_idx[k], got_val[k], check_values ? value_sort_key(got_val[k]) : 0u});
      w.push_back({row, want_idx[k], want_val[k], check_values ? value_sort_key(want_val[k]) : 0u});
    }
  }
  const auto by_key = [](const Entry& a, const Entry& b) {
    if (a.row != b.row) return a.row < b.row;
    if (a.idx != b.idx) return a.idx < b.idx;
    return a.val_key < b.val_key;
  };
  std::sort(g.begin(), g.end(), by_key);
  std::sort(w.begin(), w.end(), by_key);
  for (size_t i = 0; i < g.size(); ++i) {
    if (g[i].row != w[i].row || g[i].idx != w[i].idx) {
      char detail[256];
      std::snprintf(detail, sizeof(detail), "entry %zu: got (row %d, idx %d) want (row %d, idx %d)",
                    i, g[i].row, g[i].idx, w[i].row, w[i].idx);
      kops_test::record_failure(what, detail, __FILE__, __LINE__);
      return;
    }
    if (!check_values) continue;
    if (g[i].idx < 0 && std::isinf(w[i].val) && w[i].val < 0.0f) {
      // Reference padding: the kernel must pad with -inf as well.
      if (!(std::isinf(g[i].val) && g[i].val < 0.0f)) {
        kops_test::record_failure(what, "padding value is not -inf", __FILE__, __LINE__);
        return;
      }
    } else if (!same_value(g[i].val, w[i].val)) {
      char detail[256];
      std::snprintf(detail, sizeof(detail), "value of (row %d, idx %d): got %.9g want %.9g",
                    g[i].row, g[i].idx, static_cast<double>(g[i].val),
                    static_cast<double>(w[i].val));
      kops_test::record_failure(what, detail, __FILE__, __LINE__);
      return;
    }
  }
}

void run_topk_case(const TopkCase& c, const char* what) {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  const size_t rows = static_cast<size_t>(c.rows);
  const size_t kv = static_cast<size_t>(c.max_kv);
  const size_t topk = static_cast<size_t>(c.topk);
  const size_t idx_rows = c.buffer_rows > 0 ? static_cast<size_t>(c.buffer_rows) : rows;
  const int table_rows =
      c.table_stride > 0 ? static_cast<int>(c.table.size() / c.table_stride) : 0;

  DevArray<float> d_scores(rows * kv);
  DevArray<float> d_values(rows * topk);
  DevArray<int32_t> d_idx(idx_rows * topk);
  DevArray<int32_t> d_ends(rows);
  DevArray<int32_t> d_starts(c.starts.empty() ? 0 : c.starts.size());
  DevArray<int32_t> d_table(c.table.empty() ? 0 : c.table.size());
  DevArray<int32_t> d_request(c.request.empty() ? 0 : c.request.size());
  const size_t ws_bytes = c.workspace == 1
                              ? kops::attn::sparse_topk_workspace_bytes()
                              : (c.workspace == 2 ? 1024
                                                  : (c.workspace == 3
                                                         ? kops::attn::sparse_topk_rows_workspace_bytes(
                                                               c.rows)
                                                         : 0));
  DevArray<char> d_ws(ws_bytes);
  if (d_scores.p == nullptr || d_values.p == nullptr || d_idx.p == nullptr ||
      d_ends.p == nullptr) {
    return;
  }

  std::vector<int32_t> sentinel(idx_rows * topk, kSentinelIndex);
  kops_test::copy_in(d_scores.p, c.scores);
  kops_test::copy_in(d_idx.p, sentinel);
  kops_test::copy_in(d_ends.p, c.ends);
  if (!c.starts.empty()) kops_test::copy_in(d_starts.p, c.starts);
  if (!c.table.empty()) kops_test::copy_in(d_table.p, c.table);
  if (!c.request.empty()) kops_test::copy_in(d_request.p, c.request);

  Tensor t_idx = tensor2d(d_idx.p, DType::kInt32, c.rows, c.topk);
  Tensor t_val = c.with_values ? tensor2d(d_values.p, DType::kFloat32, c.rows, c.topk) : Tensor{};
  Tensor t_scores = tensor2d(d_scores.p, DType::kFloat32, c.rows, c.max_kv);
  Tensor t_starts =
      c.starts.empty() ? Tensor{} : tensor1d(d_starts.p, DType::kInt32, c.starts.size());
  Tensor t_ends = tensor1d(d_ends.p, DType::kInt32, c.rows);
  Tensor t_table =
      c.table.empty() ? Tensor{} : tensor2d(d_table.p, DType::kInt32, table_rows, c.table_stride);
  Tensor t_request =
      c.request.empty() ? Tensor{} : tensor1d(d_request.p, DType::kInt32, c.request.size());

  KOPS_EXPECT_OK(c.workspace == 0
                     ? kops::attn::sparse_topk(t_idx, t_val, t_scores, t_starts, t_ends, c.topk,
                                               t_table, t_request, c.page_size, c.absolute, c.slots,
                                               ctx)
                     : kops::attn::sparse_topk_ws(t_idx, t_val, t_scores, t_starts, t_ends, c.topk,
                                                  t_table, t_request, c.page_size, c.absolute,
                                                  c.slots,
                                                  kops::Workspace{d_ws.p, ws_bytes}, ctx),
                 what);
  if (!sync_ok(what)) return;

  std::vector<int32_t> want_idx;
  std::vector<float> want_val;
  reference_topk(c, &want_idx, &want_val);
  const auto got_idx = kops_test::copy_out(d_idx.p, rows * topk);
  const auto got_val = kops_test::copy_out(d_values.p, rows * topk);
  check_topk(got_idx, got_val, want_idx, want_val, c.topk, c.with_values, what);

  if (idx_rows > rows) {
    const auto tail = kops_test::copy_out(d_idx.p + rows * topk, (idx_rows - rows) * topk);
    for (size_t i = 0; i < tail.size(); ++i) {
      if (tail[i] != kSentinelIndex) {
        kops_test::record_failure(what, "wrote past num_rows", __FILE__, __LINE__);
        return;
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Cases
// -----------------------------------------------------------------------------

std::vector<float> duplicate_scores(int rows, int kv, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> step(0, 9);
  std::vector<float> s(static_cast<size_t>(rows) * kv);
  for (size_t i = 0; i < s.size(); ++i) {
    s[i] = 0.125f * static_cast<float>(step(rng)) - 1.0f;
  }
  return s;
}

/// Continuous scores in [-1, 1): no repeated value, so the 22-bit bucket that
/// holds the K-th score of a 1M row leaves ~1 candidate for the fast path.
std::vector<float> uniform_scores(int rows, int kv, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> s(static_cast<size_t>(rows) * kv);
  for (size_t i = 0; i < s.size(); ++i) s[i] = dist(rng);
  return s;
}

void test_short_rows_and_ties() {
  {
    // Rows shorter than K, a partly visible row and an empty row; duplicate
    // scores force the deterministic index tie rule.
    TopkCase c;
    c.rows = 3;
    c.max_kv = 64;
    c.topk = 512;
    c.scores = duplicate_scores(c.rows, c.max_kv, 11u);
    c.ends = {64, 9, 0};
    c.absolute = true;
    run_topk_case(c, "sparse_topk short rows");
  }
  {
    // Same scores, relative indices.
    TopkCase c;
    c.rows = 3;
    c.max_kv = 64;
    c.topk = 512;
    c.scores = duplicate_scores(c.rows, c.max_kv, 11u);
    c.ends = {64, 9, 0};
    c.absolute = false;
    run_topk_case(c, "sparse_topk short rows relative");
  }
  {
    // One row visible in the middle of the matrix: columns outside
    // [start, end) hold huge values and must never be selected.
    TopkCase c;
    c.rows = 2;
    c.max_kv = 640;
    c.topk = 512;
    c.scores.assign(static_cast<size_t>(c.rows) * c.max_kv, 1.0e9f);
    std::mt19937 rng(12u);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int row = 0; row < c.rows; ++row) {
      const int32_t start = row == 0 ? 0 : 20;
      const int32_t end = row == 0 ? 640 : 620;
      for (int32_t col = start; col < end; ++col) {
        c.scores[static_cast<size_t>(row) * c.max_kv + col] = dist(rng);
      }
    }
    c.starts = {0, 20};
    c.ends = {640, 620};
    c.absolute = true;
    run_topk_case(c, "sparse_topk row window absolute");
    c.absolute = false;
    run_topk_case(c, "sparse_topk row window relative");
  }
  {
    // Boundary ties: 300 distinct leaders then 400 equal scores; the top-512
    // must take the 212 lowest-index ties.
    TopkCase c;
    c.rows = 1;
    c.max_kv = 700;
    c.topk = 512;
    c.scores.resize(static_cast<size_t>(c.max_kv));
    for (int i = 0; i < 300; ++i) c.scores[static_cast<size_t>(i)] = 10.0f - 0.01f * i;
    for (int i = 300; i < 700; ++i) c.scores[static_cast<size_t>(i)] = 0.5f;
    c.ends = {700};
    run_topk_case(c, "sparse_topk boundary ties");
  }
  {
    // Every column equal and more columns than K: the tie buffer overflows and
    // the index-pass fallback must still pick the lowest K columns.
    TopkCase c;
    c.rows = 2;
    c.max_kv = 1000;
    c.topk = 512;
    c.scores.assign(static_cast<size_t>(c.rows) * c.max_kv, -0.25f);
    c.ends = {1000, 640};
    run_topk_case(c, "sparse_topk all-equal fallback");
  }
  {
    // Fewer columns than K and all equal: every column is selected and the rest
    // is padded with -1 / -inf.
    TopkCase c;
    c.rows = 1;
    c.max_kv = 400;
    c.topk = 512;
    c.scores.assign(static_cast<size_t>(c.max_kv), 0.5f);
    c.ends = {400};
    run_topk_case(c, "sparse_topk all-equal short");
  }
  {
    // values is optional.
    TopkCase c;
    c.rows = 2;
    c.max_kv = 64;
    c.topk = 512;
    c.scores = duplicate_scores(c.rows, c.max_kv, 21u);
    c.ends = {64, 33};
    c.with_values = false;
    run_topk_case(c, "sparse_topk no values");
  }
}

void test_topk_sizes() {
  {
    TopkCase c;
    c.rows = 1;
    c.max_kv = 1088;
    c.topk = 1024;
    c.scores = duplicate_scores(c.rows, c.max_kv, 31u);
    c.ends = {1088};
    run_topk_case(c, "sparse_topk topk 1024");
  }
  {
    // Near the 2048 boundary: one row longer than K, one shorter.
    TopkCase c;
    c.rows = 2;
    c.max_kv = 2112;
    c.topk = 2048;
    c.scores = duplicate_scores(c.rows, c.max_kv, 32u);
    c.starts = {0, 50};
    c.ends = {2112, 2000};
    c.absolute = false;
    run_topk_case(c, "sparse_topk topk 2048");
    c.absolute = true;
    run_topk_case(c, "sparse_topk topk 2048 absolute");
  }
}

/// Rows of 256 Ki and 1M columns with a workspace take the cooperative
/// multi-CTA path (or, for a tie-heavy row, its exact single-CTA fallback inside
/// the same launch). Every case is checked against the host oracle, which also
/// pins the tie rule (lowest columns win) and the padding.
void test_fast_path() {
  {
    // 1M continuous scores, topk 2048: the benchmark shape. The 22-bit bucket
    // holds ~1 entry, so the last stage resolves a single candidate.
    TopkCase c;
    c.rows = 1;
    c.max_kv = 1 << 20;
    c.topk = 2048;
    c.scores = uniform_scores(1, c.max_kv, 71u);
    c.ends = {c.max_kv};
    c.workspace = 1;
    run_topk_case(c, "sparse_topk fast path 1M random");
    // Same row without the lease: the single-CTA path must agree bit for bit.
    c.workspace = 0;
    run_topk_case(c, "sparse_topk fast path 1M random no workspace");
  }
  {
    // Constant row: every entry ties, the candidate bucket overflows on purpose
    // and the exact fallback must still pick the lowest 2048 columns.
    TopkCase c;
    c.rows = 1;
    c.max_kv = 1 << 20;
    c.topk = 2048;
    c.scores.assign(static_cast<size_t>(c.max_kv), 0.25f);
    c.ends = {c.max_kv};
    c.workspace = 1;
    run_topk_case(c, "sparse_topk fast path 1M constant");
  }
  {
    // 3000 equal leaders: inside the candidate cap, so the last stage's column
    // tie radix must take the 2048 lowest of them (columns 100..2147).
    TopkCase c;
    c.rows = 1;
    c.max_kv = 1 << 20;
    c.topk = 2048;
    c.scores = uniform_scores(1, c.max_kv, 72u);
    for (int i = 0; i < 3000; ++i) c.scores[static_cast<size_t>(100 + i)] = 4.0f;
    c.ends = {c.max_kv};
    c.workspace = 1;
    run_topk_case(c, "sparse_topk fast path 1M tie set");
  }
  {
    // Visible window that does not start at column 0, relative indices and
    // topk 512: the chunk partition must follow [start, end) and the mapping
    // must stay relative to row_starts.
    TopkCase c;
    c.rows = 1;
    c.max_kv = 262144;
    c.topk = 512;
    c.scores = uniform_scores(1, c.max_kv, 73u);
    c.starts = {1000};
    c.ends = {c.max_kv};
    c.absolute = false;
    c.workspace = 1;
    run_topk_case(c, "sparse_topk fast path window relative");
  }
  {
    // Page-slot conversion through the fast path: the above-bucket emission
    // (pass 2), the tie emission (pass 3) and the last stage all map columns
    // through the block table, and page 3 of the request is unmapped (-1).
    TopkCase c;
    c.rows = 1;
    c.max_kv = 262144;
    c.topk = 2048;
    c.scores = uniform_scores(1, c.max_kv, 74u);
    c.ends = {c.max_kv};
    c.page_size = 64;
    c.table_stride = c.max_kv / c.page_size;
    c.table.resize(static_cast<size_t>(c.table_stride));
    for (int i = 0; i < c.table_stride; ++i) c.table[static_cast<size_t>(i)] = c.table_stride - 1 - i;
    c.table[3] = -1;
    c.slots = true;
    c.absolute = true;
    c.workspace = 1;
    run_topk_case(c, "sparse_topk fast path page slots");
  }
  {
    // NaN, -0.0 / +0.0 and infinities inside the fast path: NaN must rank with
    // -inf, ±0.0 must tie, and the key fold happens in every pass.
    TopkCase c;
    c.rows = 1;
    c.max_kv = 262144;
    c.topk = 2048;
    c.scores = uniform_scores(1, c.max_kv, 76u);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int i = 0; i < 64; ++i) c.scores[static_cast<size_t>(i * 37)] = nan;
    for (int i = 0; i < 32; ++i) c.scores[static_cast<size_t>(1000 + i)] = -0.0f;
    for (int i = 0; i < 32; ++i) c.scores[static_cast<size_t>(2000 + i)] = 0.0f;
    c.scores[7] = std::numeric_limits<float>::infinity();
    c.scores[11] = -std::numeric_limits<float>::infinity();
    c.ends = {c.max_kv};
    c.workspace = 1;
    run_topk_case(c, "sparse_topk fast path nan and zeros");
  }
  {
    // Fewer visible columns than TOPK on a wide row: the fast path still has to
    // select all 1500 of them and pad the rest, ignoring the columns past `end`.
    TopkCase c;
    c.rows = 1;
    c.max_kv = 262144;
    c.topk = 2048;
    c.scores.assign(static_cast<size_t>(c.max_kv), 1.0e9f);  // past `end`: must never win
    std::mt19937 rng(77u);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int i = 0; i < 1500; ++i) c.scores[static_cast<size_t>(i)] = dist(rng);
    c.ends = {1500};
    c.workspace = 1;
    run_topk_case(c, "sparse_topk fast path short visible row");
  }
  {
    // A lease below sparse_topk_workspace_bytes() is not an error: the call
    // keeps the workspace-free path and returns the same selection.
    TopkCase c;
    c.rows = 1;
    c.max_kv = 262144;
    c.topk = 1024;
    c.scores = uniform_scores(1, c.max_kv, 75u);
    c.ends = {c.max_kv};
    c.workspace = 2;
    run_topk_case(c, "sparse_topk fast path short workspace");
  }
  {
    // Several wide rows in one cooperative launch: every row owns a scratch slice
    // and a split share of its own columns, and the grid shrank the split count so
    // all CTAs stay resident.
    TopkCase c;
    c.rows = 16;
    c.max_kv = 262144;
    c.topk = 2048;
    c.scores = uniform_scores(c.rows, c.max_kv, 81u);
    c.ends.assign(static_cast<size_t>(c.rows), c.max_kv);
    c.workspace = 3;
    run_topk_case(c, "sparse_topk fast path 16 rows x 256k");
  }
  {
    // Ragged window plus a 1M row: chunks follow each row's own [start, end) with
    // relative indices, and a row shorter than K pads instead of reading past it.
    TopkCase c;
    c.rows = 4;
    c.max_kv = 1 << 20;
    c.topk = 512;
    c.scores = uniform_scores(c.rows, c.max_kv, 82u);
    c.starts = {1000, 0, 4096, 65536};
    c.ends = {c.max_kv, 300, 700000, 131072};
    c.absolute = false;
    c.workspace = 3;
    run_topk_case(c, "sparse_topk fast path 4 ragged 1M rows");
  }
  {
    // An empty row and a one-column row share the grid with a full row: they take
    // the -1 / -inf padding branch, and the full row must still be exact (no CTA
    // may skip a grid sync because its row has no visible columns).
    TopkCase c;
    c.rows = 3;
    c.max_kv = 262144;
    c.topk = 1024;
    c.scores = uniform_scores(c.rows, c.max_kv, 83u);
    c.ends = {0, 1, c.max_kv};
    c.workspace = 3;
    run_topk_case(c, "sparse_topk fast path empty and short rows");
  }
  {
    // A tie-heavy row forces its own candidate overflow and exact single-CTA
    // reselection inside the shared launch; its neighbour must be unaffected.
    TopkCase c;
    c.rows = 3;
    c.max_kv = 262144;
    c.topk = 2048;
    c.scores = uniform_scores(c.rows, c.max_kv, 84u);
    for (int64_t i = 0; i < c.max_kv; ++i) c.scores[static_cast<size_t>(c.max_kv + i)] = 0.5f;
    c.ends.assign(static_cast<size_t>(c.rows), c.max_kv);
    c.workspace = 3;
    run_topk_case(c, "sparse_topk fast path multi-row tie fallback");
  }
  {
    // Per-row page-slot mapping through the multi-row fast path.
    TopkCase c;
    c.rows = 4;
    c.max_kv = 262144;
    c.topk = 512;
    c.scores = uniform_scores(c.rows, c.max_kv, 85u);
    c.ends.assign(static_cast<size_t>(c.rows), c.max_kv);
    c.page_size = 64;
    c.table_stride = c.max_kv / c.page_size;
    c.table.assign(static_cast<size_t>(c.rows) * c.table_stride, 0);
    for (int r = 0; r < c.rows; ++r) {
      for (int i = 0; i < c.table_stride; ++i) {
        c.table[static_cast<size_t>(r) * c.table_stride + i] = c.table_stride - 1 - i;
      }
      c.table[static_cast<size_t>(r) * c.table_stride + 3] = -1;  // page 3 unmapped for every row
    }
    c.request = {0, 1, 2, 3};
    c.slots = true;
    c.absolute = true;
    c.workspace = 3;
    run_topk_case(c, "sparse_topk fast path 4 rows page slots");
  }
}

void test_page_slots() {  {
    // token_to_request swaps the requests, page 2 of request 0 is unmapped.
    TopkCase c;
    c.rows = 2;
    c.max_kv = 64;
    c.topk = 512;
    c.scores = duplicate_scores(c.rows, c.max_kv, 41u);
    c.ends = {64, 64};
    c.table_stride = 4;
    c.table = {3, 1, 4, 2, 0, 5, -1, 7};
    c.request = {1, 0};
    c.page_size = 16;
    c.slots = true;
    c.absolute = true;
    run_topk_case(c, "sparse_topk page slots");
  }
  {
    // No token_to_request: the row index is the request.
    TopkCase c;
    c.rows = 2;
    c.max_kv = 64;
    c.topk = 512;
    c.scores = duplicate_scores(c.rows, c.max_kv, 42u);
    c.ends = {64, 64};
    c.table_stride = 4;
    c.table = {0, 1, 2, 3, 4, 5, 6, 7};
    c.page_size = 16;
    c.slots = true;
    run_topk_case(c, "sparse_topk page slots per row");
  }
  {
    // Slot conversion is driven by the absolute column even when the index
    // output is relative to row_starts.
    TopkCase c;
    c.rows = 1;
    c.max_kv = 64;
    c.topk = 512;
    c.scores = duplicate_scores(c.rows, c.max_kv, 43u);
    c.starts = {16};
    c.ends = {48};
    c.table_stride = 4;
    c.table = {6, 9, 12, 15};
    c.page_size = 16;
    c.slots = true;
    c.absolute = false;
    run_topk_case(c, "sparse_topk page slots relative");
  }
}

void test_num_rows_gate() {
  TopkCase c;
  c.rows = 3;
  c.max_kv = 640;
  c.topk = 512;
  c.buffer_rows = 5;
  c.scores = duplicate_scores(c.rows, c.max_kv, 51u);
  c.ends = {640, 500, 100};
  run_topk_case(c, "sparse_topk num_rows gate");
}

void test_zero_rows() {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  DevArray<float> d_scores(4);
  DevArray<int32_t> d_idx(4);
  DevArray<int32_t> d_ends(1);
  const std::vector<float> scores = {0.0f, 0.0f, 0.0f, 0.0f};
  const std::vector<int32_t> ends = {0};
  kops_test::copy_in(d_scores.p, scores);
  kops_test::copy_in(d_ends.p, ends);
  Tensor t_scores = tensor2d(d_scores.p, DType::kFloat32, 0, 4);
  Tensor t_idx = tensor2d(d_idx.p, DType::kInt32, 0, 512);
  Tensor t_ends = tensor1d(d_ends.p, DType::kInt32, 1);
  KOPS_EXPECT_OK(kops::attn::sparse_topk(t_idx, Tensor{}, t_scores, Tensor{}, t_ends, 512,
                                         Tensor{}, Tensor{}, 0, true, false, ctx),
                 "sparse_topk zero rows");
}

/// The fast path is a cooperative launch, so graph capture and replay must be
/// rehearsed on a row wide enough to select it.
void test_graph_capture_fast_path() {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const int max_kv = 262144, topk = 2048;
  TopkCase c;
  c.rows = 1;
  c.max_kv = max_kv;
  c.topk = topk;
  c.scores = uniform_scores(1, max_kv, 81u);
  c.ends = {max_kv};
  std::vector<int32_t> want_idx;
  std::vector<float> want_val;
  reference_topk(c, &want_idx, &want_val);

  DevArray<float> d_scores(static_cast<size_t>(max_kv));
  DevArray<float> d_values(static_cast<size_t>(topk));
  DevArray<int32_t> d_idx(static_cast<size_t>(topk));
  DevArray<int32_t> d_ends(1);
  const size_t ws_bytes = kops::attn::sparse_topk_workspace_bytes();
  DevArray<char> d_ws(ws_bytes);
  kops_test::copy_in(d_scores.p, c.scores);
  kops_test::copy_in(d_ends.p, c.ends);
  std::vector<int32_t> sentinel(static_cast<size_t>(topk), kSentinelIndex);
  kops_test::copy_in(d_idx.p, sentinel);

  const Tensor t_idx = tensor2d(d_idx.p, DType::kInt32, 1, topk);
  const Tensor t_val = tensor2d(d_values.p, DType::kFloat32, 1, topk);
  const Tensor t_scores = tensor2d(d_scores.p, DType::kFloat32, 1, max_kv);
  const Tensor t_ends = tensor1d(d_ends.p, DType::kInt32, 1);
  const kops::Workspace lease{d_ws.p, ws_bytes};

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    kops_test::record_failure("cudaStreamCreate", cudaGetErrorString(cudaGetLastError()),
                              __FILE__, __LINE__);
    return;
  }
  LaunchContext gctx = ctx;
  gctx.stream = reinterpret_cast<kops::kops_stream_t>(stream);

  if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    kops_test::record_failure("sparse_topk fast capture begin", cudaGetErrorString(cudaGetLastError()),
                              __FILE__, __LINE__);
    cudaStreamDestroy(stream);
    return;
  }
  KOPS_EXPECT_OK(kops::attn::sparse_topk_ws(t_idx, t_val, t_scores, Tensor{}, t_ends, topk,
                                            Tensor{}, Tensor{}, 0, true, false, lease, gctx),
                 "sparse_topk fast capture");
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  if (cudaStreamEndCapture(stream, &graph) != cudaSuccess ||
      cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
    kops_test::record_failure("sparse_topk fast graph instantiate",
                              cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
    cudaStreamDestroy(stream);
    return;
  }

  kops_test::copy_in(d_idx.p, sentinel);
  KOPS_TEST_CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess);
  if (sync_ok("sparse_topk fast graph launch")) {
    const auto got_idx = kops_test::copy_out(d_idx.p, static_cast<size_t>(topk));
    const auto got_val = kops_test::copy_out(d_values.p, static_cast<size_t>(topk));
    check_topk(got_idx, got_val, want_idx, want_val, topk, true, "sparse_topk fast graph replay");
  }

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
}

void test_graph_capture() {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;
  const int rows = 2, max_kv = 64, topk = 512;
  TopkCase c;
  c.rows = rows;
  c.max_kv = max_kv;
  c.topk = topk;
  c.scores = duplicate_scores(rows, max_kv, 61u);
  c.ends = {max_kv, 45};
  std::vector<int32_t> want_idx;
  std::vector<float> want_val;
  reference_topk(c, &want_idx, &want_val);

  DevArray<float> d_scores(static_cast<size_t>(rows) * max_kv);
  DevArray<float> d_values(static_cast<size_t>(rows) * topk);
  DevArray<int32_t> d_idx(static_cast<size_t>(rows) * topk);
  DevArray<int32_t> d_ends(rows);
  kops_test::copy_in(d_scores.p, c.scores);
  kops_test::copy_in(d_ends.p, c.ends);
  std::vector<int32_t> sentinel(static_cast<size_t>(rows) * topk, kSentinelIndex);
  kops_test::copy_in(d_idx.p, sentinel);

  Tensor t_idx = tensor2d(d_idx.p, DType::kInt32, rows, topk);
  Tensor t_val = tensor2d(d_values.p, DType::kFloat32, rows, topk);
  Tensor t_scores = tensor2d(d_scores.p, DType::kFloat32, rows, max_kv);
  Tensor t_ends = tensor1d(d_ends.p, DType::kInt32, rows);

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    kops_test::record_failure("cudaStreamCreate", cudaGetErrorString(cudaGetLastError()),
                              __FILE__, __LINE__);
    return;
  }
  LaunchContext gctx = ctx;
  gctx.stream = reinterpret_cast<kops::kops_stream_t>(stream);

  if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
    kops_test::record_failure("cudaStreamBeginCapture", cudaGetErrorString(cudaGetLastError()),
                              __FILE__, __LINE__);
    cudaStreamDestroy(stream);
    return;
  }
  KOPS_EXPECT_OK(kops::attn::sparse_topk(t_idx, t_val, t_scores, Tensor{}, t_ends, topk, Tensor{},
                                         Tensor{}, 0, true, false, gctx),
                 "sparse_topk capture");
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  if (cudaStreamEndCapture(stream, &graph) != cudaSuccess ||
      cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
    kops_test::record_failure("sparse_topk graph instantiate",
                              cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
    cudaStreamDestroy(stream);
    return;
  }

  kops_test::copy_in(d_idx.p, sentinel);
  KOPS_TEST_CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess);
  if (sync_ok("sparse_topk graph launch")) {
    const auto got_idx = kops_test::copy_out(d_idx.p, static_cast<size_t>(rows) * topk);
    const auto got_val = kops_test::copy_out(d_values.p, static_cast<size_t>(rows) * topk);
    check_topk(got_idx, got_val, want_idx, want_val, topk, true, "sparse_topk graph replay");
  }

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
}

void test_contracts() {
  LaunchContext ctx{};
  if (!kops_test::make_context(&ctx)) return;

  const int rows = 2, max_kv = 64, topk = 512;
  const size_t n_scores = static_cast<size_t>(rows) * max_kv;
  const size_t n_idx = static_cast<size_t>(rows) * topk;

  DevArray<float> d_scores(n_scores);
  DevArray<float> d_values(n_idx);
  DevArray<int32_t> d_idx(n_idx);
  DevArray<int32_t> d_ends(rows);
  DevArray<int32_t> d_starts(rows);
  DevArray<int32_t> d_table(static_cast<size_t>(rows) * 8);
  DevArray<int32_t> d_request(rows);
  DevArray<int64_t> d_ends64(rows);
  const std::vector<float> scores(n_scores, 0.5f);
  const std::vector<float> values(n_idx, 0.0f);
  const std::vector<int32_t> zeros_index(n_idx, 0);
  const std::vector<int32_t> ends = {max_kv, max_kv};
  const std::vector<int32_t> starts = {0, 0};
  const std::vector<int32_t> table(static_cast<size_t>(rows) * 8, 0);
  const std::vector<int32_t> request = {0, 1};
  const std::vector<int64_t> ends64(rows, max_kv);
  kops_test::copy_in(d_scores.p, scores);
  kops_test::copy_in(d_values.p, values);
  kops_test::copy_in(d_idx.p, zeros_index);
  kops_test::copy_in(d_ends.p, ends);
  kops_test::copy_in(d_starts.p, starts);
  kops_test::copy_in(d_table.p, table);
  kops_test::copy_in(d_request.p, request);
  kops_test::copy_in(d_ends64.p, ends64);

  const Tensor t_idx = tensor2d(d_idx.p, DType::kInt32, rows, topk);
  const Tensor t_val = tensor2d(d_values.p, DType::kFloat32, rows, topk);
  const Tensor t_scores = tensor2d(d_scores.p, DType::kFloat32, rows, max_kv);
  const Tensor t_starts = tensor1d(d_starts.p, DType::kInt32, rows);
  const Tensor t_ends = tensor1d(d_ends.p, DType::kInt32, rows);
  const Tensor t_table = tensor2d(d_table.p, DType::kInt32, rows, 8);
  const Tensor t_request = tensor1d(d_request.p, DType::kInt32, rows);

  // A valid call to show the fixture itself is accepted.
  KOPS_EXPECT_OK(kops::attn::sparse_topk(t_idx, t_val, t_scores, t_starts, t_ends, topk, t_table,
                                         t_request, 16, true, false, ctx),
                 "sparse_topk contract baseline");

  // topk outside the compiled set.
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, t_val, t_scores, t_starts, t_ends, 300, t_table,
                                             t_request, 16, true, false, ctx),
                     Status::kUnsupportedShape, "sparse_topk topk");
  // dtypes.
  Tensor bad_idx = t_idx;
  bad_idx.dtype = DType::kFloat32;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(bad_idx, t_val, t_scores, t_starts, t_ends, topk,
                                             t_table, t_request, 16, true, false, ctx),
                     Status::kUnsupportedDtype, "sparse_topk indices dtype");
  Tensor bad_scores = t_scores;
  bad_scores.dtype = DType::kFloat16;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, t_val, bad_scores, t_starts, t_ends, topk,
                                             t_table, t_request, 16, true, false, ctx),
                     Status::kUnsupportedDtype, "sparse_topk scores dtype");
  Tensor bad_values = t_val;
  bad_values.dtype = DType::kInt32;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, bad_values, t_scores, t_starts, t_ends, topk,
                                             t_table, t_request, 16, true, false, ctx),
                     Status::kUnsupportedDtype, "sparse_topk values dtype");
  Tensor bad_ends = t_ends;
  bad_ends.data = d_ends64.p;
  bad_ends.dtype = DType::kInt64;
  bad_ends.stride[0] = 1;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, t_val, t_scores, t_starts, bad_ends, topk,
                                             t_table, t_request, 16, true, false, ctx),
                     Status::kUnsupportedDtype, "sparse_topk row_ends dtype");
  // Shapes / layout.
  Tensor flat_idx = t_idx;
  flat_idx.ndim = 1;
  flat_idx.shape[0] = n_idx;
  flat_idx.shape[1] = 0;
  flat_idx.stride[0] = 1;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(flat_idx, t_val, t_scores, t_starts, t_ends, topk,
                                             t_table, t_request, 16, true, false, ctx),
                     Status::kInvalidArgument, "sparse_topk indices ndim");
  Tensor narrow_idx = t_idx;
  narrow_idx.shape[1] = 256;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(narrow_idx, t_val, t_scores, t_starts, t_ends, topk,
                                             t_table, t_request, 16, true, false, ctx),
                     Status::kInvalidArgument, "sparse_topk indices width");
  Tensor strided_idx = t_idx;
  strided_idx.stride[0] = topk + 4;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(strided_idx, t_val, t_scores, t_starts, t_ends, topk,
                                             t_table, t_request, 16, true, false, ctx),
                     Status::kUnsupportedLayout, "sparse_topk indices stride");
  Tensor packed_scores = t_scores;
  packed_scores.stride[1] = 2;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, t_val, packed_scores, t_starts, t_ends, topk,
                                             t_table, t_request, 16, true, false, ctx),
                     Status::kUnsupportedLayout, "sparse_topk scores inner stride");
  Tensor short_scores = t_scores;
  short_scores.stride[0] = 32;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, t_val, short_scores, t_starts, t_ends, topk,
                                             t_table, t_request, 16, true, false, ctx),
                     Status::kUnsupportedLayout, "sparse_topk scores row stride");
  Tensor offset_scores = t_scores;
  offset_scores.byte_offset = 2;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, t_val, offset_scores, t_starts, t_ends, topk,
                                             t_table, t_request, 16, true, false, ctx),
                     Status::kMisaligned, "sparse_topk scores alignment");
  Tensor short_ends = t_ends;
  short_ends.shape[0] = 1;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, t_val, t_scores, t_starts, short_ends, topk,
                                             t_table, t_request, 16, true, false, ctx),
                     Status::kInvalidArgument, "sparse_topk row_ends length");
  Tensor short_starts = t_starts;
  short_starts.shape[0] = 1;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, t_val, t_scores, short_starts, t_ends, topk,
                                             t_table, t_request, 16, true, false, ctx),
                     Status::kInvalidArgument, "sparse_topk row_starts length");
  Tensor wrong_values = t_val;
  wrong_values.shape[1] = topk / 2;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, wrong_values, t_scores, t_starts, t_ends, topk,
                                             t_table, t_request, 16, true, false, ctx),
                     Status::kInvalidArgument, "sparse_topk values shape");
  // Page-slot arguments.
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, t_val, t_scores, t_starts, t_ends, topk,
                                             Tensor{}, t_request, 16, true, true, ctx),
                     Status::kInvalidArgument, "sparse_topk slots without table");
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, t_val, t_scores, t_starts, t_ends, topk,
                                             t_table, t_request, 0, true, true, ctx),
                     Status::kInvalidArgument, "sparse_topk slots page_size");
  Tensor small_table = t_table;
  small_table.shape[1] = 2;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, t_val, t_scores, t_starts, t_ends, topk,
                                             small_table, t_request, 16, true, true, ctx),
                     Status::kUnsupportedShape, "sparse_topk block table coverage");
  Tensor short_request = t_request;
  short_request.shape[0] = 1;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, t_val, t_scores, t_starts, t_ends, topk,
                                             t_table, short_request, 16, true, true, ctx),
                     Status::kInvalidArgument, "sparse_topk token_to_request length");
  Tensor one_row_table = t_table;
  one_row_table.shape[0] = 1;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, t_val, t_scores, t_starts, t_ends, topk,
                                             one_row_table, Tensor{}, 16, true, true, ctx),
                     Status::kInvalidArgument, "sparse_topk block table rows");
  Tensor strided_table = t_table;
  strided_table.stride[1] = 2;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, t_val, t_scores, t_starts, t_ends, topk,
                                             strided_table, t_request, 16, true, true, ctx),
                     Status::kUnsupportedLayout, "sparse_topk block table stride");
  // A non-CUDA context is rejected without touching the device.
  LaunchContext amd = ctx;
  amd.backend = kops::BackendKind::kHip;
  KOPS_EXPECT_STATUS(kops::attn::sparse_topk(t_idx, t_val, t_scores, t_starts, t_ends, topk,
                                             t_table, t_request, 16, true, false, amd),
                     Status::kUnsupportedBackend, "sparse_topk backend");
}

}  // namespace

int main() {
  kops::LaunchContext probe{};
  if (!kops_test::make_context(&probe)) return 77;

  test_short_rows_and_ties();
  test_topk_sizes();
  test_fast_path();
  test_page_slots();
  test_num_rows_gate();
  test_zero_rows();
  test_graph_capture();
  test_graph_capture_fast_path();
  test_contracts();
  return kops_test::finish("test_sparse");
}

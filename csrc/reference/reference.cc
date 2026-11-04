// =============================================================================
// kernel-ops | csrc/reference/reference.cc
// fp64 host reference for the M1 golden-path kernels. Scalar loops, fixed
// evaluation order, no FMA contraction (build with -ffp-contract=off).
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "csrc/common/macros.cuh"
#include "csrc/common/status.h"
#include "csrc/common/dtype.cuh"
#include "csrc/common/op_types.cuh"
#include "csrc/common/quant.cuh"
#include "csrc/reference/ops.h"

namespace kops::ref {
namespace {

constexpr double kPi = 3.14159265358979323846;

bool bad(const void* p, int64_t n) { return p == nullptr || n <= 0; }

double act_scalar(double x, ActFn act, const GluParams& p) {
  switch (act) {
    case ActFn::kIdentity: return x;
    case ActFn::kSiLU: return x / (1.0 + std::exp(-p.gate_alpha * x));
    case ActFn::kGeLUErf: return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
    case ActFn::kGeLUTanh:
      return 0.5 * x *
             (1.0 + std::tanh(std::sqrt(2.0 / kPi) * (x + 0.044715 * x * x * x)));
    case ActFn::kQuickGeLU: return x / (1.0 + std::exp(-1.702 * x));
    case ActFn::kReLU: return x > 0.0 ? x : 0.0;
    case ActFn::kReLU2: return x > 0.0 ? x * x : 0.0;
    case ActFn::kSigmoid: return 1.0 / (1.0 + std::exp(-x));
    case ActFn::kTanh: return std::tanh(x);
    case ActFn::kSoftplus: return x > 20.0 ? x : std::log1p(std::exp(x));
    case ActFn::kSiTU:
      return p.situ_beta * std::tanh(x / p.situ_beta) * (1.0 / (1.0 + std::exp(-x)));
    case ActFn::kFatReLU: return x > p.fatrelu_threshold ? x : 0.0;
  }
  return x;
}

}  // namespace

Status rms_norm_f64(const double* in, int64_t in_row_stride, const double* weight,
                    double weight_offset, double eps, int64_t rows, int64_t hidden,
                    double* out, int64_t out_row_stride) {
  if (bad(in, rows * hidden) || bad(out, rows * hidden) || bad(weight, hidden) || hidden <= 0) {
    return detail::set_error(Status::kInvalidArgument, "rms_norm_f64: bad argument");
  }
  for (int64_t r = 0; r < rows; ++r) {
    const double* x = in + r * in_row_stride;
    double ss = 0.0;
    for (int64_t i = 0; i < hidden; ++i) ss += x[i] * x[i];
    const double rstd = 1.0 / std::sqrt(ss / static_cast<double>(hidden) + eps);
    double* y = out + r * out_row_stride;
    for (int64_t i = 0; i < hidden; ++i) y[i] = x[i] * rstd * (weight[i] + weight_offset);
  }
  return Status::kSuccess;
}

Status act_f64(const double* in, int64_t n, ActFn act, const GluParams& p, double* out) {
  if (bad(in, n) || bad(out, n)) {
    return detail::set_error(Status::kInvalidArgument, "act_f64: bad argument");
  }
  for (int64_t i = 0; i < n; ++i) out[i] = act_scalar(in[i], act, p);
  return Status::kSuccess;
}

Status act_and_mul_f64(const double* in, int64_t in_row_stride, const double* row_weight,
                       GluParams p, ActFn act, GateLayout layout, int64_t rows, int64_t d,
                       double* out) {
  if (bad(in, rows * in_row_stride) || bad(out, rows * d) || d <= 0) {
    return detail::set_error(Status::kInvalidArgument, "act_and_mul_f64: bad argument");
  }
  for (int64_t r = 0; r < rows; ++r) {
    const double* x = in + r * in_row_stride;
    double* y = out + r * d;
    const double rw = (row_weight != nullptr) ? row_weight[r] : 1.0;
    for (int64_t i = 0; i < d; ++i) {
      double gate, up;
      switch (layout) {
        case GateLayout::kGateUp: gate = x[i]; up = x[i + d]; break;
        case GateLayout::kUpGate: up = x[i]; gate = x[i + d]; break;
        case GateLayout::kInterleaved: gate = x[2 * i]; up = x[2 * i + 1]; break;
        default: gate = x[i]; up = x[i + d]; break;
      }
      if (p.gate_clamp_max > 0.0f && gate > p.gate_clamp_max) gate = p.gate_clamp_max;
      if (p.up_clamp > 0.0f) {
        if (up > p.up_clamp) up = p.up_clamp;
        if (up < -p.up_clamp) up = -p.up_clamp;
      }
      up += p.up_offset;
      if (p.situ_linear_beta > 0.0f) {
        up = p.situ_linear_beta * std::tanh(up / p.situ_linear_beta);
      }
      y[i] = rw * act_scalar(gate, act, p) * up;
    }
  }
  return Status::kSuccess;
}

Status merge_states_f64(const double* o_states, const double* lse_states, int64_t rows, int64_t n,
                        int64_t heads, int64_t head_dim, double* o, double* lse) {
  if (bad(o_states, rows * n * heads * head_dim) || bad(lse_states, rows * n * heads) ||
      bad(o, rows * heads * head_dim) || n <= 0) {
    return detail::set_error(Status::kInvalidArgument, "merge_states_f64: bad argument");
  }
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t h = 0; h < heads; ++h) {
      double m = -INFINITY;
      for (int64_t i = 0; i < n; ++i) {
        const double l = lse_states[(r * n + i) * heads + h];
        if (l > m) m = l;
      }
      double denom = 0.0;
      if (std::isfinite(m)) {
        for (int64_t i = 0; i < n; ++i) denom += std::exp(lse_states[(r * n + i) * heads + h] - m);
      }
      const double l_out = std::isfinite(m) ? m + std::log(denom) : -INFINITY;
      for (int64_t d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        if (std::isfinite(m)) {
          for (int64_t i = 0; i < n; ++i) {
            const double w =
                std::exp(lse_states[(r * n + i) * heads + h] - l_out);
            acc += w * o_states[((r * n + i) * heads + h) * head_dim + d];
          }
        }
        o[(r * heads + h) * head_dim + d] = acc;
      }
      if (lse != nullptr) lse[r * heads + h] = l_out;
    }
  }
  return Status::kSuccess;
}

namespace {

int64_t floor_div(int64_t a, int64_t b) {
  const int64_t q = a / b;
  return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

}  // namespace

Status attention_f64(const AttentionRefArgs& a) {
  if (a.q == nullptr || a.k == nullptr || a.v == nullptr || a.o == nullptr ||
      a.qo_indptr == nullptr || a.kv_indptr == nullptr || a.batch <= 0 || a.q_heads <= 0 ||
      a.kv_heads <= 0 || a.q_heads % a.kv_heads != 0 || a.head_dim_qk <= 0 ||
      a.head_dim_vo <= 0 || (a.custom_mask != nullptr && a.mask_indptr == nullptr)) {
    return detail::set_error(Status::kInvalidArgument, "attention_f64: bad argument");
  }
  const int32_t group = a.q_heads / a.kv_heads;
  const int64_t dqk = a.head_dim_qk, dvo = a.head_dim_vo;
  std::vector<double> s;
  for (int32_t b = 0; b < a.batch; ++b) {
    const int64_t q0 = a.qo_indptr[b], qn = a.qo_indptr[b + 1] - q0;
    const int64_t k0 = a.kv_indptr[b], kn = a.kv_indptr[b + 1] - k0;
    const int64_t pos0 = a.q_pos_offset != nullptr ? a.q_pos_offset[b] : kn - qn;
    s.assign(static_cast<size_t>(kn > 0 ? kn : 1), 0.0);
    for (int64_t i = 0; i < qn; ++i) {
      const int64_t qpos = pos0 + i;
      for (int32_t h = 0; h < a.q_heads; ++h) {
        const int32_t hk = h / group;
        const double* qv = a.q + ((q0 + i) * a.q_heads + h) * dqk;
        double m = -INFINITY;
        for (int64_t j = 0; j < kn; ++j) {
          bool vis = true;
          if (a.causal) vis = vis && j <= qpos;
          if (a.custom_mask != nullptr) {
            const int64_t bit = a.mask_indptr[b] + i * kn + j;
            vis = vis && ((a.custom_mask[bit >> 3] >> (bit & 7)) & 1) != 0;
          }
          if (a.window_left >= 0) {
            vis = vis && (j >= qpos - a.window_left || j < a.num_sink_tokens);
          }
          if (a.window_right >= 0) vis = vis && j <= qpos + a.window_right;
          if (a.chunk_size > 0) {
            vis = vis && floor_div(qpos, a.chunk_size) == floor_div(j, a.chunk_size);
          }
          if (!vis) {
            s[static_cast<size_t>(j)] = -INFINITY;
            continue;
          }
          const double* kv = a.k + ((k0 + j) * a.kv_heads + hk) * dqk;
          double dot = 0.0;
          for (int64_t d = 0; d < dqk; ++d) dot += qv[d] * kv[d];
          double x = a.sm_scale * dot;
          if (a.logits_softcap > 0.0) x = a.logits_softcap * std::tanh(x / a.logits_softcap);
          if (a.alibi_slopes != nullptr) x += a.alibi_slopes[h] * static_cast<double>(j - qpos);
          s[static_cast<size_t>(j)] = x;
          if (x > m) m = x;
        }
        const double sink = a.sink_logits != nullptr ? a.sink_logits[h] : -INFINITY;
        if (sink > m) m = sink;
        double* ov = a.o + ((q0 + i) * a.q_heads + h) * dvo;
        double denom = 0.0;
        if (std::isfinite(m)) {
          for (int64_t j = 0; j < kn; ++j) denom += std::exp(s[static_cast<size_t>(j)] - m);
          if (a.sink_logits != nullptr) denom += std::exp(sink - m);
        }
        for (int64_t d = 0; d < dvo; ++d) {
          double acc = 0.0;
          if (denom > 0.0) {
            for (int64_t j = 0; j < kn; ++j) {
              const double p = std::exp(s[static_cast<size_t>(j)] - m);
              if (p != 0.0) acc += p * a.v[((k0 + j) * a.kv_heads + hk) * dvo + d];
            }
            acc /= denom;
          }
          ov[d] = acc;
        }
        if (a.lse != nullptr) {
          a.lse[(q0 + i) * a.q_heads + h] = denom > 0.0 ? m + std::log(denom) : -INFINITY;
        }
      }
    }
  }
  return Status::kSuccess;
}

Status softmax_f64(const double* logits, int64_t rows, int64_t cols, double temperature,
                   double* probs) {
  if (bad(logits, rows * cols) || bad(probs, rows * cols) || temperature <= 0.0) {
    return detail::set_error(Status::kInvalidArgument, "softmax_f64: bad argument");
  }
  for (int64_t r = 0; r < rows; ++r) {
    const double* x = logits + r * cols;
    double* p = probs + r * cols;
    double m = -INFINITY;
    for (int64_t i = 0; i < cols; ++i) m = std::max(m, x[i] / temperature);
    double s = 0.0;
    for (int64_t i = 0; i < cols; ++i) {
      p[i] = std::exp(x[i] / temperature - m);
      s += p[i];
    }
    for (int64_t i = 0; i < cols; ++i) p[i] /= s;
  }
  return Status::kSuccess;
}

// ---- State / recurrent (B9) ------------------------------------------------------

namespace {
double sigmoid_d(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double softplus_d(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }
}  // namespace

Status causal_conv1d_f64(const double* x, int64_t len, int64_t dim, const double* w,
                         int32_t width, const double* bias, ActFn act, double* hist, double* y) {
  if (len < 0 || dim <= 0 || width < 2 || width > 4 || w == nullptr || hist == nullptr ||
      (len > 0 && (x == nullptr || y == nullptr)) ||
      (act != ActFn::kSiLU && act != ActFn::kIdentity)) {
    return detail::set_error(Status::kInvalidArgument, "causal_conv1d_f64: bad argument");
  }
  const int h = width - 1;
  std::vector<double> win(static_cast<size_t>(width));
  for (int64_t c = 0; c < dim; ++c) {
    for (int j = 0; j < h; ++j) win[static_cast<size_t>(j)] = hist[c * h + j];
    for (int64_t t = 0; t < len; ++t) {
      win[static_cast<size_t>(h)] = x[t * dim + c];
      double acc = bias ? bias[c] : 0.0;
      for (int j = 0; j < width; ++j) acc += w[c * width + j] * win[static_cast<size_t>(j)];
      y[t * dim + c] = act == ActFn::kSiLU ? acc * sigmoid_d(acc) : acc;
      for (int j = 0; j < h; ++j) win[static_cast<size_t>(j)] = win[static_cast<size_t>(j + 1)];
    }
    for (int j = 0; j < h; ++j) hist[c * h + j] = win[static_cast<size_t>(j)];
  }
  return Status::kSuccess;
}

Status ssm_recurrence_f64(const SsmRefArgs& a) {
  if (a.len < 0 || a.nheads <= 0 || a.headdim <= 0 || a.dstate <= 0 || a.ngroups <= 0 ||
      a.nheads % a.ngroups != 0 || a.A == nullptr || a.state == nullptr ||
      (a.len > 0 && (a.x == nullptr || a.dt == nullptr || a.B == nullptr || a.C == nullptr ||
                     a.y == nullptr))) {
    return detail::set_error(Status::kInvalidArgument, "ssm_recurrence_f64: bad argument");
  }
  const int64_t H = a.nheads, P = a.headdim, N = a.dstate, G = a.ngroups, hpg = H / G;
  for (int64_t t = 0; t < a.len; ++t) {
    for (int64_t h = 0; h < H; ++h) {
      const int64_t g = h / hpg;
      double dt = a.dt[t * H + h] + (a.dt_bias ? a.dt_bias[h] : 0.0);
      if (a.dt_softplus) dt = softplus_d(dt);
      if (a.has_dt_limit) dt = std::min(std::max(dt, a.dt_min), a.dt_max);
      for (int64_t p = 0; p < P; ++p) {
        const double xv = a.x[(t * H + h) * P + p];
        double* s = a.state + (h * P + p) * N;
        double y = 0.0;
        for (int64_t n = 0; n < N; ++n) {
          const double an = a.a_per_state ? a.A[h * N + n] : a.A[h];
          s[n] = s[n] * std::exp(dt * an) + dt * a.B[(t * G + g) * N + n] * xv;
          y += s[n] * a.C[(t * G + g) * N + n];
        }
        if (a.D) y += (a.d_per_p ? a.D[h * P + p] : a.D[h]) * xv;
        if (a.z) {
          const double zv = a.z[(t * H + h) * P + p];
          y *= zv * sigmoid_d(zv);
        }
        a.y[(t * H + h) * P + p] = y;
      }
    }
  }
  return Status::kSuccess;
}

Status linear_attention_f64(const double* q, const double* k, const double* v,
                            const double* slope, double scale, int64_t len, int64_t heads,
                            int64_t dk, int64_t dv, double* state, double* o) {
  if (len < 0 || heads <= 0 || dk <= 0 || dv <= 0 || slope == nullptr || state == nullptr ||
      (len > 0 && (q == nullptr || k == nullptr || v == nullptr || o == nullptr))) {
    return detail::set_error(Status::kInvalidArgument, "linear_attention_f64: bad argument");
  }
  for (int64_t t = 0; t < len; ++t) {
    for (int64_t h = 0; h < heads; ++h) {
      const double decay = std::exp(-slope[h]);
      double* s = state + h * dk * dv;
      const double* qt = q + (t * heads + h) * dk;
      const double* kt = k + (t * heads + h) * dk;
      const double* vt = v + (t * heads + h) * dv;
      for (int64_t i = 0; i < dk; ++i) {
        for (int64_t j = 0; j < dv; ++j) s[i * dv + j] = decay * s[i * dv + j] + kt[i] * vt[j];
      }
      for (int64_t j = 0; j < dv; ++j) {
        double acc = 0.0;
        for (int64_t i = 0; i < dk; ++i) acc += qt[i] * s[i * dv + j];
        o[(t * heads + h) * dv + j] = scale * acc;
      }
    }
  }
  return Status::kSuccess;
}

Status delta_rule_f64(const DeltaRuleRefArgs& a) {
  if (a.len < 0 || a.num_k_heads <= 0 || a.num_v_heads <= 0 || a.head_dim_k <= 0 ||
      a.head_dim_v <= 0 || a.num_v_heads % a.num_k_heads != 0 || a.state == nullptr ||
      (a.len > 0 && (a.q == nullptr || a.k == nullptr || a.v == nullptr || a.g == nullptr ||
                     a.beta == nullptr || a.o == nullptr))) {
    return detail::set_error(Status::kInvalidArgument, "delta_rule_f64: bad argument");
  }
  const int64_t Hk = a.num_k_heads, Hv = a.num_v_heads, Dk = a.head_dim_k, Dv = a.head_dim_v;
  std::vector<double> qn(static_cast<size_t>(Dk)), kn(static_cast<size_t>(Dk)), kv(static_cast<size_t>(Dv));
  for (int64_t t = 0; t < a.len; ++t) {
    for (int64_t h = 0; h < Hv; ++h) {
      const int64_t hk = h / (Hv / Hk);
      const double* qt = a.q + (t * Hk + hk) * Dk;
      const double* kt = a.k + (t * Hk + hk) * Dk;
      double sq = 0.0, sk = 0.0;
      for (int64_t i = 0; i < Dk; ++i) {
        sq += qt[i] * qt[i];
        sk += kt[i] * kt[i];
      }
      const double rq = a.use_qk_l2norm ? 1.0 / std::sqrt(sq + 1e-6) : 1.0;
      const double rk = a.use_qk_l2norm ? 1.0 / std::sqrt(sk + 1e-6) : 1.0;
      for (int64_t i = 0; i < Dk; ++i) {
        qn[static_cast<size_t>(i)] = qt[i] * rq * a.scale;
        kn[static_cast<size_t>(i)] = kt[i] * rk;
      }
      double* s = a.state + h * Dk * Dv;
      for (int64_t i = 0; i < Dk; ++i) {
        const double d = std::exp(a.per_channel ? a.g[(t * Hv + h) * Dk + i] : a.g[t * Hv + h]);
        for (int64_t j = 0; j < Dv; ++j) s[i * Dv + j] *= d;
      }
      const double beta = a.beta[t * Hv + h];
      const double* vt = a.v + (t * Hv + h) * Dv;
      for (int64_t j = 0; j < Dv; ++j) {
        double acc = 0.0;
        for (int64_t i = 0; i < Dk; ++i) acc += s[i * Dv + j] * kn[static_cast<size_t>(i)];
        kv[static_cast<size_t>(j)] = beta * (vt[j] - acc);
      }
      for (int64_t i = 0; i < Dk; ++i) {
        for (int64_t j = 0; j < Dv; ++j) s[i * Dv + j] += kn[static_cast<size_t>(i)] * kv[static_cast<size_t>(j)];
      }
      for (int64_t j = 0; j < Dv; ++j) {
        double acc = 0.0;
        for (int64_t i = 0; i < Dk; ++i) acc += s[i * Dv + j] * qn[static_cast<size_t>(i)];
        a.o[(t * Hv + h) * Dv + j] = acc;
      }
    }
  }
  return Status::kSuccess;
}

Status delta_rule_gating_f64(const double* a, const double* b, const double* A_log,
                             const double* dt_bias, double lower_bound, int64_t rows,
                             int64_t num_v_heads, int64_t per, double* g, double* beta) {
  if (rows < 0 || num_v_heads <= 0 || per <= 0 || A_log == nullptr || dt_bias == nullptr ||
      (rows > 0 && (a == nullptr || b == nullptr || g == nullptr || beta == nullptr))) {
    return detail::set_error(Status::kInvalidArgument, "delta_rule_gating_f64: bad argument");
  }
  for (int64_t t = 0; t < rows; ++t) {
    for (int64_t h = 0; h < num_v_heads; ++h) {
      const double rate = std::exp(A_log[h]);
      for (int64_t d = 0; d < per; ++d) {
        const int64_t i = (t * num_v_heads + h) * per + d;
        const double xg = a[i] + dt_bias[h * per + d];
        g[i] = lower_bound < 0.0 ? lower_bound * sigmoid_d(rate * xg) : -rate * softplus_d(xg);
      }
      beta[t * num_v_heads + h] = sigmoid_d(b[t * num_v_heads + h]);
    }
  }
  return Status::kSuccess;
}

}  // namespace kops::ref

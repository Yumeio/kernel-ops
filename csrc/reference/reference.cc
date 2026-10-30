// =============================================================================
// kernel-ops | csrc/reference/reference.cc
// fp64 host reference for the M1 golden-path kernels. Scalar loops, fixed
// evaluation order, no FMA contraction (build with -ffp-contract=off).
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>

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

}  // namespace kops::ref

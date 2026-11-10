#pragma once
// =============================================================================
// kernel-ops | tests/unit/golden.h
// Reader for the output-only KOPSGOLD format documented in tests/golden/README.md.
// No JSON parser, no third-party dependency: the manifest is a generation-time
// index; the test only needs the case file and the environment variable
// KOPS_GOLDEN_DIR.
// =============================================================================
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace kops_test {

struct Golden {
  uint32_t dtype = 0;      // 0 fp32, 1 bf16, 2 fp16, 3 u8
  uint32_t rank = 0;
  uint32_t flags = 0;      // bit0 HAS_LSE, bit1 HAS_SINK
  uint64_t seed = 0;
  uint32_t codec = 0;
  uint32_t topk = 0;
  float sink = 0.0f;
  int64_t shape[8] = {};
  std::vector<uint8_t> output;
  std::vector<float> lse;
  bool has_lse() const { return (flags & 1u) != 0u; }
  bool has_sink() const { return (flags & 2u) != 0u; }
  size_t dtype_size() const {
    return dtype == 0 ? 4u : (dtype == 3 ? 1u : 2u);
  }
  int64_t numel() const {
    int64_t n = rank == 0 ? 1 : 1;
    for (uint32_t i = 0; i < rank && i < 8; ++i) n *= shape[i];
    return n;
  }
};

inline uint16_t golden_u16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}
inline uint32_t golden_u32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
inline uint64_t golden_u64(const uint8_t* p) {
  return static_cast<uint64_t>(golden_u32(p)) |
         (static_cast<uint64_t>(golden_u32(p + 4)) << 32);
}

/// Load <KOPS_GOLDEN_DIR>/cases/<id>.kopsgold. Returns false when the directory,
/// file, magic or section bounds are invalid; prints a short reason to stderr.
inline bool load_golden(const char* id, Golden* out, const char** reason = nullptr) {
  auto fail = [&](const char* why) {
    if (reason != nullptr) *reason = why;
    return false;
  };
  if (out == nullptr || id == nullptr) return fail("null argument");
  const char* root = std::getenv("KOPS_GOLDEN_DIR");
  if (root == nullptr || *root == '\0') return fail("KOPS_GOLDEN_DIR is not set");
  const std::string path = std::string(root) + "/cases/" + id + ".kopsgold";
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return fail("cannot open golden");
  const std::streamsize size = f.tellg();
  if (size < 96) return fail("golden shorter than header");
  f.seekg(0);
  std::vector<uint8_t> file(static_cast<size_t>(size));
  if (!f.read(reinterpret_cast<char*>(file.data()), size)) return fail("golden read failed");
  const uint8_t* p = file.data();
  if (std::memcmp(p, "KOPSGOLD", 8) != 0) return fail("bad magic");
  const uint16_t version = golden_u16(p + 8);
  const uint16_t header_bytes = golden_u16(p + 10);
  if (version != 1) return fail("unsupported golden version");
  const uint32_t rank = golden_u32(p + 16);
  if (rank > 8 || header_bytes != 96 + 8 * rank) return fail("bad rank/header size");
  if (static_cast<size_t>(header_bytes) > file.size()) return fail("truncated shape array");
  Golden g;
  g.dtype = golden_u32(p + 12);
  g.rank = rank;
  g.flags = golden_u32(p + 20);
  g.seed = golden_u64(p + 24);
  g.codec = golden_u32(p + 32);
  g.topk = golden_u32(p + 36);
  g.sink = 0.0f;
  std::memcpy(&g.sink, p + 40, sizeof(float));
  for (uint32_t i = 0; i < rank; ++i) g.shape[i] = static_cast<int64_t>(golden_u64(p + 96 + 8 * i));
  const uint64_t out_off = golden_u64(p + 48);
  const uint64_t out_bytes = golden_u64(p + 56);
  const uint64_t lse_off = golden_u64(p + 64);
  const uint64_t lse_bytes = golden_u64(p + 72);
  const int64_t expect_out = g.numel() * static_cast<int64_t>(g.dtype_size());
  if (expect_out < 0 || out_bytes != static_cast<uint64_t>(expect_out)) return fail("output size");
  if (out_off < header_bytes || out_off + out_bytes > static_cast<uint64_t>(file.size())) {
    return fail("output section out of bounds");
  }
  g.output.assign(file.begin() + static_cast<std::ptrdiff_t>(out_off),
                  file.begin() + static_cast<std::ptrdiff_t>(out_off + out_bytes));
  if (g.has_lse()) {
    if (lse_bytes == 0 || lse_off < header_bytes || lse_off + lse_bytes > static_cast<uint64_t>(file.size()) ||
        (lse_bytes % 4u) != 0u) {
      return fail("LSE section out of bounds");
    }
    g.lse.resize(lse_bytes / 4u);
    std::memcpy(g.lse.data(), file.data() + lse_off, lse_bytes);
  }
  *out = std::move(g);
  return true;
}

/// SplitMix64, the deterministic input generator named in tests/golden/README.md.
struct SplitMix64 {
  uint64_t state;
  explicit SplitMix64(uint64_t seed) : state(seed) {}
  uint64_t next() {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  float unit() {
    return static_cast<float>((next() >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f;
  }
};

}  // namespace kops_test

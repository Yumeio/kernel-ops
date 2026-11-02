#pragma once
// =============================================================================
// kernel-ops | csrc/common/vec.cuh
// Aligned register bundles, 128/256-bit vector loads and stores and host-side
// vector-width selection.
// Requires: common/macros.cuh.
//   Included by common/device.cuh; kernels get Vec / load_vec / store_vec from there.
//
// A 32-byte bundle moves with one ld/st.global.v8.u32 on the Blackwell "a"
// targets (the compiler's own __CUDA_ARCH_FEAT_SM*_ALL macros select it, no
// KOPS_HAS_* flag) and with two 128-bit accesses everywhere else; the bytes
// moved are identical.
// =============================================================================
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace kops {

/// Portable widest bundle (one LDG.128 / STG.128) and the Blackwell-a widest bundle.
inline constexpr int kMaxVecBytes = 16;
inline constexpr int kMaxVecBytes256 = 32;

/// Elements of T that fit in kMaxVecBytes.
template <typename T>
inline constexpr int max_vec_size_v = kMaxVecBytes / static_cast<int>(sizeof(T));

namespace detail {

#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL) || \
    defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
#define KOPS_DETAIL_LDST256 1
#else
#define KOPS_DETAIL_LDST256 0
#endif

/// 32-byte register bundle used to move a 256-bit value through PTX.
struct alignas(kMaxVecBytes256) Bits256 {
  uint32_t w[8];
};

#if defined(__CUDACC__)
/// 32-byte global load. Plain (non-.nc) so it stays valid for in-place kernels.
KOPS_DEVICE Bits256 load_bits256(const void* ptr) {
  Bits256 r;
#if KOPS_DETAIL_LDST256
  asm volatile("ld.global.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
               : "=r"(r.w[0]), "=r"(r.w[1]), "=r"(r.w[2]), "=r"(r.w[3]), "=r"(r.w[4]), "=r"(r.w[5]),
                 "=r"(r.w[6]), "=r"(r.w[7])
               : "l"(ptr)
               : "memory");
#else
  const uint4* q = reinterpret_cast<const uint4*>(ptr);
  const uint4 lo = q[0];
  const uint4 hi = q[1];
  r.w[0] = lo.x; r.w[1] = lo.y; r.w[2] = lo.z; r.w[3] = lo.w;
  r.w[4] = hi.x; r.w[5] = hi.y; r.w[6] = hi.z; r.w[7] = hi.w;
#endif
  return r;
}

/// 32-byte global store.
KOPS_DEVICE void store_bits256(void* ptr, const Bits256& v) {
#if KOPS_DETAIL_LDST256
  asm volatile("st.global.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
               :
               : "l"(ptr), "r"(v.w[0]), "r"(v.w[1]), "r"(v.w[2]), "r"(v.w[3]), "r"(v.w[4]), "r"(v.w[5]),
                 "r"(v.w[6]), "r"(v.w[7])
               : "memory");
#else
  uint4* q = reinterpret_cast<uint4*>(ptr);
  q[0] = make_uint4(v.w[0], v.w[1], v.w[2], v.w[3]);
  q[1] = make_uint4(v.w[4], v.w[5], v.w[6], v.w[7]);
#endif
}
#endif  // __CUDACC__

}  // namespace detail

/// Register vector of N elements of T with natural alignment, capped at 32 B.
/// Bundles of up to 32 B go through load_vec / store_vec; a wider bundle (for
/// example Vec<float, 16>) only holds register values. Elements are not
/// initialised by default.
template <typename T, int N>
struct alignas((sizeof(T) * N) > kMaxVecBytes256 ? kMaxVecBytes256 : (sizeof(T) * N)) Vec {
  static_assert(N > 0 && (N & (N - 1)) == 0, "Vec: N must be a power of 2");
  T data[N];
  KOPS_HOST_DEVICE_INLINE T& operator[](int i) { return data[i]; }
  KOPS_HOST_DEVICE_INLINE const T& operator[](int i) const { return data[i]; }
};

#if defined(__CUDACC__)
/// Load an aligned bundle. @pre ptr is aligned to sizeof(T) * N bytes and addresses N readable
/// elements; a 32-byte bundle must address global memory. No bounds or alignment check.
template <typename T, int N>
KOPS_DEVICE Vec<T, N> load_vec(const T* __restrict__ ptr) {
  static_assert(sizeof(Vec<T, N>) <= kMaxVecBytes256, "load_vec: wider than 32B");
  if constexpr (sizeof(Vec<T, N>) > kMaxVecBytes) {
    static_assert(sizeof(Vec<T, N>) == sizeof(detail::Bits256), "load_vec: unsupported bundle size");
    const detail::Bits256 bits = detail::load_bits256(ptr);
    Vec<T, N> out;
    memcpy(&out, &bits, sizeof(out));
    return out;
  } else {
    return *reinterpret_cast<const Vec<T, N>*>(ptr);
  }
}

/// Store an aligned bundle. Same preconditions as load_vec.
template <typename T, int N>
KOPS_DEVICE void store_vec(T* __restrict__ ptr, const Vec<T, N>& value) {
  static_assert(sizeof(Vec<T, N>) <= kMaxVecBytes256, "store_vec: wider than 32B");
  if constexpr (sizeof(Vec<T, N>) > kMaxVecBytes) {
    static_assert(sizeof(Vec<T, N>) == sizeof(detail::Bits256), "store_vec: unsupported bundle size");
    detail::Bits256 bits;
    memcpy(&bits, &value, sizeof(bits));
    detail::store_bits256(ptr, bits);
  } else {
    *reinterpret_cast<Vec<T, N>*>(ptr) = value;
  }
}
#endif  // __CUDACC__

/// True when p is a multiple of `bytes` (null passes; nothing is dereferenced). @pre bytes != 0.
KOPS_HOST_DEVICE_INLINE bool is_aligned(const void* p, size_t bytes) {
  return reinterpret_cast<uintptr_t>(p) % bytes == 0;
}

/// Largest power-of-two element count with sizeof(T) * count <= max_bytes that divides n and keeps
/// every pointer sizeof(T) * count aligned; falls back to 1. For several rows pass the row length
/// and make every row stride a multiple of the result. n == 0 implies no access validity.
template <typename T, typename... Ptrs>
inline int pick_vec_size_up_to(int max_bytes, int64_t n, const Ptrs*... ptrs) {
  int vec = max_bytes / static_cast<int>(sizeof(T));
  if (vec < 1) vec = 1;
  while (vec & (vec - 1)) vec &= vec - 1;
  while (vec > 1) {
    [[maybe_unused]] const size_t bytes = sizeof(T) * static_cast<size_t>(vec);
    if (n % vec == 0 && (is_aligned(ptrs, bytes) && ...)) break;
    vec /= 2;
  }
  return vec;
}

/// pick_vec_size_up_to with the portable 16-byte cap.
template <typename T, typename... Ptrs>
inline int pick_vec_size(int64_t n, const Ptrs*... ptrs) {
  return pick_vec_size_up_to<T>(kMaxVecBytes, n, ptrs...);
}

}  // namespace kops

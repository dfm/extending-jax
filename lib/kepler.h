// This header defines the actual algorithm for our op. It is reused in cpu_ops.cc and
// kernels.cc.cu to expose this as a XLA custom call. The details aren't too important
// except that directly implementing this algorithm as a higher-level JAX function
// probably wouldn't be very efficient. That being said, this is not meant as a
// particularly efficient or robust implementation. It's just here to demonstrate the
// infrastructure required to extend JAX.

#ifndef _KEPLER_JAX_KEPLER_H_
#define _KEPLER_JAX_KEPLER_H_

#include <cmath>

namespace kepler_jax {

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifdef __CUDACC__
#define KEPLER_JAX_INLINE_OR_DEVICE __host__ __device__
#else
#define KEPLER_JAX_INLINE_OR_DEVICE inline

template <typename T>
inline void sincos(const T& x, T* sx, T* cx) {
  *sx = sin(x);
  *cx = cos(x);
}
#endif

template <typename T>
KEPLER_JAX_INLINE_OR_DEVICE void compute_eccentric_anomaly(const T& mean_anom, const T& ecc,
                                                           T* sin_ecc_anom, T* cos_ecc_anom) {
  const T tol = 1e-12;
  T g, E = (mean_anom < M_PI) ? mean_anom + 0.85 * ecc : mean_anom - 0.85 * ecc;
  for (int i = 0; i < 20; ++i) {
    sincos(E, sin_ecc_anom, cos_ecc_anom);
    g = E - ecc * (*sin_ecc_anom) - mean_anom;
    if (fabs(g) <= tol) return;
    E -= g / (1 - ecc * (*cos_ecc_anom));
  }
}

}  // namespace kepler_jax

#endif
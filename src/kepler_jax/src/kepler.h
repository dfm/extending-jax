#ifndef _KEPLER_JAX_KEPLER_H_
#define _KEPLER_JAX_KEPLER_H_

#include <cmath>
#include <cstdint>

namespace kepler_jax {

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

template <typename Scalar>
KEPLER_JAX_INLINE_OR_DEVICE void compute_eccentric_anomaly(const Scalar& mean_anom,
                                                           const Scalar& ecc, Scalar* sin_ecc_anom,
                                                           Scalar* cos_ecc_anom) {
  Scalar E, g, gp, gpp, gppp, d_3, d_4;

  // Initial guess
  E = (mean_anom < M_PI) ? mean_anom + 0.85 * ecc : mean_anom - 0.85 * ecc;

  // Iterate high order Householder's method for 10 iterations
  for (int i = 0; i < 10; ++i) {
    sincos(E, sin_ecc_anom, cos_ecc_anom);

    gpp = ecc * (*sin_ecc_anom);
    g = E - gpp - mean_anom;
    // if (fabs(g) < tol) {
    //   return;
    // }
    gp = 1 - ecc * (*cos_ecc_anom);
    gppp = 1 - gp;

    d_3 = -g / (gp - 0.5 * g * gpp / gp);
    d_4 = -g / (gp + 0.5 * d_3 * (gpp + d_3 * gppp / 6));
    E -= g / (gp + 0.5 * d_4 * (gpp + d_4 * (gppp / 6 - d_4 * gpp / 24)));
  }
  // If we get here, we didn't converge
}

enum Type { F32 = sizeof(float), F64 = sizeof(double) };
struct KeplerDescriptor {
  Type dtype;
  std::int64_t size;
};

}  // namespace kepler_jax

#endif
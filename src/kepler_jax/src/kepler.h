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

template <typename Scalar>
KEPLER_JAX_INLINE_OR_DEVICE Scalar get_starter(const Scalar& mean_anom, const Scalar& ecc) {
  // mean_anom must be in the range [0, pi)
  const Scalar f1 = 3 * M_PI / (M_PI - 6 / M_PI);
  const Scalar f2 = 1.6 / (M_PI - 6 / M_PI);
  const Scalar ome = 1 - ecc;
  const Scalar M2 = mean_anom * mean_anom;
  const Scalar alpha = f1 + f2 * (M_PI - mean_anom) / (1 + ecc);
  const Scalar d = 3 * ome + alpha * ecc;
  const Scalar alphad = alpha * d;
  const Scalar r = (3 * alphad * (d - ome) + M2) * mean_anom;
  const Scalar q = 2 * alphad * ome - M2;
  const Scalar q2 = q * q;
  const Scalar w = pow(fabs(r) + sqrt(q2 * q + r * r), 2.0 / 3);
  return (2 * r * w / (w * w + w * q + q2) + mean_anom) / d;
}

template <typename Scalar>
KEPLER_JAX_INLINE_OR_DEVICE Scalar refine_estimate(const Scalar& mean_anom, const Scalar& ecc,
                                                   const Scalar& E) {
  Scalar sE, cE;
  sincos(E, &sE, &cE);

  const Scalar f_2 = ecc * sE;
  const Scalar f_0 = E - f_2 - mean_anom;
  const Scalar f_1 = 1 - ecc * cE;
  const Scalar f_3 = 1 - f_1;

  const Scalar d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1);
  const Scalar d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6);
  const Scalar d_42 = d_4 * d_4;
  const Scalar dE = f_0 / (f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24);

  return E - dE;
}

template <typename Scalar>
KEPLER_JAX_INLINE_OR_DEVICE void compute_eccentric_anomaly(const Scalar& mean_anom,
                                                           const Scalar& ecc, Scalar* sin_ecc_anom,
                                                           Scalar* cos_ecc_anom) {
  // mean_anom must be in the range [0, pi) but this is not checked
  Scalar E = get_starter(mean_anom, ecc);
  E = refine_estimate(mean_anom, ecc, E);
  sincos(E, sin_ecc_anom, cos_ecc_anom);
}

}  // namespace kepler_jax

#endif
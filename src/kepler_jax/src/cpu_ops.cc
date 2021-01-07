#include <cstdint>

#include "kepler.h"
#include "pybind11_kernel_helpers.h"

namespace {

void kepler(void *out_tuple, const void **in) {
  // The inputs are pretty straightforward
  const std::int64_t N = *reinterpret_cast<const std::int64_t *>(in[0]);
  const double *mean_anom = reinterpret_cast<const double *>(in[1]);
  const double *ecc = reinterpret_cast<const double *>(in[2]);

  // With multiple outputs, the output pointer will be a list of pointers
  void **out = reinterpret_cast<void **>(out_tuple);
  double *sin_ecc_anom = reinterpret_cast<double *>(out[0]);
  double *cos_ecc_anom = reinterpret_cast<double *>(out[1]);

  for (std::int64_t n = 0; n < N; ++n) {
    kepler_jax::compute_eccentric_anomaly(mean_anom[n], ecc[n], sin_ecc_anom + n,
                                          cos_ecc_anom + n);
  }
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["kepler"] = kepler_jax::EncapsulateFunction(kepler);
  return dict;
}

PYBIND11_MODULE(cpu_ops, m) { m.def("registrations", &Registrations); }

}  // namespace

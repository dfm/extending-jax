#include "kernels.h"
#include "pybind11_kernel_helpers.h"

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["gpu_kepler"] = kepler_jax::EncapsulateFunction(kepler_jax::gpu_kepler);
  return dict;
}

PYBIND11_MODULE(gpu_ops, m) { m.def("registrations", &Registrations); }

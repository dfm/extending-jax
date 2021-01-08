#include "kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace kepler_jax;

namespace {

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["gpu_kepler"] = EncapsulateFunction(kepler_jax::gpu_kepler);
  return dict;
}

PYBIND11_MODULE(gpu_ops, m) { m.def("registrations", &Registrations); }

}  // namespace

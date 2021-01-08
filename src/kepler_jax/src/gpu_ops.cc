#include "kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace kepler_jax;

namespace {
pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["gpu_kepler_f32"] = EncapsulateFunction(gpu_kepler_f32);
  dict["gpu_kepler_f64"] = EncapsulateFunction(gpu_kepler_f64);
  return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
  m.def("registrations", &Registrations);
  m.def("build_kepler_descriptor",
        [](std::int64_t size) { return PackDescriptor(KeplerDescriptor{size}); });
}
}  // namespace

#include "kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace kepler_jax;

namespace {
pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["gpu_kepler"] = EncapsulateFunction(gpu_kepler);
  return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
  m.def("registrations", &Registrations);
  pybind11::enum_<Type>(m, "Type")
      .value("float32", Type::F32)
      .value("float64", Type::F64)
      .export_values();
  m.def("build_kepler_descriptor", [](Type dtype, std::int64_t size) {
    return PackDescriptor(KeplerDescriptor{dtype, size});
  });
}
}  // namespace

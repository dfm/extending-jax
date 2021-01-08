#include "kepler.h"
#include "pybind11_kernel_helpers.h"

using namespace kepler_jax;

namespace {

template <typename T>
inline void apply_kepler(const std::int64_t size, void **out, const void **in) {
  const T *mean_anom = reinterpret_cast<const T *>(in[1]);
  const T *ecc = reinterpret_cast<const T *>(in[2]);
  T *sin_ecc_anom = reinterpret_cast<T *>(out[0]);
  T *cos_ecc_anom = reinterpret_cast<T *>(out[1]);
  for (std::int64_t n = 0; n < size; ++n) {
    compute_eccentric_anomaly(mean_anom[n], ecc[n], sin_ecc_anom + n, cos_ecc_anom + n);
  }
}

void cpu_kepler(void *out_tuple, const void **in) {
  // Our first input is the descriptor
  const KeplerDescriptor &d =
      *UnpackDescriptor<KeplerDescriptor>(bit_cast<const char *>(in[0]), sizeof(KeplerDescriptor));
  const std::int64_t size = d.size;

  // The output is stored as a list of pointers since we have multiple outputs
  void **out = reinterpret_cast<void **>(out_tuple);

  // Dispatch based on the data type
  switch (d.dtype) {
    case kepler_jax::Type::F32:
      apply_kepler<float>(size, out, in);
      break;
    case kepler_jax::Type::F64:
      apply_kepler<double>(size, out, in);
      break;
  }
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cpu_kepler"] = EncapsulateFunction(cpu_kepler);
  return dict;
}

PYBIND11_MODULE(cpu_ops, m) {
  m.def("registrations", &Registrations);

  pybind11::enum_<Type>(m, "Type")
      .value("float32", Type::F32)
      .value("float64", Type::F64)
      .export_values();

  m.def("build_kepler_descriptor", [](Type dtype, std::int64_t size) {
    return PackDescriptor(kepler_jax::KeplerDescriptor{dtype, size});
  });
}

}  // namespace

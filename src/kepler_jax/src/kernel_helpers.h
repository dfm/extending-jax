#ifndef _KEPLER_JAX_KERNEL_HELPERS_H_
#define _KEPLER_JAX_KERNEL_HELPERS_H_

#include <stdexcept>
#include <string>

namespace kepler_jax {

template <typename T>
std::string PackDescriptorAsString(const T& descriptor) {
  return std::string((const char*)(&descriptor), sizeof(T));
}

template <typename T>
const T* UnpackDescriptor(const char* opaque, std::size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw std::runtime_error("Invalid opaque object size");
  }
  return (const T*)opaque;
}

}  // namespace kepler_jax

#endif

#ifndef _KEPLER_JAX_KERNELS_H_
#define _KEPLER_JAX_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace kepler_jax {

enum Type { F32 = sizeof(float), F64 = sizeof(double) };
struct KeplerDescriptor {
  Type dtype;
  std::int64_t size;
};

void gpu_kepler(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);

}  // namespace kepler_jax

#endif